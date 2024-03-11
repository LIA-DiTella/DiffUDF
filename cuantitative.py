import torch
from train import setup_train
import os
import gc
from pytorch3d.loss import chamfer_distance
import trimesh as tm
import numpy as np
import open3d as o3d

def metrics( mesh, pointcloud, norm, cuda_device ):
    cd, nc = chamfer_distance( 
            x = torch.from_numpy(np.asarray(mesh.vertices)).float()[None,...].to(cuda_device), 
            y = torch.from_numpy(np.asarray(pointcloud.points)).float()[None,...].to(cuda_device), 
            x_normals = torch.from_numpy(np.asarray(mesh.vertex_normals)).float()[None,...].to(cuda_device),
            y_normals = torch.from_numpy(np.asarray(pointcloud.normals)).float()[None,...].to(cuda_device),
            norm=norm
        )
    
    return cd.cpu().numpy(), nc.cpu().numpy()

if __name__=='__main__':
    net_width = 256
    net_depth = 8
    layer_nodes = [net_width] * net_depth

    dataset = 'data/deepfashion/'
    outfolder = f'results/df_subset/'
    cuda_device = 0

    if not os.path.exists(outfolder):
        os.mkdir(outfolder)

    exp_config = {
        "num_epochs": 3000,
        "s1_epochs": 2000,
        "warmup_epochs": 1000,
        "dataset": "...",
        "batch_size": 30000,
        "sampling_percentiles": [0.333, 0.666],
        "batches_per_epoch": 1,
        "checkpoint_path": outfolder,
        "experiment_name": "...",
        "epochs_to_checkpoint": 8001,
        "gt_mode": "tanh",
        "loss_s1_weights": [ 1e4, 1e4, 1e4, 1e3 ],
        "loss_s2_weights": [ 1e5, 1e5 ],
        "alpha": 10,
        "optimizer": {
            "type": "adam",
            "lr_s1": 1e-5,
            "lr_s2": 1e-7
        },
        "network": {
            "hidden_layer_nodes": layer_nodes,
            "w0": 30,
            "pretrained_dict": "None"
        },
        "resolution": 256
    }


    with open(os.path.join(outfolder, 'results.csv'), 'w+') as result_file:
        result_file.write('mesh,time,L1CD_CAP,L2CD_CAP,NC_CAP,L1CD_MU,L2CD_MU,NC_MU\n')
    
    for it, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset)):

        try:
            dataset_index = [i for i,file in enumerate(filenames) if file[-7:] == '_pc.ply'][0]
            gt_index = [i for i,file in enumerate(filenames) if file[-6:] == '_t.obj'][0]
        except:
            continue

        # comparo con la nube de puntos y no con los vertices de la malla original... tiene mas sentido
        dataset_file = os.path.join(dirpath, filenames[dataset_index])

        print(f'Training for {filenames[gt_index]}')

        experiment_name = dirpath[dirpath.rfind('/')+1:]

        exp_config['dataset'] = dataset_file[:-7]
        exp_config['experiment_name'] = experiment_name

        if os.path.exists(os.path.join(outfolder, experiment_name)):
            print(f'Skipping {experiment_name}')
            continue

        training_time, (meshMU, meshCAP) = setup_train( exp_config, cuda_device)

        torch.cuda.empty_cache()
        gc.collect()

        print('Computing chamfer distances...')
        gt_pc = o3d.io.read_point_cloud(dataset_file)

        time = training_time
        cap_mesh = meshCAP.as_open3d
        mu_mesh = meshMU.as_open3d

        cap_mesh.compute_vertex_normals(normalized=True)
        mu_mesh.compute_vertex_normals(normalized=True)

        L1CD_CAP, NC_CAP = metrics( cap_mesh, gt_pc, norm=1, cuda_device=cuda_device )
        L2CD_CAP, _ = metrics( cap_mesh, gt_pc, norm=2, cuda_device=cuda_device )
        L1CD_MU, NC_MU = metrics( mu_mesh, gt_pc, norm=1, cuda_device=cuda_device )
        L2CD_MU, _ = metrics( mu_mesh, gt_pc, norm=2, cuda_device=cuda_device )

        with open(os.path.join(outfolder, 'results.csv'), 'a') as result_file:
            result_file.write(f'{experiment_name},{time},{L1CD_CAP},{L2CD_CAP},{NC_CAP},{L1CD_MU},{L2CD_MU},{NC_MU}\n')


    

    