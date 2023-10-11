import torch
from train import setup_train
import os
import gc
from pytorch3d.loss import chamfer_distance
import trimesh as tm

if __name__=='__main__':
    net_width = 256
    net_depth = 8
    layer_nodes = [net_width] * net_depth

    dataset = 'data/DF_subset/'
    outfolder = f'results/siren'
    cuda_device = 1

    if not os.path.exists(outfolder):
        os.mkdir(outfolder)

    exp_config = {
        "description": "SIREN Learning of complete test mesh using SDF values and non-uniform sampling.",
        "num_epochs": 1500,
        "warmup_epochs": 0,
        "sampling_opts": {
            "curvature_iteration_fractions": [0.2, 0.4, 0.4],
            "curvature_percentile_thresholds": [0.6, 0.85]
        },
        "dataset": "...",
        "batch_size": 30000,
        "sampling_percentiles": [0.33, 0.66],
        "batches_per_epoch": 1,
        "checkpoint_path": "...",
        "experiment_name": "siren",
        "epochs_to_checkpoint": 4000,
        "gt_mode": "siren",
        "loss_weights": [ 5e3, 5e2, 5e2, 5e1  ],
        "optimizer": {
            "type": "adam",
            "lr": 1e-4
        },
        "network": {
            "hidden_layer_nodes": [256,256,256,256,256,256,256,256],
            "w0": 30,
            "pretrained_dict": "None"
        },
        "resolution": 256
    }



    with open(os.path.join(outfolder, 'results.csv'), 'w+') as result_file:
        result_file.write('mesh,time,L1CD,L2CD\n')

    for dirpath, dirnames, filenames in os.walk(dataset):
        try:
            dataset_index = [i for i,file in enumerate(filenames) if file[-6:] == '_p.ply'][0]
            gt_index = [i for i,file in enumerate(filenames) if file[-6:] == '_t.obj'][0]
        except:
            continue

        gt_file = os.path.join(dirpath, filenames[gt_index])
        dataset_file = os.path.join(dirpath, filenames[dataset_index])

        print(f'Training for {filenames[gt_index]}')

        experiment_name = dirpath[dirpath.rfind('/')+1:]

        exp_config['dataset'] = dataset_file
        exp_config['experiment_name'] = experiment_name

        if os.path.exists(os.path.join(outfolder, experiment_name)):
            print(f'Skipping {experiment_name}')
            continue

        training_time, mesh = setup_train( exp_config, cuda_device)

        torch.cuda.empty_cache()
        gc.collect()

        print('Computing chamfer distances...')
        gt_pc = tm.load_mesh( gt_file )

        time = training_time
        L1CD = chamfer_distance( torch.from_numpy(mesh.vertices).float()[None,...].to(cuda_device), torch.from_numpy(gt_pc.vertices).float()[None,...].to(cuda_device), norm=1)[0].cpu().numpy()
        L2CD = chamfer_distance( torch.from_numpy(mesh.vertices).float()[None,...].to(cuda_device), torch.from_numpy(gt_pc.vertices).float()[None,...].to(cuda_device), norm=2)[0].cpu().numpy()

        with open(os.path.join(outfolder, 'results.csv'), 'a') as result_file:
            result_file.write(f'{experiment_name},{time},{L1CD},{L2CD}\n')


    

    