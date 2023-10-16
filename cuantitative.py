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

    dataset = 'data/multi_garment/'
    outfolder = f'results/best_mg'
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
        "epochs_to_checkpoint": 4000,
        "gt_mode": "tanh",
        "loss_s1_weights": [ 1e4, 1e4, 1e4, 1e3 ],
        "loss_s2_weights": [ 1e5, 1e5 ],
        "alpha": 100,
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
        result_file.write('mesh,time,L1CD_CAP,L2CD_CAP,L1CD_MU,L2CD_MU\n')
    
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
        gt_pc = tm.load( dataset_file )

        time = training_time
        L1CD_CAP = chamfer_distance( torch.from_numpy(meshCAP.vertices).float()[None,...].to(cuda_device), torch.from_numpy(gt_pc.vertices).float()[None,...].to(cuda_device), norm=1)[0].cpu().numpy()
        L2CD_CAP = chamfer_distance( torch.from_numpy(meshCAP.vertices).float()[None,...].to(cuda_device), torch.from_numpy(gt_pc.vertices).float()[None,...].to(cuda_device), norm=2)[0].cpu().numpy()
        L1CD_MU = chamfer_distance( torch.from_numpy(meshMU.vertices).float()[None,...].to(cuda_device), torch.from_numpy(gt_pc.vertices).float()[None,...].to(cuda_device), norm=1)[0].cpu().numpy()
        L2CD_MU = chamfer_distance( torch.from_numpy(meshMU.vertices).float()[None,...].to(cuda_device), torch.from_numpy(gt_pc.vertices).float()[None,...].to(cuda_device), norm=2)[0].cpu().numpy()

        with open(os.path.join(outfolder, 'results.csv'), 'a') as result_file:
            result_file.write(f'{experiment_name},{time},{L1CD_CAP},{L2CD_CAP},{L1CD_MU},{L2CD_MU}\n')


    

    