import torch
from train import setup_train
import os
import gc
from pytorch3d.loss import chamfer_distance
import pandas as pd
import trimesh as tm

if __name__=='__main__':
    net_width = 256
    net_depth = 8
    layer_nodes = [net_width] * net_depth

    dataset = 'data/DF_subset/'
    outfolder = f'results/Ablation_01'
    cuda_device = 0

    if not os.path.exists(outfolder):
        os.mkdir(outfolder)

    exp_config = {
        "num_epochs": 3000,
        "s1_epochs": 2000,
        "sampling_opts": {
            "curvature_iteration_fractions": [0.2, 0.4, 0.4],
            "curvature_percentile_thresholds": [0.6, 0.85]
        },
        "dataset": "...",
        "batch_size": 20000,
        "sampling_percentiles": [0.5, 0.5, 0.0],
        "batches_per_epoch": 1,
        "checkpoint_path": outfolder,
        "experiment_name": "...",
        "epochs_to_checkpoint": 4000,
        "gt_mode": "tanh",
        "loss_s1_weights": [ 1e4, 1e4, 1e4, 1e3, 1e2 ],
        "loss_s2_weights": [ 1e5, 1e5 ],
        "alpha": 100,
        "optimizer": {
            "type": "adam",
            "lr_s1": 1e-4,
            "lr_s2": 1e-6
        },
        "network": {
            "hidden_layer_nodes": layer_nodes,
            "w0": 30,
            "pretrained_dict": "None"
        },
        "resolution": 256
    }

    results = {
        'time': {},
        'L1CD_CAP': {},
        'L2CD_CAP': {},
        'L1CD_MU': {},
        'L2CD_MU': {}
    }

    for dirpath, dirnames, filenames in os.walk(dataset):
        try:
            dataset_index = [i for i,file in enumerate(filenames) if file[-6:] == '_p.ply'][0]
            gt_index = [i for i,file in enumerate(filenames) if file[-4:] == '.obj'][0]
        except:
            continue

        gt_file = os.path.join(dirpath, filenames[gt_index])
        dataset_file = os.path.join(dirpath, filenames[dataset_index])

        print(f'Training for {filenames[gt_index]}')

        exp_config['dataset'] = dataset_file
        exp_config['experiment_name'] = dirpath[dirpath.rfind('/')+1:] #filenames[gt_index][:filenames[gt_index].rfind('.')]

        training_time, (meshMU, meshCAP) = setup_train( exp_config, cuda_device)

        torch.cuda.empty_cache()
        gc.collect()

        print('Computing chamfer distances...')
        gt_pc = tm.load_mesh( gt_file )

        results['time'][filenames[gt_index][:filenames[gt_index].rfind('.')]] = training_time
        results['L1CD_CAP'][filenames[gt_index][:filenames[gt_index].rfind('.')]] = chamfer_distance( torch.from_numpy(meshCAP.vertices).float()[None,...], torch.from_numpy(gt_pc.vertices).float()[None,...], norm=1)[0].numpy()
        results['L2CD_CAP'][filenames[gt_index][:filenames[gt_index].rfind('.')]] = chamfer_distance( torch.from_numpy(meshCAP.vertices).float()[None,...], torch.from_numpy(gt_pc.vertices).float()[None,...], norm=2)[0].numpy()
        results['L1CD_MU'][filenames[gt_index][:filenames[gt_index].rfind('.')]] = chamfer_distance( torch.from_numpy(meshMU.vertices).float()[None,...], torch.from_numpy(gt_pc.vertices).float()[None,...], norm=1)[0].numpy()
        results['L2CD_MU'][filenames[gt_index][:filenames[gt_index].rfind('.')]] = chamfer_distance( torch.from_numpy(meshMU.vertices).float()[None,...], torch.from_numpy(gt_pc.vertices).float()[None,...], norm=2)[0].numpy()

    pd.DataFrame(results).to_csv(os.path.join(outfolder, 'results.csv'))



    

    