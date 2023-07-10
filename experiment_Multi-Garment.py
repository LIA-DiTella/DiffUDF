import torch
import numpy as np
from pytorch3d.loss import chamfer_distance
from train import setup_train
from generate_pc import generate_pc
import os
import json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dense point cloud from trained model')
    parser.add_argument('-d', '--device', type=int, default=0, help='torch device')
    parser.add_argument('-m', '--meshes', type=int, default=1000, help='amount of meshes')
    parser.add_argument('-n', '--nsamples', type=int, default=20000, help='number of samples')
    parser.add_argument('-t', '--surf_thresh', type=float, default=1e-5, help='on surface threshold')

    args = parser.parse_args()

    device_torch = torch.device(args.device)

    training_config = {
        "description": "SIREN Learning of complete test mesh using SDF values and non-uniform sampling.",
        "num_epochs": 2,
        "sampling_opts": {
            "curvature_iteration_fractions": [0.2, 0.4, 0.4],
            "curvature_percentile_thresholds": [0.6, 0.85]
        },
        "dataset": "data/segmented/Multi-Garment_dataset/",
        "batch_size": 20000,
        "sampling_percentiles": [0.5, 0.45, 0.05],
        "batches_per_epoch": 3,
        "checkpoint_path": "results/ropa/",
        "experiment_name": "test_",
        "epochs_to_checkpoint": 300,
        "loss": "loss_ndf",
        "loss_weights": [5e3, 5e2, 5e1, 5e2, 0.5],
        "optimizer": {
            "type": "adam",
            "lr": 1e-4
        },
        "network": {
            "hidden_layer_nodes": [256, 256, 256, 256],
            "w0": 30
        }
    }

    corte = 0
    for dirpath, dnames, fnames in os.walk("data/segmented/Multi-Garment_dataset/"):
        for file in fnames:
            if file.endswith(".json") and corte < args.meshes:
                parameter_dict = training_config.copy()
                parameter_dict['dataset'] += file
                parameter_dict['checkpoint_path'] += file[:file.find('.')]
                parameter_dict['experiment_name'] += file[:file.find('.')]
                setup_train( parameter_dict, args.device )
                corte += 1

    
    generate_pc_config = {
        'model_path': '',
        'json': '',
        'nsamples': 100,
        'ref_steps': 2,
        'surf_thresh': 0.03,
        'device': args.device,
        'weight0': 30,
        'max_iter': 1000
    }
    
    chamfer_distances = []
    normal_cosine_sims = []

    for dirpath, dnames, fnames in os.walk( training_config['checkpoint_path'] ):
        for file in fnames:
            if file == 'model_best.pth':
                generate_pc_config['model_path'] = os.path.join(dirpath, file)
                pred_points, pred_normals = list( generate_pc(generate_pc_config) )[0]

                # estoy en .../test_ABCD/models
                params_path = dirpath[:dirpath.rfind('/')] + '/params.json'

                with open(params_path) as params_file:
                    dataset_path = json.load( params_file )['dataset']

                with open(dataset_path) as dataset_file:
                    dataset_json = json.load(dataset_file)
                    gt_vertices = np.array( dataset_json['joints'][0]['vertices'] )
                    gt_normals = np.array( dataset_json['joints'][0]['normals'] )

                chamfer_d, normal_s = chamfer_distance( 
                    torch.from_numpy(pred_points).float().unsqueeze(0).to(device_torch), 
                    torch.from_numpy(gt_vertices).float().unsqueeze(0).to(device_torch),
                    x_normals=torch.from_numpy(pred_normals).float().unsqueeze(0).to(device_torch), 
                    y_normals=torch.from_numpy(gt_normals).float().unsqueeze(0).to(device_torch) )

                chamfer_distances.append(chamfer_d)
                normal_cosine_sims.append(normal_s)

    
    print( chamfer_distances, normal_cosine_sims )


        

                
    
