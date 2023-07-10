import torch
import numpy as np
from pytorch3d.loss import chamfer_distance
from train import setup_train
from generate_pc import generate_pc
import os
import pandas as pd
import open3d as o3d
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
        "dataset": "data/Multi-Garment_dataset/Multi-Garment_dataset/",
        "batch_size": 20000,
        "sampling_percentiles": [0.5, 0.45, 0.05],
        "batches_per_epoch": 3,
        "checkpoint_path": "results/ropa/",
        "experiment_name": "test_",
        "epochs_to_checkpoint": 600,
        "loss": "loss_siren",
        "loss_weights": [ 1e3, 1e2, 2e0, 1e1 ],
        "optimizer": {
            "type": "adam",
            "lr": 1e-4
        },
        "network": {
            "hidden_layer_nodes": [256, 256, 256, 256],
            "w0": 30,
            "pretrained_dict": 'None'
        }
    }

    processed_meshes = 0
    for dirpath, dnames, fnames in os.walk("data/Multi-Garment_dataset/Multi-Garment_dataset/"):
        for file in fnames:
            if file.endswith(".json") and processed_meshes < args.meshes:
                parameter_dict = training_config.copy()
                parameter_dict['dataset'] += file
                parameter_dict['checkpoint_path'] += file[:file.find('.')]
                parameter_dict['experiment_name'] += file[:file.find('.')]
                setup_train( parameter_dict, args.device )
                processed_meshes += 1

    
    generate_pc_config = {
        'model_path': '',
        'json': '',
        'nsamples': args.nsamples,
        'ref_steps': 5,
        'surf_thresh': 0.09,
        'grad_thresh': 0.01,
        'device': args.device,
        'weight0': 30,
        'max_iter': 100,
        'hess': False
    }
    
    chamfer_distances = {}

    for dirpath, dnames, fnames in os.walk( training_config['checkpoint_path'] ):
        for file in fnames:
            if file == 'model_best.pth':
                generate_pc_config['model_path'] = os.path.join(dirpath, file)
                pred_points, pred_normals = list( generate_pc(generate_pc_config) )[0]

                p_cloud = o3d.geometry.PointCloud( )
                p_cloud.points = o3d.utility.Vector3dVector(pred_points)
                p_cloud.normals = o3d.utility.Vector3dVector(pred_normals)
                o3d.io.write_point_cloud( dirpath[:dirpath.rfind('/')] + '/reconstructions/point_cloud.ply' , p_cloud)

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
                    y_normals=torch.from_numpy(gt_normals).float().unsqueeze(0).to(device_torch))
                
                mesh_name = dataset_path[ dataset_path.rfind('/') + 1:dataset_path.rfind('.')]

                chamfer_distances[mesh_name] = [ chamfer_d.cpu().numpy(), normal_s.cpu().numpy() ]

    chamfer_df = pd.DataFrame.from_dict(chamfer_distances, orient='index', columns=['chamfer distance', 'absolute cos_sim'])

    chamfer_df.to_csv('chamfer_distances.csv')


        

                
    
