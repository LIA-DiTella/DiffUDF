from src.render_pc import Sampler
import numpy as np
import json
import torch
import open3d as o3d
import argparse
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dense point cloud from trained model')
    parser.add_argument('json_path', metavar='path/to/json', type=str,
                        help='path to input json (preprocessed mesh)')
    parser.add_argument('model_path', metavar='path/to/pth', type=str,
                        help='path to input model')
    parser.add_argument('output_path', metavar='path/to/output/pc.ply', type=str,
                        help='path to output point cloud')
    parser.add_argument('-n', '--nsamples', type=int, default=20000, help='number of samples')
    parser.add_argument('-r', '--ref_steps', type=int, default=5, help='number of refinement steps (grad desc)')
    parser.add_argument('-t', '--surf_thresh', type=float, default=1e-5, help='on surface threshold')
    parser.add_argument('-d', '--device', type=int, default=0, help='torch device')
    parser.add_argument('-w0', '--weight0', type=float, default=30, help='w0 parameter of SIREN')
    parser.add_argument('-i', '--max_iter', type=int, default=100, help='max iterations')

    args = parser.parse_args()

    with open(args.json_path) as jsonFile:
        skel = json.load(jsonFile)
        gen = Sampler( len(skel['joints'][0]['mean']) + 3, checkpoint=args.model_path, device=args.device, threshold=0.3, w0=args.weight0 )
        
        print('Generating point cloud')
        for joint in tqdm.tqdm(skel['joints']):

            
            #T = (np.block( [[np.eye(3,3), np.asarray(joint['position']).reshape((3,1))],[np.eye(1,4,k=3)] ] ) @ 
            #     np.block( [ [np.asarray(joint['base']), np.zeros((3,1))], [np.eye(1,4,k=3)]]) @ 
            #     np.diag( [1 / skel['scale']] * 3 + [1] ))
            
            points, normals = gen.generate_point_cloud( 
                code=joint['mean'], 
                num_points=args.nsamples // skel['amount_joints'], 
                num_steps=args.ref_steps, 
                surf_thresh=args.surf_thresh,
                max_iter=args.max_iter
            )
            
            #puntosTransf = (T @ np.concatenate( (points, np.ones( (points.shape[0], 1)) ), axis=1).T).T[:, :3]

            generated_points = points #puntosTransf
            generated_normals = normals
        
    p_cloud = o3d.geometry.PointCloud( )
    p_cloud.points = o3d.utility.Vector3dVector(generated_points)
    p_cloud.normals = o3d.utility.Vector3dVector(generated_normals)
    o3d.io.write_point_cloud( args.output_path, p_cloud)