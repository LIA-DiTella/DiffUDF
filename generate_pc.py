from src.render_pc import Sampler
import numpy as np
import json
import open3d as o3d
import argparse
import tqdm

def generate_pc( config ):
    with open(config['json_path']) as jsonFile:
        skel = json.load(jsonFile)
        gen = Sampler( len(skel['joints'][0]['mean']) + 3, checkpoint=config['model_path'], device=config['device'], w0=config['w0'] )
        
        print('Generating point cloud')
        for joint in tqdm.tqdm(skel['joints']):

            T = (np.block( [[np.eye(3,3), np.asarray(joint['position']).reshape((3,1))],[np.eye(1,4,k=3)] ] ) @ 
                np.block( [ [np.asarray(joint['base']), np.zeros((3,1))], [np.eye(1,4,k=3)]]) @ 
                np.diag( [1 / skel['scale']] * 3 + [1] ))
            
            points, normals = gen.generate_point_cloud( 
                code=joint['mean'], 
                num_points=config['nsamples'] // skel['amount_joints'], 
                num_steps=config['ref_steps'], 
                surf_thresh=config['surf_thresh'],
                alpha=config['alpha'],
                gt_mode=config['gt_mode'],
                max_iter=config['max_iter']
            )
            
            puntosTransf = (T @ np.concatenate( (points, np.ones( (points.shape[0], 1)) ), axis=1).T).T[:, :3]

            yield puntosTransf, normals
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dense point cloud from trained model')
    parser.add_argument('config_path', metavar='path/to/json', type=str,
                        help='path to render config')
    args = parser.parse_args()

    with open(args.config_path) as config_file:
        config_dict = json.load(config_file)

    for generated_points, generated_normals in generate_pc(config_dict):
        p_cloud = o3d.geometry.PointCloud( )
        p_cloud.points = o3d.utility.Vector3dVector(generated_points)
        p_cloud.normals = o3d.utility.Vector3dVector(generated_normals)
        o3d.io.write_point_cloud( config_dict['output_path'], p_cloud)