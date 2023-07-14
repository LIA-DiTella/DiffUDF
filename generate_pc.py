from src.render_pc import Sampler
import numpy as np
import json
import open3d as o3d
import argparse
import tqdm

def generate_pc_full( config ):
    gen = Sampler( 3, checkpoint=config['model_path'], device=config['device'], w0=config['weight0'] )
    print('Generating point cloud')
    if config['squared']:
        return gen.generate_point_cloud_squared( 
            code=[], 
            num_points=config['nsamples'], 
            num_steps=config['ref_steps'], 
            surf_thresh=config['surf_thresh'],
            grad_thresh=config['grad_thresh'],
            max_iter=config['max_iter']
        )    
    else:
        return gen.generate_point_cloud( 
            code=[], 
            num_points=config['nsamples'], 
            num_steps=config['ref_steps'], 
            surf_thresh=config['surf_thresh'],
            max_iter=config['max_iter']
        )


def generate_pc( config ):
    if config['json'] == '':
        yield generate_pc_full( config )
    else:
        with open(config['json']) as jsonFile:
            skel = json.load(jsonFile)
            gen = Sampler( len(skel['joints'][0]['mean']) + 3, checkpoint=config['model_path'], device=config['device'], w0=config['weight0'] )
            
            print('Generating point cloud')
            for joint in tqdm.tqdm(skel['joints']):

                T = (np.block( [[np.eye(3,3), np.asarray(joint['position']).reshape((3,1))],[np.eye(1,4,k=3)] ] ) @ 
                    np.block( [ [np.asarray(joint['base']), np.zeros((3,1))], [np.eye(1,4,k=3)]]) @ 
                    np.diag( [1 / skel['scale']] * 3 + [1] ))
                
                if config['squared']:
                    points, normals = gen.generate_point_cloud_squared( 
                        code=joint['mean'], 
                        num_points=config['nsamples'] // skel['amount_joints'], 
                        num_steps=config['ref_steps'], 
                        surf_thresh=config['surf_thresh'],
                        grad_thresh=config['grad_thresh'],
                        max_iter=config['max_iter']
                    )    
                else:
                    points, normals = gen.generate_point_cloud( 
                        code=joint['mean'], 
                        num_points=config['nsamples'] // skel['amount_joints'], 
                        num_steps=config['ref_steps'], 
                        surf_thresh=config['surf_thresh'],
                        max_iter=config['max_iter']
                    )
                
                puntosTransf = (T @ np.concatenate( (points, np.ones( (points.shape[0], 1)) ), axis=1).T).T[:, :3]

                yield puntosTransf, normals
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dense point cloud from trained model')
    parser.add_argument('model_path', metavar='path/to/pth', type=str,
                        help='path to input model')
    parser.add_argument('output_path', metavar='path/to/output/pc.ply', type=str,
                        help='path to output point cloud')
    parser.add_argument('-j', '--json', metavar='path/to/json', type=str, default='',
                        help='path to input json (preprocessed mesh)')
    parser.add_argument('-s', '--squared', action='store_true', help='use algorithm for squared distance field')
    parser.add_argument('-n', '--nsamples', type=int, default=20000, help='number of samples')
    parser.add_argument('-r', '--ref_steps', type=int, default=5, help='number of refinement steps (grad desc)')
    parser.add_argument('-t', '--surf_thresh', type=float, default=1e-5, help='on surface threshold')
    parser.add_argument('-g', '--grad_thresh', type=float, default=1e-5, help='grad norm threshold')
    parser.add_argument('-d', '--device', type=int, default=0, help='torch device')
    parser.add_argument('-w0', '--weight0', type=float, default=30, help='w0 parameter of SIREN')
    parser.add_argument('-i', '--max_iter', type=int, default=1000, help='max iterations')

    args = parser.parse_args()

    for generated_points, generated_normals in generate_pc(vars(args)):
        p_cloud = o3d.geometry.PointCloud( )
        p_cloud.points = o3d.utility.Vector3dVector(generated_points)
        p_cloud.normals = o3d.utility.Vector3dVector(generated_normals)
        o3d.io.write_point_cloud( args.output_path, p_cloud)