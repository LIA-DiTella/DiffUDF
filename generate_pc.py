from src.render_pc import Sampler
import json
import open3d as o3d
import argparse

def generate_pc( config ):
        gen = Sampler( 3, checkpoint=config['model_path'], device=config['device'], w0=config['w0'] )
            
        points, normals = gen.generate_point_cloud(
            num_points=config['nsamples'], 
            num_steps=config['ref_steps'], 
            surf_thresh=config['surf_thresh'],
            alpha=config['alpha'],
            gt_mode=config['gt_mode'],
            max_iter=config['max_iter']
        )
            

        device = o3d.core.Device("CUDA:"+str(config['device']))
        dtype = o3d.core.float32
        pcd = o3d.t.geometry.PointCloud(device)

        pcd.point.positions = o3d.core.Tensor(points, dtype, device)

        pcd.point.normals = o3d.core.Tensor(normals, dtype, device)

        return pcd
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dense point cloud from trained model')
    parser.add_argument('config_path', metavar='path/to/json', type=str,
                        help='path to render config')
    args = parser.parse_args()

    with open(args.config_path) as config_file:
        config_dict = json.load(config_file)

    point_cloud = generate_pc(config_dict)
    point_cloud.orient_normals_consistent_tangent_plane(10)
    o3d.t.io.write_point_cloud( config_dict['output_path'], point_cloud)