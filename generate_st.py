import torch
import numpy as np
from PIL import Image
from src.model import SIREN
from src.render_st import create_projectional_image, create_projectional_image_gt
import argparse
import json

def get_pixels_camera( width, height, fov, noise ):
    image_x = np.arange(0, width)
    image_y = np.arange(0, height)

    pixel_NDC_x = (image_x + noise) / width
    pixel_NDC_y = (image_y + noise) / height

    pixel_screen_x = 2 * pixel_NDC_x - 1
    pixel_screen_y = 2 * pixel_NDC_y -1 

    aspect_ratio = width / height
    fov_radians = fov * np.pi / 180
    pixel_camera_x = ( pixel_screen_x ) * aspect_ratio #* np.tan(fov_radians / 2)
    pixel_camera_y = (  pixel_screen_y ) #* np.tan(fov_radians / 2)

    pixel_camera_x, pixel_camera_y = np.meshgrid(pixel_camera_x, pixel_camera_y, indexing='xy')

    pixels_camera = np.concatenate( [
        pixel_camera_x[...,None],
        pixel_camera_y[...,None],
        -1 * np.ones_like(pixel_camera_x)[...,None]
        ], axis=-1
    )

    return pixels_camera

def generate_st( config_dict ):
    colores = np.zeros((config_dict['image_height'], config_dict['image_width'],3))

    for i in range(config_dict['sample_rate']):
        pixels_camera = get_pixels_camera(config_dict['image_width'], config_dict['image_height'], config_dict['fov'], np.random.normal(0.5,0.35))

        camera_pos = np.float32( config_dict['camera_pos'] )
        #a = np.array([0,0,-1])
        #b = -1 * camera_pos
        #b /= np.linalg.norm(b)
#
        #v = np.cross(a,b)
        #c = np.dot(a,b)
#
        #if np.isclose(c, -1):
        #    R = np.array([
        #        [-1,0,0],
        #        [0,-1,0],
        #        [0,0,1]
        #    ])
        #elif np.isclose(c,1):
        #    R = np.eye(3,3)
        #else:
        #    v_x = np.array([
        #        [0, -v[2], v[1]],
        #        [v[2], 0, -v[0]],
        #        [-v[1], v[0], 0]
        #    ])
#
        #    R = np.eye(3,3) + v_x + v_x@v_x * (1 / (1 + c))
        
        forward = camera_pos
        forward /= np.linalg.norm(forward)

        temp_up = np.float64(config_dict.get('up_vector', [0,-1,0]))
        right = np.cross(temp_up,forward)
        right /= np.linalg.norm(right)

        up = np.cross(forward, right)
        up /= np.linalg.norm(up)

        R = np.vstack( [right, up, forward] )
        print(R@[0,0,-1])

        ray_directions = (pixels_camera).reshape((config_dict['image_width'] * config_dict['image_height'], 3))
        ray_directions = (R @ ray_directions.T).T + np.tile(camera_pos, (len(ray_directions),1))
        ray_directions /= np.linalg.norm(ray_directions, axis=-1)[...,None]
        #ray_directions *= -1

        plane_normals = np.array([
            [1,0,0],
            [1,0,0],
            [0,1,0],
            [0,1,0],
            [0,0,1],
            [0,0,1]
        ])
        plane_positions = np.array([
            [1,0,0],
            [-1,0,0],
            [0,1,0],
            [0,-1,0],
            [0,0,1],
            [0,0,-1]
        ]) - np.tile( config_dict['camera_pos'], (6,1) )

        numerator = np.sum( plane_positions * plane_normals, axis=-1 )
        numerator = np.tile(numerator.reshape((1,6)), (len(ray_directions),1))
        denominator = ray_directions @ plane_normals.T

        ds = numerator / np.where( np.abs(denominator) < 1e-5, np.ones_like(denominator), denominator)

        intersections = (
            np.repeat(ray_directions, 6, axis=0).reshape((len(ray_directions), 6,3) ) * ds[...,None] + 
            np.tile( config_dict['camera_pos'], (len(ray_directions)*6,1) ).reshape((len(ray_directions), 6,3) )
        )

        mask_outside_intersections = np.prod( np.logical_and(intersections >= -1, intersections <= 1),axis=-1 ) * (np.abs(denominator) > 1e-5)
        valid_rays = np.sum( mask_outside_intersections , axis=-1 ).astype(bool)
        ds = np.min( np.where( np.logical_and( ds >= 0, mask_outside_intersections ), ds, np.ones_like(ds) * np.inf)[valid_rays,:], axis=-1)
        starting_pos = np.zeros_like(ray_directions)
        starting_pos[valid_rays,:] = ray_directions[valid_rays,:] * ds[...,None] + np.tile( config_dict['camera_pos'], (np.sum(valid_rays),1) )

        if config_dict['gt_mode'] == 'gt':
            colores += create_projectional_image_gt( 
                mesh_file=config_dict['mesh_path'], 
                width=config_dict["image_width"],
                height=config_dict['image_height'], 
                rays=ray_directions, 
                t0=starting_pos, 
                mask_rays=valid_rays,
                light_position=np.array(config_dict["light_pos"]),
                max_iterations=config_dict["max_iter"])
        else:
            device_torch = torch.device(config_dict["device"])
            model = SIREN(
                    n_in_features= 3,
                    n_out_features=1,
                    hidden_layer_config=config_dict["hidden_layer_nodes"],
                    w0=config_dict["w0"],
                    ww=None
            )

            model.load_state_dict( torch.load(config_dict["model_path"], map_location=device_torch))
            model.to(device_torch)

            colores += create_projectional_image( 
                model, 
                width=config_dict["image_width"],
                height=config_dict['image_height'], 
                rays=ray_directions, 
                t0=starting_pos, 
                mask_rays=valid_rays,
                surface_eps=config_dict["surf_thresh"],
                alpha=config_dict["alpha"],
                gt_mode=config_dict["gt_mode"],
                light_position=np.array(config_dict["light_pos"]),
                max_iterations=config_dict["max_iter"],
                device=device_torch
            )
    
    im = Image.fromarray((colores / config_dict['sample_rate'] * 255).astype(np.uint8))
    im.save(config_dict["output_path"], 'PNG')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ray traced image from trained model')
    parser.add_argument('config_path', metavar='path/to/json', type=str,
                        help='path to render config')
    args = parser.parse_args()

    with open(args.config_path) as config_file:
        config_dict = json.load(config_file)

    generate_st(config_dict)