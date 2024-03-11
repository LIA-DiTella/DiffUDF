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
    pixel_camera_x = ( pixel_screen_x ) * aspect_ratio * np.tan( fov_radians / 2 )
    pixel_camera_y = (  pixel_screen_y ) * np.tan( fov_radians / 2)

    pixel_camera_x, pixel_camera_y = np.meshgrid(pixel_camera_x, pixel_camera_y, indexing='xy')

    pixels_camera = np.concatenate( [
        pixel_camera_x[...,None],
        pixel_camera_y[...,None],
        -1 * np.ones_like(pixel_camera_x)[...,None]
        ], axis=-1
    )

    return pixels_camera

def generate_st( config_dict ):
    network_config = config_dict['network_config']
    rendering_config = config_dict['rendering_config']

    colores = np.zeros((rendering_config['height'], rendering_config['width'],3))

    for i in range(rendering_config['sample_rate']):
        pixels_camera = get_pixels_camera(rendering_config['height'], rendering_config['width'], rendering_config['fov'], np.random.normal(0.5,0.35))

        camera_pos = np.float32( rendering_config['camera_position'] )
        a = np.array([0,0,-1])
        b = -1 * camera_pos
        b /= np.linalg.norm(b)

        if np.isclose(a@b, -1):
           R = np.array([
               [-1,0,0],
               [0,1,0],
               [0,0,-1]
           ])
        elif np.isclose(a@b,1):
            R = np.eye(3,3)
        else:
            upVector = np.array([0,1,0]) - (np.array([0,1,0])@b) * b
            upVector /= np.linalg.norm(upVector)
            rightVector = np.cross(upVector, b)
            R = np.vstack([rightVector, upVector , b]).T

        ray_directions = (pixels_camera).reshape((rendering_config['width'] * rendering_config['height'], 3))
        ray_directions = (R @ ray_directions.T).T + np.tile(camera_pos, (len(ray_directions),1))
        ray_directions /= np.linalg.norm(ray_directions, axis=-1)[...,None]
        ray_directions *= -1

        plane_normals = np.array([
            [1,0,0],
            [1,0,0],
            [0,1,0],
            [0,1,0],
            [0,0,1],
            [0,0,1]
        ])
        p_pos = rendering_config.get('planes', [1,-1,1,-1,1,-1] )
        plane_positions = np.array([
            [p_pos[0],0,0],
            [p_pos[1],0,0],
            [0,p_pos[2],0],
            [0,p_pos[3],0],
            [0,0,p_pos[4]],
            [0,0,p_pos[5]]
        ]) - np.tile( rendering_config['camera_position'], (6,1) )

        numerator = np.sum( plane_positions * plane_normals, axis=-1 )
        numerator = np.tile(numerator.reshape((1,6)), (len(ray_directions),1))
        denominator = ray_directions @ plane_normals.T

        ds = numerator / np.where( np.abs(denominator) < 1e-5, np.ones_like(denominator), denominator)

        intersections = (
            np.repeat(ray_directions, 6, axis=0).reshape((len(ray_directions), 6,3) ) * ds[...,None] + 
            np.tile( rendering_config['camera_position'], (len(ray_directions)*6,1) ).reshape((len(ray_directions), 6,3) )
        )

        mask_outside_intersections = np.prod( np.logical_and(intersections >= -1.001, intersections <= 1.001),axis=-1 ) * (np.abs(denominator) > 1e-5)
        valid_rays = np.sum( mask_outside_intersections , axis=-1 ).astype(bool)
        ds = np.min( np.where( np.logical_and( ds >= 0, mask_outside_intersections ), ds, np.ones_like(ds) * np.inf)[valid_rays,:], axis=-1)
        starting_pos = np.zeros_like(ray_directions)
        starting_pos[valid_rays,:] = ray_directions[valid_rays,:] * ds[...,None] + np.tile( rendering_config['camera_position'], (np.sum(valid_rays),1) )

        if network_config['gt_mode'] == 'gt':
            colores += create_projectional_image_gt( 
                mesh_file=config_dict['mesh_path'], 
                width=config_dict["image_width"],
                height=config_dict['image_height'], 
                rays=ray_directions, 
                t0=starting_pos, 
                mask_rays=valid_rays,
                light_position=np.array(config_dict["light_pos"]),
                max_iterations=config_dict["max_iter"],
                specular_comp=config_dict.get('specular', False))
        else:
            device_torch = torch.device(network_config["device"])
            model = SIREN(
                    n_in_features= 3,
                    n_out_features=1,
                    hidden_layer_config=network_config["hidden_layer_nodes"],
                    w0=network_config["w0"],
                    ww=None
            )

            model.load_state_dict( torch.load(network_config["model_path"], map_location=device_torch))
            model.to(device_torch)

            colores += create_projectional_image( 
                model,
                rays=ray_directions, 
                t0=starting_pos, 
                mask_rays=valid_rays,
                network_config=network_config,
                rendering_config=rendering_config,
                device=device_torch
            )

        torch.cuda.empty_cache()
    
    im = Image.fromarray((colores / rendering_config['sample_rate'] * 255).astype(np.uint8))

    if rendering_config.get('rotation', 0)!= 0:
        im = im.rotate(rendering_config['rotation'])

    return im

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ray traced image from trained model')
    parser.add_argument('config_path', metavar='path/to/json', type=str,
                        help='path to render config')
    args = parser.parse_args()

    with open(args.config_path) as config_file:
        config_dict = json.load(config_file)

    im = generate_st(config_dict)
    im.save(config_dict["rendering_config"]["output_path"], 'PNG')