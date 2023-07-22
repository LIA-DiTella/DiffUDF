import torch
import numpy as np
from PIL import Image
from src.model import SIREN
from src.render_st import create_projectional_image
from src.util import normalize
import argparse
import json
from scipy import sparse
from scipy.optimize import linprog

## TODAVIA NO SE BANCA RED CON SELECTOR
## por ahora plano, origen y luz fija

def intersect_plane_box( points, directions ):
    A = sparse.vstack( [ sparse.diags( directions[:, i]) for i in range(3) ])
    A = sparse.vstack( [A, -A])
    A = sparse.hstack( [A, -1 * sparse.eye(A.shape[0], A.shape[0])])
    xs = np.hstack( [ points[:,i] for i in range(3)] )
    xs = np.hstack( [xs, -xs])
    b = np.ones(A.shape[0]) - xs

    c = np.concatenate( [np.ones(points.shape[0]), np.ones(A.shape[0]) * 100] )

    results = linprog(c, A, b, bounds=(0, None))

    if not results.success:
        raise ValueError(results.message)
    
    return points + directions * np.expand_dims(results.x[:points.shape[0]],-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ray traced image from trained model')
    parser.add_argument('config_path', metavar='path/to/json', type=str,
                        help='path to render config')
    args = parser.parse_args()

    with open(args.config_path) as config_file:
        config_dict = json.load(config_file)

    device_torch = torch.device(config_dict["device"])

    model = SIREN(
            n_in_features= 3,
            n_out_features=1,
            hidden_layer_config=config_dict["layers"],
            w0=config_dict["w0"],
            ww=None
    )

    model.load_state_dict( torch.load(config_dict["model_path"], map_location=device_torch))
    model.to(device_torch)

    principal_direction = normalize( -np.array(config_dict['origin']) )
    tangent_dir2 = normalize( np.eye(1,3,1).squeeze(0) - (np.eye(1,3,1).squeeze(0) @ principal_direction) * principal_direction )
    tangent_dir1 = np.cross( principal_direction, tangent_dir2 )

    base = np.array([principal_direction, tangent_dir1, tangent_dir2])
    ranges = np.linspace(1, -1, config_dict["width"])
    i_1, i_2 = np.meshgrid( ranges, ranges )
    center_plane = config_dict['origin'] + principal_direction * config_dict['distance']
    
    f = lambda v: center_plane + v[0]*base[1] + v[1]*base[2]

    plane = np.concatenate( [[ f(j) for j in i ] for i in np.concatenate( [np.expand_dims(i_1,-1),np.expand_dims(i_2,-1)], 2) ])
    directions = normalize( plane - np.tile( config_dict['origin'], (plane.shape[0],1) ))

    image = intersect_plane_box( plane, directions )

    colores = create_projectional_image( 
        model, 
        sample_count=config_dict["width"] ** 2, 
        surface_eps=config_dict["surf_thresh"],
        gradient_eps=config_dict["grad_thresh"],
        alpha=config_dict["alpha"],
        gt_mode=config_dict["gt_mode"],
        refinement_steps=config_dict["ref_steps"],
        directions=directions, 
        image=image,
        light_position=np.array(config_dict["light_pos"]),
        max_iterations=config_dict["max_iter"],
        device=device_torch
    )
    
    im = Image.fromarray((colores * 255).astype(np.uint8))
    im.save(config_dict["output_path"], 'PNG')