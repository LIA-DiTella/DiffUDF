import torch
import numpy as np
from src.model import SIREN
from PIL import Image
from src.render_st import create_projectional_image
import argparse
import json

## TODAVIA NO SE BANCA RED CON SELECTOR
## por ahora plano, origen y luz fija

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
            hidden_layer_config=[256,256,256,256],
            w0=config_dict["w0"],
            ww=None
    )

    model.load_state_dict( torch.load(config_dict["model_path"], map_location=device_torch))
    model.to(device_torch)

    ranges = np.linspace(1, -1, config_dict["width"])
    i_1, i_2 = np.meshgrid( ranges, ranges )
    image = np.concatenate(
                    np.concatenate( np.array([np.expand_dims(i_1, 2), 
                                            np.expand_dims(i_2, 2), 
                                            np.expand_dims(np.ones_like(i_1) * -1, 2)])
                                    , axis=2 ),
                    axis=0)
    
    colores = create_projectional_image( 
        model, 
        sample_count=config_dict["width"] ** 2, 
        surface_eps=config_dict["surf_thresh"],
        gradient_eps=config_dict["grad_thresh"],
        alpha=config_dict["alpha"],
        beta=config_dict["beta"],
        gt_mode=config_dict["gt_mode"],
        refinement_steps=config_dict["ref_steps"],
        origin=np.array(config_dict["origin"]), 
        image=image,
        light_position=np.array(config_dict["light_pos"]),
        max_iterations=config_dict["max_iter"],
        device=device_torch
    )
    
    im = Image.fromarray((colores * 255).astype(np.uint8))
    im.save(config_dict["output_path"], 'PNG')