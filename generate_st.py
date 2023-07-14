import torch
import numpy as np
from src.model import SIREN
from PIL import Image
from src.render_st import create_projectional_image
import argparse

## TODAVIA NO SE BANCA RED CON SELECTOR
## por ahora plano, origen y luz fija

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ray traced image from trained model')
    parser.add_argument('model_path', metavar='path/to/pth', type=str,
                        help='path to input model')
    parser.add_argument('output_path', metavar='path/to/output/pc.png', type=str,
                        help='path to output render')
    parser.add_argument('-d', '--device', type=int, default=0, help='torch device')
    parser.add_argument('-w0', '--weight0', type=float, default=30, help='w0 parameter of SIREN')
    parser.add_argument('-w', '--width', type=int, default=512, help='width of generated image')
    parser.add_argument('-t', '--surf_thresh', type=float, default=1e-3, help='on surface threshold')
    parser.add_argument('-g', '--grad_thresh', type=float, default=1e-3, help='on surface gradient norm threshold')
    parser.add_argument('-r', '--ref_steps', type=int, default=5, help='number of refinement steps (grad desc)')
    parser.add_argument('-i', '--max_iter', type=int, default=np.inf, help='max number of sphere tracing iterations')

    args = parser.parse_args()

    device_torch = torch.device(args.device)

    model = SIREN(
            n_in_features= 3,
            n_out_features=1,
            hidden_layer_config=[256,256,256,256],
            w0=args.weight0,
            ww=None
    )

    model.load_state_dict( torch.load(args.model_path, map_location=device_torch))
    model.to(device_torch)

    ranges = np.linspace(1, -1, args.width)
    i_1, i_2 = np.meshgrid( ranges, ranges )
    image = np.concatenate(
                    np.concatenate( np.array([np.expand_dims(i_1, 2), 
                                            np.expand_dims(i_2, 2), 
                                            np.expand_dims(np.ones_like(i_1) * -1, 2)])
                                    , axis=2 ),
                    axis=0)
    
    colores = create_projectional_image( 
        model, 
        sample_count=args.width ** 2, 
        surface_eps=args.surf_thresh,
        gradient_eps=args.grad_thresh,
        refinement_steps=args.ref_steps,
        origin=np.array([0,0,-2]), 
        image=image,
        light_position=np.array([1,0,-2]),
        max_iterations=args.max_iter,
        device=device_torch
    )
    
    im = Image.fromarray((colores * 255).astype(np.uint8))
    im.save(args.output_path, 'PNG')