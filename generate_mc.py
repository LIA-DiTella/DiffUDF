from src.render_mc import get_mesh_udf
from src.model import SIREN
import torch
import trimesh as tm
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate mesh through marching cubes from trained model')
    parser.add_argument('model_path', metavar='path/to/pth', type=str,
                        help='path to input model')
    parser.add_argument('output_path', metavar='path/to/output/mesh.obj', type=str,
                        help='path to output mesh')
    parser.add_argument('-n', '--nsamples', type=int, default=128, help='number of samples')
    parser.add_argument('-d', '--device', type=int, default=0, help='torch device')
    parser.add_argument('-l', '--level_set', type=float, default=0, help='level set for surface extraction')
    parser.add_argument('-w0', '--weight0', type=float, default=30, help='w0 parameter of SIREN')

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

    print('Generating mesh...')
    vertices, faces, mesh = get_mesh_udf( model, torch.Tensor([[]]).to(device_torch), N_MC=args.nsamples, device=device_torch, level_set=args.level_set )

    mesh.export(args.output_path)
    print(f'Saved to {args.output_path}')

