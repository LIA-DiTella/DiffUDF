from src.render_mc import get_mesh_udf, get_mesh_sdf
from src.render_mc_CAP import extract_geometry
from src.model import SIREN
import torch
import argparse
import numpy as np
import json

def generate_mc(model, gt_mode,device, N, output_path, alpha, bbox_min=None, bbox_max=None,  algorithm='meshudf'):
	if gt_mode != 'siren':
		if algorithm == 'meshudf':
			vertices, faces, mesh = get_mesh_udf( 
				model, 
				torch.Tensor([[]]).to(device),
				device=device,
				gt_mode=gt_mode,
				nsamples=N,
				alpha=alpha
			)
		elif algorithm == 'cap':
			mesh = extract_geometry(N, model, device, bbox_min=np.array(bbox_min), bbox_max=np.array(bbox_max))
		else:
			raise ValueError('Invalid algorithm')
	else:
		vertices, faces, mesh = get_mesh_sdf( 
			model,
			N=N,
			device=device
		)

	mesh.export(output_path)
	print(f'Saved to {output_path}')

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Generate mesh through marching cubes from trained model')
	parser.add_argument('config_path', metavar='path/to/json', type=str,
					help='path to render config')

	args = parser.parse_args()

	with open(args.config_path) as config_file:
		config_dict = json.load(config_file)	

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

	print('Generating mesh...')

	generate_mc(model, config_dict['gt_mode'], device_torch, config_dict['nsamples'], config_dict['output_path'], config_dict['alpha'], config_dict['bbox_min'], config_dict['bbox_max'], algorithm=config_dict['algorithm'])

