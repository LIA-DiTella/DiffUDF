from src.render_mc import extract_mesh_MESHUDF, extract_mesh_CAP, extract_fields, get_mesh_sdf
from src.model import SIREN
import torch
import argparse
import numpy as np
import open3d as o3d
import json

def generate_mc(model, gt_mode,device, N, output_path, alpha=None, algorithm='meshudf', from_file=None):

	if from_file is not None:
		model = SIREN(
			n_in_features= 3,
			n_out_features=1,
			hidden_layer_config=from_file["hidden_layer_nodes"],
			w0=from_file["w0"],
			ww=None
		)

		model.load_state_dict( torch.load(from_file["model_path"]))
		model.to(device)

	if algorithm == 'meshudf':
		u,g = extract_fields(model, torch.Tensor([[]]).to(device), N, gt_mode, device, alpha )
		vertices, faces, mesh = extract_mesh_MESHUDF(u, g, device, smooth_borders=True)

		mesh.export(output_path)
		print(f'Saved to {output_path}')

		return mesh

	elif algorithm == 'cap':
		u,g = extract_fields(model, torch.Tensor([[]]).to(device), N, gt_mode, device, alpha )
		mesh = extract_mesh_CAP(u.cpu().numpy(), g.cpu().numpy(), N, alpha=alpha)

		mesh.export(output_path)
		print(f'Saved to {output_path}')

		return mesh

	elif algorithm == 'both':
		u,g = extract_fields(model, torch.Tensor([[]]).to(device), N, gt_mode, device, alpha )
		vertices, faces, meshMU = extract_mesh_MESHUDF(u, g, device, smooth_borders=True)
		meshCAP = extract_mesh_CAP(u.cpu().numpy(), g.cpu().numpy(), N )

		pathMU = output_path[:output_path.rfind('.')] + '_MU' + output_path[output_path.rfind('.'):]
		pathCAP = output_path[:output_path.rfind('.')] + '_CAP' + output_path[output_path.rfind('.'):]

		meshMU.export(pathMU)
		meshCAP.export(pathCAP)
		print(f'Saved to {pathMU}, {pathCAP}')

		return meshMU, meshCAP

	elif algorithm == 'siren':
		vertices, faces, meshSIREN = get_mesh_sdf(
			model,
			N=N,
			device=device
		)

		meshSIREN.export(output_path)
		print(f'Saved to {output_path}')
		return meshSIREN
	else:
		raise ValueError('Invalid algorithm')

	

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

	generate_mc(model, config_dict['gt_mode'], device_torch, config_dict['nsamples'], config_dict['output_path'], config_dict['alpha'], algorithm=config_dict['algorithm'])

