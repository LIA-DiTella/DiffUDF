from src.render_mc import get_mesh_udf, get_mesh_sdf
from src.render_mc_CAP import extract_geometry
from src.model import SIREN
import torch
import argparse
import trimesh as tm
import json
import numpy as np

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

	del config_dict['device']
	print('Generating mesh...')

	if config_dict['gt_mode'] != 'siren':
		vertices, faces, mesh = get_mesh_udf( 
			model, 
			torch.Tensor([[]]).to(device_torch),
			device=device_torch,
			**config_dict
		)
		#mesh = tm.smoothing.filter_laplacian( mesh, iterations=2 )

		#mesh = extract_geometry(config_dict["nsamples"], model, device_torch)
	else:
		vertices, faces, mesh = get_mesh_sdf( 
			model,
			N=config_dict['nsamples'],
			device=device_torch
		)


	mesh.export(config_dict["output_path"])
	print(f'Saved to {config_dict["output_path"]}')

