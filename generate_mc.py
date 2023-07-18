from src.render_mc import get_mesh_udf
from src.model import SIREN
import torch
import argparse
import json

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
		hidden_layer_config=[256,256,256,256],
		w0=config_dict["w0"],
		ww=None
	)

	model.load_state_dict( torch.load(config_dict["model_path"], map_location=device_torch))
	model.to(device_torch)

	del config_dict['device']
	print('Generating mesh...')
	vertices, faces, mesh = get_mesh_udf( 
		model, 
		torch.Tensor([[]]).to(device_torch),
		device=device_torch,
		**config_dict
	)

	mesh.export(config_dict["output_path"])
	print(f'Saved to {config_dict["output_path"]}')

