import torch
from model import SIREN
from meshing import create_mesh
import os
import os.path as osp

model = SIREN(
        n_in_features=4,
        n_out_features=1,
        hidden_layer_config=[256, 256, 256, 256],
        w0=60,
        ww=None
)
model.load_state_dict( torch.load('results/test_complete_0_curvature_sdf/models/model_best.pth'))

#rango = [0, 0.25, 0.5, 0.75, 1.0]
rango = [0.1, 0.35, 0.65]

for i in rango:
    create_mesh(
        model,
        selector = i,
        filename= osp.join( 'results/test_complete_0_curvature_sdf', "reconstructions", f"mesh{i}.ply"),
        N=128
    )