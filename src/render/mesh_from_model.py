import torch
from model import SIREN
import numpy as np
from meshing import create_mesh
import os
import open3d as o3d
import os.path as osp

model = SIREN(
        n_in_features=4,
        n_out_features=1,
        hidden_layer_config=[256, 256, 256, 256],
        w0=60,
        ww=None
)
model.load_state_dict( torch.load('results/test5_curvature_sdf/models/model_best.pth'))

for i in np.arange(0,1, 0.1):
    vertices, faces, normals, values = create_mesh(
        model,
        distance = i,
        N=128
    )