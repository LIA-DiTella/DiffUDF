import torch
import numpy as np
from src.model import SIREN
from src.evaluate import evaluate
from src.util import normalize

class Sampler:
    def __init__(self, n_in_features, hidden_layers=[256,256,256,256], w0=30, ww=None, checkpoint = None, device =0):
        self.decoder = SIREN(
            n_in_features= n_in_features,
            n_out_features=1,
            hidden_layer_config=hidden_layers,
            w0=w0,
            ww=ww
        )
        self.features = n_in_features
        self.device = torch.device(device)
        self.decoder.to( self.device )
        self.decoder.eval()

        self.decoder.load_state_dict( torch.load(checkpoint, map_location=self.device))


    def generate_point_cloud_squared(self, code, num_steps = 5, num_points = 20000, grad_thresh=0.001, surf_thresh = 0.01, max_iter=1000 ):

        for param in self.decoder.parameters():
            param.requires_grad = False

        surface_points = np.zeros((0, 3))
        normals = np.zeros((0,3))
        iterations = 0
        while len(surface_points) < num_points and iterations < max_iter :
            samples = np.random.uniform(-1, 1, (num_points, 3) )
            gradients = np.zeros( (num_points, 3 ) )
            udfs = None
            for step in range(num_steps):
                if step == num_steps - 1:
                    hessians = np.zeros( (num_points, 3, 3))
                    udfs = evaluate( self.decoder, np.hstack( [ np.tile(code, (num_points, 1)), samples] ), gradients=gradients, hessians=hessians, device=self.device )
                else:
                    udfs = evaluate( self.decoder, np.hstack( [ np.tile(code, (num_points, 1)), samples] ), gradients=gradients, device=self.device )

                steps = np.zeros_like(udfs)
                np.sqrt(udfs, where=udfs > 0, out=steps)
                samples -= gradients * steps

            gradient_norms = np.sum( gradients ** 2, axis=1)
        
            mask_points_on_surf = np.logical_and( gradient_norms < grad_thresh, steps.flatten() < surf_thresh)

            if np.sum(mask_points_on_surf) > 0:
                samples_near_surf = samples[ mask_points_on_surf ]
                surface_points = np.vstack((surface_points, samples_near_surf))

                normals = np.vstack( ( normals, [ np.linalg.eigh(hessian)[1][:,2] for hessian in hessians[mask_points_on_surf] ]) )
            
            iterations += 1

        if iterations == max_iter:
            print(f'Max iterations reached. Only sampled {len(surface_points)} surface points.')

        return surface_points, normals
    
    def generate_point_cloud(self, code, num_steps = 5, num_points = 20000, surf_thresh = 0.01, max_iter=1000 ):

        for param in self.decoder.parameters():
            param.requires_grad = False

        surface_points = np.zeros((0, 3))
        normals = np.zeros((0,3))
        iterations = 0
        while len(surface_points) < num_points and iterations < max_iter :
            samples = np.random.uniform(-1, 1, (num_points, 3) )
            gradients = np.zeros( (num_points, 3 ) )
            udfs = None
            for step in range(num_steps):
                udfs = evaluate( self.decoder, np.hstack( [ np.tile(code, (num_points, 1)), samples] ), gradients=gradients, device=self.device )

                gradients_norm = normalize(gradients)
                samples -= udfs * gradients_norm
        
            mask_points_on_surf = udfs.flatten() < surf_thresh

            if np.sum(mask_points_on_surf) > 0:
                surface_points = np.vstack((surface_points, samples[ mask_points_on_surf ]))
                normals = np.vstack( ( normals, gradients_norm[mask_points_on_surf]) )
            
            iterations += 1

        if iterations == max_iter:
            print(f'Max iterations reached. Only sampled {len(surface_points)} surface points.')

        return surface_points, normals
