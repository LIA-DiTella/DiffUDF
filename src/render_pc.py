import torch
import numpy as np
from src.model import SIREN
from src.evaluate import evaluate
from src.util import normalize

class Sampler:
    def __init__(self, n_in_features, hidden_layers=[256,256,256,256], w0=30, ww=None,  threshold = 0.4, checkpoint = None, device =0):
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
        self.threshold = threshold


    def generate_point_cloud(self, code, num_steps = 5, num_points = 20000, surf_thresh = 0.009 ):

        for param in self.decoder.parameters():
            param.requires_grad = False

        surface_points = np.zeros((0, 3))
        samples = np.random.uniform(-1, 1, (num_points, 3) )

        i = 0
        while len(surface_points) < num_points:
            gradients = np.zeros( (num_points, len(code) + 3 ) )
            udfs = None
            for _ in range(num_steps):
                udfs = evaluate( self.decoder, np.hstack( [ np.tile(code, (num_points, 1)), samples] ), gradients=gradients, device=self.device )

                udfs = np.where( udfs <= 0, udfs, np.sqrt(udfs) )

                gradient_wrt_samples = normalize(gradients[:, -3:])

                samples -= gradient_wrt_samples * udfs
        
            if i > 0: # en la primera iteracion no sampleo... nose bien porque
                samples_near_surf = samples[udfs.squeeze(1) < surf_thresh]
                surface_points = np.vstack((surface_points, samples_near_surf))

            samples = samples[ udfs.squeeze(1) < 0.5 ]
            samples = samples[ np.random.randint( 0, samples.shape[0], num_points) ]

            samples += (self.threshold / 3) * np.random.standard_normal( samples.shape )
            i += 1

        return surface_points
