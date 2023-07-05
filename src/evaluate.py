import torch
import numpy as np
from src.diff_operators import gradient, hessian

def evaluate( model, samples, latent_vec=torch.Tensor([[]]), max_batch=64**2, output_size=1, device=torch.device(0), gradients=None, hessians=None ):
    # samples = ( amount_samples, 3 )    
    head = 0
    amount_samples = samples.shape[0]
    feature_length = latent_vec.shape[1]

    evaluations = np.zeros( (amount_samples, output_size))

    while head < amount_samples:
        
        if torch.is_tensor(samples):
            inputs_subset = samples[head:min(head + max_batch, amount_samples), :]
        else:
            inputs_subset = torch.from_numpy(samples[head:min(head + max_batch, amount_samples), :]).float()
            
        if feature_length != 0:
            batch_vecs = latent_vec.view(latent_vec.shape[0], 1, latent_vec.shape[1]).repeat(1, inputs_subset.shape[0], 1)
            inputs_subset = torch.cat([batch_vecs.reshape(-1, latent_vec.shape[1]), inputs_subset.reshape(-1, inputs_subset.shape[-1])], dim=1)

        inputs_subset = inputs_subset.to(device).unsqueeze(0)

        x, y =  model(inputs_subset).values()

        if gradients is not None:
            gradients[head:min(head + max_batch, amount_samples)] = gradient(y,x).squeeze(0).detach().cpu().numpy()[..., feature_length:]

        if hessians is not None:
            hessians[head:min(head + max_batch, amount_samples)] = hessian(y,x)[0].squeeze(0).detach().cpu().numpy()[..., feature_length:, feature_length:]
    
        evaluations[head:min(head + max_batch, amount_samples)] = y.squeeze(0).detach().cpu()
        head += max_batch

    return evaluations