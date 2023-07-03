import torch
import numpy as np
from src.diff_operators import gradient, hessian

def evaluate( model, samples, max_batch=64**2, output_size=1, device=torch.device(0), gradients=None, hessians=None ):
    # samples = ( amount_samples, features + 3 )
    head = 0        
    amount_samples = samples.shape[0]
    evaluations = np.zeros( (amount_samples, output_size))

    while head < amount_samples:
        inputs_subset = torch.from_numpy(samples[head:min(head + max_batch, amount_samples), :]).float().to(device).unsqueeze(0)
        x, y =  model(inputs_subset).values()

        if gradients is not None:
            gradients[head:min(head + max_batch, amount_samples)] = gradient(y,x).squeeze(0).detach().cpu().numpy()

        if hessians is not None:
            hessians[head:min(head + max_batch, amount_samples)] = hessian(y,x)[0].squeeze(0).detach().cpu().numpy()
    
        evaluations[head:min(head + max_batch, amount_samples)] = y.squeeze(0).detach().cpu()
        head += max_batch

    return evaluations