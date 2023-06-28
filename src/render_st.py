import numpy as np
import matplotlib.cm as cm
from src.util import normalize
import torch
from src.evaluate import evaluate

def create_orthogonal_image( model, sample_count, surface_eps, gradient_step, refinement_steps ):
    device_torch = torch.device(0)
    BORDES = [1, -1]
    OFFSETPLANO = 1
    LADO = int(np.sqrt(sample_count))

    i_1, i_2 = np.meshgrid( np.linspace(BORDES[0], BORDES[1], LADO), np.linspace(BORDES[0], BORDES[1], LADO) )
    samples = np.concatenate(
                    np.concatenate( np.array([np.expand_dims(i_1, 2), 
                                            np.expand_dims(i_2, 2), 
                                            np.expand_dims(np.ones_like(i_1) * OFFSETPLANO, 2)])
                                    , axis=2 ),
                    axis=0)


    mask = np.ones(sample_count, dtype=np.bool8)
    hits = np.zeros(sample_count, dtype=np.bool8)
    while np.sum(mask) > 0:
        max_batch = 64 ** 3
        head = 0
        
        inputs = samples[ mask ]
        udfs = np.zeros( len(inputs) )

        while head < sample_count:
            inputs_subset = torch.from_numpy(inputs[head:min(head + max_batch, sample_count), :]).float().to(device_torch)
            x, y =  model(inputs_subset).values()
        
            udfs[head:min(head + max_batch, sample_count)] = torch.where( y < 0, y,torch.sqrt( y) ).squeeze().detach().cpu()
            head += max_batch

        hits[mask] += udfs < surface_eps
        mask[mask] *= udfs > surface_eps
        samples[mask] -= np.hstack( [ np.zeros( (len(udfs[udfs > surface_eps]), 2) ), np.expand_dims( udfs[udfs > surface_eps], 1 )] )


        mask *= samples[:, 2] >= -1

    values = []
    for _ in range(refinement_steps):
        max_batch = 64 ** 3
        head = 0
        
        inputs = samples[ hits ]
        gradients = np.zeros((len(inputs), 3))

        while head < sample_count:
            inputs_subset = torch.from_numpy(inputs[head:min(head + max_batch, sample_count), :]).float().to(device_torch)
            x, y =  model(inputs_subset).values()

            y.sum().backward()
            udfs = y.squeeze().detach().cpu().numpy()
            if len(udfs) > 0:
                values.append(y.squeeze().detach().cpu().numpy())
        
            gradients[head:min(head + max_batch, sample_count)] = x.grad.detach().cpu()
            head += max_batch

        samples[hits] -= gradients * gradient_step

    cmap = cm.get_cmap('turbo')
    return cmap( (np.clip( samples[:, 2].reshape((LADO, LADO)), -1, 1) + np.ones((LADO, LADO))) / 2 )[:,:,:3], values

def create_projectional_image( model, sample_count, surface_eps, refinement_steps, origin, image, light_position, shininess=40 ):
    # image es una lista de puntos. Tengo un rayo por cada punto en la imagen. Los rayos salen con dirección norm(image_i - origin) desde el punto mismo.
    device_torch = torch.device(0)
    LADO = int(np.sqrt(sample_count))

    directions = normalize( image - np.tile( origin, (image.shape[0],1) ))

    alive = np.ones(sample_count, dtype=np.bool8)
    hits = np.zeros(sample_count, dtype=np.bool8)

    samples = image.copy()

    while np.sum(alive) > 0:
        udfs = evaluate( model, samples[ alive ], device=device_torch)

        hits[alive] += (udfs < surface_eps).squeeze(1)
        samples[alive] += directions[alive] * np.hstack([udfs, udfs, udfs])
        alive[alive] *= (udfs > surface_eps).squeeze(1)
        alive *= np.logical_and( np.all( samples > -1, axis=1 ), np.all( samples < 1, axis=1 ) )

    
    #normals = np.zeros( (np.sum(hits), 3))
    amount_hits = np.sum(hits)
    hessians = np.zeros( (amount_hits, 3, 3) )
    gradients = np.zeros((amount_hits, 3))

    for i in range(refinement_steps):
        if i == refinement_steps - 1:
            hessians = np.zeros((amount_hits, 3, 3))
            udfs = evaluate( model, samples[hits], gradients=gradients, hessians=hessians)
        else:
            udfs = evaluate( model, samples[hits], gradients=gradients)

        udfs = np.where( udfs <= 0, udfs, np.sqrt(udfs) )

        samples[hits] -= normalize(gradients) * udfs
        
        #normals += normalize(gradients)

        samples[hits] -= udfs * gradients

    #normals /= refinement_steps
    #normals = normalize(normals)
    normals = np.array( [ np.linalg.eigh(hessian)[1][:,2] for hessian in hessians ] )
    # podria ser que las normales apunten para el otro lado. las tengo que invertir si  < direccion, normal > = cos(tita) > 0
    normals *= np.where( np.expand_dims(np.sum(normals * directions[hits], axis=1),1) > 0, -1 * np.ones( (normals.shape[0], 1)), np.ones( (normals.shape[0], 1)) )

    return phong_shading(light_position, shininess, hits, samples, normals).reshape((LADO,LADO,3))  #final_samples, np.linalg.norm( gradients, axis=1)


def phong_shading(light_position, shininess, hits, samples, normals):
    light_directions = normalize( np.tile( light_position, (normals.shape[0],1) ) - samples[hits] )
    lambertian = np.max( [np.expand_dims(np.sum(normals * light_directions, axis=1),1), np.zeros((normals.shape[0],1))], axis=0 )
    
    reflect = lambda I, N : I - (2 * np.expand_dims( np.sum(N * I, axis=1),1)) * N
    R = reflect( (-1 * light_directions), normals )
    V = normalize(samples[hits])
    spec_angles = np.max( [np.sum( R * V, axis=1 ), np.zeros(normals.shape[0])], axis=0)

    specular = np.zeros_like(lambertian)
    specular[lambertian > 0] = np.expand_dims(np.power(spec_angles, shininess),1)[lambertian > 0]

    colors = np.zeros_like(samples)

    diffuse_color = np.array([0.3, 0.4, 0.7] )
    specular_color = np.array([1, 1, 1])
    colors[hits] = np.tile( diffuse_color, (normals.shape[0],1)) * lambertian + np.tile( specular_color, (normals.shape[0],1)) * specular
    
    return colors