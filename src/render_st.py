import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.util import normalize
import torch
from src.inverses import inverse
import open3d as o3d
import open3d.core as o3c
import numpy as np
from src.diff_operators import gradient, hessian, divergence, jacobian

def evaluate(model, samples, max_batch=64**2, output_size=1, device=torch.device(0), get_gradients=False, get_normals=False, get_curvatures='none'):
    head = 0
    amount_samples = samples.shape[0]

    evaluations = np.zeros( (amount_samples, output_size))
    if get_gradients:
        gradients = np.zeros((amount_samples, 3))

    if get_normals or get_curvatures:
        normals = np.zeros((amount_samples,3))
    
    if get_curvatures != 'none':
        curvatures = np.zeros((amount_samples, 1))

    while head < amount_samples:
        
        if torch.is_tensor(samples):
            inputs_subset = samples[head:min(head + max_batch, amount_samples), :]
        else:
            inputs_subset = torch.from_numpy(samples[head:min(head + max_batch, amount_samples), :]).float()

        inputs_subset = inputs_subset.to(device).unsqueeze(0)

        x, y =  model(inputs_subset).values()

        if get_gradients:
            gradient_torch = gradient(y,x)
            gradients[head:min(head + max_batch, amount_samples)] = gradient_torch.squeeze(0).detach().cpu().numpy()[..., :]

        if get_normals:
            hessians_torch = hessian(y,x)
            eigenvalues, eigenvectors = torch.linalg.eigh( hessians_torch )
            pred_normals = eigenvectors[..., 2]
            normals[head:min(head + max_batch, amount_samples)] = pred_normals[0].detach().cpu().numpy()[..., :]

            if get_curvatures == 'gaussian':
                shape_op, status = jacobian(pred_normals, x)

                extended_hessians = torch.zeros((shape_op.shape[1], 4,4)).to(device)
                extended_hessians[:, :3,:3] = shape_op[0, :,:]
                extended_hessians[:, :3, 3] = pred_normals
                extended_hessians[:, 3, :3] = pred_normals
                curvatures[head:min(head + max_batch, amount_samples)] = (-1 * torch.linalg.det(extended_hessians)).detach().cpu().numpy()[...,None]
            elif get_curvatures == 'mean':
                curvatures[head:min(head + max_batch, amount_samples)] = divergence( pred_normals, x ).detach().cpu().numpy()[...,:]
    
        evaluations[head:min(head + max_batch, amount_samples)] = y.squeeze(0).detach().cpu()
        head += max_batch

    if get_curvatures != 'none':
        return evaluations, normals, curvatures
    if get_normals:
        return evaluations, normals
    if get_gradients:
        return evaluations, gradients
        

def create_projectional_image( model, width, height, rays, t0, mask_rays, surface_eps, alpha, gt_mode, light_position, specular_comp, plot_curvatures, max_iterations=30, device=torch.device(0) ):
    # image es una lista de puntos. Tengo un rayo por cada punto en la imagen. Los rayos salen con dirección norm(image_i - origin) desde el punto mismo.
    hits = np.zeros_like(mask_rays, dtype=np.bool8)

    iteration = 0
    while np.sum(mask_rays) > 0 and iteration < max_iterations:
        gradients = np.zeros_like(t0[mask_rays])
        udfs, gradients = evaluate( model, t0[ mask_rays ], get_gradients=True, device=device)
        steps = inverse( gt_mode, np.abs(udfs), alpha )

        t0[mask_rays] += rays[mask_rays] * steps

        threshold_mask = np.abs(udfs).flatten() < surface_eps
        indomain_mask = np.logical_and( np.all( t0[mask_rays] > -1, axis=1 ), np.all( t0[mask_rays] < 1, axis=1 ))
        hits[mask_rays] += np.logical_and( threshold_mask, indomain_mask)
        mask_rays[mask_rays] *= np.logical_and( np.logical_not(threshold_mask), indomain_mask )
        
        iteration += 1

    if np.sum(hits) == 0:
        raise ValueError(f"Ray tracing did not converge in {max_iterations} iterations to any point at distance {surface_eps} or lower from surface.")

    amount_hits = np.sum(hits)
    gradients = np.zeros((amount_hits, 3))
    hessians = np.zeros((amount_hits, 3, 3))

    if gt_mode == 'siren':
        udfs, gradients = evaluate( model, t0[hits], get_gradients=True, device=device)
        normals = normalize(gradients)
        return phong_shading(light_position, specular_comp, 40, hits, t0, normals).reshape((height,width,3)) 
    else:
        if plot_curvatures != 'none':
            udfs, normals, curvatures = evaluate( model, t0[hits], get_normals=True, get_curvatures=plot_curvatures, device=device )
            # podria ser que las normales apunten para el otro lado. las tengo que invertir si  < direccion, normal > = cos(tita) > 0
            direction_alignment = np.sign(np.expand_dims(np.sum(normals * rays[hits], axis=1),1)) * -1
            normals *= direction_alignment

            if plot_curvatures == 'mean':
                curvatures *= direction_alignment / 2

            cmap = cm.get_cmap('bwr')
            curvatures = np.clip( curvatures, np.percentile(curvatures, 5), np.percentile(curvatures,95))
            curvatures -= np.min(curvatures)
            curvatures/= np.max(curvatures)

            return phong_shading(light_position, specular_comp, 40, hits, t0, normals, color_map=cmap(curvatures.squeeze(1))[:,:3]).reshape((height,width,3))     
        else:
            udfs, normals = evaluate( model, t0[hits], get_normals=True, device=device )
            direction_alignment = np.sign(np.expand_dims(np.sum(normals * rays[hits], axis=1),1)) * -1
            normals *= direction_alignment
            return phong_shading( light_position, specular_comp, 40, hits, t0, normals ).reshape((height,width,3))     


def phong_shading(light_position, specular_comp, shininess, hits, samples, normals, color_map=None):
    light_directions = normalize( np.tile( light_position, (normals.shape[0],1) ) - samples[hits] )
    lambertian = np.max( [np.expand_dims(np.sum(normals * light_directions, axis=1),1), np.zeros((normals.shape[0],1))], axis=0 )
    
    reflect = lambda I, N : I - (2 * np.expand_dims( np.sum(N * I, axis=1),1)) * N
    R = reflect( (-1 * light_directions), normals )
    V = normalize(samples[hits])
    spec_angles = np.max( [np.sum( R * V, axis=1 ), np.zeros(normals.shape[0])], axis=0)

    if specular_comp:
        specular = np.zeros_like(lambertian)
        specular[lambertian > 0] = np.expand_dims(np.power(spec_angles, shininess),1)[lambertian > 0]
    else:
        specular = 0

    colors = np.ones_like(samples)


    if color_map is None:
        diffuse_color = np.tile( np.array([0.7, 0.7, 0.7] ), (normals.shape[0],1))
        specular_color = np.tile( np.array([0.7, 0.7, 0.7] ), (normals.shape[0],1))
        ambient_color = np.tile( np.array([0.2, 0.2, 0.2] ), (normals.shape[0],1))
    else:
        diffuse_color = color_map
        specular_color = color_map
        #ambient_color = np.tile( np.array([0.2, 0.2, 0.2] ), (normals.shape[0],1))
        ambient_color = np.clip( color_map - np.tile( np.array([0.7,0.7,0.7]), (normals.shape[0],1) ), 0.1, 1)

    colors[hits] = np.clip( 
        diffuse_color * lambertian + 
        specular_color * specular +
        ambient_color , 0, 1)
    
    return colors

def create_projectional_image_gt( mesh_file, width, height, rays, t0, mask_rays, light_position, specular_comp,surface_eps=0.001, max_iterations=30 ):
    # image es una lista de puntos. Tengo un rayo por cada punto en la imagen. Los rayos salen con dirección norm(image_i - origin) desde el punto mismo.
    mesh = o3d.t.io.read_triangle_mesh(mesh_file)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    hits = np.zeros_like(mask_rays, dtype=np.bool8)
    iteration = 0
    while np.sum(mask_rays) > 0 and iteration < max_iterations:
        udfs = np.expand_dims(scene.compute_distance( o3c.Tensor(t0[mask_rays], dtype=o3c.float32) ).numpy(), -1)

        t0[mask_rays] += rays[mask_rays] * np.hstack([udfs, udfs, udfs])

        mask = udfs.squeeze(-1) < surface_eps
        hits[mask_rays] += mask
        mask_rays[mask_rays] *= np.logical_not(mask)

        mask_rays *= np.logical_and( np.all( t0 > -1.3, axis=1 ), np.all( t0 < 1.3, axis=1 ) )
        
        iteration += 1
    
    if np.sum(hits) == 0:
        raise ValueError(f"Ray tracing did not converge in {max_iterations} iterations to any point at distance {surface_eps} or lower from surface.")

    grad_eps = 0.0001
    normals = normalize( np.vstack( [
        (scene.compute_signed_distance( o3c.Tensor(t0[hits] + np.tile( np.eye(1, 3, i), (np.sum(hits),1)) * grad_eps, dtype=o3c.float32) ).numpy() -
        scene.compute_signed_distance( o3c.Tensor(t0[hits] - np.tile( np.eye(1, 3, i), (np.sum(hits),1)) * grad_eps, dtype=o3c.float32) ).numpy()) / (2*grad_eps)
        for i in range(3)]).T )
    
    normals *= np.where( np.expand_dims(np.sum(normals * rays[hits], axis=1),1) > 0, -1 * np.ones( (normals.shape[0], 1)), np.ones( (normals.shape[0], 1)) )

    return phong_shading(light_position, specular_comp, 40, hits, t0, normals).reshape((width,height,3)) 
