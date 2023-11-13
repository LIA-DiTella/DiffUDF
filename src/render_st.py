import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.util import normalize
import torch
import torch.nn.functional as F
from src.inverses import inverse
import open3d as o3d
import open3d.core as o3c
import numpy as np
from src.diff_operators import gradient, hessian, divergence, jacobian

def evaluate(model, samples, max_batch=64**2, device=torch.device(0)):
    head = 0
    amount_samples = samples.shape[0]

    evaluations = []
    inputs = []

    while head < amount_samples:
        
        if torch.is_tensor(samples):
            inputs_subset = samples[head:min(head + max_batch, amount_samples), :]
        else:
            inputs_subset = torch.from_numpy(samples[head:min(head + max_batch, amount_samples), :]).float()

        inputs_subset = inputs_subset.to(device).unsqueeze(0)

        x, y =  model(inputs_subset).values()
        
        evaluations.append( y )
        inputs.append( x )

        head += max_batch

    return inputs, evaluations


def batched_op( inputs, outputs, op, *args, **kwargs ):
    return [ op(x,y, *args, **kwargs) for x,y in zip(inputs, outputs) ]

def compute_curvature( inputs, normals, curvature='mean', device=torch.device(0)):
    if curvature == 'mean':
        return (divergence( normals, inputs ) / 2).detach().cpu()
    elif curvature == 'gaussian':
        shape_op, status = jacobian(normals, inputs)

        extended_hessians = torch.zeros((shape_op.shape[1], 4,4)).to(device)
        extended_hessians[:, :3,:3] = shape_op[0, :,:]
        extended_hessians[:, :3, 3] = normals
        extended_hessians[:, 3, :3] = normals

        return (-1 * torch.linalg.det(extended_hessians)[None, ..., None]).detach().cpu()
    else:
        return None

def compute_normals_and_cd( inputs, outputs):
    hessians_torch = hessian(outputs,inputs)
    eigenvalues, eigenvectors = torch.linalg.eigh( hessians_torch )
    pred_normals = eigenvectors[..., 2]
    
    return pred_normals, eigenvectors[...,:2].detach().cpu()

def compute_grad(inputs, outputs):
    return gradient(outputs, inputs)

def create_projectional_image( 
        model,
        rays,
        t0,
        mask_rays,
        network_config,
        rendering_config,
        device ): 
    hits = propagate_rays(model, rays, t0, mask_rays, network_config, rendering_config, device)
    grad_descent(model, t0, hits, network_config, rendering_config, device)

    inputs, udfs = evaluate( model, t0[ hits ], device=device)

    if network_config['gt_mode'] == 'siren':
        gradients = torch.hstack(batched_op(inputs, udfs, compute_grad )).squeeze(0).detach().cpu().numpy()
        udfs = udfs.squeeze(0).detach().cpu().numpy()
        normals = normalize(gradients)
        return phong_shading(
            rendering_config['light_position'], 
            rendering_config['shininess'], 
            hits, t0, normals).reshape((rendering_config['height'],rendering_config['width'],3)) 
    else:
        cmap = cm.get_cmap('bwr')
        normals_and_cd = batched_op( inputs, udfs, compute_normals_and_cd )
        normals = [ p[0] for p in normals_and_cd ]
        pcd = [ p[1] for p in normals_and_cd ]

        if rendering_config['plot_curvatures'] in ['mean', 'gaussian']:
            curvatures = batched_op( inputs, normals, compute_curvature, curvature=rendering_config['plot_curvatures'], device=device )
            curvatures = torch.hstack(curvatures).squeeze(0).numpy()

        else:
            curvatures= None

        normals = torch.hstack(normals).squeeze(0).detach().cpu().numpy()
        pcd = torch.hstack(pcd).squeeze(0).detach().cpu().numpy()

        direction_alignment = np.sign(np.expand_dims(np.sum(normals * rays[hits], axis=1),1)) * -1
        normals *= direction_alignment

        if rendering_config['plot_curvatures'] == 'mean':
            curvatures *= direction_alignment

        if curvatures is not None:
            curvatures = np.clip( curvatures, np.percentile(curvatures, rendering_config['curv_low_bound']), np.percentile(curvatures,rendering_config['curv_high_bound']))
            curvatures -= np.mean(curvatures)
            curvatures /= 2* np.max(np.abs(curvatures))
            curvatures += 0.5
            curvatures = cmap(curvatures.squeeze(1))[:,:3]

        if rendering_config['reflection_method'] == 'blinn-phong':
            return phong_shading(
                rendering_config['light_position'], 
                rendering_config['shininess'], 
                hits, t0, normals, color_map=curvatures).reshape((rendering_config['height'],rendering_config['width'],3)) 

        elif rendering_config['reflection_method'] == 'ward':
            return ward_reflectance(
                rendering_config['light_position'],
                rendering_config['camera_position'], 
                hits, 
                t0, 
                normals, 
                alpha1=rendering_config['alpha1'], 
                alpha2=rendering_config['alpha2'], 
                pc1=pcd[..., 0],
                pc2=pcd[..., 1],
                color_map=curvatures ).reshape((rendering_config['height'],rendering_config['width'],3))     


def propagate_rays(model, rays, t0, mask_rays, network_config, rendering_config, device):
    hits = np.zeros_like(mask_rays, dtype=np.bool8)

    iteration = 0
    while np.sum(mask_rays) > 0 and iteration < rendering_config['max_iterations']:
        _, udfs = evaluate( model, t0[ mask_rays ], device=device)
        udfs = torch.hstack(udfs).squeeze(0).detach().cpu().numpy()
        steps = inverse( network_config['gt_mode'], np.abs(udfs), network_config['alpha'] )
        #steps = np.abs(udfs)

        t0[mask_rays] += rays[mask_rays] * steps

        if network_config['gt_mode'] == 'siren':
            threshold_mask = udfs.flatten() < rendering_config['surface_threshold']
        else:
            #threshold_mask = np.abs(udfs).flatten() < rendering_config['surface_threshold']
            threshold_mask = np.abs(steps).flatten() < rendering_config['surface_threshold']
            
        indomain_mask = np.logical_and( np.all( t0[mask_rays] > -1, axis=1 ), np.all( t0[mask_rays] < 1, axis=1 ))
        hits[mask_rays] += np.logical_and( threshold_mask, indomain_mask)
        mask_rays[mask_rays] *= np.logical_and( np.logical_not(threshold_mask), indomain_mask )
        
        iteration += 1

    if np.sum(hits) == 0:
        raise ValueError(f"Ray tracing did not converge in {rendering_config['max_iterations']} iterations to any point at distance {rendering_config['surface_threshold']} or lower from surface.")
    return hits

def grad_descent( model, t0, mask_rays, network_config, rendering_config, device ):
    for step in range(rendering_config['gd_steps']):
        inputs, udfs = evaluate( model, t0[ mask_rays ], device=device)
        gradients = torch.hstack(batched_op(inputs, udfs, compute_grad )).squeeze(0).detach().cpu().numpy()
        gradients = normalize(gradients)
        
        udfs = torch.hstack(udfs).squeeze(0).detach().cpu().numpy()
        steps = inverse( network_config['gt_mode'], np.abs(udfs), network_config['alpha'] )

        t0[mask_rays] -= gradients * steps

def phong_shading(light_position, shininess, hits, samples, normals, color_map=None):
    light_directions = normalize( np.tile( light_position, (normals.shape[0],1) ) - samples[hits] )
    lambertian = np.max( [np.expand_dims(np.sum(normals * light_directions, axis=1),1), np.zeros((normals.shape[0],1))], axis=0 )
    
    reflect = lambda I, N : I - (2 * np.expand_dims( np.sum(N * I, axis=1),1)) * N
    R = reflect( (-1 * light_directions), normals )
    V = normalize(samples[hits])
    spec_angles = np.max( [np.sum( R * V, axis=1 ), np.zeros(normals.shape[0])], axis=0)

    specular = np.zeros_like(lambertian)
    if shininess > 0:
        specular[lambertian > 0] = np.expand_dims(np.power(spec_angles, shininess),1)[lambertian > 0]
            

    colors = np.ones_like(samples)

    if color_map is None:
        diffuse_color = np.tile( np.array([0.7, 0.7, 0.7] ), (normals.shape[0],1))
        specular_color = np.tile( np.array([0.7, 0.7, 0.7] ), (normals.shape[0],1))
        ambient_color = np.tile( np.array([0.2, 0.2, 0.2] ), (normals.shape[0],1))
    else:
        diffuse_color = color_map * 0.7
        specular_color = color_map * 0.7
        ambient_color = color_map * 0.2

    colors[hits] = np.clip( 
        diffuse_color * lambertian + 
        specular_color * specular +
        ambient_color , 0, 1)
    
    return colors

def ward_reflectance(light_position, camera_position, hits, samples, normals, alpha1, alpha2, pc1, pc2, color_map=None):
    light_directions = normalize( np.tile( light_position, (normals.shape[0],1) ) - samples[hits] )
    lambertian = np.max( [np.expand_dims(np.sum(normals * light_directions, axis=1),1), np.zeros((normals.shape[0],1))], axis=0 )
    
    reflect = lambda I, N : I - (2 * np.expand_dims( np.sum(N * I, axis=1),1)) * N
    R = reflect( (-1 * light_directions), normals )
    V = normalize(samples[hits])

    colors = np.ones_like(samples)

    viewer_direcions = normalize( np.tile( camera_position, (normals.shape[0],1) ) - samples[hits] )
    H = normalize( viewer_direcions + light_directions )
    dot = lambda x,y: np.sum( x* y, axis=-1)
    weight = 1 / (4 * np.pi * alpha1 * alpha2 * np.sqrt( dot(normals, light_directions) * dot(normals,viewer_direcions) ))
    specular = weight * np.exp(
        -2 * ( (dot(H, pc1) / alpha1)**2 + (dot(H, pc2) / alpha2)**2 ) / (1+ dot(normals, H))
    )
    specular = specular[...,None]
    specular = np.nan_to_num(specular)

    if color_map is None:
        diffuse_color = np.tile( np.array([0.7, 0.7, 0.7] ), (normals.shape[0],1))
        specular_color = np.tile( np.array([0.7, 0.7, 0.7] ), (normals.shape[0],1))
        ambient_color = np.tile( np.array([0.2, 0.2, 0.2] ), (normals.shape[0],1))
    else:
        diffuse_color = color_map * 0.7
        specular_color = color_map * 0.7
        ambient_color = color_map * 0.2

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
