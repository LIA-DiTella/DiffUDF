import numpy as np
import matplotlib.cm as cm
from src.util import normalize
import torch
from src.evaluate import evaluate
from src.inverses import inverse
import open3d as o3d
import open3d.core as o3c

def create_projectional_image( model, width, height, rays, t0, mask_rays, surface_eps, alpha, gt_mode, light_position, max_iterations=30, device=torch.device(0) ):
    # image es una lista de puntos. Tengo un rayo por cada punto en la imagen. Los rayos salen con dirección norm(image_i - origin) desde el punto mismo.
    hits = np.zeros_like(mask_rays, dtype=np.bool8)

    iteration = 0
    while np.sum(mask_rays) > 0 and iteration < max_iterations:
        gradients = np.zeros_like(t0[mask_rays])
        udfs = evaluate( model, t0[ mask_rays ], gradients=gradients, device=device)
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

    udfs = evaluate( model, t0[hits], gradients=gradients, hessians=hessians, device=device)

    if gt_mode == 'siren':
        normals = normalize(gradients)
    else:
        normals = np.array( [ np.linalg.eigh(hessian)[1][:,2] for hessian in hessians ] )
        # podria ser que las normales apunten para el otro lado. las tengo que invertir si  < direccion, normal > = cos(tita) > 0
        normals *= np.where( np.expand_dims(np.sum(normals * rays[hits], axis=1),1) > 0, -1 * np.ones( (normals.shape[0], 1)), np.ones( (normals.shape[0], 1)) )
    

    return phong_shading(light_position, 0, hits, t0, normals).reshape((height,width,3))  #final_samples, np.linalg.norm( gradients, axis=1)


def phong_shading(light_position, shininess, hits, samples, normals):
    light_directions = normalize( np.tile( light_position, (normals.shape[0],1) ) - samples[hits] )
    lambertian = np.max( [np.expand_dims(np.sum(normals * light_directions, axis=1),1), np.zeros((normals.shape[0],1))], axis=0 )
    
    reflect = lambda I, N : I - (2 * np.expand_dims( np.sum(N * I, axis=1),1)) * N
    R = reflect( (-1 * light_directions), normals )
    V = normalize(samples[hits])
    spec_angles = np.max( [np.sum( R * V, axis=1 ), np.zeros(normals.shape[0])], axis=0)

    specular = np.zeros_like(lambertian)
    specular[lambertian > 0] = np.expand_dims(np.power(spec_angles, shininess),1)[lambertian > 0]

    colors = np.ones_like(samples)

    diffuse_color = np.array([0.7, 0.7, 0.7] )
    specular_color = np.array([0.7, 0.7, 0.7])
    ambient_color = np.array( [0.2, 0.2, 0.2])
    colors[hits] = np.clip( 
        np.tile( diffuse_color, (normals.shape[0],1)) * lambertian + 
        #np.tile( specular_color, (normals.shape[0],1)) * specular +
        ambient_color , 0, 1)
    
    return colors

def create_projectional_image_gt( mesh_file, width, height, rays, t0, mask_rays, light_position, surface_eps=0.001, max_iterations=30 ):
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

    return phong_shading(light_position, 0, hits, t0, normals).reshape((width,height,3))  #final_samples, np.linalg.norm( gradients, axis=1)
