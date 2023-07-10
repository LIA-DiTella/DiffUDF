import torch
import numpy as np
from scipy.sparse import coo_matrix
import trimesh
from torch.nn import functional as F
import sys
from collections import defaultdict
import src.diff_operators as diff
from src.evaluate import evaluate

import sys
sys.path.append('src/marching_cubes')
from _marching_cubes_lewiner import udf_mc_lewiner

def get_udf_normals_grid(decoder, latent_vec, N, level_set ):
    """
    Fills a dense N*N*N regular grid by querying the decoder network
    Inputs: 
        decoder: coordinate network to evaluate
        latent_vec: conditioning vector
        N: grid size
        max_batch: number of points we can simultaneously evaluate
        fourier: are xyz coordinates encoded with fourier?
    Returns:
        df_values: (N,N,N) tensor representing distance field values on the grid
        vecs: (N,N,N,3) tensor representing gradients values on the grid, only for locations with a small
                distance field value
        samples: (N**3, 7) tensor representing (x,y,z, distance field, grad_x, grad_y, grad_z)
    """

    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 7)
    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = torch.div(overall_index, N, rounding_mode='floor') % N
    samples[:, 0] = torch.div(torch.div(overall_index, N, rounding_mode='floor'), N, rounding_mode='floor') % N
    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    samples.requires_grad = False

    samples.pin_memory()
    
    gradients = np.zeros((samples.shape[0], 3))
    udfs = torch.sqrt( torch.from_numpy( evaluate( decoder, samples[:, :3], latent_vec, gradients=gradients ) ).float().squeeze(1) )
    gradients_torch = torch.from_numpy( gradients )

    samples[..., 3] = torch.clip( torch.where( udfs > 0.09, udfs, 4 * torch.sum(gradients_torch ** 2, dim=1) ) - level_set, min=0 )
    samples[..., 4:] = - F.normalize( gradients_torch, dim= 1)

    # Separate values in DF / gradients
    df_values = samples[:, 3]
    df_values = df_values.reshape(N, N, N)
    vecs = samples[:, 4:]
    vecs = vecs.reshape(N, N, N, 3)

    return df_values, vecs

def get_mesh_udf(decoder, latent_vec, N_MC, device, smooth_borders=False, level_set=0):
    """
    Computes a triangulated mesh from a distance field network conditioned on the latent vector
    Inputs: 
        decoder: coordinate network to evaluate
        latent_vec: conditioning vector
        samples: already computed (N**3, 7) tensor representing (x,y,z, distance field, grad_x, grad_y, grad_z)
                    for a previous latent_vec, which is assumed to be close to the current one, if any
        indices: tensor representing the coordinates that need updating in the previous samples tensor (to speed
                    up iterations)
        N_MC: grid size
        fourier: are xyz coordinates encoded with fourier?
        gradient: do we need gradients?
        eps: length of the normal vectors used to derive gradients
        border_gradients: add a special case for border gradients?
        smooth_borders: do we smooth borders with a Laplacian?
    Returns:
        verts: vertices of the mesh
        faces: faces of the mesh
        mesh: trimesh object of the mesh
        samples: (N**3, 7) tensor representing (x,y,z, distance field, grad_x, grad_y, grad_z)
        indices: tensor representing the coordinates that need updating in the next iteration
    """
    ### 1: sample grid
    df_values, normals = get_udf_normals_grid(decoder, latent_vec, N=N_MC, level_set=level_set )
    df_values[df_values < 0] = 0
    ### 2: run our custom MC on it
    N = df_values.shape[0]
    voxel_size = 2.0 / (N - 1)
    verts, faces, _, _ = udf_mc_lewiner(df_values.cpu().detach().numpy(),
                                        normals.cpu().detach().numpy(),
                                        spacing=[voxel_size] * 3)
    verts = verts - 1 # since voxel_origin = [-1, -1, -1]
    ### 3: evaluate vertices DF, and remove the ones that are too far
    verts_torch = torch.from_numpy(verts).float().to(device)
    xyz = verts_torch
    pred_df_verts = evaluate(decoder, xyz, latent_vec )

    # Remove faces that have vertices far from the surface
    filtered_faces = faces[np.max(pred_df_verts[faces], axis=1)[:,0] < voxel_size / 6]
    filtered_mesh = trimesh.Trimesh(verts, filtered_faces)
    ### 4: clean the mesh a bit
    # Remove NaNs, flat triangles, duplicate faces
    filtered_mesh = filtered_mesh.process(validate=False) # DO NOT try to consistently align winding directions: too slow and poor results
    filtered_mesh.remove_duplicate_faces()
    filtered_mesh.remove_degenerate_faces()
    # Fill single triangle holes
    filtered_mesh.fill_holes()

    filtered_mesh_2 = trimesh.Trimesh(filtered_mesh.vertices, filtered_mesh.faces)
    # Re-process the mesh until it is stable:
    n_verts, n_faces, n_iter = 0, 0, 0
    while (n_verts, n_faces) != (len(filtered_mesh_2.vertices), len(filtered_mesh_2.faces)) and n_iter<10:
        filtered_mesh_2 = filtered_mesh_2.process(validate=False)
        filtered_mesh_2.remove_duplicate_faces()
        filtered_mesh_2.remove_degenerate_faces()
        (n_verts, n_faces) = (len(filtered_mesh_2.vertices), len(filtered_mesh_2.faces))
        n_iter += 1
        filtered_mesh_2 = trimesh.Trimesh(filtered_mesh_2.vertices, filtered_mesh_2.faces)

    filtered_mesh = trimesh.Trimesh(filtered_mesh_2.vertices, filtered_mesh_2.faces)

    if smooth_borders:
        # Identify borders: those appearing only once
        border_edges = trimesh.grouping.group_rows(filtered_mesh.edges_sorted, require_count=1)

        # Build a dictionnary of (u,l): l is the list of vertices that are adjacent to u
        neighbours  = defaultdict(lambda: [])
        for (u,v) in filtered_mesh.edges_sorted[border_edges]:
            neighbours[u].append(v)
            neighbours[v].append(u)
        border_vertices = np.array(list(neighbours.keys()))

        # Build a sparse matrix for computing laplacian
        pos_i, pos_j = [], []
        for k, ns in enumerate(neighbours.values()):
            for j in ns:
                pos_i.append(k)
                pos_j.append(j)

        sparse = coo_matrix((np.ones(len(pos_i)),   # put ones
                            (pos_i, pos_j)),        # at these locations
                            shape=(len(border_vertices), len(filtered_mesh.vertices)))

        # Smoothing operation:
        lambda_ = 0.3
        for _ in range(5):
            border_neighbouring_averages = sparse @ filtered_mesh.vertices / sparse.sum(axis=1)
            laplacian = border_neighbouring_averages - filtered_mesh.vertices[border_vertices]
            filtered_mesh.vertices[border_vertices] = filtered_mesh.vertices[border_vertices] + lambda_ * laplacian

    return torch.tensor(filtered_mesh.vertices).float().cuda(), torch.tensor(filtered_mesh.faces).long().cuda(), filtered_mesh