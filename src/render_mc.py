import torch
import numpy as np
from scipy.sparse import coo_matrix
import trimesh
from torch.nn import functional as F
import sys
from collections import defaultdict
from src.evaluate import evaluate
from src.inverses import inverse
import numpy as np
from skimage.measure import marching_cubes
import torch

import sys
sys.path.append('src/marching_cubes')
from _marching_cubes_lewiner import udf_mc_lewiner

# Paper MeshUDF

def get_udf_normals_grid(decoder, latent_vec, N, gt_mode, alpha ):
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
    pred_df =  evaluate( decoder, samples[:, :3], latent_vec, gradients=gradients)
    udfs = torch.from_numpy( inverse( gt_mode, pred_df, alpha, min_step=0 ) ).float()
    gradients = torch.from_numpy( gradients )
    gradients = -1 * F.normalize(gradients, dim=-1)
    
    samples[..., 3] = udfs.squeeze(1)
    samples[..., 4:] = gradients

    # Separate values in DF / gradients
    df_values = samples[:, 3]
    df_values = df_values.reshape(N, N, N)
    vecs = samples[:, 4:]
    vecs = vecs.reshape(N, N, N, 3)

    return df_values, vecs

def get_mesh_udf(decoder, latent_vec, nsamples, device, gt_mode, alpha, smooth_borders=False, **kwargs ):
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
    df_values, normals = get_udf_normals_grid(decoder, latent_vec, nsamples, gt_mode, alpha )
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
    pred_df_verts = inverse( gt_mode, evaluate(decoder, xyz, latent_vec ), alpha, min_step=0 )

    # Remove faces that have vertices far from the surface
    filtered_faces = faces #faces[np.max(pred_df_verts[faces], axis=1)[:,0] < voxel_size / 6]
    
    if len(filtered_faces) == 0:
        raise ValueError("Could not find surface in volume")
    
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


def gen_sdf_coordinate_grid(N: int, voxel_size: float,
                           device: torch.device,
                           voxel_origin: list = [-1, -1, -1]) -> torch.Tensor:
    """Creates the coordinate grid for inference and marching cubes run.

    Parameters
    ----------
    N: int
        Number of elements in each dimension. Total grid size will be N ** 3

    voxel_size: number
        Size of each voxel

    t: float, optional
        Reconstruction time. Required for space-time models. Default value is
        None, meaning that time is not a model parameter

    device: string, optional
        Device to store tensors. Default is CPU

    voxel_origin: list[number, number, number], optional
        Origin coordinates of the volume. Must be the (bottom, left, down)
        coordinates. Default is [-1, -1, -1]

    Returns
    -------
    samples: torch.Tensor
        A (N**3, 3) shaped tensor with samples' coordinates. If t is not None,
        then the return tensor is has 4 columns instead of 3, with the last
        column equalling `t`.
    """
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())

    sdf_coord = 3

    # (x,y,z,sdf) if we are not considering time
    # (x,y,z,t,sdf) otherwise
    samples = torch.zeros(N ** 3, sdf_coord + 1, device=device,
                          requires_grad=False)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    return samples


def get_mesh_sdf(
    decoder,
    N=256,
    device=torch.device(0),
    max_batch=64 ** 3,
    offset=None,
    scale=None,
):
    decoder.eval()
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not
    # the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    samples = gen_sdf_coordinate_grid(N, voxel_size, device=device)

    sdf_coord = 3

    num_samples = N ** 3
    head = 0

    while head < num_samples:
        # print(head)
        sample_subset = samples[head:min(head + max_batch, num_samples), 0:sdf_coord]

        samples[head:min(head + max_batch, num_samples), sdf_coord] = (
            decoder(sample_subset)["model_out"]
            .squeeze()
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, sdf_coord]
    sdf_values = sdf_values.reshape(N, N, N)

    verts, faces, normals, values = convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        offset,
        scale,
    )

    return verts, faces, trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    if isinstance(pytorch_3d_sdf_tensor, torch.Tensor):
        numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    else:
        numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

    # Check if the cubes contains the zero-level set
    level = 0.0
    if level < numpy_3d_sdf_tensor.min() or level > numpy_3d_sdf_tensor.max():
        print(f"Surface level must be within volume data range.")
    else:
        verts, faces, normals, values = marching_cubes(
            numpy_3d_sdf_tensor, level, spacing=[voxel_size] * 3
        )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    return mesh_points, faces, normals, values

