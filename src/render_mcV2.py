import torch
import numpy as np
import trimesh
from torch.nn import functional as F
from src.evaluate import evaluate
from src.inverses import inverse
import numpy as np
from skimage.measure import marching_cubes
import torch
import mcubes

# Paper MeshUDF

def extract_fields2(decoder, latent_vec, N, gt_mode, device, alpha ):
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
    samples = torch.zeros(N ** 3, 7).to(device)
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
    
    gradients = np.zeros((samples.shape[0], 3))
    hessians = np.zeros((gradients.shape[0],3,3))
    #pred_df = torch.from_numpy( inverse( gt_mode, np.abs(evaluate( decoder, samples[:, :3], latent_vec, device=device, gradients=gradients, hessians=hessians ) ), alpha))
    pred_df = torch.from_numpy( np.abs(evaluate( decoder, samples[:, :3], latent_vec, device=device, gradients=gradients, hessians=hessians ) ) )
    
    gradients = torch.from_numpy( gradients )
    gradients = -1 * F.normalize(gradients, dim=-1)

    eigenvalues, eigenvectors = torch.linalg.eigh( torch.from_numpy(hessians) )
    pred_normals = eigenvectors[..., 2]

    pred_normals = torch.where(
        torch.sum( gradients * pred_normals, dim=-1 )[..., None] < 0,
        torch.ones( (pred_normals.shape[0],1)) * -1,
        torch.ones( (pred_normals.shape[0],1))
    ) * pred_normals

    grad_norms = torch.linalg.norm(gradients, axis=-1)[:,None]
    
    samples[..., 3] = (pred_df).squeeze(1).to(device)
    samples[..., 4:] = torch.where(
        torch.hstack([grad_norms, grad_norms, grad_norms]) < 0.04,
        pred_normals,
        gradients
    ).to(device)

    # Separate values in DF / gradients
    df_values = samples[:, 3]
    df_values = df_values.reshape(N, N, N)
    vecs = samples[:, 4:]
    vecs = vecs.reshape(N, N, N, 3)

    return df_values, vecs

def extract_mesh_CAP2( ndf, grad, resolution ):
    bbox_min, bbox_max = np.array([-1,-1,-1]), np.array([1,1,1])
    side_length = 2 / resolution
    v_all = []
    t_all = []
    threshold = 0.0007   # accelerate extraction
    v_num = 0
    for i in range(resolution-1):
        for j in range(resolution-1):
            for k in range(resolution-1):
                ndf_loc = ndf[i:i+2]
                ndf_loc = ndf_loc[:,j:j+2,:]
                ndf_loc = ndf_loc[:,:,k:k+2]
                if np.min(ndf_loc) > threshold:
                    continue
                grad_loc = grad[i:i+2]
                grad_loc = grad_loc[:,j:j+2,:]
                grad_loc = grad_loc[:,:,k:k+2]

                res = np.ones((2,2,2))
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            pos1 = bbox_min + np.array( [i,j,k]) *  side_length
                            pos2 = bbox_min + np.array( [i + ii ,j + jj ,k + kk]) *  side_length



                            if np.dot(grad_loc[0][0][0], grad_loc[ii][jj][kk]) < 0:
                                res[ii][jj][kk] = -ndf_loc[ii][jj][kk]
                            else:
                                res[ii][jj][kk] = ndf_loc[ii][jj][kk]

                if res.min()<0:
                    vertices, triangles = mcubes.marching_cubes(
                        res, 0)
                    # print(vertices)
                    # vertices -= 1.5

                    vertices[:,0] += i
                    vertices[:,1] += j
                    vertices[:,2] += k
                    triangles += v_num
                    # vertices = 
                    # vertices[:,1] /= 3  # TODO
                    v_all.append(vertices)
                    t_all.append(triangles)

                    v_num += vertices.shape[0]
                    # print(v_num)

    v_all = np.concatenate(v_all)
    t_all = np.concatenate(t_all)
    # Create mesh
    #print(v_all.shape)
    v_all = v_all / (resolution - 1.0) * (bbox_max - bbox_min)[None, :] + bbox_min[None, :]
    #print(v_all.shape)
    mesh = trimesh.Trimesh(v_all, t_all, process=False)
    
    return mesh


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

