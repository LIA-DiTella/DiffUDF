'''From the DeepSDF repository https://github.com/facebookresearch/DeepSDF
'''

import numpy as np
import plyfile
from skimage.measure import marching_cubes
import time
import torch


def gen_mc_coordinate_grid(N: int, distance, voxel_size: float, device: str = "cpu",
                           voxel_origin: list = [-1, -1, -1]) -> torch.Tensor:
    """Creates the coordinate grid for inference and marching cubes run.

    Parameters
    ----------
    N: int
        Number of elements in each dimension. Total grid size will be N ** 3
    
    distance: float

    voxel_size: number
        Size of each voxel


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

    samples = torch.zeros(N ** 3, 4, device=device,
                          requires_grad=False)

    # transform last 3 columns
    # to be the x, y, z index
    samples[:, 3] = overall_index % N
    samples[:, 2] = (overall_index.long() / N) % N
    samples[:, 1] = ((overall_index.long() / N) / N) % N

    # transform last 3 columns
    # to be the x, y, z coordinate
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[2]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[1]
    samples[:, 3] = (samples[:, 3] * voxel_size) + voxel_origin[0]

    samples[:, 0] = distance

    return samples


def create_mesh(
    decoder,
    distance,
    filename="",
    N=256,
    max_batch=64 ** 3,
    offset=None,
    scale=None,
    device="cpu",
    silent=False
):
    decoder.eval()
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not
    # the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    samples = gen_mc_coordinate_grid(N,distance, voxel_size,  device=device )

    num_samples = N ** 3
    head = 0
    sdf_coord = 0

    start = time.time()
    while head < num_samples:
        # print(head)
        sample_subset = samples[head:min(head + max_batch, num_samples), :]
        
        samples[head:min(head + max_batch, num_samples), sdf_coord] = (
            decoder(sample_subset)["model_out"]
            .squeeze()
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, sdf_coord]
    sdf_values = sdf_values.reshape(N, N, N)
    
    end = time.time()
    if not silent:
        print(f"Sampling took: {end-start} s")

    verts, faces, normals, values = convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        offset,
        scale,
    )

    if filename:
        if not silent:
            print(f"Saving mesh to {filename}")

        save_ply(verts, faces, filename)

        if not silent:
            print("Done")

    return verts, faces, normals, values


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


def save_ply(
        verts: np.array,
        faces: np.array,
        filename: str,
        vertex_attributes: list = None
) -> None:
    """Converts the vertices and faces into a PLY format, saving the resulting
    file.

    Parameters
    ----------
    verts: np.array
        An NxD matrix with the vertices and its attributes (normals,
        curvatures, etc.). Note that we expect verts to have at least 3
        columns, each corresponding to a vertex coordinate.

    faces: np.array
        An Fx3 matrix with the vertex indices for each triangle.

    filename: str
        Path to the output PLY file.

    vertex_attributes: list of tuples
        A list with the dtypes of vertex attributes other than coordinates.

    Examples
    --------
    > # This creates a simple triangle and saves it to a file called
    > #"triagle.ply"
    > verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    > faces = np.array([[0, 1, 2]])
    > save_ply(verts, faces, "triangle.ply")

    > # Writting normal information as well
    > verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    > faces = np.array([[0, 1, 2]])
    > normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    > attrs = [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    > save_ply(verts, faces, "triangle_normals.ply", vertex_attributes=attrs)
    """
    # try writing to the ply file
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    dtypes = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if vertex_attributes is not None:
        dtypes[3:3] = vertex_attributes

    verts_tuple = np.zeros(
        (num_verts,),
        dtype=dtypes
    )

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(verts[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(
        faces_building,
        dtype=[("vertex_indices", "i4", (3,))]
    )

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(filename)
