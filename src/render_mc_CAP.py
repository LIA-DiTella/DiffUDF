import mcubes
import trimesh
import torch
import numpy as np
from src.evaluate import evaluate
from src.inverses import inverse
from src.util import normalize

def extract_fields( resolution, model, device, bbox_min, bbox_max ):
    X = np.linspace(bbox_min[0], bbox_max[0], resolution)
    Y = np.linspace(bbox_min[1], bbox_max[1], resolution)
    Z = np.linspace(bbox_min[2], bbox_max[2], resolution)
    # with torch.no_grad():

    xx, yy, zz = np.meshgrid( X, Y, Z, indexing='ij' )

    pts = np.concatenate([xx[...,None], yy[...,None], zz[...,None]], axis=-1)
    
    gradients = np.zeros((resolution**3,3))
    hessians = np.zeros((resolution**3,3,3))

    val = evaluate( model, pts.reshape(resolution**3, 3), device=device, gradients=gradients, hessians=hessians )

    eigenvalues, eigenvectors = torch.linalg.eigh( torch.from_numpy(hessians) )

    pred_normals = eigenvectors[..., 2].numpy()

    pred_normals = np.where(
        np.sum( gradients * pred_normals ) < 0,
        np.ones( (pred_normals.shape[0],1)) * -1,
        np.ones( (pred_normals.shape[0],1))
    ) * pred_normals
    grad_norms = np.linalg.norm(gradients, axis=-1)[:,None]

    u = inverse('tanh', val - np.min(val), 10).reshape((resolution, resolution, resolution))
    g = np.where(
        np.hstack([grad_norms, grad_norms, grad_norms]) < .1,
        pred_normals,
        normalize(gradients)
    ).reshape((resolution, resolution, resolution,3))

    return u, g

def extract_geometry( resolution, model, device, bbox_min, bbox_max):

    print('Extracting mesh with resolution: {}'.format(resolution))
    u, g = extract_fields( resolution, model, device, bbox_min, bbox_max)
    mesh = surface_extraction(u, g, resolution, bbox_min, bbox_max)

    return mesh


def surface_extraction(ndf, grad, resolution, bbox_min, bbox_max):
    v_all = []
    t_all = []
    threshold = 0.0037   # accelerate extraction
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
                            if np.dot(grad_loc[0][0][0], grad_loc[ii][jj][kk]) < 0:
                                res[ii][jj][kk] = -1 * ndf_loc[ii][jj][kk]
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