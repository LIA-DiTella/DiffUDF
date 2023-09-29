import mcubes
import trimesh
import torch
import numpy as np
from src.evaluate import evaluate
from src.inverses import inverse
import open3d as o3d
from scipy.spatial import KDTree
from src.util import normalize

def extract_fields( resolution, model, device, bbox_min, bbox_max, alpha ):
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

    u = inverse('tanh', np.abs(val ), alpha).reshape((resolution, resolution, resolution))
    g = np.where(
        np.hstack([grad_norms, grad_norms, grad_norms]) < .1,
        pred_normals,
        normalize(gradients)
    ).reshape((resolution, resolution, resolution,3))

    return u, g

def extract_gt_field( resolution, mesh, bbox_min, bbox_max, alpha ):
    X = np.linspace(bbox_min[0], bbox_max[0], resolution)
    Y = np.linspace(bbox_min[1], bbox_max[1], resolution)
    Z = np.linspace(bbox_min[2], bbox_max[2], resolution)
    # with torch.no_grad():

    xx, yy, zz = np.meshgrid( X, Y, Z, indexing='ij' )

    pts = np.concatenate([xx[...,None], yy[...,None], zz[...,None]], axis=-1)

    subpc = np.asarray(mesh.vertices)#[np.random.choice(len(np.asarray(mesh.vertices)), 100000, replace=False),:]
    tree = KDTree( subpc )
    print('quering...')
    distances, indexs = tree.query(pts, 1)

    u = (distances) * np.tanh(alpha * distances)
    g = (pts - subpc[indexs])
    print('extracting mesh...')

    return u, g

def extract_geometry( resolution, model, device, bbox_min, bbox_max, alpha):
    if bbox_min is None:
        bbox_min = [-1,-1,-1]
    
    if bbox_max is None:
        bbox_max = [1,1,1]

    print('Extracting mesh with resolution: {}'.format(resolution))
    u, g = extract_fields( resolution, model, device, np.array(bbox_min), np.array(bbox_max), alpha)
    mesh = surface_extraction(u, g, resolution, np.array(bbox_min), np.array(bbox_max), alpha)

    return mesh

def surface_extraction(ndf, grad, resolution, bbox_min, bbox_max, alpha):
    v_all = []
    t_all = []
    threshold = 0.008   # accelerate extraction
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

                anchor_i,anchor_j,anchor_k = np.unravel_index(np.argmax(ndf_loc),ndf_loc.shape)

                res = np.ones((2,2,2))
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            #distance = np.linalg.norm( bbox_min + np.array([i,j,k])*((bbox_max - bbox_min)/resolution) - (bbox_min + np.array([i+ii,j+jj,k+kk])*((bbox_max - bbox_min)/resolution)))
                            inverse_val = np.abs(ndf_loc[ii,jj,kk])#inverse('tanh', np.abs(ndf_loc[ii,jj,kk]), alpha)

                            if np.dot(grad_loc[anchor_i, anchor_j, anchor_k], grad_loc[ii,jj,kk]) < 0:
                                sign = -1 * np.sign(ndf_loc[anchor_i, anchor_j, anchor_k])
                                res[ii,jj,kk] = sign * inverse_val
                            else:
                                sign = np.sign(ndf_loc[anchor_i, anchor_j, anchor_k])
                                res[ii,jj,kk] = sign * inverse_val

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