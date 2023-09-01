import mcubes
import trimesh
import torch
import numpy as np
from src.evaluate import evaluate
from src.inverses import inverse
from src.util import normalize

def extract_fields( resolution, model, device):
    N = 32
    X = torch.linspace(-1, 1, resolution).split(N)
    Y = torch.linspace(-1, 1, resolution).split(N)
    Z = torch.linspace(-1, 1, resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    g = np.zeros([resolution, resolution, resolution, 3], dtype=np.float32)
    # with torch.no_grad():
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs)

                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
                gradients = torch.zeros_like(pts).detach().cpu().numpy()
                val = inverse( 'tanh', evaluate( model, pts, device=device, gradients=gradients ), 10, min_step=0 )
                gradients = normalize( gradients )

                u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val.reshape((32,32,32))
                g[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = gradients.reshape((32,32,32,3))

    return u, g

def extract_geometry( resolution, model, device):

    print('Extracting mesh with resolution: {}'.format(resolution))
    u, g = extract_fields( resolution, model, device)
    mesh = surface_extraction(u, g, resolution)

    return mesh


def surface_extraction(ndf, grad, resolution):
    v_all = []
    t_all = []
    threshold = 0.5   # accelerate extraction
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
                                res[ii][jj][kk] = -ndf_loc[ii][jj][kk]
                            else:
                                res[ii][jj][kk] = ndf_loc[ii][jj][kk]

                if res.min()<0:
                    vertices, triangles = mcubes.marching_cubes(
                        res, 0.0)
                    # print(vertices)
                    # vertices -= 1.5
                    # vertices /= 128
                    vertices[:,0] += i #/ resolution
                    vertices[:,1] += j #/ resolution
                    vertices[:,2] += k #/ resolution
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
    v_all = v_all / (resolution - 1.0) * (np.array([1]) - np.array([-1]))[None, :] + np.array([-1])[None, :]
    
    mesh = trimesh.Trimesh(v_all, t_all, process=False)
    
    return mesh