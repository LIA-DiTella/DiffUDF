import numpy as np
import json
import torch
import open3d as o3d
import open3d.core as o3c
import argparse
from src.model import SIREN
from src.evaluate import evaluate
from PIL import Image
from scipy.spatial import KDTree
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from src.util import normalize

def imagen_dist( axis, distancias, niveles, eps=0.0005, negs=False, color_map='br', min_val=-1.5, max_val=1.5, contour=False):
    masked_distancias = distancias
    for v in niveles:
        masked_distancias = np.ma.masked_inside( masked_distancias, - eps, eps )

    if negs:
        masked_distancias = np.ma.masked_less(masked_distancias, 0)
    
    pos = axis.imshow( 
        masked_distancias.reshape(np.sqrt(len(distancias)).astype(np.uint32), np.sqrt(len(distancias)).astype(np.uint32)), 
        cmap=color_map, 
        interpolation='none', 
        vmin=min_val, 
        vmax=max_val
    )

    if contour:
        axis.contour(
            masked_distancias.reshape(np.sqrt(len(distancias)).astype(np.uint32), np.sqrt(len(distancias)).astype(np.uint32)),
            levels= np.linspace(min_val,max_val,18), colors='black', linewidths=0.5)
        pos = axis.contourf(
            masked_distancias.reshape(np.sqrt(len(distancias)).astype(np.uint32), np.sqrt(len(distancias)).astype(np.uint32)),
            levels= np.linspace(min_val,max_val,18), cmap=color_map)
    

    axis.contour( 
        np.ma.masked_outside( distancias, -eps, eps ).reshape(np.sqrt(len(distancias)).astype(np.uint32), np.sqrt(len(distancias)).astype(np.uint32)),
        colors='black', linewidths=0.5
    )
        
    axis.set_xticks([])
    axis.set_yticks([])

    return pos
    
def generate_df( model_path, mesh_path, output_path, options ):

    model = SIREN(
            n_in_features= 3,
            n_out_features=1,
            hidden_layer_config=options['hidden_layer_nodes'],
            w0=options['weight0'],
            ww=None,
            activation=options.get('activation', 'sine')
    )
    model.load_state_dict( torch.load(model_path, weights_only=True ))

    SAMPLES = options['width'] ** 2
    BORDES = [1, -1]
    EJEPLANO = [2,1,0]
    OFFSETPLANO = 0.0

    device_torch = torch.device(options['device'])
    model.to(device_torch)

    mesh = o3d.t.io.read_triangle_mesh(mesh_path)

    ranges = np.linspace(BORDES[0], BORDES[1], options['width'])
    i_1, i_2 = np.meshgrid( ranges, ranges )
    samples = np.concatenate(
            np.concatenate( np.array([np.expand_dims(i_1, 2), 
                                np.expand_dims(i_2, 2), 
                                np.expand_dims(np.ones_like(i_1) * OFFSETPLANO, 2)])[EJEPLANO]
                        , axis=2 ),
            axis=0)

    gradients = np.zeros((SAMPLES, 3))
    hessians = np.zeros((SAMPLES, 3, 3))
    pred_distances = evaluate( model, samples, device=device_torch, gradients=gradients, hessians=hessians )
    pred_grad_norm = np.linalg.norm( gradients , axis=1 ).reshape((SAMPLES, 1))

    gradients = normalize(gradients)
    eigenvalues, eigenvectors = torch.linalg.eigh( torch.from_numpy(hessians) )
    pred_normals = eigenvectors[..., 2].numpy()

    pred_normals = np.where(
        np.sum( gradients * pred_normals, axis=-1 )[..., None] < 0,
        np.ones( (pred_normals.shape[0],1)) * -1,
        np.ones( (pred_normals.shape[0],1))
    ) * pred_normals
    
    normals = np.where(
        np.concatenate( [pred_grad_norm , pred_grad_norm, pred_grad_norm], axis=-1) < 0.04,
        pred_normals,
        gradients
    )

    #cyclic_cmap = cm.get_cmap('twilight_shifted')
    #grad_map = cyclic_cmap( np.arccos( normals[:, 0] ) / np.pi )[:,:3]

    normals *= np.hstack([ np.ones((len(normals), 2)), np.sign(normals[:, 2]).reshape((len(normals), 1))])
    grad_map = ( normals + np.ones_like(normals) ) / 2

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    gt_distances = (scene.compute_distance( o3c.Tensor(samples, dtype=o3c.float32) ).numpy()).reshape((SAMPLES, 1))
    if options['gt_mode'] == 'squared':
        gt_grad_norm = 2 * options['alpha'] * gt_distances
        gt_distances = options['alpha'] * (gt_distances ** 2)
    elif options['gt_mode'] == 'tanh':
        tanh = np.tanh( options['alpha'] * gt_distances ) 
        gt_grad_norm = tanh + options['alpha'] * gt_distances * (1 - tanh ** 2)
        gt_distances = gt_distances * tanh
    elif options['gt_mode'] == 'siren':
        gt_grad_norm = np.where( gt_distances < options['surf_thresh'], np.zeros_like(gt_distances), np.ones_like(gt_distances))
        gt_distances = gt_distances
    else:
        raise ValueError('gt_mode not valid')
    
    #plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,9), dpi=500)

    #max_val = np.max( np.concatenate([gt_distances,pred_distances,gt_grad_norm,pred_grad_norm]))

    color_map = 'bwr_r'

    pos = imagen_dist( axes.flat[0] ,np.clip(gt_distances, a_min=None,a_max=1.5), [0], negs=True, color_map=color_map, eps=options['surf_thresh'], contour=True)
    imagen_dist( axes.flat[1] ,np.clip(pred_distances,a_min=None, a_max=1.5), [0], negs=True, color_map=color_map, eps=options['surf_thresh'], contour=True)
    imagen_dist( axes.flat[2] ,np.clip(gt_grad_norm, a_min=None,a_max=1.5), [0], negs=True, color_map=color_map, eps=options['surf_thresh'])
    imagen_dist( axes.flat[3] ,np.clip(pred_grad_norm, a_min=None,a_max=1.5), [0], negs=True, color_map=color_map, eps=options['surf_thresh'])

    axes.flat[0].set_title(r'Ground truth slices')
    axes.flat[1].set_title(r'Predicted value slices')
    axes.flat[0].set_ylabel(r'$f$', rotation=0, labelpad=12, size='large')
    axes.flat[2].set_ylabel(r'$\left \| \nabla f \right \|$', rotation=0, labelpad=24, size='large')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(pos, cax=cbar_ax)
    fig.savefig(output_path + 'distance_fields.png')

    im = Image.fromarray((grad_map.reshape(np.sqrt(SAMPLES).astype(np.uint32), np.sqrt(SAMPLES).astype(np.uint32), 3) * 255).astype(np.uint8))
    im.save( output_path +'pred_grad.png', 'PNG')

def generate_df_pc( model_path, pc_path, output_path, options ):

    model = SIREN(
            n_in_features= 3,
            n_out_features=1,
            hidden_layer_config=options['hidden_layer_nodes'],
            w0=options['weight0'],
            ww=None,
            activation=options.get('activation', 'sine')
    )
    model.load_state_dict( torch.load(model_path, weights_only=True))

    SAMPLES = options['width'] ** 2
    BORDES = [1, -1]
    EJEPLANO = [2,1,0]
    OFFSETPLANO = 0.0

    device_torch = torch.device(options['device'])
    model.to(device_torch)

    pointcloud = o3d.io.read_point_cloud(pc_path)

    ranges = np.linspace(BORDES[0], BORDES[1], options['width'])
    i_1, i_2 = np.meshgrid( ranges, ranges )
    samples = np.concatenate(
            np.concatenate( np.array([np.expand_dims(i_1, 2), 
                                np.expand_dims(i_2, 2), 
                                np.expand_dims(np.ones_like(i_1) * OFFSETPLANO, 2)])[EJEPLANO]
                        , axis=2 ),
            axis=0)

    gradients = np.zeros((SAMPLES, 3))
    hessians = np.zeros((SAMPLES, 3, 3))
    pred_distances = evaluate( model, samples, device=device_torch, gradients=gradients, hessians=hessians )
    pred_grad_norm = np.linalg.norm( gradients , axis=1 ).reshape((SAMPLES, 1))

    gradients = normalize(gradients)
    eigenvalues, eigenvectors = torch.linalg.eigh( torch.from_numpy(hessians) )
    pred_normals = eigenvectors[..., 2].numpy()

    pred_normals = np.where(
        np.sum( gradients * pred_normals, axis=-1 )[..., None] < 0,
        np.ones( (pred_normals.shape[0],1)) * -1,
        np.ones( (pred_normals.shape[0],1))
    ) * pred_normals
    
    normals = np.where(
        np.concatenate( [pred_grad_norm , pred_grad_norm, pred_grad_norm], axis=-1) < 0.04,
        pred_normals,
        gradients
    )

    normals *= np.hstack([ np.ones((len(normals), 2)), np.sign(normals[:, 2]).reshape((len(normals), 1))])
    grad_map = ( normals + np.ones_like(normals) ) / 2

    tree = KDTree( np.asarray( pointcloud.points ) )
    gt_distances, _ = tree.query( samples )

    if options['gt_mode'] == 'squared':
        gt_grad_norm = 2 * options['alpha'] * gt_distances
        gt_distances = options['alpha'] * (gt_distances ** 2)
    elif options['gt_mode'] == 'tanh':
        tanh = np.tanh( options['alpha'] * gt_distances ) 
        gt_grad_norm = tanh + options['alpha'] * gt_distances * (1 - tanh ** 2)
        gt_distances = gt_distances * tanh
    elif options['gt_mode'] == 'siren':
        gt_grad_norm = np.where( gt_distances < options['surf_thresh'], np.zeros_like(gt_distances), np.ones_like(gt_distances))
        gt_distances = gt_distances
    else:
        raise ValueError('gt_mode not valid')
    
    #plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,9), dpi=500)

    #max_val = np.max( np.concatenate([gt_distances,pred_distances,gt_grad_norm,pred_grad_norm]))

    color_map = 'bwr_r'

    pos = imagen_dist( axes.flat[0] ,np.clip(gt_distances, a_min=None,a_max=1.5), [0], negs=True, color_map=color_map, eps=options['surf_thresh'], contour=True)
    imagen_dist( axes.flat[1] ,np.clip(pred_distances,a_min=None, a_max=1.5), [0], negs=True, color_map=color_map, eps=options['surf_thresh'], contour=True)
    imagen_dist( axes.flat[2] ,np.clip(gt_grad_norm, a_min=None,a_max=1.5), [0], negs=True, color_map=color_map, eps=options['surf_thresh'])
    imagen_dist( axes.flat[3] ,np.clip(pred_grad_norm, a_min=None,a_max=1.5), [0], negs=True, color_map=color_map, eps=options['surf_thresh'])

    axes.flat[0].set_title(r'Ground truth slices')
    axes.flat[1].set_title(r'Predicted value slices')
    axes.flat[0].set_ylabel(r'$f$', rotation=0, labelpad=12, size='large')
    axes.flat[2].set_ylabel(r'$\left \| \nabla f \right \|$', rotation=0, labelpad=24, size='large')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(pos, cax=cbar_ax)
    fig.savefig(output_path + 'distance_fields.png')

    im = Image.fromarray((grad_map.reshape(np.sqrt(SAMPLES).astype(np.uint32), np.sqrt(SAMPLES).astype(np.uint32), 3) * 255).astype(np.uint8))
    im.save( output_path +'pred_grad.png', 'PNG')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dense point cloud from trained model')
    parser.add_argument('mesh_path', metavar='path/to/mesh.obj', type=str,
                        help='path to input preprocessed mesh')
    parser.add_argument('model_path', metavar='path/to/pth', type=str,
                        help='path to input model')
    parser.add_argument('output_path', metavar='path/to/output/', type=str,
                        help='path to output folder')
    parser.add_argument('-d', '--device', type=int, default=0, help='torch device')
    parser.add_argument('-w0', '--weight0', type=float, default=30, help='w0 parameter of SIREN')
    parser.add_argument('-w', '--width', type=int, default=512, help='width of generated image')
    parser.add_argument('-t', '--surf_thresh', type=float, default=1e-3, help='on surface threshold')
    parser.add_argument('--gt_mode', type=str, default='siren', help='ground truth function')
    parser.add_argument('-a', '--alpha', type=float, default=1, help='alpha for ground truth')

    args = parser.parse_args()
    d = vars(args)
    d['hidden_layer_nodes'] = [256,256,256,256,256,256,256,256]
    d['activation']='sine'
    generate_df_pc(args.model_path, args.mesh_path, args.output_path,d )

