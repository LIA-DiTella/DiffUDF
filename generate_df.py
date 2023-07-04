import numpy as np
import json
import torch
import open3d as o3d
import open3d.core as o3c
import argparse
from src.model import SIREN
from src.evaluate import evaluate
import matplotlib.cm as cm
from PIL import Image
from src.util import normalize

def imagen_dist( path, distancias, niveles, eps=0.0005, negs=False, color_map='br'):
    def lines( distancias, niveles, eps ):
        res = np.ones_like(distancias, dtype=np.bool8)
        
        for v in niveles:
            res = np.logical_and( np.logical_not( np.logical_and( distancias > v - eps, distancias < v + eps)), res)

        return res


    mask = lines(distancias, niveles, eps)

    if color_map == 'br':
        cmap = cm.get_cmap('binary')
        colores = np.expand_dims(cmap( distancias[mask] )[:, 0], axis=1)
        colores = np.hstack( [np.ones_like(colores) - colores, np.zeros_like(colores), colores])
    else:
        cmap = cm.get_cmap(color_map)
        colores = cmap( distancias[mask] )[:, :3]

    imagen = np.ones((SAMPLES,3))
    imagen[mask.squeeze(1)] = colores

    if negs:
        imagen[distancias.squeeze(1) < 0] = np.tile( np.eye( 1, 3, k=1) * 255, (np.sum(distancias < 0),1)).astype(np.uint32)

    im = Image.fromarray((imagen.reshape(np.sqrt(SAMPLES).astype(np.uint32), np.sqrt(SAMPLES).astype(np.uint32), 3) * 255).astype(np.uint8))
    im.save(path, 'PNG')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dense point cloud from trained model')
    parser.add_argument('json_path', metavar='path/to/json', type=str,
                        help='path to input json (preprocessed mesh)')
    parser.add_argument('model_path', metavar='path/to/pth', type=str,
                        help='path to input model')
    parser.add_argument('output_path', metavar='path/to/output/', type=str,
                        help='path to output folder')
    parser.add_argument('-d', '--device', type=int, default=0, help='torch device')
    parser.add_argument('-w0', '--weight0', type=float, default=30, help='w0 parameter of SIREN')
    parser.add_argument('-w', '--width', type=int, default=512, help='width of generated image')
    parser.add_argument('-t', '--surf_thresh', type=float, default=1e-3, help='on surface threshold')
    parser.add_argument('-j', '--joint', type=int, default=0, help="joint number to render")

    args = parser.parse_args()

    SAMPLES = args.width ** 2
    BORDES = [1, -1]
    EJEPLANO = [0,2,1]
    OFFSETPLANO = 0.0

    device = o3c.Device('CPU:0')
    device_torch = torch.device(args.device)

    with open(args.json_path) as jsonFile:
        skel = json.load(jsonFile)
        features = len(skel['joints'][args.joint]['mean'])
        code = np.tile( skel['joints'][args.joint]['mean'], (SAMPLES, 1))
        mesh = o3d.t.geometry.TriangleMesh(device)
        mesh.vertex["positions"] = o3c.Tensor(np.array(skel['joints'][args.joint]['vertices']), dtype=o3c.float32)
        mesh.triangle["indices"] = o3c.Tensor(np.array(skel['joints'][args.joint]['triangles']), dtype=o3c.int32)


    model = SIREN(
            n_in_features= features + 3,
            n_out_features=1,
            hidden_layer_config=[256,256,256,256],
            w0=args.weight0,
            ww=None
    )
    model.load_state_dict( torch.load(args.model_path, map_location=device_torch))
    model.to(device_torch)

    ranges = np.linspace(BORDES[0], BORDES[1], args.width)
    i_1, i_2 = np.meshgrid( ranges, ranges )
    samples = np.concatenate(
            np.concatenate( np.array([np.expand_dims(i_1, 2), 
                                np.expand_dims(i_2, 2), 
                                np.expand_dims(np.ones_like(i_1) * OFFSETPLANO, 2)])[EJEPLANO]
                        , axis=2 ),
            axis=0)

    inputs = np.hstack([code, samples])
    gradients = np.zeros((SAMPLES, 3))

    pred_distances = evaluate( model, samples, device=device_torch, gradients=gradients )
    pred_grad_norm = np.sum( gradients ** 2, axis=1 ).reshape((SAMPLES, 1)) #np.linalg.norm(gradients, axis=1).reshape( (SAMPLES, 1))
    pred_grad_cm = ( normalize(gradients) + np.ones_like(gradients) ) / 2

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    gt_distances = (scene.compute_distance( o3c.Tensor(samples, dtype=o3c.float32) ).numpy() ** 2).reshape((SAMPLES, 1))

    imagen_dist( args.output_path + 'pred_field.png',pred_distances / np.max(pred_distances), np.linspace(0,1,10), negs=True, color_map='turbo', eps=args.surf_thresh)
    imagen_dist( args.output_path + 'gt_field.png',gt_distances / np.max(gt_distances), np.linspace(0,1,10), negs=True, color_map='turbo', eps=args.surf_thresh)
    imagen_dist( args.output_path + 'pred_grad_norm.png',pred_grad_norm / np.max(pred_grad_norm), np.linspace(0,1,10), color_map='turbo', eps=args.surf_thresh )

    im = Image.fromarray((pred_grad_cm.reshape(np.sqrt(SAMPLES).astype(np.uint32), np.sqrt(SAMPLES).astype(np.uint32), 3) * 255).astype(np.uint8))
    im.save(args.output_path +'pred_grad.png', 'PNG')
