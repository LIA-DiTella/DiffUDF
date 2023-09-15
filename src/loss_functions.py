# coding: utf-8

import torch
import torch.nn.functional as F
import src.diff_operators as dif
import numpy as np


def sdf_constraint_on_surf(udf, pred_sdf):
    return torch.where(
        udf == 0,
        #F.smooth_l1_loss(pred_sdf, torch.zeros_like(pred_sdf), beta=1e-5, reduction='none'),
        torch.abs( pred_sdf),
        #pred_sdf ** 2,
        torch.zeros_like(pred_sdf)
    )


def sdf_constraint_off_surf(udf, tdf, pred_sdf):
    return torch.where(
        udf != 0,
        #F.smooth_l1_loss(pred_sdf, gt_sdf, beta=1e-5, reduction='none'),
        #torch.abs(tdf - pred_sdf),
        #torch.exp(-1e2 * torch.nn.functional.relu(pred_sdf)),
        #( pred_sdf - gt_sdf ) ** 2,
        torch.where(
            udf < 0.1,
            torch.abs(tdf - pred_sdf),
            torch.exp(-65 * torch.abs(pred_sdf))
        ),
        torch.zeros_like(pred_sdf)
    )

def vector_aligment_on_surf(gt_sdf, gt_vectors, pred_vectors):
    return torch.where(
        gt_sdf == 0,
        1 - F.cosine_similarity(pred_vectors, gt_vectors.squeeze(0), dim=-1)[..., None],
        torch.zeros_like(gt_sdf)
    )

def eikonal_constraint(gradient):
    return (gradient.norm(dim=-1) - 1.) ** 2
    
def off_surface_without_sdf_constraint(gt_sdf, pred_sdf, radius=1e2):
    """
    This function penalizes the pred_sdf of points in gt_sdf!=0
    Used in SIREN's papers
    """
    return torch.where(
           gt_sdf == 0,
           torch.zeros_like(pred_sdf),
           torch.exp(-radius * torch.abs(pred_sdf))
        )

def principal_curvature_alignment( udf, gt_vectors, pred_normals ): # hessians, alpha ):
    surface_points_mask = torch.flatten(udf == 0)

    return torch.where(
        surface_points_mask,
        #torch.abs( torch.flatten(torch.bmm( gt_vectors[0,...].unsqueeze(1), torch.bmm( hessians[0,...], gt_vectors[0,...].unsqueeze(-1)))) - 2*alpha ),
        (1 - torch.abs(F.cosine_similarity(gt_vectors, pred_normals,dim=-1))),
        torch.zeros_like(surface_points_mask)
    )

def total_variation(  alpha, udf, gradient, coords ):
    f = 1 - torch.tanh(alpha * udf) ** 2
    return torch.where(
        udf != 0,
        torch.abs( 
            torch.linalg.norm( dif.gradient( torch.linalg.norm(gradient, dim=-1), coords ), dim=-1 )[...,None] - 
            2 * alpha * torch.abs(f - udf * torch.tanh(alpha * udf) * f)
        ),
        torch.zeros_like(udf)
    )

def grad_consistency( model, coords, gt_normals, alpha ):
    steps = torch.normal(0, 0.01, (coords.shape[0], coords.shape[1], 1)).to(coords.device)
    samples = coords + gt_normals * steps

    model_output = model(samples)
    gradients = dif.gradient(model_output['model_out'], model_output['model_in'])

    tan = torch.tanh(alpha * torch.abs(steps))

    return (
        1 - F.cosine_similarity( F.normalize(gradients, dim=-1), gt_normals * torch.sign(steps) ,dim=-1 ), 
        torch.abs(model_output['model_out'] - steps * tan),
        torch.abs( torch.linalg.norm(gradients.squeeze(0), dim=-1) - torch.abs( tan + torch.abs(steps) * alpha * (1 - tan ** 2) ).squeeze(-1) )
    )

def loss_siren(model_output, gt, loss_weights, alpha=None ):
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = dif.gradient(pred_sdf, coords).squeeze(0)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    return {
        'sdf_on_surf': sdf_constraint_on_surf(gt_sdf, pred_sdf).mean() * loss_weights[0],
        'sdf_off_surf': sdf_constraint_off_surf(gt_sdf, pred_sdf).mean() * loss_weights[1],
        'normal_constraint': vector_aligment_on_surf(gt_sdf, gt_normals, gradient).mean() * loss_weights[2] ,
        'grad_constraint': eikonal_constraint(gradient).unsqueeze(-1).mean() * loss_weights[3]
    }

def loss_squared( model_output, gt, loss_weights, alpha  ):
    udf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = dif.gradient(pred_sdf, coords).squeeze(0)
    
    gt_udf = alpha * (udf ** 2)
    principal_direction_constraint = principal_curvature_alignment(udf, gt_normals, pred_sdf, coords, alpha )
    grad_constraint = torch.abs(torch.linalg.norm(gradient, dim=-1) - 2 * alpha * udf.squeeze(-1))

    return {
        'sdf_on_surf': sdf_constraint_on_surf( gt_udf, pred_sdf).mean() * loss_weights[0],
        'sdf_off_surf': sdf_constraint_off_surf( gt_udf, pred_sdf).mean() * loss_weights[1],
        'hessian_constraint': principal_direction_constraint.mean() * loss_weights[2],
        'grad_constraint': grad_constraint.mean() * loss_weights[3]
    }

def loss_tanh_gc( model, model_input, gt, loss_weights, alpha ):
    model_output = model(model_input)
    
    udf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = dif.gradient(pred_sdf, coords)
    
    hessians = dif.hessian(pred_sdf.squeeze(-1), coords)
    eigenvalues, eigenvectors = torch.linalg.eigh( hessians )
    pred_normals = eigenvectors[..., 2]

    principal_direction_constraint = principal_curvature_alignment( udf, gt_normals, pred_normals )

    tdf = udf * torch.tanh( alpha * udf )
    tan = torch.tanh( alpha * udf )
    grad_constraint = torch.abs(torch.linalg.norm(gradient.squeeze(0), dim=-1) - torch.abs( tan + udf * alpha * (1 - tan ** 2)).squeeze(-1))

    grad_consist, sdf_near_surf, grad_constraint_near_surf = grad_consistency( model, coords[:,(udf == 0).flatten(),:], gt_normals[:,(udf == 0).flatten(),:], alpha  )
    

    return {
        'sdf_on_surf': sdf_constraint_on_surf( udf, pred_sdf).mean() * loss_weights[0],
        'sdf_off_surf': (sdf_constraint_off_surf( udf, tdf, pred_sdf ).mean()  + sdf_near_surf.mean())* loss_weights[1],
        'hessian_constraint': principal_direction_constraint.mean() * loss_weights[2],
        'grad_constraint': (grad_constraint.mean() + grad_constraint_near_surf.mean()) * loss_weights[3],
        'grad_consistency': grad_consist.mean() * loss_weights[4]
    }

def loss_tanh( model, model_input, gt, loss_weights, alpha ):
    model_output = model(model_input)
    
    udf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = dif.gradient(pred_sdf, coords)
    
    hessians = dif.hessian(pred_sdf.squeeze(-1), coords)
    eigenvalues, eigenvectors = torch.linalg.eigh( hessians )
    pred_normals = eigenvectors[..., 2]

    principal_direction_constraint = principal_curvature_alignment( udf, gt_normals, pred_normals )
    #principal_curvature_constraint = torch.where(
    #    (udf == 0).squeeze(-1),
    #    torch.abs( eigenvalues[..., 2]- 2*alpha),
    #    torch.zeros_like(udf).squeeze(-1)
    #)


    tdf = udf * torch.tanh( alpha * udf )
    tan = torch.tanh( alpha * udf )
    grad_constraint = torch.abs( torch.linalg.norm(gradient.squeeze(0), dim=-1) - torch.abs( tan + udf * alpha * (1 - tan ** 2) ).squeeze(-1) )

    return {
        'sdf_on_surf': sdf_constraint_on_surf( udf, pred_sdf).mean() * loss_weights[0],
        'sdf_off_surf': sdf_constraint_off_surf( udf, tdf, pred_sdf).mean() * loss_weights[1],
        'hessian_constraint': principal_direction_constraint.mean() * loss_weights[2],
        #'curv_constraint': principal_curvature_constraint.mean() * loss_weights[3],
        'grad_constraint': grad_constraint.mean() * loss_weights[3]
    }

def loss_tanh_tv( model, model_input, gt, loss_weights, alpha ):
    model_output = model(model_input)
    
    udf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = dif.gradient(pred_sdf, coords)
    
    hessians = dif.hessian(pred_sdf.squeeze(-1), coords)
    eigenvalues, eigenvectors = torch.linalg.eigh( hessians )
    pred_normals = eigenvectors[..., 2]

    principal_curvature_constraint = principal_curvature_alignment( udf, gt_normals, pred_normals ).mean()

    tdf = udf * torch.tanh( alpha * udf )
    tan = torch.tanh( alpha * udf )
    grad_constraint = torch.abs( torch.linalg.norm(gradient.squeeze(0), dim=-1) - torch.abs( tan + udf * alpha * (1 - tan ** 2) ).squeeze(-1) )

    return {
        'sdf_on_surf': sdf_constraint_on_surf( udf, pred_sdf).mean() * loss_weights[0],
        'sdf_off_surf': sdf_constraint_off_surf( udf, tdf, pred_sdf).mean() * loss_weights[1],
        'hessian_constraint': principal_curvature_constraint * loss_weights[2],
        'grad_constraint': grad_constraint.mean() * loss_weights[3],
        'total_variation': total_variation( alpha, udf, gradient, coords  ).mean() * loss_weights[4]
    }

def loss_tanh_hessian( model, model_input, gt, loss_weights, alpha ):
    model_output = model(model_input)
    
    udf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = dif.gradient(pred_sdf, coords)
    
    hessians = dif.hessian(pred_sdf.squeeze(-1), coords)
    hess_constraint = torch.linalg.matrix_norm( hessians[0,(udf == 0).flatten(),:] - 2*alpha*torch.bmm(gt_normals[0,(udf == 0).flatten(),:].unsqueeze(-1), gt_normals[0,(udf == 0).flatten(),:].unsqueeze(1)) ) 

    tdf = udf * torch.tanh( alpha * udf )
    tan = torch.tanh( alpha * udf )
    grad_constraint = torch.abs( torch.linalg.norm(gradient.squeeze(0), dim=-1) - torch.abs( tan + udf * alpha * (1 - tan ** 2) ).squeeze(-1) )

    return {
        'sdf_on_surf': sdf_constraint_on_surf( udf, pred_sdf).mean() * loss_weights[0],
        'sdf_off_surf': sdf_constraint_off_surf( udf, tdf, pred_sdf).mean() * loss_weights[1],
        'hessian_constraint': hess_constraint.mean() * loss_weights[2],
        'grad_constraint': grad_constraint.mean() * loss_weights[3]
    }
