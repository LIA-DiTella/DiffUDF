# coding: utf-8

import torch
import torch.nn.functional as F
import src.diff_operators as dif


def sdf_constraint_on_surf(gt_sdf, pred_sdf):
    return torch.where(
        gt_sdf == 0,
        #F.smooth_l1_loss(pred_sdf, torch.zeros_like(pred_sdf), beta=1e-5, reduction='none'),
        #torch.abs( pred_sdf ),
        #pred_sdf ** 2,
        torch.exp( 30 * pred_sdf **2 ) - 1,
        torch.zeros_like(pred_sdf)
    )


def sdf_constraint_off_surf(gt_sdf, pred_sdf):
    return torch.where(
        gt_sdf != 0,
        #F.smooth_l1_loss(pred_sdf, gt_sdf, beta=1e-5, reduction='none'),
        #torch.abs( pred_sdf - gt_sdf ),
        #( pred_sdf - gt_sdf ) ** 2,
        torch.exp( 30 * (gt_sdf - pred_sdf)**2 ) - 1,
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

def gradient_constraint( gt_sdf, gradient ):
    #return  F.smooth_l1_loss( torch.sum( gradient ** 2, dim=-1), 4 * gt_sdf.squeeze() , beta=1e-5, reduction='none')
    return  ( torch.sum( gradient ** 2, dim=-1) - 4 * gt_sdf.squeeze() ) ** 2

    
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

def off_surface_with_negative_values( pred_sdf ):
    """
    This function penalizes the pred_sdf of points in gt_sdf!=0
    Used in SIREN's paper
    """
    return torch.where(
           pred_sdf < 0, 
           torch.exp(-pred_sdf ) - torch.ones_like(pred_sdf),
           torch.zeros_like(pred_sdf)
        )

def total_variation( gradient, coords, gt_sdf ):
    return torch.where(
        gt_sdf.flatten() != 0,
        (torch.linalg.norm( dif.gradient( torch.linalg.norm( gradient, dim=-1), coords ), dim=-1 ) - 2) ** 2,
        torch.zeros_like(gt_sdf.flatten())
    )

def principal_curvature_alignment(gt_sdf, gt_vectors, pred_sdf, coords ):
    hessians, status = dif.hessian(pred_sdf, coords)

    if status == -1:
        print('STATUS: -1')
        return torch.zeros_like(gt_sdf)

    eigenvalues, eigenvectors = torch.linalg.eigh( hessians )

    return torch.where(
        gt_sdf == 0,
        1 - torch.abs(F.cosine_similarity(gt_vectors, eigenvectors[...,2], dim=-1).unsqueeze(-1)),
        torch.zeros_like(gt_sdf)
    )
    # MINIMUM
    #, torch.where(
    #    gt_sdf == 0,
    #    -1 * torch.minimum( torch.prod(eigenvalues, dim=-1), torch.Tensor([0]).to(torch.device(0)) ).unsqueeze(-1),
    #    torch.zeros_like(gt_sdf)
    #)
    # LAPLACIAN
    #torch.where(
       # gt_sdf != 0,
       # (torch.sum( eigenvalues, dim=-1 ).unsqueeze(-1) - 4 * torch.ones_like(gt_sdf)) ** 2,
       # torch.zeros_like(gt_sdf)
    #), 

def off_surface_without_sdf_constraint(gt_sdf, pred_sdf, radius=1e2):
    """
    This function penalizes the pred_sdf of points in gt_sdf!=0
    Used in SIREN's paper
    """
    return torch.where(
           gt_sdf == 0,
           torch.zeros_like(pred_sdf),
           #torch.exp(-radius * torch.abs(pred_sdf))
           torch.exp(-radius * pred_sdf)
        )
    

def loss_siren(model_output, gt, features, loss_weights):
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    indexes = torch.tensor( [features, features + 1, features + 2] ).to(pred_sdf.device)
    gradient = torch.index_select( dif.gradient(pred_sdf, coords).squeeze(0), 1, indexes)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    return {
        'sdf_on_surf': sdf_constraint_on_surf(gt_sdf, pred_sdf).mean() * loss_weights[0],
        'sdf_off_surf': sdf_constraint_off_surf(gt_sdf, pred_sdf).mean() * loss_weights[1],
        'normal_constraint': vector_aligment_on_surf(gt_sdf, gt_normals, gradient).mean() * loss_weights[2] ,
        'grad_constraint': eikonal_constraint(gradient).unsqueeze(-1).mean() * loss_weights[3]
    }

def loss_ndf(model_output, gt, features, loss_weights):
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    indexes = torch.tensor( [features, features + 1, features + 2] ).to(pred_sdf.device)
    gradient = torch.index_select( dif.gradient(pred_sdf, coords).squeeze(0), 1, indexes)
    
    #principal_curvature_constraint, laplacian_constraint, minimum_constraint = principal_curvature_alignment(gt_sdf, gt_normals, pred_sdf, coords)
    principal_curvature_constraint = principal_curvature_alignment(gt_sdf, gt_normals, pred_sdf, coords)
    grad_constraint = gradient_constraint(gt_sdf, gradient)

    return {
        'sdf_on_surf': sdf_constraint_on_surf(gt_sdf, pred_sdf).mean() * loss_weights[0],
        'sdf_off_surf': sdf_constraint_off_surf(gt_sdf, pred_sdf).mean() * loss_weights[1],
        'grad_constraint': grad_constraint.mean() * loss_weights[2],
        'hessian_constraint': principal_curvature_constraint.mean() * loss_weights[3],
        #'laplacian_constraint': laplacian_constraint.mean() * loss_weights[4],
        'total_variation': total_variation( gradient, coords, gt_sdf ).mean() * loss_weights[5],
        #'minimum_constraint': minimum_constraint.mean() * loss_weights[6]
        #'regularization_term': off_surface_without_sdf_constraint(gt_sdf, pred_sdf).mean() * loss_weights[7]
    }


def loss_curvs(model_output, gt, features):
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']
    gt_curvature = gt['curvature']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    indexes = torch.tensor( [features, features + 1, features + 2] ).to(pred_sdf.device)
    gradient = torch.index_select( dif.gradient(pred_sdf, coords), 1, indexes)

    gradient_norm = torch.norm(gradient, dim=-1)
    unit_gradient = gradient.squeeze(-1)/gradient_norm.unsqueeze(-1)
    
    pred_mean_curvature = (-0.5)*dif.divergence(unit_gradient, coords)
    curv_constraint = torch.where(
        gt_sdf == 0,
        (pred_mean_curvature - gt_curvature) ** 2,
        torch.zeros_like(pred_mean_curvature)
    )

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    return {
        'sdf_on_surf': sdf_constraint_on_surf(gt_sdf, pred_sdf).mean() * 3e3,
        'sdf_off_surf': sdf_constraint_off_surf(gt_sdf, pred_sdf).mean() * 2e2,
        'normal_constraint': vector_aligment_on_surf(gt_sdf, gt_normals, gradient).mean() *1e2 ,
        'grad_constraint': eikonal_constraint(gradient).unsqueeze(-1).mean() * 5e1,
        'curv_constraint': curv_constraint.mean() * 1e-2
    }
