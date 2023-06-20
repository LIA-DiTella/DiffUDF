# coding: utf-8

import torch
import torch.nn.functional as F
import diff_operators


def sdf_constraint_on_surf(gt_sdf, pred_sdf):
    return torch.where(
        gt_sdf == 0,
        pred_sdf ** 2,
        torch.zeros_like(pred_sdf)
    )


def sdf_constraint_off_surf(gt_sdf, pred_sdf):
    return torch.where(
        gt_sdf != 0,
        (gt_sdf - pred_sdf) ** 2,
        torch.zeros_like(pred_sdf)
    )

def sdf_constraint_neg( pred_sdf):
    return torch.where(
        pred_sdf < 0 ,
        (pred_sdf) ** 2,
        torch.zeros_like(pred_sdf)
    )

def vector_aligment_on_surf(gt_sdf, gt_vectors, pred_vectors):
    return torch.where(
        gt_sdf == 0,
        1 - F.cosine_similarity(pred_vectors, gt_vectors, dim=-1)[..., None],
        torch.zeros_like(gt_sdf)
    )


def direction_aligment_on_surf(gt_sdf, gt_dirs, pred_dirs):
    return torch.where(
        gt_sdf == 0,
        1 - (F.cosine_similarity(pred_dirs, gt_dirs, dim=-1)[..., None])**2,
        torch.zeros_like(gt_sdf)
    )

def eikonal_constraint(gradient):
    return (gradient.norm(dim=-1) - 1.) ** 2

def minimum_constraint(gt_sdf, gradient):
    return torch.where(
        gt_sdf != 0,
        (gradient.norm(dim=-1) - 1.) ** 2,
        gradient.norm(dim=-1) ** 2
    )

def gradient_constraint_off_surf( gt_sdf, gradient ):
    return (torch.diagonal(gradient @ gradient.T) - 4 * gt_sdf.squeeze() ) ** 2

def gradient_constraint_on_surf( gt_sdf, gradient ):
    return torch.where(
        gt_sdf == 0,
        (gradient.norm(dim=-1)) ** 2,
        torch.zeros_like(gt_sdf)
    )
    
def off_surface_without_sdf_constraint(gt_sdf, pred_sdf, radius=1e2):
    """
    This function penalizes the pred_sdf of points in gt_sdf!=0
    Used in SIREN's paper
    """
    return torch.where(
           gt_sdf == 0,
           torch.zeros_like(pred_sdf),
           torch.exp(-radius * torch.abs(pred_sdf))
        )


def loss(model_output, gt, features):
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    indexes = torch.tensor( [features, features + 1, features + 2] ).to(pred_sdf.device)
    gradient = torch.index_select( diff_operators.gradient(pred_sdf, coords), 1, indexes)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    return {
        'sdf_on_surf': sdf_constraint_on_surf(gt_sdf, pred_sdf).mean() * 3e3,
        'sdf_off_surf': sdf_constraint_off_surf(gt_sdf, pred_sdf).mean() * 2e2,
        'normal_constraint': vector_aligment_on_surf(gt_sdf, gt_normals, gradient).mean() * 5e2 ,
        'grad_constraint': eikonal_constraint(gradient).unsqueeze(-1).mean() * 5e1, 
    }

def loss_ndf(model_output, gt, features):
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    indexes = torch.tensor( [features, features + 1, features + 2] ).to(pred_sdf.device)
    gradient = torch.index_select( diff_operators.gradient(pred_sdf, coords), 1, indexes)

    return {
        'sdf_on_surf': sdf_constraint_on_surf(gt_sdf, pred_sdf).mean() * 5e3,
        'sdf_off_surf': sdf_constraint_off_surf(gt_sdf, pred_sdf).mean() * 2e2,
        #'normal_constraint': vector_aligment_on_surf(gt_sdf, gt_normals, gradient).mean() * 5e1,
        'grad_constraint': gradient_constraint_off_surf(gt_sdf, gradient).unsqueeze(-1).mean() * 5e1,
        'grad_constraint_on_surf': gradient_constraint_on_surf(gt_sdf, gradient).unsqueeze(-1).mean() * 5e1
        #'neg_constraing': sdf_constraint_neg(pred_sdf).mean() * 5e4
    }


def loss_curvs(model_output, gt, features):
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']
    gt_curvature = gt['curvature']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    indexes = torch.tensor( [features, features + 1, features + 2] ).to(pred_sdf.device)
    gradient = torch.index_select( diff_operators.gradient(pred_sdf, coords), 1, indexes)

    gradient_norm = torch.norm(gradient, dim=-1)
    unit_gradient = gradient.squeeze(-1)/gradient_norm.unsqueeze(-1)
    
    pred_mean_curvature = (-0.5)*diff_operators.divergence(unit_gradient, coords)
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
