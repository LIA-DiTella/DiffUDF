# coding: utf-8

import torch
import torch.nn.functional as F
import src.diff_operators as dif


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


def vector_aligment_on_surf_both_dirs(gt_sdf, gt_vectors, pred_vectors):
    return torch.where(
        gt_sdf == 0,
        1 - torch.pow(F.cosine_similarity(pred_vectors, gt_vectors, dim=-1)[..., None], 6 ),
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

def gradient_constraint( gt_sdf, gradient ):
    # return torch.linalg.norm(gradient, dim=-1) - 2*torch.sqrt(gt_sdf).squeeze()  ## porque esto esta mal?
    return (torch.diagonal(gradient @ gradient.T) - 4*gt_sdf.squeeze() ) ** 2 

def gradient_constraint_on_surf( gt_sdf, gradient ):
    return torch.where(
        gt_sdf == 0,
        (gradient.norm(dim=-1)) ** 2,
        torch.zeros_like(gt_sdf)
    )

def laplacian_constraint(gt_sdf, pred_sdf, x):
    return torch.where(
        gt_sdf == 0,
        (dif.laplace(pred_sdf, x) - 4) ** 2,
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


def principal_curvature_alignment(gt_sdf, gt_vectors, pred_sdf, coords ):
    hessians, status = dif.hessian(pred_sdf, coords)
    eigenvalues, eigenvectors = torch.linalg.eigh( hessians )

    return torch.where(
        gt_sdf == 0,
        1 - torch.pow(F.cosine_similarity(gt_vectors.squeeze(0), torch.bmm(hessians.squeeze(0), eigenvectors[...,2].squeeze(0).unsqueeze(-1)).squeeze(-1) / eigenvalues[...,2].squeeze(0).unsqueeze(1), dim=-1)[..., None], 6 ),
        torch.zeros_like(gt_sdf)
    )

def loss(model_output, gt, features):
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    indexes = torch.tensor( [features, features + 1, features + 2] ).to(pred_sdf.device)
    gradient = torch.index_select( dif.gradient(pred_sdf, coords), 1, indexes)

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
    gradient = torch.index_select( dif.gradient(pred_sdf, coords).squeeze(0), 1, indexes)


    return {
        'sdf_on_surf': sdf_constraint_on_surf(gt_sdf, pred_sdf).mean() * 1e5,
        'sdf_off_surf': sdf_constraint_off_surf(gt_sdf, pred_sdf).mean() * 1e5,
        #'normal_constraint': vector_aligment_on_surf(gt_sdf, gt_normals, gradient).mean() * 5e2,
        'grad_constraint': gradient_constraint(gt_sdf, gradient).unsqueeze(-1).mean() * 5e1,
        #'grad_constraint': gradient_constraint_on_surf(gt_sdf, gradient).unsqueeze(-1).mean() * 5e1,
        #'regularization_term': off_surface_with_negative_values( pred_sdf ).mean() * 5e3,
        'hessian_constraint': principal_curvature_alignment(gt_sdf, gt_normals, pred_sdf, coords).mean() * 5e2
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
