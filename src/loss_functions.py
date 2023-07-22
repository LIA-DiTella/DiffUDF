# coding: utf-8

import torch
import torch.nn.functional as F
import src.diff_operators as dif


def sdf_constraint_on_surf(gt_sdf, pred_sdf):
    return torch.where(
        gt_sdf == 0,
        #F.smooth_l1_loss(pred_sdf, torch.zeros_like(pred_sdf), beta=1e-5, reduction='none'),
        torch.abs( pred_sdf),
        #pred_sdf ** 2,
        torch.zeros_like(pred_sdf)
    )


def sdf_constraint_off_surf(gt_sdf, pred_sdf):
    return torch.where(
        gt_sdf != 0,
        #F.smooth_l1_loss(pred_sdf, gt_sdf, beta=1e-5, reduction='none'),
        torch.abs(gt_sdf - pred_sdf),
        #( pred_sdf - gt_sdf ) ** 2,
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

def total_variation_constraint( gradient, coords, alpha, udf ):
    return (torch.linalg.norm( dif.gradient( torch.sum( gradient ** 2, dim=-1), coords ), dim=-1 ) - 8 * (alpha ** 2) * udf.squeeze(-1) ) ** 2


def principal_curvature_alignment( udf, gt_vectors, pred_sdf, coords, alpha, gt_curvatures ):
    hessians, status_hess = dif.hessian(pred_sdf, coords)

    if status_hess == -1:
        print('STATUS: -1')
        return torch.zeros_like(udf)

    eigenvalues, eigenvectors = torch.linalg.eigh( hessians )

    #shape_op, status_jac = dif.jacobian(eigenvectors[..., 2] * torch.sign(F.cosine_similarity(eigenvectors[...,2], gt_vectors, dim=-1)).unsqueeze(-1), coords)
    divergence = dif.divergence( torch.bmm( hessians[0,...], gt_vectors[0,...].unsqueeze(-1)), coords)

    return torch.where(
        udf == 0,
        1 - torch.abs(F.cosine_similarity(gt_vectors, eigenvectors[..., 2], dim=-1)).unsqueeze(-1),
        torch.zeros_like(udf)
    ),torch.where(
        udf.squeeze(-1) == 0,
        torch.abs( eigenvalues[..., 2] - alpha * 2) + torch.abs( eigenvalues[..., 0] ) + torch.abs( eigenvalues[..., 1] ),
        torch.zeros_like(udf.squeeze(-1))
    ),torch.where(
        udf == 0,
        torch.abs(divergence - 2*gt_curvatures),
        torch.zeros_like(udf)
    )

def loss_siren(model_output, gt, features, loss_weights, alpha=None ):
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

def loss_squared( model_output, gt, features, loss_weights, alpha  ):
    udf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    indexes = torch.tensor( [features, features + 1, features + 2] ).to(pred_sdf.device)
    gradient = torch.index_select( dif.gradient(pred_sdf, coords).squeeze(0), 1, indexes)
    
    gt_udf = alpha * (udf ** 2)
    principal_direction_constraint, principal_curvature_constraint = principal_curvature_alignment(udf, gt_normals, pred_sdf, coords, alpha)
    grad_constraint = torch.abs(torch.linalg.norm(gradient, dim=-1) - 2 * alpha * udf.squeeze(-1))

    return {
        'sdf_on_surf': sdf_constraint_on_surf( gt_udf, pred_sdf).mean() * loss_weights[0],
        'sdf_off_surf': sdf_constraint_off_surf( gt_udf, pred_sdf).mean() * loss_weights[1],
        'hessian_constraint': principal_direction_constraint.mean() * loss_weights[2],
        'grad_constraint': grad_constraint.mean() * loss_weights[3],
        'curvature_constraint': principal_curvature_constraint.mean() * loss_weights[4],
        'total_variation': 
            torch.where( 
                udf.squeeze(-1) != 0, 
                torch.abs( torch.linalg.norm( dif.gradient( torch.linalg.norm( gradient, dim=-1), coords ), dim=-1 ) - 2 * alpha), 
                torch.zeros_like(udf).squeeze(-1)
            ).mean() * loss_weights[5]
    }

def loss_tanh( model_output, gt, features, loss_weights, alpha ):
    udf = gt['sdf']
    gt_normals = gt['normals']
    gt_curvatures = gt['curvature']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    indexes = torch.tensor( [features, features + 1, features + 2] ).to(pred_sdf.device)
    gradient = torch.index_select( dif.gradient(pred_sdf, coords).squeeze(0), 1, indexes)
    
    gt_udf = udf * torch.tanh( alpha * udf )
    principal_direction_constraint, principal_curvature_constraint, surface_curvature_constraint = principal_curvature_alignment( udf, gt_normals, pred_sdf, coords, alpha, gt_curvatures )
    tan = torch.tanh( alpha * udf )
    grad_constraint = torch.abs( torch.linalg.norm(gradient, dim=-1) - torch.abs( tan + udf * alpha * (1 - tan ** 2) ).squeeze(-1) )

    return {
        'sdf_on_surf': sdf_constraint_on_surf( gt_udf, pred_sdf).mean() * loss_weights[0],
        'sdf_off_surf': sdf_constraint_off_surf( gt_udf, pred_sdf).mean() * loss_weights[1],
        'hessian_constraint': principal_direction_constraint.mean() * loss_weights[2],
        'grad_constraint': grad_constraint.mean() * loss_weights[3],
        #'curvature_constraint': principal_curvature_constraint.mean() * loss_weights[4],
        #'total_variation': 
        #    torch.where( 
        #        udf.squeeze(-1) != 0, 
        #        torch.linalg.norm( dif.gradient( torch.linalg.norm( gradient, dim=-1), coords ), dim=-1 ), 
        #        torch.zeros_like(udf).squeeze(-1)).mean() * loss_weights[5],
        'surface_curvature_constraint': surface_curvature_constraint.mean() * loss_weights[6]
    }
