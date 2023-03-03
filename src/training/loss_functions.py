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


def on_surface_normal_constraint(gt_sdf, gt_normals, grad):
    """
    This function return a number that measure how far gt_normals
    and grad are aligned in the zero-level set of sdf.
    """
    return torch.where(
           gt_sdf == 0,
           1 - F.cosine_similarity(grad, gt_normals, dim=-1)[..., None],
           torch.zeros_like(grad[..., :1])
    )

def loss(model_output, gt):
    """Uses true SDF value off surface and tries to fit the mean curvatures
    on the 0 level-set.

    Parameters
    ----------
    X: dict[str=>torch.Tensor]
        Model output with the following keys: 'model_in' and 'model_out'
        with the model input and SDF values respectively.

    gt: dict[str=>torch.Tensor]
        Ground-truth data with the following keys: 'sdf', 'normals', and
        'curvature' with the actual SDF values, the input data normals, and
        gaussian curvatures, respectively.

    Returns
    -------
    loss: dict[str=>torch.Tensor]
        The calculated loss values for each constraint.
    """
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    indexes = torch.tensor( [1,2,3] )
    gradient = torch.index_select( diff_operators.gradient(pred_sdf, coords), 1, indexes)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    return {
        'sdf_on_surf': sdf_constraint_on_surf(gt_sdf, pred_sdf).mean() * 3e3,
        'sdf_off_surf': sdf_constraint_off_surf(gt_sdf, pred_sdf).mean() * 2e2,
        'normal_constraint': vector_aligment_on_surf(gt_sdf, gt_normals, gradient).mean() *1e2 ,
        'grad_constraint': eikonal_constraint(gradient).unsqueeze(-1).mean() * 5e1,
    }

def loss_curvs(model_output, gt):
    """Uses true SDF value off surface and tries to fit the mean curvatures
    on the 0 level-set.

    Parameters
    ----------
    X: dict[str=>torch.Tensor]
        Model output with the following keys: 'model_in' and 'model_out'
        with the model input and SDF values respectively.

    gt: dict[str=>torch.Tensor]
        Ground-truth data with the following keys: 'sdf', 'normals', and
        'curvature' with the actual SDF values, the input data normals, and
        gaussian curvatures, respectively.

    Returns
    -------
    loss: dict[str=>torch.Tensor]
        The calculated loss values for each constraint.
    """
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']
    gt_curvature = gt['curvature']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    parameter_count = torch.tensor([coords.shape]).squeeze()[1]
    indexes = torch.tensor( [ parameter_count - 3, parameter_count - 2, parameter_count - 1 ] )
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
