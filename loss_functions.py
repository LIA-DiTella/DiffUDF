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


def sdf_sitzmann(X, gt):
    """Loss function employed in Sitzmann et al. for SDF experiments [1].

    Parameters
    ----------
    X: dict[str=>torch.Tensor]
        Model output with the following keys: 'model_in' and 'model_out'
        with the model input and SDF values respectively.

    gt: dict[str=>torch.Tensor]
        Ground-truth data with the following keys: 'sdf' and 'normals', with
        the actual SDF values and the input data normals, respectively.

    Returns
    -------
    loss: dict[str=>torch.Tensor]
        The calculated loss values for each constraint.

    References
    ----------
    [1] Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B.,
    & Wetzstein, G. (2020). Implicit Neural Representations with Periodic
    Activation Functions. ArXiv. Retrieved from http://arxiv.org/abs/2006.09661
    """
    gt_sdf = gt["sdf"]
    gt_normals = gt["normals"]

    coords = X["model_in"]
    pred_sdf = X["model_out"]

    grad = diff_operators.gradient(pred_sdf, coords)

    # Initial-boundary constraints
    sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(grad, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(grad[..., :1]))

    # PDE constraints
    grad_constraint = torch.abs(grad.norm(dim=-1) - 1)

    return {
        "sdf_constraint": torch.abs(sdf_constraint).mean() * 3e3,
        "inter_constraint": inter_constraint.mean() * 1e2,
        "normal_constraint": normal_constraint.mean() * 1e2,
        "grad_constraint": grad_constraint.mean() * 5e1,
    }


def true_sdf(X, gt):
    """Uses true SDF value for off-surface points.

    Parameters
    ----------
    X: dict[str=>torch.Tensor]
        Model output with the following keys: 'model_in' and 'model_out'
        with the model input and SDF values respectively.

    gt: dict[str=>torch.Tensor]
        Ground-truth data with the following keys: 'sdf' and 'normals', with
        the actual SDF values and the input data normals, respectively.

    Returns
    -------
    loss: dict[str=>torch.Tensor]
        The calculated loss values for each constraint.
    """
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = X['model_in']
    pred_sdf = X['model_out']

    indexes = torch.tensor( [1,2,3], device=pred_sdf.device )
    gradient = torch.index_select( diff_operators.gradient(pred_sdf, coords), 2, indexes)

    
    # Initial-boundary constraints
    sdf_on_surf = sdf_constraint_on_surf(gt_sdf, pred_sdf)
    sdf_off_surf = sdf_constraint_off_surf(gt_sdf, pred_sdf)
    normal_constraint = vector_aligment_on_surf(gt_sdf, gt_normals, gradient)

    # PDE constraints
    grad_constraint = eikonal_constraint(gradient).unsqueeze(-1)

    return {
       'sdf_on_surf': sdf_on_surf.mean() * 3e3,
       'sdf_off_surf': sdf_off_surf.mean() * 2e2,
       'normal_constraint': normal_constraint.mean() * 1e2, #* 1e1,
       'grad_constraint': grad_constraint.mean() * 5e1 #1e1
    }


def mean_curvature_sdf(model_output, gt):
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
    gt_curvature = gt["curvature"]

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    indexes = torch.tensor( [1,2,3] )
    gradient = torch.index_select( diff_operators.gradient(pred_sdf, coords), 2, indexes)

   # mean curvature
    pred_curvature = diff_operators.divergence(gradient, coords)
    curv_constraint = torch.where(
        gt_sdf == 0,
        (pred_curvature - gt_curvature) ** 2,
        torch.zeros_like(pred_curvature)
    )

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    return {
        'sdf_on_surf': sdf_constraint_on_surf(gt_sdf, pred_sdf).mean() * 3e3,
        'sdf_off_surf': sdf_constraint_off_surf(gt_sdf, pred_sdf).mean() * 2e2,
        'normal_constraint': vector_aligment_on_surf(gt_sdf, gt_normals, gradient).mean() *1e2 ,#* 1e1,
        'grad_constraint': eikonal_constraint(gradient).unsqueeze(-1).mean() * 5e1,
        'curv_constraint': curv_constraint.mean() * 1e-1
    }
