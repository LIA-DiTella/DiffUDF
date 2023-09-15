import numpy as np

def inverse( gt_mode, pred_df, alpha, min_step=0.01 ):
    inverse_function = {
        'siren': inv_siren,
        'squared': inv_squared,
        'tanh': inv_tanh
    }
    return inverse_function[gt_mode](pred_df, alpha, min_step)

def inv_squared( pred_df, alpha, min_step ):
    inverse = np.ones_like( pred_df ) * min_step
    np.sqrt( pred_df, out=inverse, where=pred_df > 0)
    inverse /= np.sqrt(alpha)

    return inverse

def inv_tanh( pred_df, alpha, min_step ):
    return np.where( pred_df < 0.1, np.sqrt(pred_df / alpha ), pred_df )

def inv_siren( pred_df, alpha, min_step ):
    return np.where( pred_df > 0, pred_df, np.ones_like(pred_df) * min_step ) 