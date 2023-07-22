# coding: utf-8
import json
import os
import os.path as osp
import shutil
import logging
import numpy as np


def create_output_paths(checkpoint_path, experiment_name, overwrite=True):
    """Helper function to create the output folders. Returns the resulting path.
    """
    full_path = osp.join(".", checkpoint_path, experiment_name)
    if osp.exists(full_path) and overwrite:
        shutil.rmtree(full_path)
    elif osp.exists(full_path):
        logging.warning("Output path exists. Not overwritting.")
        return full_path

    os.makedirs(osp.join(full_path, "models"))
    os.makedirs(osp.join(full_path, "reconstructions"))
    return full_path


def load_experiment_parameters(parameters_path):
    try:
        with open(parameters_path, "r") as fin:
            parameter_dict = json.load(fin)
    except FileNotFoundError:
        logging.warning("File '{parameters_path}' not found.")
        return {}
    return parameter_dict

def normalize( arr ):
    if len(arr.shape) == 1:
        return arr / np.linalg.norm(arr)
    
    norm_arr = np.linalg.norm( arr, axis=1 )
    return arr / np.vstack( [norm_arr, norm_arr, norm_arr] ).T