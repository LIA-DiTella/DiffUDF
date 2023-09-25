#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import json
import os
import os.path as osp
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from src.dataset import PointCloud
from src.loss_functions import loss_siren, loss_s1, loss_s2
from src.model import SIREN
from src.util import create_output_paths, load_experiment_parameters
from generate_df import generate_df
from generate_mc import generate_mc
import open3d as o3d


def train_model(dataset, model, device, config) -> torch.nn.Module:
    epochs = config["epochs"]
    epochs_til_checkpoint = config.get("epochs_to_checkpoint", 0)

    log_path = config["log_path"]
    optim = config["optimizer"]

    model.to(device)

    # Creating the summary storage folder
    summary_path = osp.join(log_path, 'summaries')
    if not osp.exists(summary_path):
        os.makedirs(summary_path)
    writer = SummaryWriter(summary_path)

    losses = dict()
    best_loss = np.inf
    best_weights = None

    loss_fn = loss_s1
    loss_weights = config['loss_s1_weights']

    for epoch in range(epochs):
        if epoch == config['s1_epochs']:
            print('Starting second step...')
            for g in optim.param_groups:
                g['lr'] = config['lr_s2']

            loss_weights = config['loss_s2_weights']
            loss_fn = loss_s2            

        running_loss = dict()
        for input_data, normals, sdf in iter(dataset):
            # zero the parameter gradients
            optim.zero_grad()
            
            # forward + backward + optimize
            input_data = input_data.to( device )
            normals = normals.to(device)
            sdf = sdf.to(device)
            
            loss = loss_fn( 
                model, 
                input_data, 
                {'normals': normals, 'sdf': sdf}, 
                loss_weights,
                config["alpha"]
            )

            train_loss = torch.zeros((1, 1), device=device)
            for it, l in loss.items():
                train_loss += l
                # accumulating statistics per loss term
                if it not in running_loss:
                    running_loss[it] = l.item()
                else:
                    running_loss[it] += l.item()

            train_loss.backward()
            optim.step()

            writer.add_scalar("train_loss", train_loss.item(), epoch)

        # accumulate statistics
        for it, l in running_loss.items():
            if it in losses:
                losses[it][epoch] = l
            else:
                losses[it] = [0.] * epochs
                losses[it][epoch] = l
            writer.add_scalar(it, l, epoch)

        epoch_loss = 0
        for k, v in running_loss.items():
            epoch_loss += v
        epoch_loss /=+ dataset.batchesPerEpoch
        print(f"Epoch: {epoch} - Loss: {epoch_loss}")

        # Saving the best model after warmup.
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(
                best_weights,
                osp.join(log_path, "models", "model_best.pth")
            )

        # saving the model at checkpoints
        if epoch and epochs_til_checkpoint and (not \
           epoch % epochs_til_checkpoint or epoch == config['s1_epochs'] - 1):
            print(f"Saving model for epoch {epoch}")
            torch.save(
                model.state_dict(),
                osp.join(log_path, "models", f"model_{epoch}.pth")
            )
            print(f"Generating mesh")
            generate_mc( 
                model=model, 
                gt_mode=config["gt_mode"], 
                device=device, 
                N=config.get('resolution', 256), 
                output_path=osp.join(log_path, "reconstructions", f'mc_mesh_{epoch}.obj'), 
                alpha=config['alpha'], 
                algorithm='cap'
            )

        else:
            torch.save(
                model.state_dict(),
                osp.join(log_path, "models", "model_current.pth")
            )

    return losses, best_weights

def setup_train( parameter_dict, cuda_device ):

    device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
    seed = 123 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    full_path = create_output_paths(
        parameter_dict["checkpoint_path"],
        parameter_dict["experiment_name"],
        overwrite=False
    )

    # Saving the parameters to the output path
    with open(osp.join(full_path, "params.json"), "w+") as fout:
        json.dump(parameter_dict, fout, indent=4)

    sampling_config = parameter_dict["sampling_opts"]
    dataset = PointCloud(
        meshPath= parameter_dict["dataset"],
        batchSize= parameter_dict["batch_size"],
        samplingPercentiles=parameter_dict["sampling_percentiles"],
        batchesPerEpoch = parameter_dict["batches_per_epoch"],
        curvatureFractions=sampling_config["curvature_iteration_fractions"],
        curvaturePercentiles=sampling_config["curvature_percentile_thresholds"]
    )

    network_params = parameter_dict["network"]
    model = SIREN(
        n_in_features= 3,
        n_out_features=1,
        hidden_layer_config=network_params["hidden_layer_nodes"],
        w0=network_params["w0"],
        ww=network_params.get("ww", None)
    )
    print(model)

    if network_params['pretrained_dict'] != 'None':
        model.load_state_dict(torch.load(network_params['pretrained_dict'], map_location=device))

    opt_params = parameter_dict["optimizer"]
    if opt_params["type"] == "adam":
        optimizer = torch.optim.Adam(
            lr=opt_params["lr_s1"],
            params=model.parameters()
        )
    else:
        raise ValueError('Unknown optimizer')
    

    config_dict = {
        "epochs": parameter_dict["num_epochs"],
        "s1_epochs": parameter_dict["s1_epochs"],
        "batch_size": parameter_dict["batch_size"],
        "epochs_to_checkpoint": parameter_dict["epochs_to_checkpoint"],
        "gt_mode": parameter_dict["gt_mode"],
        "log_path": full_path,
        "optimizer": optimizer,
        "lr_s1": opt_params["lr_s1"],
        "lr_s2": opt_params["lr_s2"],
        "loss_s1_weights": parameter_dict["loss_s1_weights"],
        "loss_s2_weights": parameter_dict["loss_s2_weights"],
        "alpha": parameter_dict["alpha"],
        "resolution": parameter_dict.get('resolution', 256)
    }

    losses, best_weights = train_model(
        dataset,
        model,
        device,
        config_dict,
    )
    loss_df = pd.DataFrame.from_dict(losses)
    loss_df.to_csv(osp.join(full_path, "losses.csv"), sep=";", index=None)

    # saving the final model.
    torch.save(
        model.state_dict(),
        osp.join(full_path, "models", "model_final.pth")
    )

    print('Generating distance field slices')

    df_options = {
        'device': cuda_device,
        'surf_thresh': 1e-3,
        'joint': 0,
        'width': 512,
        'weight0': network_params["w0"],
        'gt_mode': parameter_dict["gt_mode"],
        'alpha': parameter_dict['alpha'],
        'hidden_layer_nodes': network_params["hidden_layer_nodes"]
    }

    print('Generating mesh')
    generate_df( osp.join(full_path, "models", "model_best.pth"), parameter_dict['dataset'], osp.join(full_path, "reconstructions/"), df_options)

    mc_options = {
        'w0': network_params["w0"],
        'model_path': osp.join(full_path, "models", "model_best.pth"),
        'hidden_layer_nodes': network_params["hidden_layer_nodes"]
    }

    generate_mc( 
        model=None, 
        gt_mode=parameter_dict["gt_mode"],
        device=cuda_device, 
        N=parameter_dict.get('resolution', 256), 
        output_path=osp.join(full_path, "reconstructions", f'mc_mesh_best.obj'),
        alpha=parameter_dict['alpha'],
        from_file = mc_options,
        algorithm='cap'
    )

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        usage="python main.py path_to_experiments.json cuda_device"
    )

    p.add_argument(
        "experiment_path", type=str,
        help="Path to the JSON experiment description file"
    )
    p.add_argument(
        "device", type=int, help="Cuda device"
    )
    args = p.parse_args()
    parameter_dict = load_experiment_parameters(args.experiment_path)
    if not bool(parameter_dict):
        raise ValueError("JSON experiment not found")
    
    setup_train( parameter_dict, args.device )


