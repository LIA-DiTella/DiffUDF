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
from src.loss_functions import loss_siren, loss_tanh
from src.model import SIREN
from src.util import create_output_paths, load_experiment_parameters
from generate_df import generate_df
from generate_mc import generate_mc
import open3d as o3d

def update_learning_rate( epoch, warmup_epochs, optimizer, total_epochs, initial_learning_rate ):
    lr =  (epoch / warmup_epochs) if epoch < warmup_epochs else 0.5 * (np.cos((epoch - warmup_epochs)/(total_epochs - warmup_epochs) * np.pi) + 1) 
    lr = lr * initial_learning_rate

    for g in optimizer.param_groups:
        g['lr'] = lr

    return lr

def train_model(dataset, model, device, config) -> torch.nn.Module:
    epochs = config["epochs"]
    warmup_epochs = config.get("warmup_epochs", 0)

    epochs_til_checkpoint = config.get("epochs_to_checkpoint", 0)

    log_path = config["log_path"]
    loss_fn = config["loss_fn"]
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
    for epoch in range(epochs):            
        running_loss = dict()
        current_lr = None
        for input_data, normals, sdf in iter(dataset):
            
            current_lr = update_learning_rate( epoch, warmup_epochs, optim, epochs, config['init_lr'])
            # zero the parameter gradients
            optim.zero_grad()
            
            # forward + backward + optimize
            input_data = input_data.to( device )
            normals = normals.to(device)
            sdf = sdf.to(device)
            
            loss = loss_fn( model, input_data, {'normals': normals, 'sdf': sdf}, config['loss_weights'], config["alpha"])

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
        print(f"Epoch: {epoch} - Loss: {epoch_loss} - Learning Rate: {current_lr:.2e}")

        # Saving the best model after warmup.
        if epoch > warmup_epochs and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(
                best_weights,
                osp.join(log_path, "models", "model_best.pth")
            )

        # saving the model at checkpoints
        if epoch and epochs_til_checkpoint and not \
           epoch % epochs_til_checkpoint:
            print(f"Saving model for epoch {epoch}")
            torch.save(
                model.state_dict(),
                osp.join(log_path, "models", f"model_{epoch}.pth")
            )
            print(f"Generating mesh")
            generate_mc( model, config["gt_mode"], device, 256, 
                        osp.join(log_path, "reconstructions", f'mc_mesh_{epoch}.obj'), alpha=config['alpha'])

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
        model.load_state_dict(torch.load(network_params['pretrained_dict']))

    opt_params = parameter_dict["optimizer"]
    if opt_params["type"] == "adam":
        optimizer = torch.optim.Adam(
            lr=opt_params["lr"],
            params=model.parameters()
        )
    elif opt_params["type"] == "lbfgs":
        optimizer = torch.optim.LBFGS(
            lr=opt_params["lr"],
            params=model.parameters()
        )
    

    if parameter_dict["loss"] == "loss_siren":
        loss_fn = loss_siren
    elif parameter_dict["loss"] == "loss_tanh":
        loss_fn = loss_tanh
    else:
        raise ValueError("Loss unknown")

    config_dict = {
        "epochs": parameter_dict["num_epochs"],
        "warmup_epochs": parameter_dict.get("warmup_epochs", 0),
        "batch_size": parameter_dict["batch_size"],
        "epochs_to_checkpoint": parameter_dict["epochs_to_checkpoint"],
        "gt_mode": parameter_dict["gt_mode"],
        "log_path": full_path,
        "optimizer": optimizer,
        "init_lr": opt_params["lr"],
        "loss_fn": loss_fn,
        "loss_weights": parameter_dict["loss_weights"],
        "alpha": parameter_dict["alpha"]
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

    generate_df( osp.join(full_path, "models", "model_best.pth"), parameter_dict['dataset'], osp.join(full_path, "reconstructions/"), df_options)


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


