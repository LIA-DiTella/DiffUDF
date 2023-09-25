import torch
from train import setup_train
import os

if __name__=='__main__':
    net_width = 32
    net_depth = 8
    layer_nodes = [net_width] * net_depth

    folder = 'data/Preprocess/ropa/'
    outfolder = f'results/ropa_{net_width}x{net_depth}'
    cuda_device = 0

    if not os.path.exists(outfolder):
        os.mkdir(outfolder)



    exp_config = {
        "num_epochs": 3000,
        "s1_epochs": 2000,
        "sampling_opts": {
            "curvature_iteration_fractions": [0.2, 0.4, 0.4],
            "curvature_percentile_thresholds": [0.6, 0.85]
        },
        "dataset": "...",
        "batch_size": 20000,
        "sampling_percentiles": [0.5, 0.5, 0.0],
        "batches_per_epoch": 1,
        "checkpoint_path": outfolder,
        "experiment_name": "...",
        "epochs_to_checkpoint": 2500,
        "gt_mode": "tanh",
        "loss_s1_weights": [ 1e4, 1e4, 1e4, 1e3, 1e2 ],
        "loss_s2_weights": [ 1e5, 1e5, 1e3, 1e4, 1e2, 1e2 ],
        "alpha": 100,
        "optimizer": {
            "type": "adam",
            "lr_s1": 1e-4,
            "lr_s2": 1e-6
        },
        "network": {
            "hidden_layer_nodes": layer_nodes,
            "w0": 30,
            "pretrained_dict": "None"
        },
        "resolution":256
    }

    for dirpath, dirnames, filenames in os.walk(folder):
        for file in filenames:
            if file[-4:] == '.ply':
                if file == 'saco.ply' or file == 'camisa.ply':
                    print(f'Training for {file}')
                    exp_config['dataset'] = os.path.join(dirpath, file)
                    exp_config['experiment_name'] = file[:-4]

                    setup_train( exp_config, cuda_device)
                    torch.cuda.empty_cache()