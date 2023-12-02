import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

import nni
import hydra
import mlflow
import shutil
import timeit
import numpy as np
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from unsupervised.model.model import GNN
from unsupervised.model.dataloader import DataLoaderMasking, make_loaders 
from unsupervised.utils.distribution import setup_multinodes, cleanup_multinodes
from unsupervised.utils.utils import fix_random_seed, get_device
from unsupervised.utils.std_logger import Logger
from unsupervised.trainer import Masking_Trainer
from unsupervised.util import MaskAtom

@hydra.main(config_path="configs", config_name="pretrain_masking.yaml", version_base='1.2')
def main(cfg: DictConfig):
    
    orig_cwd = hydra.utils.get_original_cwd()
    
    global_rank = 0
    local_rank = 0
    world_size = 0
    
    if not cfg.mode.nni and cfg.logger.log:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])
        
    # setup distuibution data parallel
    if cfg.mode.ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        # os.environ['NCCL_DEBUG'] = 'INFO'
        # os.environ['NCCL_SHM_DISABLE'] = '1'
        # os.environ["NCCL_SOCKET_IFNAME"] = "eno1"
        random_seed = cfg.train.random_seed + local_rank
        setup_multinodes(local_rank, world_size)
        device = torch.device("cuda", local_rank)
    else:
        random_seed = cfg.train.random_seed
        device = get_device(cfg)

    if global_rank == 0:
        print("setting random seed: {}".format(random_seed))

    fix_random_seed(random_seed, cuda_deterministic=True)

    if global_rank == 0:
        print("num layer: {} mask rate: {:.2f} mask edge: {:.2f}".format(cfg.train.num_layer, cfg.train.mask_rate, cfg.train.mask_edge))

    # set up dataset and transform function.
    dataloaders = make_loaders(ddp=cfg.mode.ddp,
                               dataset=os.path.join(orig_cwd, cfg.train.dataset),
                               world_size=world_size,
                               global_rank=global_rank,
                               batch_size=cfg.train.batch_size,
                               num_workers=cfg.train.num_workers,
                               transform=MaskAtom(num_atom_type=119, num_edge_type=5, mask_rate=cfg.train.mask_rate, mask_edge=cfg.train.mask_edge),
                               loader=DataLoaderMasking)

    # set up models, one for pre-training and one for context embeddings
    model = GNN(cfg.train.num_layer, cfg.train.emb_dim, JK=cfg.train.JK, drop_ratio=cfg.train.dropout_ratio, gnn_type=cfg.train.gnn_type).to(device)
    linear_pred_atoms = torch.nn.Linear(cfg.train.emb_dim, 119).to(device)
    linear_pred_bonds = torch.nn.Linear(cfg.train.emb_dim, 4).to(device)

    model_list = [model, linear_pred_atoms, linear_pred_bonds]

    if cfg.mode.ddp:
        model_list = [DDP(model, device_ids=[global_rank], output_device=global_rank) for model in model_list]

    # set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.decay)
    optimizer_linear_pred_atoms = optim.Adam(linear_pred_atoms.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.decay)
    optimizer_linear_pred_bonds = optim.Adam(linear_pred_bonds.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.decay)
    
    optimizer_list = [optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds]    
    criterion = nn.CrossEntropyLoss()

    if not cfg.mode.nni and cfg.logger.log and global_rank == 0:
        # log hyper-parameters
        for p, v in cfg.train.items():
            mlflow.log_param(p, v)
    
    trainer = Masking_Trainer(
                            cfg=cfg, 
                            global_rank=global_rank, 
                            world_size=world_size, 
                            model_list=model_list, 
                            dataloaders=dataloaders, 
                            criterion=criterion, 
                            optimizer_list=optimizer_list, 
                            device=device,
                            output_dir=cfg.logger.log_dir)
    trainer.run()
    

    if cfg.logger.log:
        if global_rank == 0:
            Logger.info("finished training......")
    #         Logger.info("start evaluating......")
    #         Logger.info("loading best weights from {}......".format(trainer.best_model_path))
            
    # model = load_weights(model, os.path.join(cfg.logger.log_dir, trainer.best_model_path, 'model'), device)
    # linear_pred_atoms = load_weights(linear_pred_atoms, os.path.join(cfg.logger.log_dir, trainer.best_model_path, 'linear_pred_atoms'), device)
    # linear_pred_bonds = load_weights(linear_pred_bonds, os.path.join(cfg.logger.log_dir, trainer.best_model_path, 'linear_pred_bonds'), device)
    
    # # final evaluate
    # split = 'test'
    # evaluator = Masking_Evaluator(model, linear_pred_atoms, linear_pred_bonds, 
    #                               dataloaders[split], criterion, device, cfg)
    # test_metrics = evaluator.run()
    
    # if global_rank == 0:
    #     for metric_name, metric_v in test_metrics.items():
    #         if isinstance(metric_v,  (float, np.float, int, np.int)):
    #             metric_v = round(metric_v,5)
    #         elif isinstance(metric_v,  str):
    #             metric_v = "\n" + metric_v
    #         Logger.info("{} | {}: {}".format(split, metric_name, metric_v))
            
    #     if cfg.mode.nni:
    #         # report final result
    #         test_metrics["default"] = test_metrics["test_{}".format(cfg.task[cfg.task.type].default_metric)]
    #         nni.report_final_result(test_metrics)

    #     if not cfg.mode.nni and cfg.logger.log:            
    #         for metric_name, metric_v in test_metrics.items():
    #             if isinstance(metric_v, (float, np.float64, int, np.int32, np.int64)):
    #                 mlflow.log_metric("test_final/{}".format(metric_name), metric_v, step=1)
    #             elif isinstance(metric_v, str):
    #                 mlflow.log_text(metric_v, "test_final/report.txt")
    
    if cfg.mode.ddp:
        cleanup_multinodes()
    
if __name__ == "__main__":
    main()
