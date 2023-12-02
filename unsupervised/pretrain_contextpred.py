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
import numpy as np
from omegaconf import DictConfig

from unsupervised.model.model import GNN
from unsupervised.model.dataloader import DataLoaderSubstructContext, make_loaders 
from unsupervised.util import ExtractSubstructureContextPair
from unsupervised.utils.distribution import setup_multinodes, cleanup_multinodes
from unsupervised.utils.utils import fix_random_seed, get_device, is_parallel, load_weights
from unsupervised.utils.std_logger import Logger
from unsupervised.trainer import Contextpred_Trainer
from unsupervised.evaluator import Contextpred_Evaluator


@hydra.main(config_path="configs", config_name="pretrain_contextpred.yaml", version_base='1.2')
def main(cfg: DictConfig):

    orig_cwd = hydra.utils.get_original_cwd()
    
    global_rank = 0
    local_rank = 0
    world_size = 0
    
    if not cfg.mode.nni and cfg.logger.log:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])
        # mlflow.set_tag("mlflow.runName", "run_name") # conflic with ddp
        
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

    fix_random_seed(random_seed, cuda_deterministic=True)

    l1 = cfg.train.num_layer - 1
    l2 = l1 + cfg.train.csize
    
    if global_rank == 0:
        print("num layer: {} l1: {} l2: {} mode: {}".format(cfg.train.num_layer, l1, l2, cfg.train.mode))

    #set up dataset and transform function.
    dataloaders = make_loaders(ddp=cfg.mode.ddp,
                               dataset=os.path.join(orig_cwd, cfg.train.dataset),
                               world_size=world_size,
                               global_rank=global_rank,
                               batch_size=cfg.train.batch_size,
                               num_workers=cfg.train.num_workers,
                               transform=ExtractSubstructureContextPair(cfg.train.num_layer, l1, l2),
                               loader=DataLoaderSubstructContext)
    # dataset = MoleculeDataset("dataset/" + cfg.train.dataset, dataset=cfg.train.dataset, transform = ExtractSubstructureContextPair(cfg.train.num_layer, l1, l2))
    # loader = DataLoaderSubstructContext(dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers = cfg.train.num_workers)

    #set up models, one for pre-training and one for context embeddings
    model_substruct = GNN(cfg.train.num_layer, cfg.train.emb_dim, JK = cfg.train.JK, drop_ratio = cfg.train.dropout_ratio, gnn_type = cfg.train.gnn_type).to(device)
    model_context = GNN(int(l2 - l1), cfg.train.emb_dim, JK = cfg.train.JK, drop_ratio = cfg.train.dropout_ratio, gnn_type = cfg.train.gnn_type).to(device)
    
    if cfg.mode.ddp:
        model_substruct = DDP(model_substruct, device_ids=[local_rank], output_device=local_rank)
        model_context = DDP(model_context, device_ids=[local_rank], output_device=local_rank)
        
    # set up optimizer for the two GNNs
    optimizer_substruct = optim.Adam(model_substruct.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.decay)
    optimizer_context = optim.Adam(model_context.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.decay)
    criterion = nn.BCEWithLogitsLoss()

    if not cfg.mode.nni and cfg.logger.log and global_rank == 0:
        # log hyper-parameters
        for p, v in cfg.train.items():
            mlflow.log_param(p, v)
    
    trainer = Contextpred_Trainer(
        cfg=cfg,
        global_rank=global_rank, 
        world_size=world_size, 
        model_substruct=model_substruct,
        model_context=model_context,
        optimizer_substruct=optimizer_substruct,
        optimizer_context=optimizer_context,
        dataloaders=dataloaders, 
        criterion=criterion, 
        device=device,
        output_dir=cfg.logger.log_dir)
    
    trainer.run()
    

    if cfg.logger.log:
        if global_rank == 0:
            Logger.info("finished training......")
            Logger.info("start evaluating......")
    #         Logger.info("loading best weights from {}......".format(trainer.best_model_path))
    #     model_substruct = load_weights(model_substruct, os.path.join(cfg.logger.log_dir, trainer.best_model_path, 'model_substruct'), device)
    #     model_context = load_weights(model_context, os.path.join(cfg.logger.log_dir, trainer.best_model_path, 'model_context'), device)

    # evaluator = Contextpred_Evaluator(model_substruct, model_context, 
    #                               dataloaders['test'], criterion, device, cfg)
    # test_metrics = evaluator.run()
    
    # if global_rank == 0:
    #     for metric_name, metric_v in test_metrics.items():
    #         if isinstance(metric_v,  (float, np.float, int, np.int)):
    #             metric_v = round(metric_v,5)
    #         elif isinstance(metric_v,  str):
    #             metric_v = "\n" + metric_v
    #         Logger.info("test | {}: {}".format(metric_name, metric_v))
            
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
    #cycle_index(10,2)
    main()
