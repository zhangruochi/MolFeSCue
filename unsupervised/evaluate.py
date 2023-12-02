import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import mlflow
import numpy as np
from omegaconf import OmegaConf

from unsupervised.model.dataloader import DataLoaderMasking, make_loaders #, DataListLoader
from unsupervised.utils.utils import fix_random_seed, get_device, load_model_masking, load_model_contextpred, is_parallel, load_weights
from unsupervised.evaluator import Masking_Evaluator
from unsupervised.util import MaskAtom
from unsupervised.utils.std_logger import Logger 


def evaluate_masking():
    cfg = OmegaConf.load('./configs/pretrain_masking.yaml')
    
    global_rank = 0
    local_rank = 0
    world_size = 1
    random_seed = cfg.train.random_seed
    device = get_device(cfg)
    fix_random_seed(random_seed, cuda_deterministic=True)
    dataloaders = make_loaders(ddp=False,
                               dataset=cfg.train.dataset,
                               world_size=world_size,
                               global_rank=global_rank,
                               batch_size=cfg.train.batch_size,
                               num_workers=cfg.train.num_workers,
                               transform=MaskAtom(num_atom_type=119, num_edge_type=5, mask_rate=cfg.train.mask_rate, mask_edge=cfg.train.mask_edge),
                               loader=DataLoaderMasking)
    
    model, linear_pred_atoms, linear_pred_bonds = load_model_masking(cfg.inference.model_path, device)
    criterion = nn.CrossEntropyLoss()
    
    split = 'valid'
    evaluator = Masking_Evaluator(model, linear_pred_atoms, linear_pred_bonds, 
                                  dataloaders[split], criterion, device, cfg)
    test_metrics = evaluator.run(split)
    print(test_metrics)
    
if __name__ =='__main__':
    evaluate_masking()