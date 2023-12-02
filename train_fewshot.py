#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/fewshot_admet/train.py
# Project: /home/richard/projects/MolFeSCue
# Created Date: Tuesday, June 28th 2022, 6:47:18 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Sat Dec 02 2023
# Modified By: Ruochi Zhang
# -----
# Copyright (c) 2022 Bodkin World Domination Enterprises
#
# MIT License
#
# Copyright (c) 2022 HILAB
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----
###

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import mlflow
import nni
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from fsadmet.model.model_loaders import model_preperation
from fsadmet.utils.nni_utils import update_cfg
from fsadmet.utils.helpers import fix_random_seed
from fsadmet.utils.trainer import Trainer
from fsadmet.utils.distribution import setup_multinodes


@hydra.main(config_path="conf", config_name="conf")
def main(cfg: DictConfig):

    # Training settings

    orig_cwd = hydra.utils.get_original_cwd()
    global_rank = 0
    local_rank = 0
    world_size = 0

    if not cfg.mode.nni and cfg.logger.log:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    os.environ['NCCL_DEBUG'] = 'INFO'
    # os.environ['NCCL_SHM_DISABLE'] = '1'
    # os.environ["NCCL_SOCKET_IFNAME"] = "eno1"
    setup_multinodes(local_rank, world_size)

    if cfg.mode.nni:
        # use nni params
        cfg = update_cfg(cfg)

    fix_random_seed(cfg.train.random_seed, cuda_deterministic=True)

    if cfg.logger.log:

        # log hyper-parameters
        for p, v in cfg.data.items():
            mlflow.log_param(p, v)

        for p, v in cfg.model.items():
            mlflow.log_param(p, v)

        for p, v in cfg.meta.items():
            mlflow.log_param(p, v)

        for p, v in cfg.train.items():
            mlflow.log_param(p, v)

        for p, v in cfg.tasks[cfg.tasks.name].items():
            mlflow.log_param(p, v)

    device = torch.device("cuda", local_rank)
    meta_model = model_preperation(orig_cwd, cfg).to(device)

    trainer = Trainer(meta_model, cfg, device = device)

    trainer.run()


if __name__ == "__main__":
    main()
