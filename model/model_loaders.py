#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/molgen/models/model_loaders.py
# Project: /home/richard/projects/fsadmet/model
# Created Date: Wednesday, October 6th 2021, 12:23:13 am
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Sat Dec 02 2023
# Modified By: Ruochi Zhang
# -----
# Copyright (c) 2021 Bodkin World Domination Enterprises
#
# MIT License
#
# Copyright (c) 2021 HILAB
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

from omegaconf import DictConfig
import torch

from .graph_model import GNN_graphpred
from ..utils.std_logger import Logger
from .meta_model import MetaGraphModel, MetaSeqModel
from transformers import AutoModel, AutoTokenizer


def model_preperation(orig_cwd: str, cfg: DictConfig) -> torch.nn.Module:

    if cfg.model.backbone == "gnn":

        base_learner = GNN_graphpred(cfg.model.gnn.num_layer,
                            cfg.model.gnn.emb_dim,
                            1,
                            JK=cfg.model.gnn.JK,
                            drop_ratio=cfg.model.gnn.dropout_ratio,
                            graph_pooling=cfg.model.gnn.graph_pooling,
                            gnn_type=cfg.model.gnn.gnn_type)

        if cfg.model.gnn.pretrained:
            Logger.info("load pretrained model from {} ......".format(cfg.model.gnn.pretrained))
            base_learner.from_pretrained(os.path.join(orig_cwd, cfg.model.gnn.pretrained))
        
        model = MetaGraphModel(base_learner, cfg.meta.selfsupervised_weight,
                        cfg.model.gnn.emb_dim)

    elif cfg.model.backbone == "seq":

        base_learner = AutoModel.from_pretrained(os.path.join(orig_cwd, cfg.model.seq.pretrained))
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(orig_cwd, cfg.model.seq.pretrained))
        
        model = MetaSeqModel(base_learner, tokenizer)

    Logger.info("load model successful! ......\n")
    Logger.info(model)

    return model