#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/FSADMET/loss_func.py
# Project: /home/richard/projects/fsadmet/utils
# Created Date: Friday, July 1st 2022, 12:59:08 am
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


import torch
import random
import torch.nn as nn
from .train_utils import build_negative_edges
from pytorch_metric_learning import miners, losses

from pytorch_metric_learning.miners import BaseMiner
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
import numpy as np


class HardPairMiner(BaseMiner):
    def __init__(self, alpha_s, alpha_e, beta, **kwargs):
        super().__init__(**kwargs)
        self.alpha_s = alpha_s
        self.alpha_e = alpha_e 
        self.beta = beta
        self.f_t = lambda step: self.alpha_s * np.exp(-beta * step) + self.alpha_e
        

    def mine(self, embeddings, labels, t, ref_emb = None, ref_labels = None):

        mat = self.distance(embeddings, ref_emb)        
        a1, p, a2, n = lmu.get_all_pairs_indices(labels, ref_labels)
        
        neg_pairs = mat[a2, n]        
        alpha = self.f_t(t)
        pos_mask = torch.range(0, len(a1) - 1, dtype=torch.long)
        neg_mask = torch.argsort(neg_pairs, descending=False)[: int(len(neg_pairs) * alpha)]
        
        return a1[pos_mask], p[pos_mask], a2[neg_mask], n[neg_mask]
    

class LossGraphFunc():
    def __init__(self, selfsupervised_weight, contrastive_weight, alpha_s, alpha_e, beta):
        self.selfsupervised_weight = selfsupervised_weight
        self.contrastive_weight = contrastive_weight
        self.criterion = nn.BCEWithLogitsLoss()

        if contrastive_weight > 0:
            self.miner = HardPairMiner(alpha_s=alpha_s, alpha_e=alpha_e, beta=beta)
            self.contrastive = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)

        if self.selfsupervised_weight > 0:
            self.bond_criterion = nn.BCEWithLogitsLoss()
            self.mask_criterion = nn.CrossEntropyLoss()


    def __call__(self, meta_model, batch, node_emb, graph_emb, pred, step):

        self_atom_loss = torch.tensor(0)
        self_bond_loss = torch.tensor(0)
        contrastive_loss = torch.tensor(0)

        y = batch.y.view(pred.shape).to(torch.float64)

        # label loss
        loss = torch.sum(self.criterion(pred.double(),
                                    y)) / pred.size()[0]

        if self.contrastive_weight > 0:
            hard_pairs = self.miner.mine(graph_emb, y.squeeze(), step)
            contrastive_loss = self.contrastive(graph_emb, y.squeeze(), hard_pairs)
            loss += ( self.contrastive_weight * contrastive_loss )

        # selfsupervised loss
        if self.selfsupervised_weight > 0:

            # edge reconstruction loss
            positive_score = torch.sum(
                node_emb[batch.edge_index[0, ::2]] *
                node_emb[batch.edge_index[1, ::2]],
                dim=1)

            negative_edge_index = build_negative_edges(batch)
            negative_score = torch.sum(
                node_emb[negative_edge_index[0]] *
                node_emb[negative_edge_index[1]],
                dim=1)

            self_bond_loss = torch.sum(
                self.bond_criterion(positive_score,
                                torch.ones_like(positive_score)) +
                self.bond_criterion(negative_score,
                                torch.zeros_like(negative_score))
            ) / negative_edge_index[0].size()[0]

            ## add bond loss to total loss
            loss += (self.selfsupervised_weight * self_bond_loss)

            # atom prediction loss
            mask_num = random.sample(range(0,
                                            node_emb.size()[0]), y.shape[0])

            pred_emb = meta_model.masking_linear(
                node_emb[mask_num])

            self_atom_loss = self.mask_criterion(pred_emb, batch.x[mask_num, 0])

            ## add atom loss to total loss
            loss += (self.selfsupervised_weight * self_atom_loss)

        return loss, self_atom_loss, self_bond_loss, contrastive_loss


class LossSeqFunc():
    def __init__(self, contrastive_weight, alpha_s, alpha_e, beta):
        self.contrastive_weight = contrastive_weight
        self.alpha_s = alpha_s
        self.alpha_e = alpha_e 
        self.beta = beta
        self.f_t = lambda step: self.alpha_s * np.exp(-beta * step) + self.alpha_e

        self.criterion = nn.BCEWithLogitsLoss() 

        if contrastive_weight > 0:
            self.miner = HardPairMiner(alpha_s=alpha_s, alpha_e=alpha_e, beta=beta)
            self.contrastive = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)


    def __call__(self, y, pred, seq_emb, step):

        contrastive_loss = torch.tensor(0)
        loss = torch.sum(self.criterion(pred,
                                    y.float())) / pred.size()[0]

        if self.contrastive_weight > 0:
            hard_pairs = self.miner.mine(seq_emb, y.squeeze(), step)
            contrastive_loss = self.contrastive(seq_emb, y.squeeze(), hard_pairs)
            loss += ( self.contrastive_weight * contrastive_loss )
            
        return loss, contrastive_loss