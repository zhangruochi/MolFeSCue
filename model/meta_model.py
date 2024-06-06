#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/fewshot_admet/model/meta_model.py
# Project: /home/richard/projects/fsadmet/model
# Created Date: Tuesday, June 28th 2022, 6:30:24 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Tue Jun 04 2024
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

import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer


class attention(nn.Module):

    def __init__(self, dim):
        super(attention, self).__init__()
        self.layers = nn.Sequential(nn.Linear(dim, 100), nn.ReLU(),
                                    nn.Linear(100, 1))
        self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, x):
        # # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        # x = x.view(x.size(0), -1)
        x = self.layers(x)
        x = self.softmax_layer(torch.transpose(x, 1, 0))
        return x


class Interact_attention(nn.Module):

    def __init__(self, dim, num_tasks):
        super(Interact_attention, self).__init__()
        self.layers = nn.Sequential(nn.Linear(num_tasks * dim, dim), nn.Tanh())

    def forward(self, x):
        x = self.layers(x)
        return x


class MetaGraphModel(nn.Module):

    def __init__(self, base_model, selfsupervised_weight, emb_dim):
        super(MetaGraphModel, self).__init__()

        self.base_model = base_model
        self.emb_dim = emb_dim
        self.selfsupervised_weight = selfsupervised_weight

        if self.selfsupervised_weight > 0:
            self.masking_linear = nn.Linear(self.emb_dim, 119)


class MetaSeqModel(nn.Module):

    def __init__(self, base_model, tokenizer):
        super(MetaSeqModel, self).__init__()

        self.base_model = base_model
        self.tokenizer = tokenizer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 768))
        self.final_layer = nn.Linear(768, 1)

    def forward(self, x):
        pred_y = self.base_model(**x)[0]  # [10,153,768]
        pool_output = self.global_avg_pool(pred_y).squeeze(1)  # [10,768]
        pred = self.final_layer(pool_output)  # 10x1

        return pred, pool_output
