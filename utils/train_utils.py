#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/FSADMET/utils/train_utils.py
# Project: /home/richard/projects/fsadmet/utils
# Created Date: Wednesday, June 29th 2022, 12:12:49 pm
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

import random
import numpy as np
import torch

from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)


def update_params(base_model, loss, update_lr):
    grads = torch.autograd.grad(loss, base_model.parameters(), allow_unused=True)

    # Replace None gradients with zeros
    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, base_model.parameters())]

    return parameters_to_vector(grads), parameters_to_vector(
        base_model.parameters(
        )) - parameters_to_vector(grads) * update_lr


def build_negative_edges(batch):
    font_list = batch.edge_index[0, ::2].tolist()
    back_list = batch.edge_index[1, ::2].tolist()

    all_edge = {}
    for count, front_e in enumerate(font_list):
        if front_e not in all_edge:
            all_edge[front_e] = [back_list[count]]
        else:
            all_edge[front_e].append(back_list[count])

    negative_edges = []
    for num in range(batch.x.size()[0]):
        if num in all_edge:
            for num_back in range(num, batch.x.size()[0]):
                if num_back not in all_edge[num] and num != num_back:
                    negative_edges.append((num, num_back))
        else:
            for num_back in range(num, batch.x.size()[0]):
                if num != num_back:
                    negative_edges.append((num, num_back))

    negative_edge_index = torch.tensor(np.array(
        random.sample(negative_edges, len(font_list))).T,
                                       dtype=torch.long)

    return negative_edge_index


