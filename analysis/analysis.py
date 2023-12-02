#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/fsadmet/analysis/analysis.py
# Project: /home/richard/projects/fsadmet/analysis
# Created Date: Tuesday, November 22nd 2022, 4:14:15 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Tue Nov 22 2022
# Modified By: Ruochi Zhang
# -----
# Copyright (c) 2022 Bodkin World Domination Enterprises
# 
# MIT License
# 
# Copyright (c) 2022 Ruochi Zhang
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
import sys
import os

sys.path.append("..")

from matplotlib import pyplot as plt
import numpy as np
import shutil

from dataset.utils import _load_tox21_dataset
from vis_utils import get_fingerprint,  vis_embedding, vis_neg_pos_ratio

save_dir = "result"
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir)


neg_num_list = []
pos_num_list = []

for i in range(1, 12+1):
    task = i
    input_path = "/home/richard/projects/fsadmet/data/tox21/new/{}/raw/tox21.json".format(task)
    smiles_list, rdkit_mol_objs_list, labels = _load_tox21_dataset(input_path)

    fps = get_fingerprint(rdkit_mol_objs_list)
    vis_embedding(fps, labels, save_path = os.path.join(save_dir, "task_{}_embed.png".format(task)) )
    
    labels = np.squeeze(labels)
    num_1 = np.sum(labels)
    num_0 = len(labels) - num_1

    neg_num_list.append(int(num_0))
    pos_num_list.append(int(num_1))


vis_neg_pos_ratio(pos_num_list, neg_num_list, save_path = os.path.join(save_dir, "neg_pos_ratio.png"))