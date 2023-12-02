#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/FSADMET/unit_test/dataset_test.py
# Project: /home/richard/projects/fsadmet/unit_test
# Created Date: Wednesday, June 29th 2022, 3:19:32 pm
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
import torch
import sys
import os

orig_cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(orig_cwd))

from pathlib import Path
from omegaconf import OmegaConf
from FSADMET.dataset.dataset import MoleculeDataset



# def test_all_tox21_tasks():

#     cfg = OmegaConf.load(os.path.join(orig_cwd, "conf.yaml"))


#     for task_id in range(cfg.tasks.tox21.num_tasks):
#         root = os.path.join(orig_cwd, "data/tox21/new/{}".format(task_id))
#         dataset = MoleculeDataset(root, dataset="tox21")

#         print(len(dataset))

#     assert True

def test_samples():
    cfg = OmegaConf.load(os.path.join(orig_cwd, "conf.yaml"))
    root = os.path.join(orig_cwd, "data/tox21/new/{}".format(0))
    dataset = MoleculeDataset(root, dataset="tox21")

    for i in range(len(dataset)):

        if i >= 5:
            break

        mol_data = dataset[i]

        assert mol_data.x.shape[-1] == cfg.data.num_atom_features
        assert mol_data.edge_attr.shape[-1] == cfg.data.num_bond_features

        print(
            mol_data
        )  # Data(id=[1], edge_index=[2, 94], edge_attr=[94, 2], y=[1], x=[43, 2])
