#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/fsadmet/utils/nni_utils.py
# Project: /home/richard/projects/MolFeSCue/utils
# Created Date: Wednesday, November 16th 2022, 12:18:54 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Fri Jun 07 2024
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

import nni


def update_cfg(cfg):
    # get trialID
    trial_id = nni.get_trial_id()
    # initialize the params
    optimized_params = nni.get_next_parameter()
    if optimized_params:
        # update the config before training
        for p in cfg.model.gnn:
            if p in optimized_params:
                cfg.model.gnn[p] = optimized_params[p]

        cfg.train.random_seed = optimized_params["random_seed"]
        cfg.train.meta_lr = optimized_params["meta_lr"]
        cfg.train.update_lr = optimized_params["update_lr"]
        cfg.train.update_step_test = optimized_params["update_step_test"]
        cfg.train.decay = optimized_params["decay"]
        cfg.meta.contrastive_weight = optimized_params["contrastive_weight"]
        cfg.logger.log_dir = "outputs_{}".format(trial_id)

    return cfg
