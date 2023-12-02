#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/fewshot_admet/data/splitdata.py
# Project: /home/richard/projects/fsadmet/data
# Created Date: Tuesday, June 28th 2022, 5:18:45 pm
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

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import numpy as np
from itertools import compress

import random

import json

name = 'sider' # tox21


f = open(os.path.join(BASE_DIR, '{}/raw/{}.csv'.format(name,name)), 'r').readlines()[1:]
np.random.shuffle(f)


if __name__ == "__main__":
    tasks = {}
    
    # Below needs to be modified according to different original datasets
    for index, line in enumerate(f):
        line=line.strip()
        l = line.split(",")
        size=len(l)
        if size<2:
            continue
        '''
        toxcast, sider -> smi = l[0]; for i in range(1, size) 
        tox 21 -> smi = l[-1]; for i in range(12):
        muv -> smi = l[-1]; for i in range(17):
        '''
        smi = l[0] # modify to data
        for i in range(1, size):
            cur_item = l[i].strip()
            if i not in tasks:
                tasks[i] = [[],[]]
            if cur_item == "0.0" or cur_item == "0" or cur_item==0:
                tasks[i][0].append(smi)
            elif cur_item == "1.0" or cur_item == "1" or cur_item==1:
                tasks[i][1].append(smi)
    #until here

    cnt_tasks=[]
    for i in tasks:
        root = name + "/new/" + str(i)
        os.makedirs(root, exist_ok=True)
        os.makedirs(root + "/raw", exist_ok=True)
        os.makedirs(root + "/processed", exist_ok=True)

        file = open(root + "/raw/" + name + ".json", "w")
        file.write(json.dumps(tasks[i]))
        file.close()
        print('task:',i,len(tasks[i][0]), len(tasks[i][1]))
        cnt_tasks.append([len(tasks[i][0]), len(tasks[i][1])])
    print(cnt_tasks)