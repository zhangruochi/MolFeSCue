#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/fsadmet/analysis/vis_utils.py
# Project: /home/richard/projects/fsadmet/analysis
# Created Date: Tuesday, November 22nd 2022, 3:45:22 pm
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

from rdkit import Chem
from rdkit.Chem import DataStructs
import numpy as np

from sklearn.manifold import MDS, TSNE
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def get_fingerprint(mols_list):
    fps = []
    for x in mols_list:
        fp = Chem.RDKFingerprint(x) 
        arr = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp,arr)
        fps.append(arr)
    return fps


def vis_neg_pos_ratio(pos_num_list, neg_num_list, save_path):
    plt.figure(figsize = (12,8))
    name_list = ["task_{}".format(_) for _ in range(1, 12+1)]
    x = list(range(len(pos_num_list)))
    total_width, n = 0.6, 2
    width = total_width / n
    plt.bar(x, pos_num_list, width=width, label="postive mol")
    for a,b in zip(x,pos_num_list):   #柱子上的数字显示
        plt.text(a,b,'%d'%b,ha='center',va='bottom',fontsize=18)
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, neg_num_list, width=width, label="negative mol", tick_label = name_list)
    for a,b in zip(x,neg_num_list):   #柱子上的数字显示
        plt.text(a,b,'%.d'%b,ha='center',va='bottom',fontsize=18)

    plt.legend(fontsize=8)
    plt.savefig(save_path)

def vis_embedding(mol_fp_arr_list, labels, save_path):
    enc_seqs = np.stack(mol_fp_arr_list, axis = 0)
    DIMRED = TSNE
    s = DIMRED(n_components=2, random_state=3).fit_transform(enc_seqs)

    df = pd.DataFrame(data = None)
    df["TSNE1"]=s[:,0]
    df["TSNE2"]=s[:,1]
    df["labels"]=np.squeeze(labels)

    plt.figure(figsize = (12, 8), dpi=200)
    sns.scatterplot(x="TSNE1", y="TSNE2", hue="labels",data=df,alpha=0.6)
    plt.savefig(save_path)
    
    