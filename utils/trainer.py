#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/fsadmet/trainer.py
# Project: /home/richard/projects/fsadmet/utils
# Created Date: Wednesday, June 29th 2022, 1:34:26 pm
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
# import nni
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import numpy as np

from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as DataLoaderChem
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
import torch.optim as optim
import torch

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from ..dataset.dataset import MoleculeDataset
from ..dataset.dataset_chem import MoleculeDataset as MoleculeDatasetChem
from ..dataset.utils import my_collate_fn
from ..model.samples import sample_datasets, sample_test_datasets
from .train_utils import update_params
from .loss import LossGraphFunc, LossSeqFunc
from .std_logger import Logger
import nni


class Trainer(object):
    def __init__(self, meta_model, cfg, device):

        self.meta_model = meta_model
        self.device = device
        self.cfg = cfg
        
        # meta-learning parameters
        self.dataset_name = cfg.data.dataset
        self.data_path_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), cfg.data.path)

        self.num_tasks = cfg.tasks[self.dataset_name].num_tasks
        self.num_train_tasks = cfg.tasks[self.dataset_name].num_train_tasks
        self.num_test_tasks = cfg.tasks[self.dataset_name].num_test_tasks
        self.n_way = cfg.tasks[self.dataset_name].n_way
        self.m_support = cfg.tasks[self.dataset_name].m_support
        self.k_query = cfg.tasks[self.dataset_name].k_query

        # training paramters
        self.batch_size = cfg.train.batch_size
        self.meta_lr = cfg.train.meta_lr
        self.update_lr = cfg.train.update_lr
        self.update_step = cfg.train.update_step
        self.update_step_test = cfg.train.update_step_test
        self.eval_epoch = cfg.train.eval_epoch
     
        ## --------  criterion ----------

        if cfg.model.backbone == "gnn":
            self.loss_func = LossGraphFunc(cfg.meta.selfsupervised_weight,
                                  cfg.meta.contrastive_weight, cfg.meta.alpha_s, cfg.meta.alpha_e, cfg.meta.beta)
        elif cfg.model.backbone == "seq":
            self.loss_func = LossSeqFunc(cfg.meta.contrastive_weight, cfg.meta.alpha_s, cfg.meta.alpha_e, cfg.meta.beta)


        ## --------  optiomizaer ----------
        if cfg.model.backbone == "gnn":
            model_param_group = []

            model_param_group.append(
                {"params": self.meta_model.base_model.gnn.parameters()})

            model_param_group.append({
                "params":
                self.meta_model.base_model.graph_pred_linear.parameters(),
                "lr":
                cfg.train.lr * cfg.train.lr_scale
            })

            if cfg.model.gnn.graph_pooling == "attention":
                model_param_group.append({
                    "params":
                    self.meta_model.base_model.pool.parameters(),
                    "lr":
                    cfg.train.lr * cfg.train.lr_scale
                })
            
            if cfg.meta.selfsupervised_weight:
                model_param_group.append({"params": self.meta_model.masking_linear.parameters()})

            self.optimizer = optim.Adam(model_param_group,
                                lr=cfg.train.meta_lr,
                                weight_decay=cfg.train.decay)

        elif cfg.model.backbone == "seq":
            self.optimizer = optim.Adam(self.meta_model.parameters(),
                                lr=cfg.train.meta_lr,
                                weight_decay=cfg.train.decay)

        ## --------  optimizaer ----------

        ## --------  trainer ----------

        self.epochs = cfg.train.epochs
        self.num_workers = cfg.data.num_workers
        ## --------  trainer ----------

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                            self.epochs,
                                                            gamma=cfg.train.lr_decay_ratio,
                                                            last_epoch=-1,
                                                            verbose=False)

    def train_epoch(self, epoch):

        # samples dataloaders
        support_loaders = []
        query_loaders = []

        self.meta_model.base_model.train()

        for task in range(self.num_train_tasks):
            # for task in tasks_list:

            if self.cfg.model.backbone == "gnn":
                dataset = MoleculeDataset(os.path.join(self.data_path_root, self.dataset_name, "new", str(task+1)),
                                        dataset=self.dataset_name)
                collate_fn = None
            elif self.cfg.model.backbone == "seq":
                dataset = MoleculeDatasetChem(os.path.join(self.data_path_root, self.dataset_name, "new", str(task+1)),
                                        dataset=self.dataset_name)
                collate_fn = my_collate_fn
                DataLoader = DataLoaderChem

            support_dataset, query_dataset = sample_datasets(
                dataset, self.dataset_name, task, self.n_way, self.m_support, self.k_query)

            support_loader = DataLoader(support_dataset,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        num_workers=self.num_workers,
                                        collate_fn = collate_fn)

            query_loader = DataLoader(query_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      num_workers=self.num_workers,
                                      collate_fn = collate_fn)

            support_loaders.append(support_loader)
            query_loaders.append(query_loader)

        for k in range(0, self.update_step):

            old_params = parameters_to_vector(
                self.meta_model.base_model.parameters())


            # use this loss to save all the losses of query set
            losses_q = torch.tensor([0.0]).to(self.device)

            for task in range(self.num_train_tasks):

                losses_s = torch.tensor([0.0]).to(self.device)

                losses_self_atom = 0
                losses_self_bond = 0
                losses_contr = 0
                
                # training support
                for _, batch in enumerate(
                        tqdm(
                            support_loaders[task],
                            desc=
                            "Training | Epoch: {} | UpdateK: {} | Task: {} | Support Iteration"
                            .format(epoch, k, task+1))):

                    if self.cfg.model.backbone == "gnn":
                        batch = batch.to(self.device)
                        pred, graph_emb, node_emb = self.meta_model.base_model(
                            batch.x, batch.edge_index, batch.edge_attr,
                            batch.batch)
                        y = batch.y.view(pred.shape).to(torch.float64)

                        loss, self_atom_loss, self_bond_loss, contrastive_loss = self.loss_func(self.meta_model, batch, node_emb, graph_emb, pred, self.epochs)

                        losses_s += loss
                        losses_self_atom += self_atom_loss.item()
                        losses_self_bond += self_bond_loss.item()
                        losses_contr += contrastive_loss.item()


                    elif self.cfg.model.backbone == "seq":
                        input_tensor = self.meta_model.tokenizer(batch[0], max_length = 512, add_special_tokens= True, truncation= True, padding=True,return_tensors='pt').to(self.device)
                        pred, seq_embed = self.meta_model(input_tensor)# last_hidden_states
                        y = torch.tensor(batch[1]).to(self.device)
                        loss , contrastive_loss = self.loss_func(y, pred, seq_embed, self.epochs)

                        losses_s += loss
                        losses_contr += contrastive_loss.item()    

                if not self.cfg.mode.nni and self.cfg.logger.log:
                    mlflow.log_metric("Training/support_task_{}_loss".format(task+1),
                                      losses_s.item(),
                                      step=epoch * self.update_step + k)
                    mlflow.log_metric("Training/support_task_{}_loss_self_atom".format(task+1),
                                      losses_self_atom,
                                      step=epoch * self.update_step + k)
                    mlflow.log_metric("Training/support_task_{}_loss_self_bond".format(task+1),
                                      losses_self_bond,
                                      step=epoch * self.update_step + k)
                    mlflow.log_metric("Training/support_task_{}_loss_contr".format(task+1),
                                      losses_contr,
                                      step=epoch * self.update_step + k)


                _, new_params = update_params(self.meta_model.base_model,
                                                     losses_s,
                                                     update_lr=self.update_lr)

                # update parameters of base model by new_params leant from support set
                vector_to_parameters(new_params,
                                     self.meta_model.base_model.parameters())


                # use this loss to save the loss on a single query task
                this_loss_q = torch.tensor([0.0]).to(self.device)

                losses_self_atom = 0
                losses_self_bond = 0
                losses_contr = 0

                # training query task set
                for _, batch in enumerate(
                        tqdm(
                            query_loaders[task],
                            desc=
                            "Training | Epoch: {} | UpdateK: {} | Task: {} | Query Iteration"
                            .format(epoch, k, task+1))):

                    if self.cfg.model.backbone == "gnn":
                        batch = batch.to(self.device)
                        pred, graph_emb, node_emb = self.meta_model.base_model(
                            batch.x, batch.edge_index, batch.edge_attr,
                            batch.batch)

                        loss_q, self_atom_loss, self_bond_loss, contrastive_loss = self.loss_func(self.meta_model, batch, node_emb, graph_emb, pred, self.epochs)

                        this_loss_q += loss_q

                        losses_self_atom += self_atom_loss.item()
                        losses_self_bond += self_bond_loss.item()
                        losses_contr += contrastive_loss.item()
 
                    elif self.cfg.model.backbone == "seq":
                        input_tensor = self.meta_model.tokenizer(batch[0], max_length = 512, add_special_tokens= True, truncation= True,padding=True,return_tensors='pt').to(self.device)
                        pred, seq_embed = self.meta_model(input_tensor)
                        y = torch.tensor(batch[1]).to(self.device)
                        loss_q , contrastive_loss = self.loss_func(y, pred, seq_embed, self.epochs)

                        this_loss_q += loss_q
                        losses_contr += contrastive_loss.item()
                        

                if not self.cfg.mode.nni and self.cfg.logger.log:
                    mlflow.log_metric("Training/query_task_{}_loss".format(task+1),
                                      this_loss_q.item(),
                                      step=epoch * self.update_step + k)
                    mlflow.log_metric("Training/query_task_{}_loss_self_atom".format(task+1),
                                      losses_self_atom,
                                      step=epoch * self.update_step + k)
                    mlflow.log_metric("Training/query_task_{}_loss_self_bond".format(task+1),
                                      losses_self_bond,
                                      step=epoch * self.update_step + k)
                    mlflow.log_metric("Training/query_task_{}_loss_contr".format(task+1),
                                      losses_contr,
                                      step=epoch * self.update_step + k)

                if task == 0:
                    losses_q = this_loss_q
                else:
                    losses_q = torch.cat((losses_q, this_loss_q), 0)

                vector_to_parameters(old_params,
                                     self.meta_model.base_model.parameters())


            loss_q = torch.sum(losses_q) / self.num_train_tasks

            if not self.cfg.mode.nni and self.cfg.logger.log:
                mlflow.log_metric("Training/weighted_query_loss",
                                    loss_q.item(),
                                    step=epoch * self.update_step + k)

            self.optimizer.zero_grad()
            loss_q.backward()
            self.optimizer.step()
            # self.lr_scheduler.step()
            
            cur_lr = self.lr_scheduler.optimizer.state_dict(
                )['param_groups'][0]['lr']

            if not self.cfg.mode.nni and self.cfg.logger.log:
                mlflow.log_metric("lr",
                                    cur_lr,
                                    step=epoch)
        return []

    def test(self, epoch):

        accs = []
        rocs = []

        old_params = parameters_to_vector(self.meta_model.base_model.parameters())

        for task in range(self.num_test_tasks):

            if self.cfg.model.backbone == "gnn":
                dataset = MoleculeDataset(os.path.join(
                    self.data_path_root, self.dataset_name, "new",
                    str(self.num_tasks - task)),
                                        dataset=self.dataset_name)
                collate_fn = None 
            elif self.cfg.model.backbone == "seq":
                dataset = MoleculeDatasetChem(os.path.join(
                    self.data_path_root, self.dataset_name, "new",
                    str(self.num_tasks - task)),
                                        dataset=self.dataset_name)
                collate_fn = my_collate_fn
                DataLoader = DataLoaderChem
            
            support_dataset, query_dataset = sample_test_datasets(
                dataset, self.dataset_name, self.num_tasks - task - 1, self.n_way,
                self.m_support, self.k_query)

            support_loader = DataLoader(support_dataset,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        num_workers=self.num_workers,
                                        collate_fn = collate_fn)
            query_loader = DataLoader(query_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      num_workers=self.num_workers,
                                      collate_fn = collate_fn)

            self.meta_model.eval()

            for k in range(0, self.update_step_test):
                loss = torch.tensor([0.0]).to(self.device)

                losses_self_atom = 0
                losses_self_bond = 0
                losses_contr = 0
                
                for step, batch in enumerate(
                        tqdm(support_loader,
                             desc="Testing | Epoch: {} | UpdateK: {} | Task: {} | Support Iteration".
                             format(epoch, k, task))):


                    if self.cfg.model.backbone == "gnn":
                        batch = batch.to(self.device)

                        pred, graph_emb, node_emb = self.meta_model.base_model(
                            batch.x, batch.edge_index, batch.edge_attr,
                            batch.batch)

                        test_loss_s, self_atom_loss, self_bond_loss, contrastive_loss = self.loss_func(self.meta_model, batch, node_emb, graph_emb, pred, self.epochs)

                        loss += test_loss_s

                        losses_self_atom += self_atom_loss.item()
                        losses_self_bond += self_bond_loss.item()
                        losses_contr += contrastive_loss.item()

                    elif self.cfg.model.backbone == "seq":
                        input_tensor = self.meta_model.tokenizer(batch[0], max_length = 512, add_special_tokens= True, truncation= True,padding=True,return_tensors='pt').to(self.device)
                        pred, seq_embed = self.meta_model(input_tensor)
                        y = torch.tensor(batch[1]).to(self.device)
                        test_loss_s, contrastive_loss = self.loss_func(y, pred, seq_embed, self.epochs)
                        loss += test_loss_s
                        losses_contr += contrastive_loss.item()

                
                if not self.cfg.mode.nni and self.cfg.logger.log:
                    mlflow.log_metric(
                        "Testing/support_task_{}_loss".format(task),
                        loss.item(),
                        step=epoch * self.update_step_test + k)

                    mlflow.log_metric("Testing/support_task_{}_loss_self_atom".format(task),
                                      losses_self_atom,
                                      step=epoch * self.update_step_test + k)
                    mlflow.log_metric("Testing/support_task_{}_loss_self_bond".format(task),
                                      losses_self_bond,
                                      step=epoch * self.update_step_test + k)
                    mlflow.log_metric("Testing/support_task_{}_loss_contr".format(task),
                                      losses_contr,
                                      step=epoch * self.update_step_test + k)

                new_grad, new_params = update_params(
                    self.meta_model.base_model, loss, update_lr=self.update_lr)

                vector_to_parameters(new_params, self.meta_model.base_model.parameters())

            y_true = []
            y_scores = []

            for _, batch in enumerate(
                    tqdm(query_loader,
                         desc=
                         "Testing | Epoch: {} | UpdateK: {} | Task: {} | Query Iteration".
                         format(epoch, k, task))):

                if self.cfg.model.backbone == "gnn":
                    batch = batch.to(self.device)
                    pred, graph_emb, node_emb = self.meta_model.base_model(
                        batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    
                elif self.cfg.model.backbone == "seq":
                    input_tensor = self.meta_model.tokenizer(batch[0], max_length = 512, add_special_tokens= True, truncation= True,padding=True,return_tensors='pt').to(self.device)
                    pred, seq_embed = self.meta_model(input_tensor)

        
                pred = torch.sigmoid(pred)
                pred = torch.where(pred > 0.5, torch.ones_like(pred), pred)
                pred = torch.where(pred <= 0.5, torch.zeros_like(pred), pred)
                y_scores.append(pred)
                y_true.append(torch.tensor(batch[1]).to(self.device))
                
            y_true = torch.cat(y_true, dim=0).cpu().detach().numpy()[:,0]
            y_scores = torch.cat(y_scores, dim=0).cpu().detach().numpy()[:,0]
            roc_score = roc_auc_score(y_true, y_scores)
            acc_score = accuracy_score(y_true, y_scores)

            if not self.cfg.mode.nni and self.cfg.logger.log:
                mlflow.log_metric("Testing/query_task_{}_acc".format(task),
                                  acc_score,
                                  step=epoch)
                mlflow.log_metric("Testing/query_task_{}_auc".format(task),
                                  roc_score,
                                  step=epoch)

            accs.append(acc_score)
            rocs.append(roc_score)

        if not self.cfg.mode.nni and self.cfg.logger.log:
            mlflow.log_metric("Testing/query_mean_acc",
                                  np.mean(accs),
                                  step=epoch)
            mlflow.log_metric("Testing/query_mean_auc", np.mean(rocs),
                                step=epoch)
             
            vector_to_parameters(old_params, self.meta_model.base_model.parameters())

        return accs, rocs

    def run(self):

        for epoch in range(1, self.epochs + 1):
            self.train_epoch(epoch)

            if epoch % self.eval_epoch == 0:
                accs, rocs = self.test(epoch)

                Logger.info("downstream task accs: {}".format([round(_,3) for _ in accs]))
                Logger.info("downstream task aucs: {}".format([round(_,3) for _ in rocs]))

            if self.cfg.mode.nni:
                nni.report_intermediate_result(np.mean(rocs))

        if self.cfg.mode.nni:
            nni.report_final_result(np.mean(rocs))