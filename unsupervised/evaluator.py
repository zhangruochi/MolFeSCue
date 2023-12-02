import torch
import numpy as np
from tqdm import tqdm
import os

from torch.distributed import ReduceOp

from unsupervised.util import compute_accuracy
from unsupervised.utils.std_logger import Logger


class Masking_Evaluator():
    def __init__(self, model, linear_pred_atoms, linear_pred_bonds, test_loader, criterion, device, cfg=None):
        self.model = model
        self.linear_pred_atoms = linear_pred_atoms
        self.linear_pred_bonds = linear_pred_bonds
        self.device = device
        self.criterion = criterion
        self.test_loader = test_loader
        self.cfg = cfg

    def run(self, split='test'):
        self.model.eval()
        self.linear_pred_atoms.eval()
        self.linear_pred_bonds.eval()
        
        loss_accum = torch.tensor(0.).to(self.device)
        acc_node_accum = torch.tensor(0.).to(self.device)
        acc_edge_accum = torch.tensor(0.).to(self.device)
        
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.test_loader, desc="Evaluate")):
                batch = batch.to(self.device)
                node_rep = self.model(batch.x, batch.edge_index, batch.edge_attr)
                pred_node = self.linear_pred_atoms(node_rep[batch.masked_atom_indices])
                loss = self.criterion(pred_node.double(), batch.mask_node_label[:,0])
                
                acc_node = torch.tensor(compute_accuracy(pred_node, batch.mask_node_label[:,0])).to(self.device)
                acc_node_accum += acc_node
                
                if self.cfg.train.mask_edge:
                    masked_edge_index = batch.edge_index[:, batch.connected_edge_indices]
                    edge_rep = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
                    pred_edge = self.linear_pred_bonds(edge_rep)
                    loss += self.criterion(pred_edge.double(), batch.mask_edge_label[:,0])

                    acc_edge = torch.tensor(compute_accuracy(pred_edge, batch.mask_edge_label[:,0])).to(self.device)
                    acc_edge_accum += acc_edge
                    
                loss_accum += torch.tensor(loss.item()).to(self.device)
                
            if self.cfg.mode.ddp:
                torch.distributed.barrier()
                torch.distributed.all_reduce(loss_accum, op=ReduceOp.SUM)
                torch.distributed.all_reduce(acc_node_accum, op=ReduceOp.SUM)
                torch.distributed.all_reduce(acc_edge_accum, op=ReduceOp.SUM)
                
                loss_accum /= torch.distributed.get_world_size()
                acc_node_accum /= torch.distributed.get_world_size()
                acc_edge_accum /= torch.distributed.get_world_size()
                
            metrics = {
                '{}_loss'.format(split): loss_accum.item()/(step+1),
                '{}_acc_node'.format(split): acc_node_accum.item()/(step+1),
                '{}_acc_edge'.format(split): acc_edge_accum.item()/(step+1)}
        return metrics

                
class Contextpred_Evaluator():
    def __init__(self, model_substruct, model_context, test_loader, criterion, device, cfg=None):
        self.momodel_substructdel = model_substruct
        self.model_context = model_context
        self.device = device
        self.criterion = criterion
        self.test_loader = test_loader
        self.cfg = cfg

    def run(self, split='test'):
        self.momodel_substructdel.eval()
        self.model_context.eval()
        
        balanced_loss_accum = torch.tensor(0.)
        acc_accum = torch.tensor(0.)
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.test_loader, desc="Eval_{}".format(split))):
                batch = batch.to(self.device)
                # creating substructure representation
                substruct_rep = self.model_substruct(batch.x_substruct, batch.edge_index_substruct, batch.edge_attr_substruct)[batch.center_substruct_idx]
                ### creating context representations
                overlapped_node_rep = self.model_context(batch.x_context, batch.edge_index_context, batch.edge_attr_context)[batch.overlap_context_substruct_idx]
                #Contexts are represented by 
                if self.cfg.train.mode == "cbow":
                    # positive context representation
                    context_rep = self.pool_func(overlapped_node_rep, batch.batch_overlapped_context, mode = self.cfg.train.context_pooling)
                    # negative contexts are obtained by shifting the indicies of context embeddings
                    neg_context_rep = torch.cat([context_rep[self.cycle_index(len(context_rep), i+1)] for i in range(self.cfg.train.neg_samples)], dim = 0)
                    
                    pred_pos = torch.sum(substruct_rep * context_rep, dim = 1)
                    pred_neg = torch.sum(substruct_rep.repeat((self.cfg.train.neg_samples, 1))*neg_context_rep, dim = 1)

                elif self.cfg.train.mode == "skipgram":

                    expanded_substruct_rep = torch.cat([substruct_rep[i].repeat((batch.overlapped_context_size[i],1)) for i in range(len(substruct_rep))], dim = 0)
                    pred_pos = torch.sum(expanded_substruct_rep * overlapped_node_rep, dim = 1)

                    #shift indices of substructures to create negative examples
                    shifted_expanded_substruct_rep = []
                    for i in range(self.cfg.train.neg_samples):
                        shifted_substruct_rep = substruct_rep[self.cycle_index(len(substruct_rep), i+1)]
                        shifted_expanded_substruct_rep.append(torch.cat([shifted_substruct_rep[i].repeat((batch.overlapped_context_size[i],1)) for i in range(len(shifted_substruct_rep))], dim = 0))

                    shifted_expanded_substruct_rep = torch.cat(shifted_expanded_substruct_rep, dim = 0)
                    pred_neg = torch.sum(shifted_expanded_substruct_rep * overlapped_node_rep.repeat((self.cfg.train.neg_samples, 1)), dim = 1)

                else:
                    raise ValueError("Invalid mode!")

                loss_pos = self.criterion(pred_pos.double(), torch.ones(len(pred_pos)).to(pred_pos.device).double())
                loss_neg = self.criterion(pred_neg.double(), torch.zeros(len(pred_neg)).to(pred_neg.device).double())

                loss = loss_pos + self.cfg.train.neg_samples*loss_neg

                balanced_loss_accum += torch.tensor(loss_pos.detach().cpu().item() + loss_neg.detach().cpu().item()).to(self.device)
                acc_accum += 0.5* (torch.tensor(torch.sum(pred_pos > 0).detach().cpu().item())/len(pred_pos).to(self.device)
                                   + torch.tensor(torch.sum(pred_neg < 0).detach().cpu().item()).to(self.device)/len(pred_neg))
        
        if self.cfg.mode.ddp:
            torch.distributed.barrier()
            torch.distributed.all_reduce(balanced_loss_accum, op=ReduceOp.SUM)
            torch.distributed.all_reduce(acc_accum, op=ReduceOp.SUM)
            
            balanced_loss_accum /= (step * torch.distributed.get_world_size()).item()
            acc_accum /= (step * torch.distributed.get_world_size()).item()

                
        metrics = {'{}_balanced_loss'.format(split): balanced_loss_accum/(step+1),
                   '{}_accuracy'.format(split): acc_accum/(step+1)}
        
        return metrics