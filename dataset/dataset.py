#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/fewshot_admet/dataset/test.py
# Project: /home/richard/projects/fsadmet/dataset
# Created Date: Tuesday, June 28th 2022, 3:27:07 pm
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
from xml.etree.ElementTree import canonicalize
from rdkit import Chem

import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import numpy as np
import pandas as pd

import os
from itertools import repeat, product, chain

from .utils import _load_tox21_dataset, _load_sider_dataset


allowable_features = {
    'possible_atomic_num_list':
    list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


def mol_to_graph_data(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    num_atom_features = 2  # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [
            allowable_features['possible_atomic_num_list'].index(
                atom.GetAtomicNum())
        ] + [
            allowable_features['possible_chirality_list'].index(
                atom.GetChiralTag())
        ]
        atom_features_list.append(atom_feature)

    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [
                allowable_features['possible_bonds'].index(bond.GetBondType())
            ] + [
                allowable_features['possible_bond_dirs'].index(
                    bond.GetBondDir())
            ]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                    dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, SMILES = Chem.MolToSmiles(mol, canonical = True ))

    return data



class MoleculeDataset(InMemoryDataset):
    """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
    def __init__(self,
                 root,
                 dataset,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 empty=False):


        self.root = root
        self.dataset = dataset

        super(MoleculeDataset, self).__init__(root, transform, pre_transform,
                                                 pre_filter)


        # print(self.raw_dir) # /data/zhangruochi/projects/fewshot_admet/data/tox21/new/0/raw
        # print(self.raw_paths) # ['/data/zhangruochi/projects/fewshot_admet/data/tox21/new/0/raw/tox21.json']
        # print(self.processed_paths) # ['/data/zhangruochi/projects/fewshot_admet/data/tox21/new/0/processed/geometric_data_processed.pt']
        # print(self.processed_dir) # /data/zhangruochi/projects/fewshot_admet/data/tox21/new/0/processed

        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        data_smiles_list = []
        data_list = []
        if self.dataset == "tox21":
            smiles_list, rdkit_mol_objs, labels = \
                _load_tox21_dataset(self.raw_paths[0])

            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                ## convert aromatic bonds to double bonds
                #Chem.SanitizeMol(rdkit_mol,
                #sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
                
        elif self.dataset == 'sider':
            smiles_list, rdkit_mol_objs, labels = \
                _load_sider_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                #print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        else:
            raise ValueError('Invalid dataset name')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]


        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'smiles.csv'),
                                  index=False,
                                  header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


    def get(self, idx):
        res_data = Data()

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]

            if isinstance(item, torch.Tensor):
                s = list(repeat(slice(None), item.dim()))
                s[res_data.__cat_dim__(key,
                                       item)] = slice(slices[idx],
                                                      slices[idx + 1])
            else:
                s = slices[idx].item()
                
            res_data[key] = item[s]

        return res_data


if __name__ == "__main__":
    root = "/data/zhangruochi/projects/fewshot_admet/data/tox21/new/{}".format(0)
    MoleculeDataset(root, dataset="tox21")
