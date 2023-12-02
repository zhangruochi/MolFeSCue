import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler

from unsupervised.util import MaskAtom
from unsupervised.splitters import random_split
from unsupervised.model.loader import MoleculeDataset
from unsupervised.model.batch import BatchSubstructContext, BatchMasking, BatchAE
    
class DataLoaderSubstructContext(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderSubstructContext, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchSubstructContext.from_data_list(data_list),
            **kwargs)

class DataLoaderMasking(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderMasking, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchMasking.from_data_list(data_list),
            **kwargs)


class DataLoaderAE(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderAE, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchAE.from_data_list(data_list),
            **kwargs)

def make_loaders(ddp, dataset, world_size, global_rank, batch_size, num_workers, transform, loader):
    dataset = MoleculeDataset(root=dataset, dataset=os.path.basename(dataset), transform=transform)
    train_dataset, valid_dataset, test_dataset = random_split(dataset, task_idx=None, null_value=0, frac_train=0.9,frac_valid=0.05, frac_test=0.05)
    if ddp:
        train_smapler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank)
        valid_smapler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=global_rank)
        test_smapler = DistributedSampler(test_dataset, num_replicas=world_size, rank=global_rank)
        train_loader = loader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=train_smapler)
        valid_loader = loader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=valid_smapler)
        test_loader = loader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = num_workers, pin_memory=True, sampler=test_smapler)

    else:
        train_loader = loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)
        valid_loader = loader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)
        test_loader = loader(test_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)

    dataloaders = {'train': train_loader,
                'valid': valid_loader,
                'test': test_loader}

    return dataloaders

