## Installation

## Dataset

For the chemistry dataset, download from chem [data](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip) (2.5GB), unzip it, and put it under under `/data`.

## Train
### DDP Train
```bash
bash masking.sh
bash contextpred.sh
```
### Single GPU Train
```bash
python pretrain_masking.py
python contextpred.py
```
## Get Node Level Representation
```bash
python get_rep.py
```