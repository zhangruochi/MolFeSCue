
# MolFeSCue: Enhancing Molecular Property Prediction in Data-Limited and Imbalanced Contexts using Few-Shot and Contrastive Learning

## Introduction
This is the source code and dataset for the above paper. If you have any questions, please contact [zrc720@gmail.com](zrc720@gmail.com)

This study addresses the challenge of predicting molecular properties, a crucial task in fields like drug discovery, material science, and computational chemistry. The main obstacles in this area are the scarcity of annotated data and the imbalance in class distributions, which complicate the development of accurate and robust predictive models. To overcome these challenges, the study introduces the MolFeSCue framework, which integrates pre-trained molecular models within a few-shot learning approach. A key feature of this framework is the use of a novel dynamic contrastive loss function, designed to enhance model performance, particularly in situations with class imbalance. MolFeSCue stands out for its ability to quickly generalize from a limited number of samples and its effective use of the contrastive loss function to derive meaningful molecular representations from imbalanced datasets. 

## Create environment
```
conda env create -f environment.yaml
```

## Create Dataset

The original datasets are downloaded from [Data](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip). We utilize data/splitdata.py to split the datasets according to the molecular properties and save them in different files in the data/[DatasetName]/new.

```python
cd data && python splitdata.py
```

## Launch MLFlow

```python
mlflow server --default-artifact-root file://./mlruns --host 0.0.0.0 --port 5055
```
## Training 

```
chmod -R 777 train.sh
./train.sh
```


