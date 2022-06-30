# GNN-exp-pipeline

Framework and script for running experiments on graph classification datasets.

## Models

#### GIN
 - Graph Isomorphism Network from [How Powerful are Graph Neural Networks](https://arxiv.org/pdf/1810.00826.pdf) paper (Xu et al.)
 
### Config file fields and format

An example config file can be found [here]()

device: 'cuda' | 'cpu'
dataset: 'wico'
transform: transform classname

kfolds: number of folds for cross validation >=2
batch_size: size of training batches
epochs: training epochs


