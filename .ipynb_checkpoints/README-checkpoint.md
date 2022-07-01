# GNN-exp-pipeline

Framework and script for running experiments on graph classification datasets.

## Models

#### GIN
 - Graph Isomorphism Network from [How Powerful are Graph Neural Networks](https://arxiv.org/pdf/1810.00826.pdf) paper (Xu et al.)
 - Implemented with [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html)
Required config fields:

`hidden_units: list of embedding dimension for each layer (List(int))`
`train_eps: When true, epsilon will be a trainable parameter in neighborhood aggregation function (True | False)`
`aggregation: aggregation scheme (sum | mean)`
`dropout: dropout (float 0.0-1.0)`

### Config file fields and format

An example config file can be found [here](https://github.com/mperozek11/GNN-exp-pipeline/blob/main/config/test_config.yml)

device: 'cuda' | 'cpu'
dataset: 'wico'
transform: transform classname

kfolds: number of folds for cross validation >=2
batch_size: size of training batches
epochs: training epochs


