# GNN-exp-pipeline

Framework and script for running experiments on graph classification datasets.
## Instructions to Run Experiment
 - Write config file such that each posibility for each parameter is included in a list in the configuration file. Example [here](https://github.com/mperozek11/GNN-exp-pipeline/blob/main/config/top_level_experiment.yml)
 - Run the following command: 

 `python exp_set_runner.py -c <config file path> -o <output directory>`

 - Results will be saved in new directory under specified output directory named for the experiment start date/time. Results include:
    - Individual results file for each configuration.
    - Overview file with results summary (e.g. best performing models, total runtime)

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


