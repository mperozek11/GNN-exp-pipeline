import torch
import numpy as np
from tensorflow import keras
from sklearn.utils import class_weight


def target_to_categorical(dataset, target_dim):

    if type(dataset[0]) == list: # traditional NN
        dataset[1] = torch.tensor(keras.utils.to_categorical(dataset[1], target_dim))

    else: # pyg Data class
        for i in range(len(dataset)):
            dataset[i].y = torch.tensor(keras.utils.to_categorical(dataset[i].y, target_dim))

    return dataset

def get_class_weights(calculate, y):
    if calculate:
        classes = np.unique(np.array(y))
            # below are two different weighting schemes which yeild the same proportions but use different strategies

            # === 1 ===
            # prevs = []
            # for c in classes:
            #     prev = len((y == c).nonzero()[0])
            #     prevs.append(prev)
            # most_ex = sorted(prevs)[-1] # get most prevelant class number of examples
            # self.class_weights = most_ex / torch.Tensor(prevs)

            # === 2 ===
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y)
        return torch.tensor(class_weights, dtype=torch.float)
    else:
        return None

def to_device(dataset, device):
    if type(dataset[0]) == list:
        for i in range(len(dataset)):
            dataset[i][0] = dataset[i][0].to(device) # graph features
            dataset[i][1] = dataset[i][1].to(device) # graph labels

    else:
        for i in range(len(dataset)):
            dataset[i] = dataset[i].to(device) # pyg Data object
            
#function that takes in data and transforms/gets features from networkx graphs
data_list = []
first_part = []

def avg(pr):
    values = [v for _, v in pr.items()]
    avg = sum(values)/len(values)
    return avg

def stnd_dev(pr):
    values = [v for _, v in pr.items()]
    stnd_dev = np.std(values)
    return stnd_dev

def count_triangles(graph):
    undirected = graph.to_undirected()
    
    triangles = len(nx.triangles(undirected))
    all_cliques= nx.enumerate_all_cliques(undirected)
    triad_cycles= [x for x in all_cliques if len(x)==3]
    
    if triangles == 0:
        return 0
    if len(triad_cycles) == 0:
        return 0
    
    ratio = len(triad_cycles)/triangles
    return(ratio)

def runInParallel(*fns):
    proc = []
    for fn in fns:
        p = Process(target=fn)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()
    
def data_manip(wico):  
    
    for data in wico:
        graph = to_networkx(data)
    
        #source nodes
        source = [x for x in graph.nodes() if graph.out_degree(x)==1 and graph.in_degree(x)==0]
        #target nodes
        sink = [x for x in graph.nodes() if graph.out_degree(x)==0 and graph.in_degree(x)==1]
        #connected components
        largest = max(nx.strongly_connected_components(graph), key=len)
        
    
        #page rank features
        pr = nx.pagerank(graph, alpha=0.85)
        avg_pr = avg(pr)
        stnd_pr = stnd_dev(pr)
        #triangles = count_triangles(graph)
        
        #graph feature list
        data_list.append([torch.Tensor(((nx.number_of_nodes(graph), nx.number_of_edges(graph), nx.average_clustering(graph), 
                                         len(largest), len(source), len(sink), max(pr), min(pr), avg_pr, stnd_pr))), torch.Tensor((data.y, 1 ,1))]) 
    return(data_list)
    