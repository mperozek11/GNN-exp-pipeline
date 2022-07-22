import torch
import numpy as np
import networkx as nx
from multiprocessing import Process
from torch_geometric.utils import to_networkx


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
        data_list.append([torch.Tensor(((
                nx.number_of_nodes(graph),
                nx.number_of_edges(graph),
                nx.average_clustering(graph),
                len(largest),
                len(source),
                len(sink),
                max(pr),
                min(pr),
                avg_pr,
                stnd_pr))), torch.Tensor((data.y, 1 ,1))]) 
    return(data_list)
    