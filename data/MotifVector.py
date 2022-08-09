import os
import numpy as np
import pathlib
import torch

ROOT = pathlib.Path().resolve().as_posix()

#gets matrice id from motif file and stores in given list


def id_counter(file, id_list):
    f = open(file, "r")
    lines = f.readlines()
    
    for index, line in enumerate(lines):
        if index >= 31:
            word_list = []
            
            for word in line.split(" "):
                word_list.append(word)
        
            if word_list[0] != '' and word_list[0] != '\n':
                id_list.append(word_list[0])
        
            elif len(word_list) > 1:
                if word_list[1] != '':
                    id_list.append(word_list[1])
        return(id_list)       

#turns motif data into vector with features      
def vector_maker(file, id_dict, vector_list):
    f = open(file, "r")
    lines = f.readlines()

    features = []
    for i in range(len(id_dict)):
        features.append([0, 0, 0, 0])
        
    for index, line in enumerate(lines):
        if index >= 31:
            word_list = []
            for word in line.split(" "):
                if word != '':
                    word_list.append(word)
        
            if word_list[0] != '' and word_list[0] != '\n':
                if word_list[0] in id_dict:
                    minilist = [1]
                    if word_list[4] == 'undefined':
                        minilist.append(0)
                    try:
                        minilist.append(float(word_list[4]))
                    except ValueError as e:
                        minilist.append(0)
                    if word_list[5] != 'undefined':
                        minilist.append(float(word_list[5]))
                    if word_list[5] == 'undefined':
                        minilist.append(0)
                    
                    last = word_list[len(word_list)-1]
                    last = last[:len(last)-1]
                    if last == 'undefined':
                        minilist.append(0)
                    minilist.append(float(last))
                    features[id_dict[str(word_list[0])]] = minilist     
        
        vector_list.append(features)
    return(vector_list)

  
#-----------------pretty sure these don't go here?
consp_files = f'{ROOT}/data/motif_wico/5G_motif'
non_consp_files = f'{ROOT}/data/motif_wico/Non_motif'
other_files = f'{ROOT}/data/motif_wico/Other_motif'


if __name__ == "__main__":
    #-----------5G files (0)-----------------------------------
    id_list_5G = []
    for filename in os.listdir(consp_files):
        another_str = consp_files + '/' + filename
        id_counter(another_str, id_list_5G)

    x_5G = np.unique(np.array(id_list_5G))

    id_dict_5G = {}
    index = 0
    for idx in x_5G:
        id_dict_5G[idx] = index
        index +=1
        
    motif_list_5G = []
    for filename in os.listdir(consp_files):
        another_str = consp_files + '/' + filename
        vector_maker(another_str, id_dict_5G, motif_list_5G)

    #------ other files (1)--------------------------------------
    id_list_other = []
    for filename in os.listdir(other_files):
        another_str = other_files + '/' + filename
        id_counter(another_str, id_list_other)

    x_other = np.unique(np.array(id_list_other))

    id_dict_other = {}
    index_o = 0
    for idx in x_other:
        id_dict_other[idx] = index_o
        index_o +=1
        
    motif_list_other = []
    for filename in os.listdir(other_files):
        another_str = other_files + '/' + filename
        vector_maker(another_str, id_dict_other, motif_list_other)
        
    #------ non-consp files (2) --------------------------------------
    id_list_non = []
    for filename in os.listdir(non_consp_files):
        another_str = non_consp_files + '/' + filename
        id_counter(another_str, id_list_non)

    x_non = np.unique(np.array(id_list_non))

    id_dict_non = {}
    index_non = 0
    for idx in x_non:
        id_dict_non[idx] = index_non
        index_non +=1
        
    motif_list_non = []
    for filename in os.listdir(non_consp_files):
        another_str = non_consp_files + '/' + filename
        vector_maker(another_str, id_dict_non, motif_list_non)


    
    data_list = []
    #first tensor is vector list, second is subgraph second part is wheter the data is 5g non or other
    for graph in motif_list_5G:
        #tensor_graph = (torch.Tensor(graph)).long()
        tensor_graph = torch.flatten(torch.Tensor(graph))
        data_list.append([tensor_graph, torch.Tensor((0, 1, 1))]) 
        
    for graph in motif_list_other: 
        tensor_graph = torch.flatten(torch.Tensor(graph))
        data_list.append([tensor_graph, torch.Tensor((1, 1, 1))]) 

    for graph in motif_list_non: 
        tensor_graph = torch.flatten(torch.Tensor(graph))
        data_list.append([tensor_graph, torch.Tensor((2, 1, 1))])

    torch.save(data_list, f'{ROOT}/data/motif_wico/motif_wico.pt')

