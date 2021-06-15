import torch
import sys

import numpy as np


def encode(graphs, id_encoding):
    '''
        Encodes categorical variables such as structural identifiers and degree features.
    '''
    id_scope = 'local' if 'subgraph_counts' in graphs[0][0].edata else 'global'
    if id_scope == 'local':
        encoder_ids, d_id = None, [1]*graphs[0][0].edata['subgraph_counts'].shape[1]
    else:
        encoder_ids, d_id = None, [1]*graphs[0][0].ndata['subgraph_counts'].shape[1]
    if id_encoding is not None:
        id_encoding_fn = getattr(sys.modules[__name__], id_encoding)
        if id_scope == 'local':
            ids = [graph[0].edata['subgraph_counts'] for graph in graphs]
        else:
            ids = [graph[0].ndata['subgraph_counts'] for graph in graphs]
        encoder_ids = id_encoding_fn(ids)
        encoded_ids = encoder_ids.fit(ids)
        d_id = encoder_ids.d
    
#     encoder_degrees, d_degree = None, []
#     if degree_encoding is not None:
#         degree_encoding_fn = getattr(sys.modules[__name__], degree_encoding)
#         degrees = [graph.degrees.unsqueeze(1) for graph in graphs]
#         encoder_degrees = degree_encoding_fn(degrees, **(kwargs['degree']))
#         encoded_degrees = encoder_degrees.fit(degrees)
#         d_degree = encoder_degrees.d
        
    for g, graph in enumerate(graphs):
        if id_encoding is not None:
            if id_scope == 'local':
                graph[0].edata['subgraph_counts'] = encoded_ids[g]
            else:
                graph[0].ndata['subgraph_counts'] = encoded_ids[g]
#         if degree_encoding is not None:
#             setattr(graph, 'degrees', encoded_degrees[g])
                            
    return graphs, d_id


class one_hot_unique:
    
    def __init__(self, tensor_list):
        tensor_list = torch.cat(tensor_list, 0)
        self.d = list()
        self.corrs = dict()
        for col in range(tensor_list.shape[1]):
            uniques, corrs = np.unique(tensor_list[:, col], return_inverse=True, axis=0)
            self.d.append(len(uniques))
            self.corrs[col] = corrs
        return       
            
    def fit(self, tensor_list):
        pointer = 0
        encoded_tensors = list()
        for tensor in tensor_list:
            n = tensor.shape[0]
            for col in range(tensor.shape[1]):
                translated = torch.LongTensor(self.corrs[col][pointer:pointer+n]).unsqueeze(1)
                encoded = torch.cat((encoded, translated), 1) if col > 0 else translated
            encoded_tensors.append(encoded)
            pointer += n
        return encoded_tensors
        

class one_hot_max:
    
    def __init__(self, tensor_list):
        tensor_list = torch.cat(tensor_list,0)
        self.d = [int(tensor_list[:,i].max()+1) for i in range(tensor_list.shape[1])]
    
    def fit(self, tensor_list):
        return tensor_list

