import torch
import numpy as np
from utils_graph_processing import subgraph_isomorphism_vertex_counts, subgraph_isomorphism_edge_counts
from torch_geometric.utils import remove_self_loops


def subgraph_counts2ids(count_fn, data, subgraph_dicts, subgraph_params):
    
    #### Remove self loops and then assign the structural identifiers by computing subgraph isomorphisms ####
    
    if hasattr(data, 'edge_features'):
        edge_index, edge_features = remove_self_loops(data.edge_index, data.edge_features)
        setattr(data, 'edge_features', edge_features)
    else:
        edge_index = remove_self_loops(data.edge_index)[0]
             
    num_nodes = data.x.shape[0]
    identifiers = None
    for subgraph_dict in subgraph_dicts:
        kwargs = {'subgraph_dict': subgraph_dict, 
                  'induced': subgraph_params['induced'],
                  'num_nodes': num_nodes,
                  'directed': subgraph_params['directed']}
        counts = count_fn(edge_index, **kwargs)
        identifiers = counts if identifiers is None else torch.cat((identifiers, counts),1) 
    setattr(data, 'edge_index', edge_index)
    setattr(data, 'identifiers', identifiers.long())
    
    return data
