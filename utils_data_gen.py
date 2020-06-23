import glob
import re
import os
import torch
import torch_geometric as torch_geo
from torch_geometric.utils import sort_edge_index, remove_self_loops
from torch_geometric.data import Data
from utils_data_prep import load_data, load_zinc_data, load_ogb_data, load_g6_graphs
from utils_graph_processing import automorphism_orbits

import multiprocessing as mp
import time

from utils_misc import isnotebook
if isnotebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
    
def find_id_filename(data_folder, id_type, induced, k):
    
    ### look for existing precomputed datasets with larger k
    
    if induced:
        pattern = os.path.join(data_folder, '{}_induced_[0-9]*.pt'.format(id_type))
    else:
        pattern = os.path.join(data_folder, '{}_[0-9]*.pt'.format(id_type))
    filenames = glob.glob(pattern)
    for name in filenames:
        k_found = int(re.findall(r'\d+', name)[-1])
        if k_found >= k:
            return name, k_found
    return None, None

def downgrade_k(dataset, k, orbit_partition_sizes, k_min):
    
    ### donwgrade dataset by keeping only the orbits of the requested substructures
    
    feature_vector_size = sum(orbit_partition_sizes[0:k-k_min+1])
    
    graphs_ptg = list()
    for data in dataset:
        new_data = Data()
        for attr in data.__iter__():
            name, value = attr
            setattr(new_data, name, value)
        setattr(new_data, 'identifiers', data.identifiers[:, 0:feature_vector_size])
        graphs_ptg.append(new_data)

    return graphs_ptg


def generate_dataset(data_path, 
                     dataset_name,
                     k, 
                     extract_ids_fn, 
                     count_fn,
                     automorphism_fn, 
                     regression, 
                     k_min, 
                     id_type,  
                     multiprocessing = False, num_processes = 1, **subgraph_params):

    ### compute the orbits of earch substructure in the list, as well as the vertex automorphism count
    
    subgraph_dicts = None
    
    if id_type in ['custom', 'nonisomorphic_trees', 'diamond_graph', 'cycle_path_graph', 'path_cycle_graph', 'all_simple_graphs']:
        
        orbit_partition_sizes = []
        for edge_list in subgraph_params['edge_list']:
            subgraph, orbit_partition, orbit_membership, aut_count = \
                                                automorphism_fn(named_graph=None,
                                                                    k=None,
                                                                    edge_list=edge_list)
            subgraph_dicts.append({'subgraph':subgraph, 'orbit_partition': orbit_partition, 
                                   'orbit_membership': orbit_membership, 'aut_count': aut_count})
            orbit_partition_sizes.append(len(orbit_partition))
    else:
        subgraph_dicts = []
        orbit_partition_sizes = []
        for subgraph_size in range(k_min, k+1):
            subgraph, orbit_partition, orbit_membership, aut_count = \
                                                automorphism_fn(named_graph=id_type, 
                                                                    k=subgraph_size,
                                                                    edge_list=None)
            subgraph_dicts.append({'subgraph':subgraph, 'orbit_partition': orbit_partition, 
                                   'orbit_membership': orbit_membership, 'aut_count': aut_count})
            orbit_partition_sizes.append(len(orbit_partition))
            
    ### load and preprocess dataset
    
    if 'ogb' in data_path:
        graphs, num_classes = load_ogb_data(data_path, dataset_name, False)
        num_node_type, num_edge_type = None, None
    elif dataset_name == 'ZINC':
        graphs, num_classes, num_node_type, num_edge_type = load_zinc_data(data_path, dataset_name, False)
    elif dataset_name in ['sr16622', 'sr251256', 'sr261034', 'sr281264', 'sr291467', 'sr351668', 'sr351899', 'sr361446', 'sr401224', 'kfi_example']:
        graphs, num_classes = load_g6_graphs(data_path, dataset_name)
        num_node_type, num_edge_type = None, None
    else:
        graphs, num_classes = load_data(data_path, dataset_name, False)
        num_node_type, num_edge_type = None, None
        
     ### parallel computation of subgraph isomoprhisms & creation of data structure
        
    if multiprocessing:     
        print("Preparing dataset in parallel...")
        start = time.time()
        from joblib import delayed, Parallel
        graphs_ptg = Parallel(n_jobs=num_processes, verbose = 10)(delayed(_prepare)(graph, subgraph_dicts, 
                    subgraph_params, k, id_type, regression, 
                    dataset_name, extract_ids_fn, count_fn) for graph in graphs)
        print('Done ({:.2f} secs).'.format(time.time()-start))
        
    ### single-threaded computation of subgraph isomoprhisms & creation of data structure
    else:
        graphs_ptg = list()
        for i, data in tqdm(enumerate(graphs)):
            new_data = Data()
            setattr(new_data, 'edge_index', data.edge_mat)
            setattr(new_data, 'x', data.node_features)
            setattr(new_data, 'graph_size', data.node_features.shape[0])
            if new_data.edge_index.shape[1] == 0:
                setattr(new_data, 'degrees', torch.zeros((new_data.graph_size,)))
            else:
                setattr(new_data, 'degrees', torch_geo.utils.degree(new_data.edge_index[0]))

            if hasattr(data, 'edge_features'):
                setattr(new_data, 'edge_features', data.edge_features)

            if regression or dataset_name in {'ogbg-molpcba', 'ogbg-molhiv'}:
                setattr(new_data, 'y', torch.tensor(data.label).unsqueeze(0).float())
            else:
                setattr(new_data, 'y', torch.tensor(data.label).unsqueeze(0).long())
            if new_data.edge_index.shape[1] == 0 and count_fn.__name__ == 'subgraph_isomorphism_edge_counts':
                setattr(new_data, 'identifiers', torch.zeros((0, sum(orbit_partition_sizes))).long())
            else:
                new_data = extract_ids_fn(count_fn, new_data, subgraph_dicts, subgraph_params['induced'])


            graphs_ptg.append(new_data)

    return graphs_ptg, num_classes, num_node_type, num_edge_type, orbit_partition_sizes


# ------------------------------------------------------------------------
        
def _prepare(data, subgraph_dicts, subgraph_params, id_type, k, regression, dataset_name, ex_fn, cnt_fn):

    new_data = Data()
    setattr(new_data, 'edge_index', data.edge_mat)
    setattr(new_data, 'x', data.node_features)
    setattr(new_data, 'graph_size', data.node_features.shape[0])
    if new_data.edge_index.shape[1] == 0:
        setattr(new_data, 'degrees', torch.zeros((new_data.graph_size,)))
    else:
        setattr(new_data, 'degrees', torch_geo.utils.degree(new_data.edge_index[0]))

    if hasattr(data, 'edge_features'):
        setattr(new_data, 'edge_features', data.edge_features)    

    if regression or dataset_name in {'ogbg-molpcba', 'ogbg-molhiv'}:
        setattr(new_data, 'y', torch.tensor(data.label).unsqueeze(0).float())
    else:
        setattr(new_data, 'y', torch.tensor(data.label).unsqueeze(0).long())
    if new_data.edge_index.shape[1] == 0 and cnt_fn.__name__ == 'subgraph_isomorphism_edge_counts':
        setattr(new_data, 'identifiers', torch.zeros((0, sum(orbit_partition_sizes))).long())
    else:
        new_data = ex_fn(cnt_fn, new_data, subgraph_dicts, subgraph_params['induced'])
            

    return new_data

# --------------------------------------------------------------------------------------    