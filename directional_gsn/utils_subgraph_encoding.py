import os
from utils_graph_processing import subgraph_isomorphism_edge_counts, subgraph_isomorphism_vertex_counts, induced_edge_automorphism_orbits, edge_automorphism_orbits, automorphism_orbits
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import glob
import re
import types
from tqdm import tqdm

from ogb.graphproppred import DglGraphPropPredDataset


def get_custom_edge_list(ks, substructure_type=None, filename=None):
    '''
        Instantiates a list of `edge_list`s representing substructures
        of type `substructure_type` with sizes specified by `ks`.
    ''' 
    if substructure_type is None and filename is None:
        raise ValueError('You must specify either a type or a filename where to read substructures from.')
    edge_lists = []
    for k in ks:
        if substructure_type is not None:
            graphs_nx = getattr(nx, substructure_type)(k)
        else:
            graphs_nx = nx.read_graph6(os.path.join(filename, 'graph{}c.g6'.format(k)))
        if isinstance(graphs_nx, list) or isinstance(graphs_nx, types.GeneratorType):
            edge_lists += [list(graph_nx.edges) for graph_nx in graphs_nx]
        else:
            edge_lists.append(list(graphs_nx.edges))
    return edge_lists

def prepare_subgraph_params(args):
    
    ###### choose the function that computes the automorphism group and the orbits #######
    if args['edge_automorphism'] == 'induced':
        args['automorphism_fn'] = induced_edge_automorphism_orbits if  args['id_scope'] == 'local' else automorphism_orbits
    elif args['edge_automorphism'] == 'line_graph':
        args['automorphism_fn'] = edge_automorphism_orbits if  args['id_scope'] == 'local' else automorphism_orbits
    else:
        raise NotImplementedError

    ###### choose the function that computes the subgraph isomorphisms #######
    args['count_fn'] = subgraph_isomorphism_edge_counts if args['id_scope'] == 'local'else subgraph_isomorphism_vertex_counts

    ###### choose the substructures: usually loaded from networkx,
    ###### except for 'all_simple_graphs' where they need to be precomputed,
    ###### or when a custom edge list is provided in the input by the user
    if args['id_type'] in ['cycle_graph',
                           'path_graph',
                           'complete_graph',
                           'binomial_tree',
                           'star_graph',
                           'nonisomorphic_trees']:
        args['k'] = args['k'][0]
        k_max = args['k']
        k_min = 2 if args['id_type'] == 'star_graph' else 3
        args['edge_list'] = get_custom_edge_list(list(range(k_min, k_max + 1)), args['id_type'])         

    elif args['id_type'] in ['cycle_graph_chosen_k',
                             'path_graph_chosen_k', 
                             'complete_graph_chosen_k',
                             'binomial_tree_chosen_k',
                             'star_graph_chosen_k',
                             'nonisomorphic_trees_chosen_k']:
        args['edge_list'] = get_custom_edge_list(args['k'], args['id_type'].replace('_chosen_k',''))
        
    elif args['id_type'] in ['all_simple_graphs']:
        args['k'] = args['k'][0]
        k_max = args['k']
        k_min = 3
        filename = os.path.join(args['root_folder'], 'all_simple_graphs')
        args['edge_list'] = get_custom_edge_list(list(range(k_min, k_max + 1)), filename=filename)
        
    elif args['id_type'] in ['all_simple_graphs_chosen_k']:
        filename = os.path.join(args['root_folder'], 'all_simple_graphs')
        args['edge_list'] = get_custom_edge_list(args['k'], filename=filename)
        
    elif args['id_type'] in ['diamond_graph']:
        args['k'] = None
        graph_nx = nx.diamond_graph()
        args['edge_list'] = [list(graph_nx.edges)]

    elif args['id_type'] == 'custom':
        assert args['edge_list'] is not None, "Custom edge list must be provided."

    else:
        raise NotImplementedError("Identifiers {} are not currently supported.".format(args['id_type']))
    
    return args



def prepare_dataset(path, 
                    name, 
                    **subgraph_params):
    
    id_scope = subgraph_params['id_scope']
    id_type = subgraph_params['id_type']
    k = subgraph_params['k']

    data_folder = os.path.join(path, 'processed', id_scope)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if id_type != 'custom':
        if subgraph_params['induced']:
            if subgraph_params['directed_orbits'] and id_scope == 'local':
                data_file = os.path.join(data_folder, '{}_induced_directed_orbits_{}.pt'.format(id_type, k))
            else:
                data_file = os.path.join(data_folder, '{}_induced_{}.pt'.format(id_type, k))
        else:
            if subgraph_params['directed_orbits'] and id_scope == 'local':
                data_file = os.path.join(data_folder, '{}_directed_orbits_{}.pt'.format(id_type, k))
            else:
                data_file = os.path.join(data_folder, '{}_{}.pt'.format(id_type, k))
        maybe_load = True
    else:
        data_file = None  # we don't save custom substructure counts
        maybe_load = False
        
    loaded = False

    # try to load, possibly downgrading
    if maybe_load:

        if os.path.exists(data_file):  # load
            graphs_dgl, orbit_partition_sizes, split_idx = load_dataset(data_file)
            loaded = True

        else:  # try downgrading. Currently works only when for each k there is only one substructure in the family
            if id_type in ['cycle_graph',
                           'path_graph',
                           'complete_graph',
                           'binomial_tree',
                           'star_graph']:
                k_min = 2 if id_type == 'star_graph' else 3
                succeded, graphs_dgl, orbit_partition_sizes, split_idx = try_downgrading(data_folder, 
                                                                              id_type, 
                                                                              subgraph_params['induced'], 
                                                                              subgraph_params['directed_orbits'] 
                                                                              and id_scope == 'local',
                                                                              k, 
                                                                              k_min,
                                                                              id_scope)
                if succeded:  # save the dataset
                    print("Saving dataset to {}".format(data_file))
                    torch.save((graphs_dgl, orbit_partition_sizes, split_idx), data_file)
                    loaded = True

    if not loaded:

        graphs_dgl, orbit_partition_sizes, split_idx = generate_dataset(path,
                                                                        name,
                                                                        **subgraph_params)
        if data_file is not None:
            print("Saving dataset to {}".format(data_file))
            torch.save((graphs_dgl, orbit_partition_sizes, split_idx), data_file)

    return graphs_dgl, split_idx

def load_dataset(data_file):
    '''
        Loads dataset from `data_file`.
    '''
    print("Loading dataset from {}".format(data_file))
    return torch.load(data_file)


def try_downgrading(data_folder, id_type, induced, directed_orbits, k, k_min, id_scope):
    '''
        Extracts the substructures of size up to the `k`, if a collection of substructures
        with size larger than k has already been computed.
    '''
    found_data_filename, k_found = find_id_filename(data_folder, id_type, induced, directed_orbits, k)
    if found_data_filename is not None:
        graphs_dgl, orbit_partition_sizes, split_idx = load_dataset(found_data_filename)
        print("Downgrading k from dataset {}...".format(found_data_filename))
        graphs_dgl, orbit_partition_sizes = downgrade_k(graphs_dgl, k, orbit_partition_sizes, k_min, id_scope)
        return True, graphs_dgl, orbit_partition_sizes, split_idx
    else:
        return False, None, None, None


def find_id_filename(data_folder, id_type, induced, directed_orbits, k):
    '''
        Looks for existing precomputed datasets in `data_folder` with counts for substructure 
        `id_type` larger `k`.
    '''
    if induced:
        if directed_orbits:
            pattern = os.path.join(data_folder, '{}_induced_directed_orbits_[0-9]*.pt'.format(id_type))
        else:
            pattern = os.path.join(data_folder, '{}_induced_[0-9]*.pt'.format(id_type))
    else:
        if directed_orbits:
            pattern = os.path.join(data_folder, '{}_directed_orbits_[0-9]*.pt'.format(id_type))
        else:
            pattern = os.path.join(data_folder, '{}_[0-9]*.pt'.format(id_type))
    filenames = glob.glob(pattern)
    for name in filenames:
        k_found = int(re.findall(r'\d+', name)[-1])
        if k_found >= k:
            return name, k_found
    return None, None

def downgrade_k(dataset, k, orbit_partition_sizes, k_min, id_scope):
    '''
        Donwgrades `dataset` by keeping only the orbits of the requested substructures.
    '''
    feature_vector_size = sum(orbit_partition_sizes[0:k-k_min+1])
    graphs_dgl = list()
    for datapoint in dataset:
        g, label = datapoint
        if id_scope == 'global':
            g.ndata['subgraph_counts'] = g.ndata['subgraph_counts'][:, 0:feature_vector_size]
        else:
            g.edata['subgraph_counts'] = g.edata['subgraph_counts'][:, 0:feature_vector_size]
        graphs_dgl.append((g, label))
        
    return graphs_dgl, orbit_partition_sizes[0:k-k_min+1]



def generate_dataset(path, 
                     name,
                     **subgraph_params):

    id_scope = subgraph_params['id_scope']
    count_fn = subgraph_params['count_fn']
    automorphism_fn = subgraph_params['automorphism_fn']
    multiprocessing = subgraph_params['multiprocessing']
    num_processes = subgraph_params['num_processes']

    ### compute the orbits of earch substructure in the list, as well as the vertex automorphism count
    
    subgraph_dicts = []
    orbit_partition_sizes = []
    if 'edge_list' not in subgraph_params:
        raise ValueError('Edge list not provided.')
    for edge_list in subgraph_params['edge_list']:
        subgraph, orbit_partition, orbit_membership, aut_count = \
                                            automorphism_fn(edge_list=edge_list,
                                                           directed=subgraph_params['directed'],
                                                           directed_orbits=subgraph_params['directed_orbits'])
        subgraph_dicts.append({'subgraph':subgraph, 'orbit_partition': orbit_partition, 
                               'orbit_membership': orbit_membership, 'aut_count': aut_count})
        orbit_partition_sizes.append(len(orbit_partition))
        
    ### load and preprocess dataset
    dataset = DglGraphPropPredDataset(name=name, root=path)
    split_idx = dataset.get_idx_split()
        
     ### parallel computation of subgraph isomoprhisms & creation of data structure
        
    if multiprocessing:     
        print("Preparing dataset in parallel...")
        start = time.time()
        from joblib import delayed, Parallel
        graphs_dgl = Parallel(n_jobs=num_processes, verbose=10)(delayed(_prepare)(g,
                                                                                  subgraph_dicts, 
                                                                                  subgraph_params,
                                                                                  count_fn,
                                                                                  id_scope) for g in dataset)
        print('Done ({:.2f} secs).'.format(time.time() - start))
        
    ### single-threaded computation of subgraph isomoprhisms & creation of data structure
    else:
        graphs_dgl = list()
        for i, datapoint in tqdm(enumerate(dataset)):
            g, label = datapoint
            g = _prepare(g, 
                         subgraph_dicts,
                         subgraph_params, 
                         count_fn, 
                         id_scope)
            graphs_dgl.append((g, label))

    return graphs_dgl, orbit_partition_sizes, split_idx


# ------------------------------------------------------------------------
        
def _prepare(g, subgraph_dicts, subgraph_params, count_fn, id_scope):

    edge_index = torch.stack(g.edges())
    num_nodes = g.number_of_nodes()
    
    identifiers = None
    for subgraph_dict in subgraph_dicts:
        kwargs = {'subgraph_dict': subgraph_dict, 
                  'induced': subgraph_params['induced'],
                  'num_nodes': num_nodes,
                  'directed': subgraph_params['directed']}
        counts = count_fn(edge_index, **kwargs)
        identifiers = counts if identifiers is None else torch.cat((identifiers, counts),1) 
        
    if id_scope == 'global':
        g.ndata['subgraph_counts'] = identifiers.long()
    else:
        g.edata['subgraph_counts'] = identifiers.long()
    
    return g


# --------------------------------------------------------------------------------------   

