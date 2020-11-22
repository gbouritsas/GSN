import os
from utils_graph_processing import subgraph_isomorphism_edge_counts, subgraph_isomorphism_vertex_counts, induced_edge_automorphism_orbits, edge_automorphism_orbits, automorphism_orbits
from utils_ids import subgraph_counts2ids
from utils_data_gen import generate_dataset
from utils_graph_learning import multi_class_accuracy
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.data import Data
import glob
import re
import types

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

def process_arguments(args):
    
    extract_id_fn = subgraph_counts2ids

    ###### choose the function that computes the automorphism group and the orbits #######
    if args['edge_automorphism'] == 'induced':
        automorphism_fn = induced_edge_automorphism_orbits if  args['id_scope'] == 'local' else automorphism_orbits
    elif args['edge_automorphism'] == 'line_graph':
        automorphism_fn = edge_automorphism_orbits if  args['id_scope'] == 'local' else automorphism_orbits
    else:
        raise NotImplementedError

    ###### choose the function that computes the subgraph isomorphisms #######
    count_fn = subgraph_isomorphism_edge_counts if args['id_scope'] == 'local'else subgraph_isomorphism_vertex_counts

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
        args['custom_edge_list'] = get_custom_edge_list(list(range(k_min, k_max + 1)), args['id_type'])         

    elif args['id_type'] in ['cycle_graph_chosen_k',
                             'path_graph_chosen_k', 
                             'complete_graph_chosen_k',
                             'binomial_tree_chosen_k',
                             'star_graph_chosen_k',
                             'nonisomorphic_trees_chosen_k']:
        args['custom_edge_list'] = get_custom_edge_list(args['k'], args['id_type'].replace('_chosen_k',''))
        
    elif args['id_type'] in ['all_simple_graphs']:
        args['k'] = args['k'][0]
        k_max = args['k']
        k_min = 3
        filename = os.path.join(args['root_folder'], 'all_simple_graphs')
        args['custom_edge_list'] = get_custom_edge_list(list(range(k_min, k_max + 1)), filename=filename)
        
    elif args['id_type'] in ['all_simple_graphs_chosen_k']:
        filename = os.path.join(args['root_folder'], 'all_simple_graphs')
        args['custom_edge_list'] = get_custom_edge_list(args['k'], filename=filename)
        
    elif args['id_type'] in ['diamond_graph']:
        args['k'] = None
        graph_nx = nx.diamond_graph()
        args['custom_edge_list'] = [list(graph_nx.edges)]

    elif args['id_type'] == 'custom':
        assert args['custom_edge_list'] is not None, "Custom edge list must be provided."

    else:
        raise NotImplementedError("Identifiers {} are not currently supported.".format(args['id_type']))
    
    # define if degree is going to be used as a feature and when (for each layer or only at initialization)
    if args['inject_degrees']:
        args['degree_as_tag'] = [args['degree_as_tag'] for _ in range(args['num_layers'])]
    else:
        args['degree_as_tag'] = [args['degree_as_tag']] + [False for _ in range(args['num_layers']-1)]
        
    # define if existing features are going to be retained when the degree is used as a feature
    args['retain_features'] = [args['retain_features']] + [True for _ in range(args['num_layers']-1)]
        
    # replicate d_out dimensions if the rest are not defined (msg function, mlp hidden dimension, encoders, etc.)
    # and repeat hyperparams for every layer
    if args['d_msg'] == -1:
        args['d_msg'] = [None for _ in range(args['num_layers'])]
    elif args['d_msg'] is None:
        args['d_msg'] = [args['d_out'] for _ in range(args['num_layers'])]
    else:
        args['d_msg'] = [args['d_msg'] for _ in range(args['num_layers'])]    
    
    if args['d_h'] is None:
        args['d_h'] = [[args['d_out']] * (args['num_mlp_layers'] - 1) for _ in range(args['num_layers'])]
    else:
        args['d_h'] = [[args['d_h']] * (args['num_mlp_layers'] - 1) for _ in range(args['num_layers'])]
        
    if args['d_out_edge_encoder'] is None:
        args['d_out_edge_encoder'] = [args['d_out'] for _ in range(args['num_layers'])]
    else:
        args['d_out_edge_encoder'] = [args['d_out_edge_encoder'] for _ in range(args['num_layers'])]
        
    if args['d_out_node_encoder'] is None:
        args['d_out_node_encoder'] = args['d_out']
    else:
        pass
    
    if args['d_out_id_embedding'] is None:
        args['d_out_id_embedding'] = args['d_out']
    else:
        pass
        
    if args['d_out_degree_embedding'] is None:
        args['d_out_degree_embedding'] = args['d_out']
    else:
        pass
    
    # virtual node configuration for ogb datasets
    if args['vn']:
        
        if args['d_out_vn_encoder'] is None:
            args['d_out_vn_encoder'] = args['d_out']
        else:
            pass
        
        if args['d_out_vn'] is None:
            args['d_out_vn'] = [args['d_out'] for _ in range(args['num_layers']-1)]
        else:
            args['d_out_vn'] = [args['d_out_vn'] for _ in range(args['num_layers']-1)]
    else:
        pass

    # repeat hyperparams for every layer
    args['d_out'] = [args['d_out'] for _ in range(args['num_layers'])]
    
    args['train_eps'] = [args['train_eps'] for _ in range(args['num_layers'])]
    
    if len(args['final_projection']) == 1:
        args['final_projection'] = [args['final_projection'][0] for _ in range(args['num_layers'])] + [True]
        
    args['bn'] = [args['bn'] for _ in range(args['num_layers'])]
    args['dropout_features'] = [args['dropout_features'] for _ in range(args['num_layers'])] + [args['dropout_features']]
    
    # loss function & metrics
    if args['loss_fn'] == 'CrossEntropyLoss':
        assert args['regression'] is False, "Can't use Cross-Entropy loss in regression."
        loss_fn = nn.CrossEntropyLoss()
    elif args['loss_fn'] == 'BCEWithLogitsLoss':
        assert args['regression'] is False, "Can't use binary Cross-Entropy loss in regression."
        loss_fn = nn.BCEWithLogitsLoss()
    elif args['loss_fn'] == 'MSELoss':
        loss_fn = nn.MSELoss()
    elif args['loss_fn'] == 'L1Loss':
        loss_fn = nn.L1Loss()
    else:
        raise NotImplementedError
        
    if args['prediction_fn'] == 'multi_class_accuracy':
        assert args['regression'] is False, "Can't use Classification Accuracy metric in regression."
        prediction_fn = multi_class_accuracy
    elif args['prediction_fn'] == 'MSELoss':
        prediction_fn = nn.MSELoss(reduction='sum')
    elif args['prediction_fn'] == 'L1Loss':
        prediction_fn = nn.L1Loss(reduction='sum')
    elif args['prediction_fn'] == 'None':
        prediction_fn = None
    else:
        raise NotImplementedError
        
    if args['regression']:
        perf_opt = np.argmin
    else:
        perf_opt = np.argmax 
        
    return args, extract_id_fn, count_fn, automorphism_fn, loss_fn, prediction_fn, perf_opt


def prepare_dataset(path, 
                    dataset, 
                    name, 
                    id_scope, 
                    id_type,
                    k, 
                    regression,
                    extract_ids_fn, 
                    count_fn,
                    automorphism_fn,
                    multiprocessing,
                    num_processes,
                    **subgraph_params):

    if dataset in ['bioinformatics', 'social', 'chemical', 'ogb', 'SR_graphs']:
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
    else:
        raise NotImplementedError("Dataset family {} is not currently supported.".format(dataset))

    # try to load, possibly downgrading
    if maybe_load:

        if os.path.exists(data_file):  # load
            graphs_ptg, num_classes, orbit_partition_sizes = load_dataset(data_file)
            loaded = True

        else:  # try downgrading. Currently works only when for each k there is only one substructure in the family
            if id_type in ['cycle_graph',
                           'path_graph',
                           'complete_graph',
                           'binomial_tree',
                           'star_graph']:
                k_min = 2 if id_type == 'star_graph' else 3
                succeded, graphs_ptg, num_classes, orbit_partition_sizes = try_downgrading(data_folder, 
                                                                                           id_type, 
                                                                                           subgraph_params['induced'], 
                                                                                           subgraph_params['directed_orbits'] 
                                                                                           and id_scope == 'local',
                                                                                           k, k_min)
                if succeded:  # save the dataset
                    print("Saving dataset to {}".format(data_file))
                    torch.save((graphs_ptg, num_classes, orbit_partition_sizes), data_file)
                    loaded = True

    if not loaded:

        graphs_ptg, num_classes, num_node_type, num_edge_type, orbit_partition_sizes = generate_dataset(path,
                                                                                                        name,
                                                                                                        k, 
                                                                                                        extract_ids_fn, 
                                                                                                        count_fn,
                                                                                                        automorphism_fn,
                                                                                                        regression, 
                                                                                                        id_type,
                                                                                                        multiprocessing,
                                                                                                        num_processes, 
                                                                                                        **subgraph_params)
        if data_file is not None:
            print("Saving dataset to {}".format(data_file))
            torch.save((graphs_ptg, num_classes, orbit_partition_sizes), data_file)

        if num_node_type is not None:
            torch.save((num_node_type, num_edge_type), os.path.join(path, 'processed', 'num_feature_types.pt'))

    return graphs_ptg, num_classes, orbit_partition_sizes


def load_dataset(data_file):
    '''
        Loads dataset from `data_file`.
    '''
    print("Loading dataset from {}".format(data_file))
    dataset_obj = torch.load(data_file)
    graphs_ptg = dataset_obj[0]
    num_classes = dataset_obj[1]
    orbit_partition_sizes = dataset_obj[2]

    return graphs_ptg, num_classes, orbit_partition_sizes


def try_downgrading(data_folder, id_type, induced, directed_orbits, k, k_min):
    '''
        Extracts the substructures of size up to the `k`, if a collection of substructures
        with size larger than k has already been computed.
    '''
    found_data_filename, k_found = find_id_filename(data_folder, id_type, induced, directed_orbits, k)
    if found_data_filename is not None:
        graphs_ptg, num_classes, orbit_partition_sizes = load_dataset(found_data_filename)
        print("Downgrading k from dataset {}...".format(found_data_filename))
        graphs_ptg, orbit_partition_sizes = downgrade_k(graphs_ptg, k, orbit_partition_sizes, k_min)
        return True, graphs_ptg, num_classes, orbit_partition_sizes
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

def downgrade_k(dataset, k, orbit_partition_sizes, k_min):
    '''
        Donwgrades `dataset` by keeping only the orbits of the requested substructures.
    '''
    feature_vector_size = sum(orbit_partition_sizes[0:k-k_min+1])
    graphs_ptg = list()
    for data in dataset:
        new_data = Data()
        for attr in data.__iter__():
            name, value = attr
            setattr(new_data, name, value)
        setattr(new_data, 'identifiers', data.identifiers[:, 0:feature_vector_size])
        graphs_ptg.append(new_data)
    return graphs_ptg, orbit_partition_sizes[0:k-k_min+1]
    
    