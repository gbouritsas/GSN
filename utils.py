import os

from utils_graph_processing import subgraph_isomorphism_edge_counts, subgraph_isomorphism_vertex_counts, induced_edge_automorphism_orbits, edge_automorphism_orbits, automorphism_orbits

from utils_ids import subgraph_counts2ids
from utils_data_gen import find_id_filename, downgrade_k, generate_dataset

from utils_graph_learning import multi_class_accuracy
import torch
import torch.nn as nn
import numpy as np

from torch_geometric.data import Data

import networkx as nx




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

    if args['id_type'] in ['cycle_graph', 'path_graph', 'complete_graph', 'binomial_tree']:
        args['k'] = args['k'][0]
        k_min = 3

    elif args['id_type'] in ['star_graph']:
        args['k'] = args['k'][0]
        k_min = 2            

    elif args['id_type'] in ['diamond_graph']:
        k_min = None
        args['k'] = None
        graph_nx = nx.diamond_graph()
        args['custom_edge_list'] = [list(graph_nx.edges)]

    elif args['id_type'] in ['all_simple_graphs']:
        k_min = 3
        args['k'] = args['k'][0]
        args['custom_edge_list'] = []
        for k in range(k_min, args['k']+1):
            graphs_nx = nx.read_graph6(os.path.join(args['root_folder'], 'synthetic', 'all_simple_graphs', 'graph'+str(k)+'c.g6'))
            if type(graphs_nx) is list:
                args['custom_edge_list'] += [list(graph_nx.edges) for graph_nx in graphs_nx]
            else:
                args['custom_edge_list'] += [list(graphs_nx.edges)]

    elif args['id_type'] == 'nonisomorphic_trees':
        args['k'] = args['k'][0]
        args['custom_edge_list'] = []
        for k in range(k_min, args['k']+1):
            graphs_nx = getattr(nx, args['id_type'])(k)
            args['custom_edge_list'] += [list(graph_nx.edges) for graph_nx in graphs_nx]

    elif args['id_type'] in ['cycle_path_graph', 'path_cycle_graph']:
        k_cycle, k_path = (args['k'][0], args['k'][1]) if args['id_type'] == 'cycle_path_graph' else (args['k'][1], args['k'][0])
        args['custom_edge_list'] = []
        for k in range(k_min, k_cycle+1):
            graph_nx = nx.cycle_graph(k)
            args['custom_edge_list'] += [list(graph_nx.edges)]
        for k in range(k_min, k_path+1):
            graph_nx = nx.path_graph(k)
            args['custom_edge_list'] += [list(graph_nx.edges)]

    elif args['id_type'] == 'custom':
        assert args['custom_edge_list'] is not None, "custom edge list should be provided"

    else:
        raise NotImplementedError("Identifiers {} are not currently supported.".format(args['id_type']))
    
    # Define if degree is going to be used as a feature and when (for each layer or only at initialization)
    if args['inject_degrees']:
        args['degree_as_tag'] = [args['degree_as_tag'] for _ in range(args['num_layers'])]
    else:
        args['degree_as_tag'] = [args['degree_as_tag']] + [False for _ in range(args['num_layers']-1)]
        
    # Define if existing features are going to be retained when the degree is used as a feature
    args['retain_features'] = [args['retain_features']] + [True for _ in range(args['num_layers']-1)]
        
    # Replicate d_out dimensions if the rest are not defined (msg function, mlp hidden dimension, encoders, etc.)
    # and repeat hyperparams for every layer
    if args['d_msg'] == -1:
        args['d_msg'] = [None for _ in range(args['num_layers'])]
    elif args['d_msg'] is None:
        args['d_msg'] = [args['d_out'] for _ in range(args['num_layers'])]
    else:
        args['d_msg'] = [args['d_msg'] for _ in range(args['num_layers'])]    
    
    if args['d_h'] is None:
        args['d_h'] = [[args['d_out']] for _ in range(args['num_layers'])]
    else:
        args['d_h'] = [[args['d_h']] for _ in range(args['num_layers'])]
        
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
    
    # Virtual node configuration for ogb datasets
    if args['vn']:
        if args['d_out_vn'] is None:
            args['d_out_vn'] = [args['d_out'] for _ in range(args['num_layers']-1)]
        else:
            args['d_out_vn'] = [args['d_out_vn'] for _ in range(args['num_layers']-1)]
    else:
        pass
    

    # Repeat hyperparams for every layer
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
        
    args['name'] = args['dataset_name']
        
    return args, extract_id_fn, count_fn, automorphism_fn, k_min, loss_fn, prediction_fn, perf_opt


def prepare_dataset(path, 
                    dataset, 
                    name, 
                    id_scope, 
                    id_type,
                    k, 
                    regression, 
                    k_min,
                    extract_ids_fn, 
                    count_fn,
                    automorphism_fn,
                    multiprocessing,
                    num_processes,
                    **subgraph_params):
    
    
    if dataset in ['bioinformatics', 'social', 'chemical', 'ogb', 'SR_graphs', 'KFI_graphs']:
        data_folder = os.path.join(path, 'processed', id_scope)
        if id_type != 'custom':
            if subgraph_params['induced']:
                data_file = os.path.join(data_folder, '{}_induced_{}.pt'.format(id_type, k))
            else:
                data_file = os.path.join(data_folder, '{}_{}.pt'.format(id_type, k))
        else:
            # TODO: take into account different custom edge_lists
            raise NotImplementedError("Custom edge list saving & loading needs to be fixed")
            
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
    else:
        raise NotImplementedError("Dataset kind {} is not currently supported.".format(dataset))


    if not os.path.exists(data_file):
        ###### Look for already precomputed datasets containing a superset of the current chosen substructures #######
        found_data_filename, graphs_ptg, num_classes, orbit_partition_sizes = try_downgrading(data_folder, id_type, subgraph_params['induced'], k, k_min)
        if found_data_filename is not None:
            print("Saving dataset to {}".format(data_file))
            torch.save((graphs_ptg, num_classes, orbit_partition_sizes), data_file)
        else:
            ###### Data preprocessing - subgraph isomorphism counting #######
            graphs_ptg, num_classes, num_node_type, num_edge_type, orbit_partition_sizes = \
                                                            generate_dataset(path, name, k, 
                                                                             extract_ids_fn, 
                                                                             count_fn,
                                                                             automorphism_fn,
                                                                             regression, k_min, 
                                                                             id_type,
                                                                             multiprocessing,
                                                                             num_processes, 
                                                                             **subgraph_params)
            print("Saving dataset to {}".format(data_file))
            torch.save((graphs_ptg, num_classes, orbit_partition_sizes), data_file)

            if num_node_type is not None:
                torch.save((num_node_type, num_edge_type), 
                           os.path.join(path, 'processed', 'num_feature_types.pt'))
    else:
        print("Loading dataset from {}".format(data_file))
        dataset_obj = torch.load(data_file)

        graphs_ptg = dataset_obj[0]
        num_classes = dataset_obj[1]
        # backwards compatibility
        if len(dataset_obj) == 3:
            orbit_partition_sizes = dataset_obj[2]
        else:
            orbit_partition_sizes = [1 for _ in range(k_min,k+1)]

    return graphs_ptg, num_classes, orbit_partition_sizes



def try_downgrading(data_folder, id_type, induced, k, k_min):
    
    ### When a collection of substructures with size k larger than the current chosen one has already been computed,
    ### this code extracts the substructures of size up to the current k, in order to avoid doing the precomputation again

    found_data_filename, k_found = find_id_filename(data_folder, id_type, induced, k)

    if found_data_filename is not None:
        dataset_obj = torch.load(found_data_filename)
        dataset = dataset_obj[0]
        num_classes = dataset_obj[1]
        # backwards compatibility
        if len(dataset_obj) == 3:
            orbit_partition_sizes = dataset_obj[2]
        else:
            orbit_partition_sizes = [1 for _ in range(k_min,k_found+1)]


        print("Downgrading k from dataset {}.".format(found_data_filename))
        graphs_ptg = downgrade_k(dataset, k, orbit_partition_sizes, k_min)

        return found_data_filename, graphs_ptg, num_classes, orbit_partition_sizes[0:k-k_min+1]
    else:
        return None, None, None, None
    
    