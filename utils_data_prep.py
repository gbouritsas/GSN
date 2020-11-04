import os
import csv
import pickle
from collections import namedtuple
import networkx as nx
import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedKFold
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.utils import to_undirected

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            Code obtained from here: https://github.com/weihua916/powerful-gnns
            
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0


def load_data(path, name, degree_as_tag):
    '''
        Code obtained from here: https://github.com/weihua916/powerful-gnns
    
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('%s/%s.txt' % (path, name), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    #add labels and edge_mat       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)


def load_zinc_data(path, name, degree_as_tag, num_atom_type=28, num_bond_type=4):
    
     ### splits and preprocessing according to https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/main_molecules_graph_regression.py
    
    assert name.upper() == 'ZINC'
    Graph = namedtuple('Graph', ['node_features', 'edge_mat', 'edge_features', 'label'])
    
    def _prepare(molecule):
    
        node_features = molecule['atom_type'].long()
        
        adj = molecule['bond_type']
        edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
        edge_idxs_in_adj = edge_list.split(1, dim=1)
        edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
        
        label = molecule['logP_SA_cycle_normalized']
        graph = Graph(node_features, edge_list.permute(1, 0), edge_features, label)

        return graph

    data = list()
    for split_name in ['train', 'val', 'test']:
        with open(os.path.join(path,'molecules','{}.pickle'.format(split_name)), "rb") as f:
            split_data = pickle.load(f)
        
        # loading the sampled indices from file ./zinc_molecules/<split>.index
        with open(os.path.join(path, 'indices', '{}.index'.format(split_name)), "r") as f:
            data_idx = [list(map(int, idx)) for idx in csv.reader(f)]
        
        split_data = [ split_data[i] for i in data_idx[0] ]
            
        for molecule in split_data:
            data.append(_prepare(molecule))

    return data, 1, num_atom_type, num_bond_type


def load_ogb_data(path, name, degree_as_tag):
    
     ### splits and preprocessing according to https://github.com/snap-stanford/ogb
        
    def add_zeros(data):
        data.x = torch.zeros(data.num_nodes, dtype=torch.long)
        return data
    
    transform = add_zeros if name == 'ogbg-ppa' else None
    print('Applying transform {} to dataset {}.'.format(transform, name))
    dataset = PygGraphPropPredDataset(name=name, root=path, transform=transform)
    Graph = namedtuple('Graph', ['node_features', 'edge_mat', 'edge_features', 'label'])
    graph_list = list()
    for datum in dataset:
        graph = Graph(datum.x, datum.edge_index, datum.edge_attr, datum.y)
        graph_list.append(graph)
    num_classes = dataset.num_classes if name == 'ogbg-ppa' else dataset.num_tasks
    return graph_list, num_classes


def load_g6_graphs(path, name):
    
     ### code used to load SR graphs obtained from here http://users.cecs.anu.edu.au/~bdm/data/graphs.html
     ### we don't split the data, because no training is performed (the network is used with random weights for the SR experiment)

    dataset = nx.read_graph6(os.path.join(path, name+'.g6'))
    Graph = namedtuple('Graph', ['node_features', 'edge_mat','label'])
    graph_list = list()
    for i,datum in enumerate(dataset):
        x = torch.ones(datum.number_of_nodes(),1)
        edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1,0))
        graph = Graph(x, edge_index, torch.tensor(i).long())
        graph_list.append(graph)
    num_classes = len(dataset)
    
    return graph_list, num_classes


def separate_data(graph_list, seed, fold_idx):
    
    ### Code obtained from here: https://github.com/weihua916/powerful-gnns
    
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    if hasattr(graph_list[0], 'label'):
        labels = [graph.label for graph in graph_list]
    elif hasattr(graph_list[0], 'y'):
        labels = [graph.y for graph in graph_list]
    else:
        raise NotImplementedError
        
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list

def separate_data_given_split(graph_list, path, fold_idx):
    
    ### Splits data based on pre-computed splits
    
    assert -1 <= fold_idx and fold_idx < 10, "Parameter fold_idx must be from -1 to 9, with -1 referring to the special model selection split."

    train_filename = os.path.join(path, '10fold_idx', 'train_idx-{}.txt'.format(fold_idx+1))
    test_filename = os.path.join(path, '10fold_idx', 'test_idx-{}.txt'.format(fold_idx+1))
    val_filename = os.path.join(path, '10fold_idx', 'val_idx-{}.txt'.format(fold_idx+1))
    train_idx = np.loadtxt(train_filename, dtype=int)
    test_idx = np.loadtxt(test_filename, dtype=int)
        
    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]
    val_graph_list = None                           
    
    if os.path.exists(val_filename):
        val_idx = np.loadtxt(val_filename, dtype=int)
        val_graph_list = [graph_list[i] for i in val_idx]

    return train_graph_list, test_graph_list, val_graph_list
