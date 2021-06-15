import time
import dgl
import torch
from torch.utils.data import Dataset
import random as rd
from ogb.graphproppred import DglGraphPropPredDataset
from ogb.graphproppred import Evaluator


from scipy import sparse as sp
import numpy as np
import itertools
import torch.utils.data

##### MODIFIED CODE HERE
from utils_subgraph_encoding import prepare_dataset
from utils_one_hot_encoding import encode
##### MODIFIED CODE HERE


def positional_encoding(g, pos_enc_dim, norm):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    if norm == 'none':
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1), dtype=float)
        L = N * sp.eye(g.number_of_nodes()) - A
    elif norm == 'sym':
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(g.number_of_nodes()) - N * A * N
    elif norm == 'walk':
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1., dtype=float)
        L = sp.eye(g.number_of_nodes()) - N * A

    # # Eigenvectors with numpy
    # EigVal, EigVec = np.linalg.eig(L.toarray())
    # idx = EigVal.argsort() # increasing order
    # EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float()

    # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim, which='SR', tol=1e-2)
    EigVec = EigVec[:, EigVal.argsort()]  # increasing order
    scalar_field = torch.from_numpy(np.real(EigVec[:, :pos_enc_dim])).float()
    g.ndata['eig'] = scalar_field if 'eig' not in g.ndata else torch.cat((g.ndata['eig'], scalar_field), dim=1)

    return g


class HIVDGL(torch.utils.data.Dataset):
    def __init__(self, data, split, norm='norm', pos_enc_dim=0,  directions='eig', **subgraph_params):
        self.split = split
        
        ##### MODIFIED CODE HERE
        self.data = [data[split_ind] for split_ind in self.split]
        ##### MODIFIED CODE HERE
        
        self.graph_lists = []
        self.graph_labels = []
        for g in self.data:
            if g[0].number_of_nodes() > 5:
                self.graph_lists.append(g[0])
                self.graph_labels.append(g[1])
        self.n_samples = len(self.graph_lists)
        
        new_graph_lists = []
        for g in self.graph_lists:
            for direction in directions:
                if direction == 'eig':
                    g = positional_encoding(g, 4, norm=norm)
                    if pos_enc_dim > 0:
                        g = self._add_positional_encodings(g, pos_enc_dim)
                elif direction == 'subgraphs':
                    g = self.get_subgraphs(g, subgraph_params['id_scope'])
                elif direction == 'edge_feat':
                    g = self.get_edge_feat(g)
                else:
                    raise NotImplementedError('direction {} is not currently supported.'.format(direction))
            if 'eig' in g.ndata:
                g.ndata['eig'] = g.ndata['eig'].float()
            if 'eig' in g.edata:
                g.edata['eig'] = g.edata['eig'].float()
            new_graph_lists.append(g)
        self.graph_lists = new_graph_lists

         
    def get_subgraphs(self, g, id_scope):
        if id_scope=='global':
            scalar_field = g.ndata['subgraph_counts']
            g.ndata['eig'] = scalar_field if 'eig' not in g.ndata else torch.cat((g.ndata['eig'], scalar_field), dim=1)
        else:
            vector_field = g.edata['subgraph_counts']
            #import pdb;pdb.set_trace()
            g.edata['eig'] = vector_field if 'eig' not in g.edata else torch.cat((g.edata['eig'], vector_field), dim=1)
        return g
            
    def get_edge_feat(self, g):
        vector_field = g.edata['feat']
        g.edata['eig'] = vector_field if 'eig' not in g.edata else torch.cat((g.edata['eig'], vector_field), dim=1)
        return g
            

    def _add_positional_encodings(self, g, pos_enc_dim):
        g.ndata['pos_enc'] = g.ndata['eig'][:,1:pos_enc_dim+1]
        return g

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]


class HIVDataset(Dataset):
    def __init__(self,
                 name, 
                 pos_enc_dim=0, 
                 norm='none',
                 path='dataset/ogbg-molhiv',
                 directions=['subgraphs'],
                 verbose=True,
                 **subgraph_params):
        start = time.time()
        if verbose:
            print("[I] Loading dataset %s..." % (name))
        self.name = name
        
        ##### MODIFIED CODE HERE
        if 'subgraphs' in directions:
            self.dataset, self.split_idx = prepare_dataset(path, 
                                           name,
                                           **subgraph_params)
            print("One hot encoding substructure counts... ", end='')
            self.dataset, self.d_id = encode(self.dataset, subgraph_params['id_encoding'])
        else:
            self.dataset = DglGraphPropPredDataset(name = name, root=path)
            self.split_idx = self.dataset.get_idx_split()
            self.d_id = None
            
        self.train = HIVDGL(self.dataset, self.split_idx['train'],
                            norm=norm, pos_enc_dim=pos_enc_dim, directions=directions, **subgraph_params)
        self.val = HIVDGL(self.dataset, self.split_idx['valid'], 
                          norm=norm, pos_enc_dim=pos_enc_dim, directions=directions, **subgraph_params)
        self.test = HIVDGL(self.dataset, self.split_idx['test'], 
                           norm=norm, pos_enc_dim=pos_enc_dim, directions=directions, **subgraph_params)
        ##### MODIFIED CODE HERE
        #import pdb;pdb.set_trace()
        self.evaluator = Evaluator(name='ogbg-molhiv')

        if verbose:
            print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
            print("[I] Finished loading.")
            print("[I] Data load time: {:.4f}s".format(time.time() - start))

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.cat(labels).long()
        tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        snorm_n = torch.cat(tab_snorm_n).sqrt()
        tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels, snorm_n, snorm_e

    def _add_self_loops(self):
        # function for adding self loops
        # this function will be called only if self_loop flag is True

        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]