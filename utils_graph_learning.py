import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import degree

from models_misc import mlp
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 


def multi_class_accuracy(y_hat, y, reduction='sum'):
    
    pred = y_hat.max(1)[1]
    if reduction == 'sum':
        acc = pred.eq(y).sum().float()
    elif reduction == 'mean':
        acc = pred.eq(y).mean().float()
    else:
        raise NotImplementedError('Reduction {} not currently implemented.'.format(reduction))
    return acc


def global_add_pool_sparse(x, batch):
    
    #-------------- global sum pooling
    index = torch.stack([batch, torch.tensor(list(range(batch.shape[0])), device=x.device)], 0)  
    x_sparse = torch.sparse.FloatTensor(index, x, torch.Size([torch.max(batch)+1, x.shape[0], x.shape[1]]))
        
    return torch.sparse.sum(x_sparse, 1).to_dense()


def global_mean_pool_sparse(x, batch):
    
    #-------------- global average pooling
    index = torch.stack([batch, torch.tensor(list(range(batch.shape[0])), device=x.device)], 0)  
    x_sparse = torch.sparse.FloatTensor(index, x, torch.Size([torch.max(batch)+1, x.shape[0], x.shape[1]]))

    graph_sizes = degree(batch).float()
    graph_sizes[graph_sizes==0.0] = 1.0

    return torch.sparse.sum(x_sparse, 1).to_dense() / graph_sizes.unsqueeze(1)


class DiscreteEmbedding(torch.nn.Module):

    def __init__(self, encoder_name, d_in_features, d_in_encoder, d_out_encoder, **kwargs):

        super(DiscreteEmbedding, self).__init__()
        
        #-------------- various different embedding layers
        kwargs['init'] = None if 'init' not in kwargs else kwargs['init']
    
        self.encoder_name = encoder_name
        # d_in_features: input feature size (e.g. if already one hot encoded), 
        # d_in_encoder: number of unique values that will be encoded (size of embedding vocabulary)
        
        #-------------- fill embedding with zeros
        if encoder_name == 'zero_encoder':
            self.encoder = zero_encoder(d_out_encoder)
            d_out = d_out_encoder

        #-------------- linear pojection
        elif encoder_name == 'linear':
            self.encoder = nn.Linear(d_in_features,  d_out_encoder, bias=True)
            d_out = d_out_encoder

        #-------------- mlp
        elif encoder_name == 'mlp':
            self.encoder = mlp(d_in_features,
                               d_out_encoder,           
                               d_out_encoder,
                               kwargs['seed'],
                               kwargs['activation_mlp'],
                               kwargs['bn_mlp'])
            d_out = d_out_encoder

        #-------------- multi hot encoding of categorical data
        elif encoder_name == 'one_hot_encoder':
            self.encoder = one_hot_encoder(d_in_encoder)
            d_out = sum(d_in_encoder)

        #-------------- embedding of categorical data (linear projection without bias of one hot encodings)
        elif encoder_name == 'embedding':
            self.encoder = multi_embedding(d_in_encoder, d_out_encoder, kwargs['aggr'], kwargs['init'])
            if kwargs['aggr'] == 'concat':
                d_out = len(d_in_encoder) * d_out_encoder
            else:
                d_out = d_out_encoder
                
        #-------------- for ogb: multi hot encoding of node features
        elif encoder_name == 'atom_one_hot_encoder':
            full_atom_feature_dims = get_atom_feature_dims() if kwargs['features_scope'] == 'full' else get_atom_feature_dims()[:2]
            self.encoder = one_hot_encoder(full_atom_feature_dims)
            d_out = sum(full_atom_feature_dims)
        
        #-------------- for ogb: multi hot encoding of edge features
        elif encoder_name  == 'bond_one_hot_encoder':
            full_bond_feature_dims = get_bond_feature_dims() if kwargs['features_scope'] == 'full' else  get_bond_feature_dims()[:2]
            self.encoder  = one_hot_encoder(full_bond_feature_dims)
            d_out = sum(full_bond_feature_dims)
                
        #-------------- for ogb: embedding of node features
        elif encoder_name == 'atom_encoder':
            self.encoder  = AtomEncoder(d_out_encoder)
            d_out = d_out_encoder

        #-------------- for ogb: embedding of edge features
        elif encoder_name  == 'bond_encoder':
            self.encoder  = BondEncoder(emb_dim = d_out_encoder)
            d_out = d_out_encoder

        #-------------- no embedding, use as is
        elif encoder_name == 'None':
            self.encoder  = None
            d_out = d_in_features

        else:
            raise NotImplementedError('Encoder {} is not currently supported.'.format(encoder_name))
            
        self.d_out = d_out
        
        return

    def forward(self, x):
        
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        if self.encoder is not None:
            x = x.float() if self.encoder_name ==  'linear' or self.encoder_name == 'mlp' else x.long()
            return self.encoder(x)
        else:
            return x.float()    


class multi_embedding(torch.nn.Module):
    
    def __init__(self, d_in, d_out, aggr = 'concat', init=None):
        
        super(multi_embedding, self).__init__()
        
        #-------------- embedding of multiple categorical features. Summation or concatenation of the embeddings is allowed
        
        self.d_in = d_in
        self.aggr = aggr
        self.encoder = []
        for i in range(len(d_in)):
            self.encoder.append(nn.Embedding(d_in[i], d_out))
            if init == 'zeros':
                print('### INITIALIZING EMBEDDING TO ZERO ###')
                torch.nn.init.constant_(self.encoder[i].weight.data, 0)
            else:
                torch.nn.init.xavier_uniform_(self.encoder[-1].weight.data)
        self.encoder = nn.ModuleList(self.encoder)   
        
        return 

    def forward(self, tensor):
        
        for i in range(tensor.shape[1]):
            embedding_i = self.encoder[i](tensor[:,i])
            if self.aggr == 'concat':
                embedding = torch.cat((embedding, embedding_i),1) if i>0 else embedding_i
            elif self.aggr == 'sum':
                embedding = embedding + embedding_i if i>0 else embedding_i
            else:
                raise NotImplementedError('multi embedding aggregation {} is not currently supported.'.format(self.aggr))
        
        return embedding


class one_hot_encoder(torch.nn.Module):
    
    def __init__(self, d_in):
        
        super(one_hot_encoder, self).__init__()
        
        self.d_in = d_in
        
        return 

    def forward(self, tensor):
        
        for i in range(tensor.shape[1]):
            onehot_i = torch.zeros((tensor.shape[0], self.d_in[i]), device=tensor.device)
            onehot_i.scatter_(1, tensor[:,i:i+1], 1)
            onehot = torch.cat((onehot, onehot_i), 1) if i>0 else onehot_i
        
        return onehot
    
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.d_in)
    

class zero_encoder(torch.nn.Module):
    
    def __init__(self, d_out):
        
        super(zero_encoder, self).__init__()
        
        self.d_out = d_out
        
        return 

    def forward(self, tensor):
        
        return torch.zeros((tensor.shape[0], self.d_out), device=tensor.device)
    
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.d_out) 


class central_encoder(nn.Module):
    
    def __init__(self, nb_encoder, d_ef, extend=True):
        
        super(central_encoder, self).__init__()
        
        #-------------- For the neighbor aggregation: central node embedding
        #-------------- This is a way to create a dummy variable that represents self loops.
        #-------------- Useful when working with edge features or GSN-e
        #-------------- Two ways are allowed: extra dummy variable (one hot or embedding) or a vector filled with zeros
        
        self.extend = extend
        self.nb_encoder = nb_encoder
        
        if self.extend:
            print('##### EXTENDING EDGE FEATURE DIMENSIONS #####')
        
        if 'one_hot_encoder' in nb_encoder:
            if self.extend:
                self.encoder = DiscreteEmbedding('one_hot_encoder', 1, [d_ef+1], None)
                self.d_out = d_ef+1
            else:
                self.d_out = d_ef
        else:
            self.d_out = d_ef
            if self.extend:
                self.encoder = DiscreteEmbedding('embedding',  None, [1], d_ef, aggr='sum')
            else:
                pass
            
        return

    def forward(self, x_nb, num_nodes):
        
        if 'one_hot_encoder' in self.nb_encoder:
            if self.extend:
                zero_extension = torch.zeros((x_nb.shape[0], 1), device=x_nb.device)
                x_nb = torch.cat((zero_extension, x_nb), -1)
                x_central = torch.zeros((num_nodes,1), device=x_nb.device).long()
                x_central = self.encoder(x_central)
            else:
                x_central = torch.zeros((num_nodes, self.d_out), device=x_nb.device)
        else:
            if self.extend:
                x_central = torch.zeros((num_nodes,1), device=x_nb.device).long()
                x_central = self.encoder(x_central)
            else:
                x_central = torch.zeros((num_nodes, self.d_out), device=x_nb.device)
            
        return x_central, x_nb
    
    
