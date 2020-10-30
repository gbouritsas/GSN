import torch
import torch.nn as nn
from torch_geometric.utils import degree

from models_misc import mlp

class MPNN_sparse(nn.Module):
    
    def __init__(self,
                 d_in,
                 d_degree,
                 degree_as_tag,
                 retain_features,
                 d_msg,
                 d_up,
                 d_h,
                 seed,
                 activation_name,
                 bn,
                 aggr='add',
                 msg_kind='general',
                 eps=0,
                 train_eps=False,
                 flow='source_to_target',
                 **kwargs):

        super(MPNN_sparse, self).__init__()
        
        d_msg = d_in if d_msg is None else d_msg

        self.flow = flow
        self.aggr = aggr
        self.msg_kind = msg_kind
        
        self.degree_as_tag = degree_as_tag
        self.retain_features = retain_features
    
        if degree_as_tag:
            d_in = d_in + d_degree if retain_features else d_degree

        if msg_kind == 'gin':
            msg_input_dim = None
            self.initial_eps = eps
            if train_eps:
                self.eps = torch.nn.Parameter(torch.Tensor([eps]))
            else:
                self.register_buffer('eps', torch.Tensor([eps]))
            self.eps.data.fill_(self.initial_eps)
            self.msg_fn = None
            update_input_dim = d_in
            
        elif msg_kind == 'general':
            msg_input_dim = 2 * d_in
            self.msg_fn = mlp(
                msg_input_dim,
                d_msg,
                d_h,
                seed,
                activation_name,
                bn)
            update_input_dim = d_in + d_msg

        self.update_fn = mlp(
            update_input_dim,
            d_up,
            d_h,
            seed,
            activation_name,
            bn)

        return
    
    def forward(self, x, edge_index, **kwargs):

        # prepare input features
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        
        degrees = kwargs['degrees']
        degrees = degrees.unsqueeze(-1) if degrees.dim() == 1 else degrees
        if self.degree_as_tag:
            x = torch.cat([x, degrees], -1) if self.retain_features else degrees

        if self.msg_kind == 'gin':
            self_msg = x
            out = self.update_fn((1 + self.eps) * self_msg + self.propagate(edge_index=edge_index, x=x))   
        elif self.msg_kind == 'general':
            out = self.update_fn(torch.cat((x, self.propagate(edge_index=edge_index, x=x)), -1))

        return out
    
    def propagate(self, edge_index, x):
        
        select = 0 if self.flow == 'target_to_source' else 1 
        aggr_dim = 1 - select
        n_nodes = x.shape[0]
        
        edge_index_i, edge_index_j = edge_index[select, :], edge_index[1 - select, :]
        x_i, x_j = x[edge_index_i, :], x[edge_index_j, :]
        
        msgs = self.message(x_i, x_j)
        msgs = torch.sparse.FloatTensor(edge_index, msgs, torch.Size([n_nodes, n_nodes, msgs.shape[1]]))
        
        if self.aggr == 'add':
            message = torch.sparse.sum(msgs, aggr_dim).to_dense()
            
        elif self.aggr == 'mean':
            degrees = degree(edge_index[select])
            degrees[degrees==0.0] = 1.0
            message = torch.sparse.sum(msgs, aggr_index).to_dense()
            message = message / degrees.unsqueeze(1)
        
        else:
            raise NotImplementedError("Aggregation kind {} is not currently supported.".format(self.aggr))
        
        return message
    
    def message(self, x_i, x_j):
            
        if self.msg_kind == 'gin':
            msg_j = x_j
        elif self.msg_kind == 'general':
            msg_j = self.msg_fn(torch.cat((x_i, x_j), -1))
        else:
            raise NotImplementedError("Message kind {} is not currently supported.".format(self.msg_kind))
        return msg_j
    
    def __repr__(self):
        return '{}(msg_fn = {}, update_fn = {})'.format(self.__class__.__name__, self.msg_fn, self.update_fn)


