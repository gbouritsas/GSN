import torch
import torch.nn as nn
import torch.nn.functional as F
    
from graph_filters.GSN_sparse import GSN_sparse
from graph_filters.GSN_edge_sparse import GSN_edge_sparse
from graph_filters.MPNN_sparse import MPNN_sparse
from graph_filters.MPNN_edge_sparse import MPNN_edge_sparse

from models_misc import mlp, choose_activation
from utils_graph_learning import global_add_pool_sparse, global_mean_pool_sparse, DiscreteEmbedding

class MLPSubstructures(torch.nn.Module):

    def __init__(self,
                 in_features,
                 out_features, 
                 encoder_ids,
                 d_in_id,
                 in_edge_features=None, 
                 d_in_node_encoder=None,
                 d_in_edge_encoder=None,
                 encoder_degrees=None,
                 d_degree=None, 
                 **kwargs):

        super(MLPSubstructures, self).__init__()
        
        seed = kwargs['seed']
        

        #-------------- Initializations

        self.model_name = kwargs['model_name']
        self.readout = kwargs['readout'] if kwargs['readout'] is not None else 'sum'
        self.dropout_features = kwargs['dropout_features']
        self.bn = kwargs['bn']
        self.degree_as_tag = kwargs['degree_as_tag'] 
        self.retain_features = kwargs['retain_features']
        self.id_scope = kwargs['id_scope']
        
        d_out = kwargs['d_out']
        d_h = kwargs['d_h']
        activation_mlp = kwargs['activation_mlp']
        bn_mlp = kwargs['bn_mlp']
        jk_mlp = kwargs['jk_mlp']
        degree_embedding = kwargs['degree_embedding'] if kwargs['degree_as_tag'][0] else 'None'

        encoders_kwargs = {'seed':seed,
                           'activation_mlp': activation_mlp,
                           'bn_mlp': bn_mlp,
                           'aggr': kwargs['multi_embedding_aggr']}


        #-------------- Input node embedding
        self.input_node_encoder = DiscreteEmbedding(kwargs['input_node_encoder'], 
                                                    in_features,
                                                    d_in_node_encoder,
                                                    kwargs['d_out_node_encoder'],
                                                    **encoders_kwargs)
        d_in = self.input_node_encoder.d_out


        #-------------- Edge embedding (for each GNN layer)
        self.edge_encoder = DiscreteEmbedding(kwargs['edge_encoder'], 
                                               in_edge_features,
                                               d_in_edge_encoder,
                                               kwargs['d_out_edge_encoder'][0],
                                               **encoders_kwargs)
        d_ef = self.edge_encoder.d_out
          
        #-------------- Identifier embedding (for each GNN layer)
        self.id_encoder = DiscreteEmbedding(kwargs['id_embedding'], 
                                             len(d_in_id),
                                             d_in_id,
                                             kwargs['d_out_id_embedding'],
                                             **encoders_kwargs)
        d_id = self.id_encoder.d_out
            

        #-------------- Degree embedding            
        self.degree_encoder = DiscreteEmbedding(degree_embedding,
                                                1,
                                                d_degree,
                                                kwargs['d_out_degree_embedding'],
                                                **encoders_kwargs)
        d_degree = self.degree_encoder.d_out

        
        #-------------- edge-wise MLP w/ bn
        
        if self.degree_as_tag[0] and self.retain_features[0]==True:
            mlp_input_dim = 2*(d_in+d_degree)
        elif self.degree_as_tag[0] and self.retain_features[0]==False:
            mlp_input_dim = 2*d_degree
        else:
            mlp_input_dim = 2*d_in

        if self.id_scope == 'global':
            mlp_input_dim += 2*d_id
        else:
            mlp_input_dim += d_id

        if kwargs['edge_encoder'] != 'None':
            mlp_input_dim += d_ef

        filter_fn = mlp(mlp_input_dim, d_out[0], d_h[0], seed, activation_mlp, bn_mlp) 
        self.conv = filter_fn
        self.batch_norms = nn.BatchNorm1d(d_out[0]) if self.bn[0] else None

        if jk_mlp:
            final_jk_layer = mlp(d_out[0], out_features, d_h[0], seed, activation_mlp, bn_mlp)
        else:
            final_jk_layer = nn.Linear(d_out[0], out_features)
            
        self.lin_proj = final_jk_layer
        
        #-------------- Readout
        if self.readout == 'sum':
            self.global_pool = global_add_pool_sparse
        elif self.readout == 'mean': 
            self.global_pool = global_mean_pool_sparse
        else:
            raise ValueError("Invalid graph pooling type.")
                
                
        #-------------- Activation fn (same across the network)
        self.activation = choose_activation(kwargs['activation'])
                
        return
        

    def forward(self, data, print_flag=False, return_intermediate=False):
        
        degrees = self.degree_encoder(data.degrees)
        identifiers = self.id_encoder(data.identifiers)
            
        edge_index = data.edge_index                                  
        x = self.input_node_encoder(data.x)   
        
        if hasattr(data, 'edge_features'): 
            e = self.edge_encoder(data.edge_features) 
            
        if self.degree_as_tag[0] and self.retain_features[0]==True:
            x = torch.cat((x, degrees), 1)
        elif self.degree_as_tag[0] and self.retain_features[0]==False:
            x = degrees
        else:
            x = x
            
        x_i, x_j = x[edge_index[0]], x[edge_index[1]]    
        x_in = torch.cat((x_i,x_j),1)

        if self.id_scope == 'global':
            identifiers_i, identifiers_j = identifiers[edge_index[0]], identifiers[edge_index[1]]    
            x_in =  torch.cat((x_in, identifiers_i, identifiers_j), 1)
        else:
            x_in =  torch.cat((x_in, identifiers), 1)

        if hasattr(data, 'edge_features'):
            x_in =  torch.cat((x_in, e), 1)
            
        x = self.conv(x_in)
        if self.bn[0]:
            x = self.batch_norms(x)
        x = self.activation(x)
        
        x_global = self.global_pool(x, data.batch[edge_index[0]])

#         import pdb;pdb.set_trace()
        prediction = F.dropout(self.lin_proj(x_global), p=self.dropout_features[0], training=self.training)
                
        if return_intermediate:
            return prediction, None
        else:
            return prediction
    
