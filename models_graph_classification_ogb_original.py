import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_filters.GSN_edge_sparse_ogb import GSN_edge_sparse_ogb
from graph_filters.MPNN_edge_sparse_ogb import MPNN_edge_sparse_ogb

from models_misc import mlp, choose_activation
from utils_graph_learning import global_add_pool_sparse, global_mean_pool_sparse, DiscreteEmbedding

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder



class GNN_OGB(torch.nn.Module):

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

        super(GNN_OGB, self).__init__()

        seed = kwargs['seed']

        #-------------- Initializations

        self.model_name = kwargs['model_name']
        self.readout = kwargs['readout'] if kwargs['readout'] is not None else 'sum'
        self.dropout_features = kwargs['dropout_features']
        self.bn = kwargs['bn']
        self.final_projection = kwargs['final_projection']
        self.residual = kwargs['residual']
        self.inject_ids = kwargs['inject_ids']
        self.vn = kwargs['vn']

        id_scope = kwargs['id_scope']
        d_msg = kwargs['d_msg']
        d_out = kwargs['d_out']
        d_h = kwargs['d_h']
        aggr = kwargs['aggr'] if kwargs['aggr'] is not None else 'add'
        flow = kwargs['flow'] if kwargs['flow'] is not None else 'target_to_source'
        msg_kind = kwargs['msg_kind'] if kwargs['msg_kind'] is not None else 'general'
        train_eps = kwargs['train_eps'] if kwargs['train_eps'] is not None else [False for _ in range(len(d_out))]
        activation_mlp = kwargs['activation_mlp']
        bn_mlp = kwargs['bn_mlp']
        jk_mlp = kwargs['jk_mlp']               
        degree_embedding = kwargs['degree_embedding'] if kwargs['degree_as_tag'][0] else 'None'
        degree_as_tag = kwargs['degree_as_tag']
        retain_features = kwargs['retain_features']

        encoders_kwargs = {'seed':seed,
                           'activation_mlp': activation_mlp,
                           'bn_mlp': bn_mlp, 
                           'aggr': kwargs['multi_embedding_aggr'],
                           'features_scope': kwargs['features_scope']}


        #-------------- Input node embedding
        self.input_node_encoder = DiscreteEmbedding(kwargs['input_node_encoder'], 
                                                           in_features,
                                                           d_in_node_encoder,
                                                           kwargs['d_out_node_encoder'],
                                                           **encoders_kwargs)
        d_in = self.input_node_encoder.d_out


        #-------------- Virtual node embedding
        if self.vn:
            vn_encoder_kwargs = copy.deepcopy(encoders_kwargs)
            vn_encoder_kwargs['init'] = 'zeros'
            self.vn_encoder = DiscreteEmbedding(kwargs['input_vn_encoder'], 
                                                1,
                                                [1],
                                                kwargs['d_out_vn_encoder'],
                                                **vn_encoder_kwargs)
            d_in_vn = self.vn_encoder.d_out


        #-------------- Edge embedding (for each GNN layer)
        self.edge_encoder = []
        d_ef = []
        for i in range(len(d_out)):
            edge_encoder_layer = DiscreteEmbedding(kwargs['edge_encoder'], 
                                                    in_edge_features,
                                                    d_in_edge_encoder,
                                                    kwargs['d_out_edge_encoder'][i],
                                                    **encoders_kwargs)
            self.edge_encoder.append(edge_encoder_layer)
            d_ef.append(edge_encoder_layer.d_out)

        self.edge_encoder  = nn.ModuleList(self.edge_encoder)  

        # -------------- Identifier embedding (for each GNN layer)
        self.id_encoder = []
        d_id = []
        num_id_encoders = len(d_out) if kwargs['inject_ids'] else 1
        for i in range(num_id_encoders):
            id_encoder_layer = DiscreteEmbedding(kwargs['id_embedding'], 
                                                 len(d_in_id),
                                                 d_in_id,
                                                 kwargs['d_out_id_embedding'],
                                                 **encoders_kwargs)
            self.id_encoder.append(id_encoder_layer)
            d_id.append(id_encoder_layer.d_out)
            
        self.id_encoder  = nn.ModuleList(self.id_encoder) 

        #-------------- Degree embedding            
        self.degree_encoder = DiscreteEmbedding(degree_embedding, 
                                                1,
                                                d_degree,
                                                kwargs['d_out_degree_embedding'],
                                                **encoders_kwargs)
        d_degree = self.degree_encoder.d_out


        #-------------- GNN layers w/ bn 
        self.conv = []
        self.batch_norms = []
        self.mlp_vn = []
        for i in range(len(d_out)):

            if i > 0 and self.vn:
                #-------------- vn msg function     
                mlp_vn_temp = mlp(d_in_vn, kwargs['d_out_vn'][i-1], d_h[i], seed, activation_mlp, bn_mlp)
                self.mlp_vn.append(mlp_vn_temp)
                d_in_vn= kwargs['d_out_vn'][i-1]

            kwargs_filter = {
                 'd_in': d_in,
                 'd_degree': d_degree,
                 'degree_as_tag': degree_as_tag[i],
                 'retain_features': retain_features[i],
                 'd_msg': d_msg[i],
                 'd_up': d_out[i],
                 'd_h': d_h[i],
                 'seed': seed,
                 'activation_name': activation_mlp,
                 'bn': bn_mlp,
                 'aggr': aggr,
                 'msg_kind': msg_kind,
                 'eps': 0,
                 'train_eps': train_eps[i],
                 'flow': flow,
                 'd_ef': d_ef[i],
                 'edge_embedding': kwargs['edge_encoder'],
                 'id_embedding': kwargs['id_embedding'],
                 'extend_dims': kwargs['extend_dims']}


            use_ids = ((i > 0 and kwargs['inject_ids']) or (i == 0)) and (self.model_name == 'GSN_edge_sparse_ogb')

            if use_ids:
                filter_fn = GSN_edge_sparse_ogb
                kwargs_filter['d_id'] = d_id[i] if self.inject_ids else d_id[0]
                kwargs_filter['id_scope'] = id_scope
            else:
                filter_fn = MPNN_edge_sparse_ogb
            self.conv.append(filter_fn(**kwargs_filter))

            bn_layer = nn.BatchNorm1d(d_out[i]) if self.bn[i] else None
            self.batch_norms.append(bn_layer)

            d_in = d_out[i]

        self.conv = nn.ModuleList(self.conv)
        self.batch_norms = nn.ModuleList(self.batch_norms)
        if kwargs['vn']:
            self.mlp_vn = nn.ModuleList(self.mlp_vn)


        #-------------- Readout 
        if self.readout == 'sum':
            self.global_pool = global_add_pool_sparse
        elif self.readout == 'mean': 
            self.global_pool = global_mean_pool_sparse
        else:
            raise ValueError("Invalid graph pooling type.")

        #-------------- Virtual node aggregation operator
        if self.vn:
            if kwargs['vn_pooling'] == 'sum':
                self.global_vn_pool = global_add_pool_sparse
            elif kwargs['vn_pooling'] == 'mean':
                self.global_vn_pool = global_mean_pool_sparse
            else:
                raise ValueError("Invalid graph virtual node pooling type.")

        self.lin_proj = nn.Linear(d_out[-1], out_features)


        #-------------- Activation fn (same across the network)

        self.activation = choose_activation(kwargs['activation'])

        return


    def forward(self, data, return_intermediate=False):

        #-------------- Code adopted from https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol.
        #-------------- Modified accordingly to allow for the existence of structural identifiers

        kwargs = {}
        kwargs['degrees'] = self.degree_encoder(data.degrees)

        #-------------- edge index, initial node features enmbedding, initial vn embedding
        edge_index = data.edge_index 
        if self.vn:
            vn_embedding = self.vn_encoder(torch.zeros(data.batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))                                                    
        x = self.input_node_encoder(data.x)   
        x_interm = [x]

        for i in range(0, len(self.conv)):

            #-------------- encode ids (different for each layer)
            kwargs['identifiers'] = self.id_encoder[i](data.identifiers) if self.inject_ids else self.id_encoder[0](data.identifiers)

            #-------------- edge features embedding (different for each layer)    
            if hasattr(data, 'edge_features'): 
                kwargs['edge_features'] = self.edge_encoder[i](data.edge_features)
            else:
                kwargs['edge_features'] = None

            if self.vn:
                x_interm[i] = x_interm[i] + vn_embedding[data.batch]

            x = self.conv[i](x_interm[i], edge_index, **kwargs)          

            x = self.batch_norms[i](x) if self.bn[i] else x

            if  i == len(self.conv) - 1:
                x = F.dropout(x, self.dropout_features[i], training = self.training)
            else:
                x = F.dropout(self.activation(x), self.dropout_features[i], training = self.training)

            if self.residual:
                x += x_interm[-1]

            x_interm.append(x)

            if i < len(self.conv) - 1 and self.vn:
                vn_embedding_temp = self.global_vn_pool(x_interm[i], data.batch) + vn_embedding
                vn_embedding = self.mlp_vn[i](vn_embedding_temp)

                if self.residual:
                    vn_embedding = vn_embedding + F.dropout(self.activation(vn_embedding), self.dropout_features[i], training = self.training)
                else:
                    vn_embedding = F.dropout(self.activation(vn_embedding), self.dropout_features[i], training = self.training)

        prediction = 0
        for i in range(0,len(self.conv)+1):
            if self.final_projection[i]:
                prediction += x_interm[i]

        x_global = self.global_pool(prediction, data.batch)

        return self.lin_proj(x_global)
