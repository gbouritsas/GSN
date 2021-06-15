EPS = 1e-5
import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregators import AGGREGATORS
from .layers import MLP, FCLayer
from .scalers import SCALERS
from dgl.nn.pytorch.glob import mean_nodes, sum_nodes

class DGNLayerSimple(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, graph_norm, batch_norm, aggregators, scalers, residual, avg_d,
                 posttrans_layers=1):
        super().__init__()

        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.aggregators = aggregators
        self.scalers = scalers
        self.batchnorm_h = nn.BatchNorm1d(out_dim)

        self.posttrans = MLP(in_size=(len(aggregators) * len(scalers)) * in_dim, hidden_size=out_dim,
                             out_size=out_dim, layers=posttrans_layers, mid_activation='relu', last_activation='none')
        self.avg_d = avg_d
        if in_dim != out_dim:
            self.residual = False

    def pretrans_edges(self, edges):
        if 'eig' in edges.src:
            vector_field = edges.src['eig'] - edges.dst['eig']
        else:
            vector_field = None
        if 'eig' in edges.data:
            vector_field = edges.data['eig'] if vector_field is None else torch.cat((vector_field, edges.data['eig']), dim=1)
        return {'e': edges.src['h'], 'vector_field': vector_field}

    def message_func(self, edges):
        return {'e': edges.data['e'], 'vector_field': edges.data['vector_field'].to('cuda' if torch.cuda.is_available() else 'cpu')}

    def reduce_func(self, nodes):
        h_in = nodes.data['h']
        h = nodes.mailbox['e']
        vector_field = nodes.mailbox['vector_field']
        D = h.shape[-2]

        # aggregators and scalers
        h = torch.cat([aggregate(h, vector_field, h_in) for aggregate in self.aggregators], dim=1)
        if len(self.scalers) > 1:
            h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=1)

        return {'h': h}

    def posttrans_nodes(self, nodes):
        return self.posttrans(nodes.data['h'])

    def forward(self, g, h, e, snorm_n):
        h_in = h
        g.ndata['h'] = h

        # pretransformation
        g.apply_edges(self.pretrans_edges)

        # aggregation
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']

        # posttransformation
        h = self.posttrans(h)

        # graph and batch normalization and residual
        if self.graph_norm:
            h = h * snorm_n
        if self.batch_norm:
            h = self.batchnorm_h(h)
        h = F.relu(h)
        if self.residual:
            h = h_in + h

        h = F.dropout(h, self.dropout, training=self.training)
        return h


class DGNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, graph_norm, batch_norm, aggregators, scalers, avg_d, type_net,
                 residual, towers=5, divide_input=True, edge_features=None, edge_dim=None, pretrans_layers=1,
                 posttrans_layers=1):
        super().__init__()

        # retrieve the aggregators and scalers functions
        aggregators = [AGGREGATORS[aggr] for aggr in aggregators.split()]
        scalers = [SCALERS[scale] for scale in scalers.split()]

        if type_net == 'simple':
            self.model = DGNLayerSimple(in_dim=in_dim, out_dim=out_dim, dropout=dropout, graph_norm=graph_norm,
                                        batch_norm=batch_norm, residual=residual, aggregators=aggregators,
                                        scalers=scalers, avg_d=avg_d, posttrans_layers=posttrans_layers)
        elif type_net == 'complex':
            self.model = DGNLayerComplex(in_dim=in_dim, out_dim=out_dim, dropout=dropout, graph_norm=graph_norm,
                                         batch_norm=batch_norm, aggregators=aggregators, residual=residual,
                                         scalers=scalers, avg_d=avg_d, edge_features=edge_features, edge_dim=edge_dim,
                                         pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers)
        elif type_net == 'towers':
            self.model = DGNLayerTower(in_dim=in_dim, out_dim=out_dim, aggregators=aggregators, scalers=scalers,
                                       avg_d=avg_d, dropout=dropout, graph_norm=graph_norm,
                                       batch_norm=batch_norm, towers=towers, pretrans_layers=pretrans_layers,
                                       posttrans_layers=posttrans_layers, divide_input=divide_input,
                                       residual=residual, edge_features=edge_features, edge_dim=edge_dim)
