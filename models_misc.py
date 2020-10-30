import torch
import torch.nn as nn


def choose_activation(activation):
    if activation =='elu':
        return nn.ELU()
    elif activation =='relu':
        return nn.ReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'identity':
        return lambda x: x
    else:
        raise NotImplementedError
        
            
class mlp(torch.nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 d_k,
                 seed,
                 activation='elu',
                 batch_norm=False):
        super(mlp, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.d_k = d_k
        self.seed = seed
        self.activation_name = activation
        self.batch_norm = batch_norm

        self.fc = []
        self.bn = []

        d_in = [in_features]
        d_k = d_k + [out_features]
        for i in range(0, len(d_k)):
            self.fc.append(nn.Linear(d_in[i], d_k[i], bias=True))
            d_in = d_in + [d_k[i]]
            if self.batch_norm and i!=len(d_k)-1:
                self.bn.append(nn.BatchNorm1d((d_k[i])))

        self.fc = nn.ModuleList(self.fc)
        self.bn = nn.ModuleList(self.bn)
        self.activation = choose_activation(activation)


    def forward(self, x):
        for i in range(0, len(self.fc)-1):
            if self.batch_norm:
                x = self.activation(self.bn[i](self.fc[i](x)))
            else:
                x = self.activation(self.fc[i](x))
        x = self.fc[-1](x)
        return x
