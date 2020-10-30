import torch
import sys
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, MinMaxScaler, StandardScaler

import numpy as np


def encode(graphs, id_encoding, degree_encoding=None, **kwargs):
    '''
        Encodes categorical variables such as structural identifiers and degree features.
    '''
    encoder_ids, d_id = None, [1]*graphs[0].identifiers.shape[1]
    if id_encoding is not None:
        id_encoding_fn = getattr(sys.modules[__name__], id_encoding)
        ids = [graph.identifiers for graph in graphs]
        encoder_ids = id_encoding_fn(ids, **(kwargs['ids']))
        encoded_ids = encoder_ids.fit(ids)
        d_id = encoder_ids.d
    
    encoder_degrees, d_degree = None, []
    if degree_encoding is not None:
        degree_encoding_fn = getattr(sys.modules[__name__], degree_encoding)
        degrees = [graph.degrees.unsqueeze(1) for graph in graphs]
        encoder_degrees = degree_encoding_fn(degrees, **(kwargs['degree']))
        encoded_degrees = encoder_degrees.fit(degrees)
        d_degree = encoder_degrees.d
        
    for g, graph in enumerate(graphs):
        if id_encoding is not None:
            setattr(graph, 'identifiers', encoded_ids[g])
        if degree_encoding is not None:
            setattr(graph, 'degrees', encoded_degrees[g])
                            
    return graphs, encoder_ids, d_id, encoder_degrees, d_degree


class one_hot_unique:
    
    def __init__(self, tensor_list, **kwargs):
        tensor_list = torch.cat(tensor_list, 0)
        self.d = list()
        self.corrs = dict()
        for col in range(tensor_list.shape[1]):
            uniques, corrs = np.unique(tensor_list[:, col], return_inverse=True, axis=0)
            self.d.append(len(uniques))
            self.corrs[col] = corrs
        return       
            
    def fit(self, tensor_list):
        pointer = 0
        encoded_tensors = list()
        for tensor in tensor_list:
            n = tensor.shape[0]
            for col in range(tensor.shape[1]):
                translated = torch.LongTensor(self.corrs[col][pointer:pointer+n]).unsqueeze(1)
                encoded = torch.cat((encoded, translated), 1) if col > 0 else translated
            encoded_tensors.append(encoded)
            pointer += n
        return encoded_tensors
        

class one_hot_max:
    
    def __init__(self, tensor_list, **kwargs):
        tensor_list = torch.cat(tensor_list,0)
        self.d = [int(tensor_list[:,i].max()+1) for i in range(tensor_list.shape[1])]
    
    def fit(self, tensor_list):
        return tensor_list

    
# NB: this encoding scheme has been implemented, but never tested in experiments: use at your own risk.  
'''
class minmax:
    
    def __init__(self, tensor_list, **kwargs):
        
        range_scaler = [0.0, 1.0] if kwargs['range'] is None else kwargs['range']
        self.encoder = MinMaxScaler(feature_range=range_scaler)
        self.d = [1 for i in range(tensor_list[0].shape[1])]
        
    def fit(self, tensor_list):
        
        catted = torch.cat(tensor_list, 0).cpu().float().numpy()
        self.encoder.fit(catted)
        translated = self.encoder.transform(catted)
        
        pointer = 0
        encoded_tensors = list()
        for tensor in tensor_list:
            n = tensor.shape[0]
            encoded = torch.FloatTensor(translated[pointer:pointer+n,:])
            encoded_tensors.append(encoded)
            pointer += n
            
        return encoded_tensors
'''
    

# NB: this encoding scheme has been implemented, but never tested in experiments: use at your own risk.
'''
class binning:
    
    def __init__(self, tensor_list, **kwargs):
    
        self.n_bins = kwargs['bins'][0]
        self.strategy = kwargs['strategy']
    
    
    def fit(self, tensor_list):
        
        
        catted = torch.cat(tensor_list, 0)
        translated = None
        d = []
        for col in range(catted.shape[1]):
            tensor_column = catted[:, col].unsqueeze(1).cpu().numpy()
            print(col, np.unique(tensor_column))
#             B = min([self.n_bins[col], len(np.unique(tensor_column))])
            B = min([self.n_bins, len(np.unique(tensor_column))])
            if B == 1:
                result = torch.ones(tensor_column.shape)
            else:
                encoder = KBinsDiscretizer(n_bins=B, encode='ordinal', strategy=self.strategy)
                d.append(encoder.n_bins)
                encoder.fit(tensor_column)
                result = encoder.transform(tensor_column)
                result = torch.LongTensor(result)
            translated = result if col == 0 else torch.cat((translated, result), 1)
            
        pointer = 0
        encoded_tensors = list()
        for tensor in tensor_list:
            n = tensor.shape[0]
            encoded_tensors.append(translated[pointer:pointer+n])
            pointer += n
            
        self.d = d
        return encoded_tensors
'''

