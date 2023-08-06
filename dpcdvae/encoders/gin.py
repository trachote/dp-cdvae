# coding=utf-8
from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.nn import global_max_pool, SAGPooling
from torch_scatter import scatter

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul

from gnn.gin import GINEConv
from gnn.mlp import MultiLayerPerceptron
from .dimenet import DimeNetppEncoder
#from torch_geometric.nn import GINEConv

from common.data_utils import get_pbc_distances

class GINEncoder(nn.Module):

    def __init__(self, hidden_dim, latent_dim, num_convs=3, 
                 edge_dim=1, activation='relu', short_cut=True, 
                 concat_hidden=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_convs = num_convs
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.node_emb = nn.Embedding(100, hidden_dim)

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None 
        
        self.edge_emb = nn.Sequential(nn.Linear(edge_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim))
        
        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            #out_dim = hidden_dim if i < self.num_convs - 1 else latent_dim
            self.convs.append(GINEConv(MultiLayerPerceptron(hidden_dim, [hidden_dim, hidden_dim], \
                                    activation=activation)))#, activation=activation))
            
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(),
                               nn.Linear(hidden_dim, latent_dim))
        #self.pool = SAGPooling(hidden_dim)
        

    def forward(self, data):
        """
        Input:
            data: (torch_geometric.data.Data): batched graph
            node_attr: node feature tensor with shape (num_node, hidden)
            edge_attr: edge feature tensor with shape (num_edge, hidden)
        Output:
            node_attr
            graph feature
        """
        z = data.atom_types - 1
        edge_index = data.edge_index
        
        if data.edge_attr is not None:
            edge_attr = data.edge_attr
        else:
            #edge_attr = torch.ones((edge_index.size(-1), self.hidden_dim), device=z.device)
            out = get_pbc_distances(
            data.frac_coords,
            data.edge_index,
            data.lengths,
            data.angles,
            data.to_jimages,
            data.num_atoms,
            data.num_bonds,
            return_offsets=True
            )

            edge_index = out["edge_index"]
            edge_attr = out["distances"].view(-1, 1)
            
        edge_attr = self.edge_emb(edge_attr) # (num_edge, hidden)
        node_attr = self.node_emb(z)    # (num_node, hidden)
 
        hiddens = []
        conv_input = node_attr # (num_node, hidden)

        for conv_idx, conv in enumerate(self.convs):
            hidden = conv(conv_input, edge_index, edge_attr)
            if conv_idx < len(self.convs) - 1 and self.activation is not None:
                hidden = self.activation(hidden)
            assert hidden.shape == conv_input.shape                
            if self.short_cut and hidden.shape == conv_input.shape:
                hidden = hidden + conv_input

            hiddens.append(hidden)
            conv_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
            
        out = global_max_pool(node_feature, data.batch)
        #out = self.pool(node_feature, edge_index, batch=data.batch)[0]
        #out = scatter(node_feature, data.batch, dim=0, reduce='max')
        out = self.fc(out)
        return out
    

class DimeGINEncoder(nn.Module):
    def __init__(self, hparams):
        super(DimeGINEncoder, self).__init__()
        latent_dim = hparams.latent_dim
        dime_params = {k: v for k, v in hparams.encoder1.items() if k != "_target_"}
        gine_params = {k: v for k, v in hparams.encoder2.items() if k != "_target_"}
        self.dime = DimeNetppEncoder(**dime_params)
        self.gine = GINEncoder(**gine_params)
        
        self.fc = nn.Sequential(nn.Linear(2*latent_dim, 2*latent_dim),
                                nn.ReLU(),
                                nn.Linear(2*latent_dim, 2*latent_dim),
                                nn.ReLU(),
                                nn.Linear(2*latent_dim, latent_dim))
       
    def forward(self, data):
        z1 = self.dime(data)
        z2 = self.gine(data)
        z = torch.cat([z1, z2], dim=-1)
        return self.fc(z)
