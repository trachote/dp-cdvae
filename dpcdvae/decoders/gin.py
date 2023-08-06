from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.nn import global_max_pool, SAGPooling
from torch_scatter import scatter

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul

from dpcdvae.utils import get_timestep_embedding, default_init
from gnn.gin import GINEConv
from gnn.mlp import MultiLayerPerceptron
#from torch_geometric.nn import GINEConv

from common.data_utils import get_pbc_distances, frac_to_cart_coords, radius_graph_pbc

class GINDecoder(nn.Module):

    def __init__(self, hidden_dim, latent_dim, num_convs=3,
                 edge_dim=1, activation='relu', short_cut=True,
                 concat_hidden=False, time_dim = 128,
                 radius=12., max_neighbors=40, embed_time=True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_convs = num_convs
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.time_dim = time_dim
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.embed_time = embed_time

        #self.node_emb = nn.Embedding(100, hidden_dim)
        self.node_emb = nn.Sequential(nn.Linear(100, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim))

        if (self.time_dim > 0) and self.embed_time:
            self.fc_time = nn.Sequential(nn.Linear(self.time_dim, self.time_dim * 4),
                                     nn.ReLU(),
                                     nn.Linear(self.time_dim * 4, self.time_dim)
                                     )
            for i in [0, 2]:
                self.fc_time[i].weight.data = default_init()(self.fc_time[i].weight.data.shape)
                nn.init.zeros_(self.fc_time[i].bias)

        self.fc_node = nn.Sequential(nn.Linear(2*hidden_dim + self.time_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim))

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

        #self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
        #                       nn.ReLU(),
        #                       nn.Linear(hidden_dim, latent_dim))
        #self.pool = SAGPooling(hidden_dim)


    def forward(self, z, t, coords, atom_types, lengths, angles, num_atoms):
        """
        Input:
            data: (torch_geometric.data.Data): batched graph
            node_attr: node feature tensor with shape (num_node, hidden)
            edge_attr: edge feature tensor with shape (num_edge, hidden)
        Output:
            node_attr
            graph feature
        """

        if True:
            cart_coords = frac_to_cart_coords(coords, lengths, angles, num_atoms)
            edge_index, to_jimages, num_bonds = radius_graph_pbc(
            cart_coords, lengths, angles, num_atoms, self.radius, self.max_neighbors,
            device=num_atoms.device)

            out = get_pbc_distances(
            cart_coords,
            edge_index,
            lengths,
            angles,
            to_jimages,
            num_atoms,
            num_bonds,
            coord_is_cart=True,
            return_offsets=True,
            )

            edge_index = out["edge_index"]
            edge_attr = out["distances"].view(-1, 1)

        edge_attr = self.edge_emb(edge_attr) # (num_edge, hidden)
        #node_attr = self.node_emb(A)    # (num_node, hidden)
        feats = [z]
        if (self.time_dim > 0) and self.embed_time:
            time_emb = get_timestep_embedding(t.squeeze(), self.time_dim)
            time_emb = self.fc_time(time_emb).repeat_interleave(num_atoms, dim=0)
            feats.append(time_emb)
        atom_types = self.node_emb(atom_types)
        feats.append(atom_types)
        node_attr = self.fc_node(torch.cat(feats, dim=-1))

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

        return node_feature
    