import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, ModuleList, Linear, Embedding
from torch_geometric.nn import MessagePassing, radius_graph, global_max_pool
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_scatter import scatter_add
from math import pi as PI

from .geodiff_utils import get_edge_encoder, MLP

#from ..common import MeanReadout, SumReadout, MultiLayerPerceptron
from common.data_utils import lattice_params_to_matrix_torch, frac_to_cart_coords

class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class AsymmetricSineCosineSmearing(Module):

    def __init__(self, num_basis=50):
        super().__init__()
        num_basis_k = num_basis // 2
        num_basis_l = num_basis - num_basis_k
        self.register_buffer('freq_k', torch.arange(1, num_basis_k + 1).float())
        self.register_buffer('freq_l', torch.arange(1, num_basis_l + 1).float())

    @property
    def num_basis(self):
        return self.freq_k.size(0) + self.freq_l.size(0)

    def forward(self, angle):
        # If we don't incorporate `cos`, the embedding of 0-deg and 180-deg will be the
        #  same, which is undesirable.
        s = torch.sin(angle.view(-1, 1) * self.freq_k.view(1, -1))  # (num_angles, num_basis_k)
        c = torch.cos(angle.view(-1, 1) * self.freq_l.view(1, -1))  # (num_angles, num_basis_l)
        return torch.cat([s, c], dim=-1)

    
class SymmetricCosineSmearing(Module):

    def __init__(self, num_basis=50):
        super().__init__()
        self.register_buffer('freq_k', torch.arange(1, num_basis+1).float())

    @property
    def num_basis(self):
        return self.freq_k.size(0)

    def forward(self, angle):
        return torch.cos(angle.view(-1, 1) * self.freq_k.view(1, -1))   # (num_angles, num_basis)


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff, smooth):
        super(CFConv, self).__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff
        self.smooth = smooth

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_length, edge_attr):
        if self.smooth:
            C = 0.5 * (torch.cos(edge_length * PI / self.cutoff) + 1.0)
            C = C * (edge_length <= self.cutoff) * (edge_length >= 0.0)     # Modification: cutoff
        else:
            C = (edge_length <= self.cutoff).float()
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff, smooth):
        super(InteractionBlock, self).__init__()
        mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, mlp, cutoff, smooth)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_length, edge_attr):
        x = self.conv(x, edge_index, edge_length, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class SchNet(Module):

    def __init__(self, hidden_channels=256, num_filters=128,
                num_interactions=6, edge_channels=100, cutoff=10.0, smooth=False,
                **kwargs):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.edge_channels = edge_channels
        self.cutoff = cutoff

        self.embedding = Embedding(100, hidden_channels, max_norm=10.0)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, edge_channels,
                                     num_filters, cutoff, smooth)
            self.interactions.append(block)

        self.edge_encoder = get_edge_encoder('gaussian')
        self.hidden_d_mlp = MLP(356, [356, 512, 256])
        self.final_mlp = MLP(256, [256, 256])

    def forward(self, batch, embed_node=True, transform=False):
        ## Node attr embedding
        atom_types = batch.atom_types
        if embed_node:
            assert atom_types.dim() == 1 and atom_types.dtype == torch.long
            h = self.embedding(atom_types)
        else:
            h = atom_types

        ## Get edge info
        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles).to(atom_types.device)
        edge_index = batch.edge_index
        dr_ij = []
        ne = torch.tensor([g.num_edges for g in batch.to_data_list()])

        for i, (n, ns) in enumerate(zip(ne, ne.cumsum(0))):
            dr = batch.frac_coords[edge_index[0][ns-n:ns]] \
                    - batch.frac_coords[edge_index[1][ns-n:ns]] \
                    + batch.to_jimages[ns-n:ns]
            dr = torch.einsum('ea,ab->eb', dr, lattices[i])
            dr_ij.append(dr)
        dr_ij = torch.cat(dr_ij, dim=0).view(-1, 3)
        edge_length = torch.linalg.norm(dr_ij, dim=-1).view(-1, 1)
        
        if batch.edge_attr is None:
            edge_attr = get_coulomb_attr(batch, edge_length, self.edge_channels)
            #edge_attr = torch.ones((edge_length.shape[0], self.edge_channels)).to(atom_types.device)
        else:
            edge_attr = batch.edge_attr
        
        ## Edge embedding
        edge_attr = self.edge_encoder(edge_length, edge_attr)

        ## SchNet
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)
        
        if transform:
            ## Hidden as a function of distance: h(d)_ij
            h_row, h_col = h[edge_index[0]], h[edge_index[1]]
            h_pair = torch.cat([h_row*h_col, edge_attr], dim=-1)
            h_out = self.hidden_d_mlp(h_pair)
            
            ## h(d)_ij -> h(r)_i
            na = batch.frac_coords.size(0)
            dd_dr = (1. / edge_length) * dr_ij # (ne, 3)
            dd_dr = torch.linalg.norm(dd_dr, dim=-1).view(-1, 1) # (ne, 1)
            h_out = scatter_add(dd_dr * h_out, edge_index[0], dim=0, dim_size=na) \
                    + scatter_add( - dd_dr * h_out, edge_index[1], dim=0, dim_size=na) # (na, 3)
        
        else:
            h_out = h

        ## pool and more emb
        h_out = global_max_pool(h_out, batch.batch)
        h_out = self.final_mlp(h_out)
        return h_out


def get_coulomb_attr(batch, edge_length, num_bins):
    if hasattr(batch, 'cart_coords'):
        cart_coords = batch.cart_coords
    else:
        cart_coords = frac_to_cart_coords(batch.frac_coords,
                                          batch.lengths,
                                          batch.angles,
                                          batch.num_atoms)

    edge_index = batch.edge_index
    atom_A = batch.atom_types[edge_index[0]].repeat(num_bins, 1).T
    atom_B = batch.atom_types[edge_index[1]].repeat(num_bins, 1).T
    edge_length = edge_length.repeat(1, num_bins)
    d = torch.linspace(0., 5., num_bins).repeat(atom_A.shape[0], 1).to(atom_A.device)
    edge_attr = atom_A / (d + 1e-8).abs() + atom_B / (edge_length - d + 1e-8).abs()
    edge_attr = F.normalize(edge_attr) * atom_A * atom_B / 1e3
    #print(f"attr: {edge_attr.min()}, {edge_attr.max()}")
    return edge_attr
    
