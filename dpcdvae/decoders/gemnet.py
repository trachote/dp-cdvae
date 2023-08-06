import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from gnn.embeddings import MAX_ATOMIC_NUM
from gnn.gemnet.gemnet import GemNetT
from gnn.mlp import FourierFeatures
from .gin import GINDecoder

from dpcdvae.utils import get_timestep_embedding, default_init
from common.data_utils import bound_frac, lattice_params_from_matrix

class GemNetTDecoder(nn.Module):
    """Decoder with GemNetT."""

    def __init__(
        self,
        hidden_dim=128,
        latent_dim=256,
        max_neighbors=20,
        radius=6.,
        scale_file=None,
        condition_time=None,
        num_targets=1,
        regress_logvars=False,
        time_dim=128,
        #timesteps=1000,
        noisy_atom_types=False,
        fourier_feats=False,
        regress_atoms=False,
    ):
        super(GemNetTDecoder, self).__init__()
        self.cutoff = radius
        self.max_num_neighbors = max_neighbors
        self.regress_logvars = regress_logvars
        self.condition_time = condition_time
        self.noisy_atom_types = noisy_atom_types
        self.fourier_feats = fourier_feats
        self.regress_atoms = regress_atoms
        
        if condition_time == 'None':
            self.time_dim = 0
        elif condition_time == 'constant':
            self.time_dim = 1
        elif condition_time == 'embed':
            self.time_dim = time_dim
            #self.timesteps = timesteps
            # Condition on noise levels.
            #self.fc_time = nn.Embedding(self.timesteps, self.time_dim)
            self.fc_time = nn.Sequential(nn.Linear(self.time_dim, self.time_dim * 4),
                                         nn.ReLU(),
                                         nn.Linear(self.time_dim * 4, self.time_dim)
                                        )
            for i in [0, 2]:
                self.fc_time[i].weight.data = default_init()(self.fc_time[i].weight.data.shape)
                nn.init.zeros_(self.fc_time[i].bias)

        if self.fourier_feats:
            first, last, step = 3, 8, 1
            fourier_dim = 2 * (last - first + 1) * 3
            self.fourier_pos = FourierFeatures(first, last, step)
#             if self.noisy_atom_types:
#                 first, last, step = 5, 6, 1
#                 fourier_dim += 2 * (last - first + 1) * MAX_ATOMIC_NUM
#                 self.fourier_type = FourierFeatures(first, last, step)
        else:
            fourier_dim = 0
            
        if self.noisy_atom_types:
            noisy_atom_dim = hidden_dim
            self.noisy_atom_emb = nn.Sequential(nn.Linear(MAX_ATOMIC_NUM, noisy_atom_dim * 4),
                                         nn.ReLU(),
                                         nn.Linear(noisy_atom_dim * 4, noisy_atom_dim)
                                        )
            for i in [0, 2]:
                nn.init.xavier_uniform_(self.noisy_atom_emb[i].weight.data)
                nn.init.zeros_(self.noisy_atom_emb[i].bias)
        else:
            noisy_atom_dim = 0
        
        self.gemnet = GemNetT(
            num_targets=num_targets,
            latent_dim=latent_dim,
            emb_size_atom=hidden_dim,
            emb_size_edge=hidden_dim,
            regress_forces=True,
            regress_logvars=self.regress_logvars,
            cutoff=self.cutoff,
            max_neighbors=self.max_num_neighbors,
            otf_graph=True,
            scale_file=scale_file,
            condition_time=self.condition_time,
            time_dim=self.time_dim,
            noisy_atom_types=False, #self.noisy_atom_types,
            extra_dim=fourier_dim+noisy_atom_dim,
        )
        atom_hidden_dim = hidden_dim + latent_dim + self.time_dim + fourier_dim + noisy_atom_dim
        self.fc_atom = nn.Linear(atom_hidden_dim, MAX_ATOMIC_NUM)
#         self.fc_atom = nn.Sequential(nn.Linear(atom_hidden_dim, atom_hidden_dim),
#                                      nn.ReLU(),
#                                      nn.Linear(atom_hidden_dim, MAX_ATOMIC_NUM)
#                                      )

    def forward(self, z, t, pred_frac_coords, pred_atom_types, num_atoms,
                lengths, angles, noisy_atom_types=None):
        """
        args:
            z: (N_cryst, num_latent)
            pred_frac_coords: (N_atoms, 3)
            pred_atom_types: (N_atoms, ), need to use atomic number e.g. H = 1
            num_atoms: (N_cryst,)
            lengths: (N_cryst, 3)
            angles: (N_cryst, 3)
        returns:
            atom_frac_coords: (N_atoms, 3)
            atom_types: (N_atoms, MAX_ATOMIC_NUM)
        """
        
        if self.condition_time == 'embed':
            time_emb = get_timestep_embedding(t.squeeze(), self.time_dim)
            #time_emb = F.one_hot((t*self.timesteps).long(), self.timesteps)
            time_emb = self.fc_time(time_emb)
        elif self.condition_time == 'constant':
            time_emb = t
        else:
            time_emb = None
        
        extra_feats = []
        if self.fourier_feats:
            extra_feats.append(self.fourier_pos(pred_frac_coords))
        if self.noisy_atom_types:
            assert noisy_atom_types != None
            extra_feats.append(self.noisy_atom_emb(noisy_atom_types))
            
        extra_feats = torch.cat(extra_feats, dim=-1) if len(extra_feats) > 0 else None
        
        pred_frac_coords = bound_frac(pred_frac_coords)
        # (num_atoms, hidden_dim) (num_crysts, 3)
        _, h, pred_cart_coord_diff = self.gemnet(
            z=z,
            time_emb=time_emb,
            pos=pred_frac_coords,
            atom_types=pred_atom_types,
            num_atoms=num_atoms,
            lengths=lengths,
            angles=angles,
            edge_index=None,
            to_jimages=None,
            num_bonds=None,
            extra_feats=extra_feats,
        )
        if not self.regress_atoms:
            pred_atom_types = self.fc_atom(h)
        else:
            pred_atom_types = h
        return pred_cart_coord_diff, pred_atom_types
