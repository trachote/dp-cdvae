import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gnn.embeddings import MAX_ATOMIC_NUM
from models.gnn.gemnet.gemnet import GemNetT
from models.sde_utils import get_timestep_embedding, default_init
from torch_scatter import scatter


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
        time_dim=128,
    ):
        super(GemNetTDecoder, self).__init__()
        self.cutoff = radius
        self.max_num_neighbors = max_neighbors
        self.condition_time = condition_time
        
        if condition_time == 'None':
            self.time_dim = 0
        elif condition_time == 'constant':
            self.time_dim = 1
        elif condition_time == 'embed':
            self.time_dim = time_dim
            # Condition on noise levels.
            self.fc_time = nn.Sequential(nn.Linear(self.time_dim, self.time_dim * 4),
                                         nn.ReLU(),
                                         nn.Linear(self.time_dim * 4, self.time_dim)
                                        )
            for i in [0, 2]:
                self.fc_time[i].weight.data = default_init()(self.fc_time[i].weight.data.shape)
                nn.init.zeros_(self.fc_time[i].bias)

        self.gemnet = GemNetT(
            num_targets=1,
            latent_dim=latent_dim,
            emb_size_atom=hidden_dim,
            emb_size_edge=hidden_dim,
            regress_forces=True,
            cutoff=self.cutoff,
            max_neighbors=self.max_num_neighbors,
            otf_graph=True,
            scale_file=scale_file,
            condition_time=self.condition_time,
            time_dim=self.time_dim,
        )
        self.fc_atom = nn.Linear(hidden_dim, MAX_ATOMIC_NUM)

    def forward(self, z, t, pred_frac_coords, pred_atom_types, num_atoms,
                lengths, angles):
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
            time_emb = get_timestep_embedding(t, self.time_dim)
            time_emb = self.fc_time(time_emb)
        elif self.condition_time == 'constant':
            time_emb = t
        else:
            time_emb = None
        
        # (num_atoms, hidden_dim) (num_crysts, 3)
        _, h, pred_cart_coord_diff = self.gemnet(
            z=z,
            time_emb=time_emb,
            frac_coords=pred_frac_coords,
            atom_types=pred_atom_types,
            num_atoms=num_atoms,
            lengths=lengths,
            angles=angles,
            edge_index=None,
            to_jimages=None,
            num_bonds=None,
        )
        pred_atom_types = self.fc_atom(h)
        return pred_cart_coord_diff, pred_atom_types
