from gnn.gemnet.gemnet import GemNetT
from torch import nn

class GemNetTEncoder(nn.Module):
    """Wrapper for GemNetT."""

    def __init__(
        self,
        num_targets,
        hidden_size,
        otf_graph=False,
        cutoff=6.0,
        max_num_neighbors=20,
        scale_file=None,
    ):
        super(GemNetTEncoder, self).__init__()
        self.num_targets = num_targets
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.otf_graph = otf_graph

        self.gemnet = GemNetT(
            num_targets=num_targets,
            latent_dim=0,
            emb_size_atom=hidden_size,
            emb_size_edge=hidden_size,
            regress_forces=False,
            cutoff=self.cutoff,
            max_neighbors=self.max_num_neighbors,
            otf_graph=self.otf_graph,
            scale_file=scale_file,
        )

    def forward(self, data):
        # (num_crysts, num_targets)
        output = self.gemnet(
            z=None,
            time_emb=None,
            frac_coords=data.frac_coords,
            atom_types=data.atom_types,
            num_atoms=data.num_atoms,
            lengths=data.lengths,
            angles=data.angles,
            edge_index=data.edge_index,
            to_jimages=data.to_jimages,
            num_bonds=data.num_bonds
        )
        return output
