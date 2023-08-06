import omegaconf
import torch
import pandas as pd
from omegaconf import ValueNode, OmegaConf
from torch.utils.data import Dataset

from torch_geometric.data import Data

import sys
sys.path.insert(1, '.')

# from cdvae.common.utils import PROJECT_ROOT
from common.data_utils import (
    preprocess, preprocess_tensors, add_scaled_lattice_prop)

import pickle


class CrystDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode,
                 prop: ValueNode, niggli: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode,
                 lattice_scale_method: ValueNode, prop_classes: ValueNode,
                 **kwargs):

        super().__init__()
        self.path = path
        self.name = name
        # self.df = pd.read_csv(path)
        self.prop = prop
        self.prop_classes = prop_classes
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method


#        open file
        with open(self.path.replace(".csv", "_processed"), 'rb') as f:
            self.cached_data = pickle.load(f)

        # self.cached_data = preprocess(
        #     self.path,
        #     preprocess_workers,
        #     niggli=self.niggli,
        #     primitive=self.primitive,
        #     graph_method=self.graph_method,
        #     prop_list=[prop])

        if len(self.cached_data[0]['graph_arrays'] )==8:
            # added cart coords
            self.v = 1
        else:
            # original version
            self.v = 0

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method, self.v)

        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        # scaler is set in DataModule set stage
        # prop = self.scaler.transform(data_dict[self.prop])

        prop = torch.stack([self.scaler[i].transform(data_dict[self.prop[i]]) for i in range(len(self.prop))])

        if self.prop_classes:
            prop_classes = torch.stack([torch.tensor(data_dict[i]) for i in self.prop_classes]).long().view(1, -1)
        else:
            prop_classes = []


        if self.v:
            (frac_coords, cart_coords, atom_types, lengths, angles, edge_indices,
            to_jimages, num_atoms) = data_dict['graph_arrays']
        else:
            (frac_coords, atom_types, lengths, angles, edge_indices,
            to_jimages, num_atoms) = data_dict['graph_arrays']       
            cart_coords = 0             


        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            cart_coords=torch.Tensor(cart_coords),            
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop.view(1, -1), # (1, N) N = number of properties
#            z=prop_classes.view(1, -1)-1
            z=prop_classes
        )
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"


def genCrystalDatasetFile(name: ValueNode, path: ValueNode,
                 prop: ValueNode, niggli: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode,
                 lattice_scale_method: ValueNode, prop_classes: ValueNode,
                 **kwargs):

    cached_data = preprocess(
        path,
        preprocess_workers,
        niggli=niggli,
        primitive=primitive,
        graph_method=graph_method,
        prop_list=prop +prop_classes)

    with open(path.replace(".csv", "_processed"), 'wb') as f:
        pickle.dump(cached_data, f)


class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method, 1)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, cart_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # (frac_coords, atom_types, lengths, angles, edge_indices,
        #  to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            cart_coords=torch.Tensor(cart_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"


# @hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
# def main(cfg: omegaconf.DictConfig):
#     from torch_geometric.data import Batch
#     from cdvae.common.data_utils import get_scaler_from_data_list
#     dataset: CrystDataset = hydra.utils.instantiate(
#         cfg.data.datamodule.datasets.train, _recursive_=False
#     )
#     lattice_scaler = get_scaler_from_data_list(
#         dataset.cached_data,
#         key='scaled_lattice')
#     scaler = get_scaler_from_data_list(
#         dataset.cached_data,
#         key=dataset.prop)

#     dataset.lattice_scaler = lattice_scaler
#     dataset.scaler = scaler
#     data_list = [dataset[i] for i in range(len(dataset))]
#     batch = Batch.from_data_list(data_list)
#     return batch

def main(cfg):
    print("START")
    train_dataset = genCrystalDatasetFile(**cfg.data.datamodule.datasets.train)
    val_dataset = genCrystalDatasetFile(**cfg.data.datamodule.datasets.val[0])
    test_dataset = genCrystalDatasetFile(**cfg.data.datamodule.datasets.test[0])

if __name__ == "__main__":

    data = OmegaConf.load("./conf/data/mp_20_class.yaml")
    cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.create(OmegaConf.to_yaml({
        'PROJECT_ROOT':'/home/cue/projects/cdvae-3', 'data':data
    })), resolve=True))

    main(cfg)
