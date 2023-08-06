from typing import Any, Dict

import numpy as np
import math
import copy
import json
import os
import glob
from omegaconf import DictConfig, OmegaConf

import torch
from torch import sqrt
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm

from common.datamodule import CrystDataModule
from common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume,
    frac_to_cart_coords, min_distance_sqr_pbc, bound_frac, 
    lattice_params_to_matrix_torch, lattice_params_from_matrix)

from gnn.embeddings import MAX_ATOMIC_NUM, KHOT_EMBEDDINGS
#from dpcdvae.utils import gaussian_kld


def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers - 1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


class BaseModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.hparams = cfg.model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hparams.data = cfg.data
        self.current_epoch = 0
        self.logs = {'train':[], 'val':[], 'test':[]}
        self.train_checkpoint_path = None
        self.val_checkpoint_path = None
        self.min_val_loss = float('inf')
        self.min_val_epoch = 0
        self.train_log = None
        self.val_log = None
        self.test_log = None
        self.model_name = "model"
        self.datamodule = None
        
    def init(self, load=False, test=False):
        self.init_optimizer()
        self.init_scheduler()

        self.to(self.device)

        # continue train, load = True else load = False
        checkpoint = self.load_checkpoint(load)

        if checkpoint==False:
            print("No checkpoint: Save hparams.yaml")
            with open(self.cfg.output_dir + "/hparams.yaml", "w") as f:
                f.write(OmegaConf.to_yaml(cfg=self.cfg))

        self.init_datamodule(load)

        self.init_dataloader(load, test)

    def init_optimizer(self):
        print(f"Instantiating Optimizer <{self.cfg.optim.optimizer._target_}>")
        # if self.cfg.optim.optimizer._target_=='Adam':
        self.optimizer = torch.optim.Adam(self.parameters(), **{k:v for k, v in self.cfg.optim.optimizer.items() if k!='_target_'})        

    def init_scheduler(self):
        print(f"Instantiating LR Scheduler <{self.cfg.optim.lr_scheduler._target_}>")
        print(f"LR Scheduler: ", self.cfg.optim.use_lr_scheduler)
        # if self.cfg.optim.optimizer.optimizer._target_=='ReduceLROnPlateau':
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **{k:v for k, v in self.cfg.optim.lr_scheduler.items() if k!='_target_'})

    def init_datamodule(self, load):
        if self.cfg.data.datamodule._target_=='CrystDataModule':
            print(f"Set up datamodule")
            self.datamodule = CrystDataModule(**{k:v for k, v in self.cfg.data.datamodule.items() if k!='_target_'})

            if load!=True:
                print(">>> save scalers")
                self.lattice_scaler = self.datamodule.lattice_scaler.copy()
                self.param_decoder.lattice_scaler = self.datamodule.lattice_scaler.copy()
                self.scaler = self.datamodule.scaler.copy()
                torch.save(self.datamodule.lattice_scaler, self.cfg.output_dir  + '/lattice_scaler.pt')
                torch.save(self.datamodule.scaler, self.cfg.output_dir  + '/prop_scaler.pt')

        if load==True:
            print(">>> Load scalers")
            self.lattice_scaler = torch.load(self.cfg.output_dir  + '/lattice_scaler.pt')
            self.scaler = torch.load(self.cfg.output_dir  + '/prop_scaler.pt')   
     

    def init_dataloader(self, load, test):
        if self.datamodule is not None:
            if load:
                if test:
                    self.datamodule.setup('test')
                    self.test_dataloader = self.datamodule.test_dataloader()[0]
                else:
                    self.datamodule.setup()
                    self.test_dataloader = self.datamodule.val_dataloader()[0]
            else:
                self.datamodule.setup("fit")
                self.train_dataloader = self.datamodule.train_dataloader()
                self.val_dataloader = self.datamodule.val_dataloader()[0]
                self.test_dataloader = None

    def train_start(self):
        print(">>>TRAINING START<<<")
        pass

    def train_end(self, e):
        print(">>>TRAINING END<<<")
        self.logging(e)
        self.train_checkpoint_path = self.save_checkpoint(model_checkpoint_path=self.train_checkpoint_path, suffix="train")     

    def clip_grad_value_(self):
        torch.nn.utils.clip_grad_value_(self.parameters(), clip_value=0.5)

    def train_step_end(self, e):
        # for examination
        pass

    def val_step_end(self, e):
        # for examination
        pass

    def train_epoch_start(self, e):
        self.clear_log_dict()
        self.train_log = None
        self.val_log = None
        self.test_log = None

    def val_epoch_start(self, e):
        pass    

    def train_epoch_end(self, e):
        log_dict = {'epoch':e}
        log_dict.update({k:np.mean([x[k].item() if torch.is_tensor(x[k]) else x[k] for x in self.logs['train']]) for k in self.logs['train'][0].keys()})

        with open(self.cfg.output_dir + "/train_metrics.json", 'a') as f:
            f.write(json.dumps({k:v for k, v in log_dict.items()}))
            f.write('\r\n')

        self.train_log = log_dict

    def val_epoch_end(self, e):
        log_dict = {'epoch':e}
        log_dict.update({k:np.mean([x[k].item() if torch.is_tensor(x[k]) else x[k] for x in self.logs['val']]) for k in self.logs['val'][0].keys()})

        with open(self.cfg.output_dir + "/val_metrics.json", 'a') as f:
            f.write(json.dumps({k:v for k, v in log_dict.items()}))
            f.write('\r\n')

        self.val_log = log_dict

        if self.val_log['val_loss'] < self.min_val_loss:
            self.min_val_loss = self.val_log['val_loss']
            self.min_val_epoch = e
            self.val_checkpoint_path = self.save_checkpoint(model_checkpoint_path=self.val_checkpoint_path, suffix="val")

    def test_epoch_end(self):
        log_dict = {}
        log_dict.update({k:np.mean([x[k].item() if torch.is_tensor(x[k]) else x[k] for x in self.logs['test']]) for k in self.logs['test'][0].keys()})

        with open(self.cfg.output_dir + "/test_metrics.json", 'a') as f:
            f.write(json.dumps({k:v for k, v in log_dict.items()}))
            f.write('\r\n')

        self.test_log = log_dict

    def train_val_epoch_end(self, e):

        if e % self.cfg.logging.log_freq_every_n_epoch == 0:
            self.logging(e)

        if e % self.cfg.checkpoint_freq_every_n_epoch == 0:
            self.train_checkpoint_path = self.save_checkpoint(model_checkpoint_path=self.train_checkpoint_path, suffix="train")

        self.current_epoch += 1

    def load_checkpoint(self, load):
        checkpoint = False
        # ckpts = list(self.cfg.output_dir.glob(f'*={self.model_name}=*.ckpt'))
        ckpts = list(glob.glob(f'{self.cfg.output_dir}/*={self.model_name}=*.ckpt'))   
        print(ckpts, f'{self.cfg.output_dir}/*={self.model_name}=*.ckpt')     
        if len(ckpts) > 0:

            ckpt_epochs = np.array([int(ckpt.split('/')[-1].split('.')[0].split('=')[1]) for ckpt in ckpts])
            ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
            print(f">>>>> Load model from checkpoint {ckpt}")

            if torch.cuda.is_available():
                ckpt = torch.load(ckpt)
            else:
                ckpt = torch.load(ckpt, map_location=torch.device('cpu'))
            
            self.current_epoch = ckpt['epoch'] + 1
            
            print(f">>>>> Update model from train checkpoint") 
            self.load_state_dict(ckpt['model_state_dict'])
            
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except:
                print("ERROR: Loading state dict")

            if self.cfg.optim.use_lr_scheduler:
                try:
                    self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                except:
                    print("ERROR: Loading scheduler")                

            self.train_checkpoint_path = f"{self.cfg.output_dir}/epoch={ckpt['epoch']}={self.model_name}=train.ckpt"

            ckpts = list(glob.glob(f'{self.cfg.output_dir}/*={self.model_name}=val.ckpt'))
            # list(self.cfg.output_dir.glob(f'*={self.model_name}=val.ckpt'))

            if len(ckpts) > 0:

                ckpt_epochs = np.array([int(ckpt.split('/')[-1].split('.')[0].split('=')[1]) for ckpt in ckpts])
                ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
                
                print(f">>>>> Load val model from checkpoint {ckpt}")                

                if torch.cuda.is_available():
                    ckpt = torch.load(ckpt)
                else:
                    ckpt = torch.load(ckpt, map_location=torch.device('cpu'))                

                if load:
                    print(f">>>>> Update model from val checkpoint") 
                    self.load_state_dict(ckpt['model_state_dict'])

                self.min_val_epoch = ckpt['epoch']
                self.min_val_loss = torch.tensor(ckpt['val_loss'])

                print("min val epoch: ", self.min_val_epoch)
                print("min val loss: ", self.min_val_loss.item())            

                self.val_checkpoint_path = f"{self.cfg.output_dir}/epoch={ckpt['epoch']}={self.model_name}=val.ckpt"
            checkpoint = True
        else:
            print(f">>>>> New Training")        

        return checkpoint

    def save_checkpoint(self, model_checkpoint_path, suffix="val", logs={}):
        model_checkpoint = {
            'model_state_dict':copy.deepcopy(self.state_dict()), 
            'optimizer_state_dict':copy.deepcopy(self.optimizer.state_dict()), 
            'scheduler_state_dict':copy.deepcopy(self.scheduler.state_dict()),
            'epoch':self.current_epoch, 
            'train_loss':self.train_log['train_loss'], 
            'val_loss':self.val_log['val_loss'] if self.val_log else None 
        }

        model_checkpoint.update(logs)

        new_model_checkpoint_path = f"{self.cfg.output_dir}/epoch={self.current_epoch}={self.model_name}={suffix}.ckpt"

        if new_model_checkpoint_path != model_checkpoint_path:
            if model_checkpoint_path and os.path.exists(model_checkpoint_path):
                os.remove(model_checkpoint_path)
            print("Save model checkpoint: ", new_model_checkpoint_path)
            print("\tmodel checkpoint train loss: ", model_checkpoint['train_loss'])
            print("\tmodel checkpoint val loss: ", model_checkpoint['val_loss'])
            torch.save(model_checkpoint, new_model_checkpoint_path)

        return new_model_checkpoint_path

    def early_stopping(self, e):
        if e - self.min_val_epoch > self.cfg.data.early_stopping_patience_epoch:
            print("Early stopping")
            return True

        return False
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        teacher_forcing = (
                self.current_epoch <= self.hparams.teacher_forcing_max_epoch)
        outputs = self(batch, teacher_forcing, training=True)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='train')
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            prefix='train'
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='val')
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            prefix='val'
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='test')
        self.log_dict(
            log_dict,
            prefix='test'
        )
        return loss

    def compute_stats(self, batch, outputs, prefix):
        num_atom_loss = outputs['num_atom_loss']
        lattice_loss = outputs['lattice_loss']
        coord_loss = outputs['coord_loss']
        type_loss = outputs['type_loss']
        kld_loss = outputs['kld_loss']
        composition_loss = outputs['composition_loss']
        coord_cost = 10 if self.current_epoch > self.hparams.mul_cost_coord_epoch else 1

        loss = (
                self.hparams.cost_natom * num_atom_loss +
                self.hparams.cost_lattice * lattice_loss +
                coord_cost * self.hparams.cost_coord * coord_loss +
                self.hparams.cost_type * type_loss +
                self.hparams.beta * kld_loss +
                self.hparams.cost_composition * composition_loss
        )

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_natom_loss': num_atom_loss,
            f'{prefix}_lattice_loss': lattice_loss,
            f'{prefix}_coord_loss': coord_loss,
            f'{prefix}_type_loss': type_loss,
            f'{prefix}_kld_loss': kld_loss,
            f'{prefix}_composition_loss': composition_loss,
        }

        if prefix != 'train':
            # validation/test loss only has coord and type
            loss = (
                    self.hparams.cost_coord * coord_loss +
                    self.hparams.cost_type * type_loss)

            # evaluate num_atom prediction.
            pred_num_atoms = outputs['pred_num_atoms'].argmax(dim=-1)
            num_atom_accuracy = (
                                        pred_num_atoms == batch.num_atoms).sum() / batch.num_graphs

            # evalute lattice prediction.
            pred_lengths_and_angles = outputs['pred_lengths_and_angles']
            self.lattice_scaler.match_device(pred_lengths_and_angles)
            scaled_preds = self.lattice_scaler.inverse_transform(
                pred_lengths_and_angles)
            pred_lengths = scaled_preds[:, :3]
            pred_angles = scaled_preds[:, 3:]

            if self.hparams.data.lattice_scale_method == 'scale_length':
                pred_lengths = pred_lengths * \
                               batch.num_atoms.view(-1, 1).float() ** (1 / 3)
            lengths_mard = mard(batch.lengths, pred_lengths)
            angles_mae = torch.mean(torch.abs(pred_angles - batch.angles))

            pred_volumes = lengths_angles_to_volume(pred_lengths, pred_angles)
            true_volumes = lengths_angles_to_volume(
                batch.lengths, batch.angles)
            volumes_mard = mard(true_volumes, pred_volumes)

            # evaluate atom type prediction.
            pred_atom_types = outputs['pred_atom_types']
            target_atom_types = outputs['target_atom_types']
            type_accuracy = pred_atom_types.argmax(
                dim=-1) == (target_atom_types - 1)
            type_accuracy = scatter(type_accuracy.float(
            ), batch.batch, dim=0, reduce='mean').mean()

            log_dict.update({
                f'{prefix}_loss': loss,
                f'{prefix}_natom_accuracy': num_atom_accuracy,
                f'{prefix}_lengths_mard': lengths_mard,
                f'{prefix}_angles_mae': angles_mae,
                f'{prefix}_volumes_mard': volumes_mard,
                f'{prefix}_type_accuracy': type_accuracy,
            })

        return log_dict, loss    
    
    def logging(self, e):
        print(f"Epoch {e:5d}:")
        print(f"\tTrain Loss:{self.train_log['train_loss']}")
        if self.val_log:
            print(f"\tVal Loss:{self.val_log['val_loss']}")
        print(f"\tLR:{self.optimizer.param_groups[0]['lr']}")

    def log_dict(self, log_dict, prefix, on_step=False, on_epoch=False, prog_bar=False):
        self.logs[prefix].append(log_dict)

    def clear_log_dict(self):
        for x in self.logs:
            self.logs[x] = []
        self.train_log = []
        self.val_log = []


    def kld_reparam(self, hidden):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return mu, log_var, z
    
    def gaussian_kld(self, q_mu, q_logvar, p_mu, p_logvar, batch_idx, is_logvar=False, reduce='sum'):
        '''
        KLD(q || p)
        '''
        if not is_logvar:
            q_logvar = 2*torch.log(q_logvar)
            p_logvar = 2*torch.log(p_logvar)
        kld = 0.5 * (p_logvar - q_logvar - 1 \
                     + (torch.exp(q_logvar) + (q_mu - p_mu)**2) / torch.exp(p_logvar))
#         kld = torch.log(p_sigma / q_sigma) \
#               + 0.5 * (q_sigma**2 + (q_mu - p_mu)**2) / (p_sigma**2) \
#               - 0.5
        return scatter(kld, batch_idx, dim=0, reduce=reduce)

    def sub_mean(self, x, batch_idx, num_atoms, dim=0, recenter=True):
        if recenter:
            self.mean = scatter(x, batch_idx, dim=dim, reduce='mean')
            self.mean = self.mean.repeat_interleave(num_atoms, dim=0)
            x = x - self.mean
        else:
            self.mean = 0.
        return x

    def add_mean(self, x, recenter=True):
        if recenter:
            x = x + self.mean
        return x

    def shortest_side(self, x):
        neg = x < 0.
        dcm = x % ((-1)**neg)
        p = (-1) ** (dcm < (0.5*(-1)**neg))
        x_fold = p * torch.div(x, (-1)**neg, rounding_mode='floor') + dcm
        return x_fold

    def generate_rand_init(self, pred_composition_per_atom, pred_lengths,
                           pred_angles, num_atoms, batch):
        rand_frac_coords = torch.rand(num_atoms.sum(), 3,
                                      device=num_atoms.device)
        pred_composition_per_atom = F.softmax(pred_composition_per_atom,
                                              dim=-1)
        rand_atom_types = self.sample_composition(
            pred_composition_per_atom, num_atoms)
        return rand_frac_coords, rand_atom_types

    def sample_composition(self, composition_prob, num_atoms):
        """
        Samples composition such that it exactly satisfies composition_prob
        """
        batch = torch.arange(
            len(num_atoms), device=num_atoms.device).repeat_interleave(num_atoms)
        assert composition_prob.size(0) == num_atoms.sum() == batch.size(0)
        composition_prob = scatter(
            composition_prob, index=batch, dim=0, reduce='mean')

        all_sampled_comp = []

        for comp_prob, num_atom in zip(list(composition_prob), list(num_atoms)):
            comp_num = torch.round(comp_prob * num_atom)
            atom_type = torch.nonzero(comp_num, as_tuple=True)[0] + 1
            atom_num = comp_num[atom_type - 1].long()

            sampled_comp = atom_type.repeat_interleave(atom_num, dim=0)

            # if the rounded composition gives less atoms, sample the rest
            if sampled_comp.size(0) < num_atom:
                left_atom_num = num_atom - sampled_comp.size(0)

                left_comp_prob = comp_prob - comp_num.float() / num_atom

                left_comp_prob[left_comp_prob < 0.] = 0.
                left_comp = torch.multinomial(
                    left_comp_prob, num_samples=left_atom_num, replacement=True)
                # convert to atomic number
                left_comp = left_comp + 1
                sampled_comp = torch.cat([sampled_comp, left_comp], dim=0)

            sampled_comp = sampled_comp[torch.randperm(sampled_comp.size(0))]
            sampled_comp = sampled_comp[:num_atom]
            all_sampled_comp.append(sampled_comp)

        all_sampled_comp = torch.cat(all_sampled_comp, dim=0)
        assert all_sampled_comp.size(0) == num_atoms.sum()
        return all_sampled_comp

    def predict_num_atoms(self, z):
        return self.fc_num_atoms(z)

    def predict_property(self, z):
        self.scaler.match_device(z)
        return self.scaler.inverse_transform(self.fc_property(z))

    def predict_property_class(self, z):
        return torch.stack([self.fc_property_class[i](z) for i in range(self.len_prop_classes)], -1)

    def predict_lattice(self, z, num_atoms):
        self.lattice_scaler.match_device(z)
        pred_lengths_and_angles = self.fc_lattice(z)  # (N, 6)
        scaled_preds = self.lattice_scaler.inverse_transform(
            pred_lengths_and_angles)
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:, 3:]
        if self.hparams.data.lattice_scale_method == 'scale_length':
            pred_lengths = pred_lengths * num_atoms.view(-1, 1).float() ** (1 / 3)
        # <pred_lengths_and_angles> is scaled.
        return pred_lengths_and_angles, pred_lengths, pred_angles

    def predict_composition(self, z, num_atoms):
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        pred_composition_per_atom = self.fc_composition(z_per_atom)
        return pred_composition_per_atom
    
    def l2_loss(self, eps_out, eps, batch_idx, norm=False, is_nodewise=True):
        if norm:
            denorm = self.diffalgo.n_dims * eps_out.shape[1]
        else:
            denorm = 1.
            
        loss = torch.sum((eps_out - eps) ** 2 / denorm, dim=1)        
        if is_nodewise:
            loss = scatter(loss, batch_idx, reduce='mean').mean()
        else:
            loss = loss.mean()
        return loss 

    def num_atom_loss(self, pred_num_atoms, num_atoms):
        return F.cross_entropy(pred_num_atoms, num_atoms)

    def lattice_loss(self, pred_lengths_and_angles, batch):
        self.lattice_scaler.match_device(pred_lengths_and_angles)
        if self.hparams.data.lattice_scale_method == 'scale_length':
            target_lengths = batch.lengths / \
                             batch.num_atoms.view(-1, 1).float() ** (1 / 3)
        target_lengths_and_angles = torch.cat(
            [target_lengths, batch.angles], dim=-1)
        target_lengths_and_angles = self.lattice_scaler.transform(
            target_lengths_and_angles)
        return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    def composition_loss(self, pred_composition_per_atom, target_atom_types, batch_idx):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(pred_composition_per_atom,
                               target_atom_types, reduction='none')
        return scatter(loss, batch_idx, reduce='mean').mean()

    def coord_loss(self, pred_coord_diff, noisy_coords,
                   alpha_t, beta_t, batch, loss_type='coord'):
        alpha_t, beta_t = alpha_t.unsqueeze(-1), beta_t.unsqueeze(-1)
        noisy_coords = frac_to_cart_coords(noisy_coords, batch.lengths,
                                           batch.angles, batch.num_atoms)
        noisy_coords = self.add_mean(noisy_coords, recenter=self.recenter)
        target_coords = frac_to_cart_coords(batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        _, target_coord_diff = min_distance_sqr_pbc(target_coords, noisy_coords,
                                                    batch.lengths, batch.angles,
                                                    batch.num_atoms, self.device,
                                                    return_vector=True)
        target_coord_diff = (target_coord_diff - (sqrt(alpha_t) - 1) * target_coords)            
        target_coord_diff = self.coords(target_coord_diff, batch, input_type='cart')

        loss_per_atom = torch.sum(
            (target_coord_diff - pred_coord_diff) ** 2, dim=1)
        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()

    def type_loss(self, pred_atom_types, target_atom_types,
                  sigma_type_t, batch_idx):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(
            pred_atom_types, target_atom_types, reduction='none')
        # rescale loss according to noise
        loss = loss / sigma_type_t
        return scatter(loss, batch_idx, reduce='mean').mean()

    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        return kld_loss

    def sample(self, num_samples, ld_kwargs):
        z = torch.randn(num_samples, self.hparams.hidden_dim,
                        device=self.device)
        samples = self.langevin_dynamics(z, ld_kwargs)
        return samples


class DPCDVAE(BaseModel):
    def __init__(self, encoder, param_decoder, diffalgo, diffnet, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.hparams = cfg.model
        self.hparams.data = cfg.data
        self.hparams.algo = "dpcdvae"
        self.model_name = "dpcdvae"
        self.recenter = cfg.model.recenter
        self.logs = {'train': [], 'val': [], 'test': []}
        self.T = self.hparams.num_noise_level
        
        self.encoder = encoder
        self.param_decoder = param_decoder
        self.diffalgo = diffalgo
        self.diffnet = diffnet      

        self.fc_mu = nn.Linear(self.hparams.latent_dim,
                               self.hparams.latent_dim)
        self.fc_var = nn.Linear(self.hparams.latent_dim,
                                self.hparams.latent_dim)

        self.embedding = torch.zeros(100, 92)
        for i in range(100):
            self.embedding[i] = torch.tensor(KHOT_EMBEDDINGS[i + 1])

        # obtain from datamodule.
        self.lattice_scaler = None
        self.scaler = None        

    def forward(self, batch, teacher_forcing, training, model_classifier=None):
        batch_idx = batch.batch
        atom_types = batch.atom_types
        num_atoms = batch.num_atoms
        
        # Encode        
        hidden = self.encoder(batch)
        mu, log_var, z = self.kld_reparam(hidden)
        
        # Decode lattice, num_atoms, and composition (without perturbing noise)
        (pred_num_atoms, pred_lengths_and_angles, pred_lengths, pred_angles,
         pred_composition_per_atom) = self.param_decoder(
            z, batch.num_atoms, batch.lengths, batch.angles, teacher_forcing)
        
        # Perturb features
        frac_coords = self.sub_mean(batch.frac_coords, batch.batch, batch.num_atoms, recenter=self.recenter)
        composition_probs = F.softmax(pred_composition_per_atom.detach(), dim=-1)
        noisy_feats, eps_r, times = self.diffalgo.perturb_sample(frac_coords,
                                                                atom_types,
                                                                composition_probs,
                                                                num_atoms)        
        r_t, A_t = noisy_feats
        t_norm = times[0] / self.T
        t_type = times[1].repeat_interleave(num_atoms, dim=0)
        sigma_type_t = self.diffalgo.type_sigmas[t_type]

        # Predict noise and atomic types
        Z_t = torch.multinomial(A_t, num_samples=1).squeeze(1) + 1
        eps_theta, A_theta = self.diffnet(z, t_norm, r_t, Z_t, 
                                          num_atoms,
                                          pred_lengths, 
                                          pred_angles)

        # Compute losses
        ## eps and type losses
        coord_loss = self.l2_loss(eps_theta, eps_r, batch_idx)
        type_loss = self.type_loss(A_theta, 
                                   atom_types,
                                   sigma_type_t, 
                                   batch_idx)
        
        ## Other losses
        num_atom_loss = self.num_atom_loss(pred_num_atoms, num_atoms)
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, batch)
        composition_loss = self.composition_loss(
            pred_composition_per_atom, batch.atom_types, batch_idx)
        
        ## KLD loss
        kld_loss = self.kld_loss(mu, log_var)

        return {
            'num_atom_loss': num_atom_loss,
            'lattice_loss': lattice_loss,
            'composition_loss': composition_loss,
            'coord_loss': coord_loss,
            'type_loss': type_loss,
            'kld_loss': kld_loss,
            'pred_num_atoms': pred_num_atoms,
            'pred_lengths_and_angles': pred_lengths_and_angles,
            'pred_lengths': pred_lengths,
            'pred_angles': pred_angles,
            'pred_cart_coord_diff': eps_theta,
            'pred_atom_types': A_theta,
            'pred_composition_per_atom': pred_composition_per_atom,
            'target_frac_coords': batch.frac_coords,
            'target_atom_types': batch.atom_types,
            'rand_frac_coords': r_t,
            'rand_atom_types': Z_t,
            'z': z,
        }

    @torch.no_grad()
    def langevin_dynamics(self, z, ld_kwargs, gt_num_atoms=None, gt_atom_types=None, model_classifier=None,
                          labels=None):
        """
        decode crystral structure from latent embeddings.
        ld_kwargs: args for doing annealed langevin dynamics sampling:
            n_step_each:  number of steps for each sigma level.
            step_lr:      step size param.
            min_sigma:    minimum sigma to use in annealed langevin dynamics.
            save_traj:    if <True>, save the entire LD trajectory.
            disable_bar:  disable the progress bar of langevin dynamics.
        gt_num_atoms: if not <None>, use the ground truth number of atoms.
        gt_atom_types: if not <None>, use the ground truth atom types.
        """
        if ld_kwargs.save_traj:
            all_frac_coords = []
            all_unbounded_coords = []
            all_pred_cart_coord_diff = []
            all_noise_cart = []
            all_atom_types = []

        num_atoms, _, lengths, angles, A_t = self.param_decoder(z, gt_num_atoms)
        if gt_num_atoms is not None:
            num_atoms = gt_num_atoms
        batch_idx = torch.arange(z.shape[0], device=z.device).repeat_interleave(num_atoms)

        # init coords.
        r_t, _ = self.diffalgo.get_sample_noise(num_atoms, gt_atom_types)
        rf_t = bound_frac(r_t)

        # obtain atom types.
        A_t = F.softmax(A_t, dim=-1)
        if gt_atom_types is None:
            Z_t = self.sample_composition(A_t, num_atoms)
        else:
            Z_t = gt_atom_types
        
        # Predictor 
        t_min = 1
        for j in tqdm(reversed(range(t_min, self.T)), total=(self.T - t_min), disable=ld_kwargs.disable_bar):
            t = torch.tensor([j] * num_atoms.shape[0]).long().to(z.device)
            eps_r, _ = self.diffalgo.get_sample_noise(num_atoms, gt_atom_types)
            a, b, sigma = self.get_predictor_params(t.repeat_interleave(num_atoms, dim=0), 
                                                    predictor='modified', v=0)
            
            # Denoise fractional coordinates
            eps_theta, A_t = self.diffnet(
                z, t / self.T, r_t, Z_t, num_atoms, lengths, angles)
            
            rf_t = self.sub_mean(rf_t, batch_idx, num_atoms, recenter=self.recenter)
            first_term = a * rf_t
            second_term = b * eps_theta
            third_term = sigma * eps_r
            r_t = first_term + second_term + third_term
            rf_t = bound_frac(r_t)

            if gt_atom_types is None:
                Z_t = torch.argmax(A_t, dim=1) + 1

            if ld_kwargs.save_traj:
                all_frac_coords.append(rf_t)
                all_unbounded_coords.append(r_t)
                all_pred_cart_coord_diff.append(second_term)
                all_noise_cart.append(third_term)
                all_atom_types.append(A_t)

        if gt_atom_types is not None:
            A_t = F.one_hot(gt_atom_types - 1, num_classes=MAX_ATOMIC_NUM)

        output_dict = {'num_atoms': num_atoms, 'lengths': lengths, 'angles': angles,
                       'frac_coords': rf_t, 'atom_types': A_t,
                       'unbounded_coords': r_t, 'is_traj': False}

        if ld_kwargs.save_traj:
            output_dict.update(dict(
                all_frac_coords=torch.stack(all_frac_coords, dim=0),
                all_unbounded_coords=torch.stack(all_unbounded_coords, dim=0),
                all_atom_types=torch.stack(all_atom_types, dim=0),
                all_second_terms=torch.stack(
                    all_pred_cart_coord_diff, dim=0),
                all_third_terms=torch.stack(all_noise_cart, dim=0),
                is_traj=True))
        
        return output_dict    
           
    def noise_params(self, t, expand_dim=False):
        assert t[t==0].sum() == 0
        s = t - 1
        alpha_s = self.diffalgo.alphas[s]
        beta_s = self.diffalgo.betas[s]
        alpha_t = self.diffalgo.alphas[t]
        beta_t = self.diffalgo.betas[t]
        if expand_dim:
            return alpha_s.unsqueeze(-1), alpha_t.unsqueeze(-1), beta_s.unsqueeze(-1), beta_t.unsqueeze(-1)
        else:
            return alpha_s, alpha_t, beta_s, beta_t
        
    def get_predictor_params(self, t, v=0, predictor='modified'):
        alpha_s, alpha_t, _, beta_t = self.noise_params(t, expand_dim=True)        
        if predictor == 'normal': 
            a = 1 / sqrt(1 - beta_t)
            b = -a * beta_t / sqrt(1 - alpha_t)            
        elif predictor == 'modified': 
            # Use alpha_t (cumprod) instead of 1 - beta_t
            a = 1 / alpha_t.sqrt()
            b = -a * (1 - alpha_t).sqrt()             
        sigma = v * sqrt(beta_t) + (1-v) * sqrt(((1 - alpha_s) / (1 - alpha_t)) * beta_t)
        return a, b, sigma
    