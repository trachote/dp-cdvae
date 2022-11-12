from typing import Any, Dict

# import hydra
import numpy as np
import omegaconf
import torch
# import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm


from common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume,
    frac_to_cart_coords, min_distance_sqr_pbc)

from models.cdvae.embeddings import MAX_ATOMIC_NUM
from models.cdvae.embeddings import KHOT_EMBEDDINGS
from models.cdvae.gnn import DimeNetPlusPlusWrap
from models.cdvae.decoder import GemNetTDecoder

from models.model import MODEL
from models.cdvae.model import CDVAE, build_mlp



def get_score_fn(sde, model, train=False, continuous=False):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        model: A score model.
        train: `True` for training and `False` for evaluation.
        continuous: If `True`, the score-based model is expected to directly take continuous time steps.

    Returns:
        A score function.
    """
#   model_fn = get_model_fn(model, train=train)
    
    model_fn = model

    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
        def score_fn(x, t):
        # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, sde_lib.subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                score, _ = model_fn(x, labels)
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.N - 1)
                score = model_fn(x, labels)
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

            score = -score / std[:, None, None, None]
            return score

    elif isinstance(sde, sde_lib.VESDE):
        def score_fn(x, t):
            if continuous:
                labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = sde.T - t
                labels *= sde.N - 1
                labels = torch.round(labels).long()

            score, _ = model_fn(x, labels)
            return score, _

    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    return score_fn


def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
    """Create a loss function for training with arbirary SDEs.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        train: `True` for training loss and `False` for evaluation loss.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
        likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
        eps: A `float` number. The smallest time step to sample from.

    Returns:
        A loss function.
    """

    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def forward_fn(model, frac_coords, pred_lengths, pred_angles, num_atoms):
        # score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
        score_fn = get_score_fn(sde, model, train=train, continuous=continuous)    
        t = torch.rand(frac_coords.shape[0], device=frac_coords.device) * (sde.T - eps) + eps
        z = torch.randn_like(frac_coords)

        cart_coords = frac_to_cart_coords(frac_coords, pred_lengths, pred_angles, num_atoms)

        mean, std = sde.marginal_prob(cart_coords, t)
        perturbed_data = mean + std[:, None] * z

        noisy_frac_coords = cart_to_frac_coords(perturbed_data, pred_lengths, pred_angles, num_atoms) 

        score, _ = score_fn(noisy_frac_coords, t)

        return {'pred_cart_coor_diff':score, 'pred_atom_types':_, 'noisy_frac_coords':noisy_frac_coords, 'std':std, 't':t}

    def loss_fn(params, batch):

        noisy_cart_coords = frac_to_cart_coords(params['noisy_frac_coords'], batch.lengths, batch.angles, batch.num_atoms)
        target_cart_coords = frac_to_cart_coords(batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        _, target_cart_coord_diff = min_distance_sqr_pbc(
            target_cart_coords, noisy_cart_coords, batch.lengths, batch.angles, batch.num_atoms, device=batch.frac_coords.device, return_vector=True
        )

        if not likelihood_weighting:
            target_cart_coord_diff = target_cart_coord_diff / params['std'][:, None]
            pred_cart_coord_diff = params['pred_cart_coor_diff']

            # losses = torch.square(score * params['std'][:, None] + z)
            # losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
            loss_per_atom = 0.5 * torch.sum((target_cart_coord_diff - pred_cart_coord_diff)**2, dim=1)
        else:
            target_cart_coord_diff = target_cart_coord_diff / params['std'][:, None]**2
            pred_cart_coord_diff = params['pred_cart_coor_diff'] / params['std'][:, None]
            g2 = sde.sde(torch.zeros_like(batch), params['t'])[1] ** 2
            # losses = torch.square(score + z / std[:, None])
            loss_per_atom = 0.5 * torch.sum((target_cart_coord_diff - pred_cart_coord_diff)**2, dim=1) * g2
            # losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        # loss = torch.mean(losses)
        # return loss

        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()



    return forward_fn, loss_fn


def get_smld_loss_fn(vesde, train, reduce_mean=False):
    """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
    assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

    # Previous SMLD models assume descending sigmas
    smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def forward_fn(model, frac_coords, pred_lengths, pred_angles, num_atoms):

        model_fn = model

        labels = torch.randint(0, vesde.N, (frac_coords.shape[0],), device=frac_coords.device)
        sigmas = smld_sigma_array.to(frac_coords.device)[labels]
        noise = torch.randn_like(frac_coords) * sigmas[:, None] #cart_noises_per_atom

        cart_coords = frac_to_cart_coords(frac_coords, pred_lengths, pred_angles, num_atoms)

        perturbed_data = noise + cart_coords #cart_coords = cart_coords + cart_noises_per_atom

        noisy_frac_coords = cart_to_frac_coords(perturbed_data, pred_lengths, pred_angles, num_atoms)        

        score, _ = model_fn(noisy_frac_coords, labels)

        return {'pred_cart_coor_diff':score, 'pred_atom_types':_, 'noisy_frac_coords':noisy_frac_coords, 'sigmas':sigmas}

    def loss_fn(params, batch):

        noisy_cart_coords = frac_to_cart_coords(params['noisy_frac_coords'], batch.lengths, batch.angles, batch.num_atoms)
        target_cart_coords = frac_to_cart_coords(batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        _, target_cart_coord_diff = min_distance_sqr_pbc(
            target_cart_coords, noisy_cart_coords, batch.lengths, batch.angles, batch.num_atoms, device=batch.frac_coords.device, return_vector=True
        )

        target_cart_coord_diff = target_cart_coord_diff / params['sigmas'][:, None]**2
        pred_cart_coord_diff = params['pred_cart_coor_diff'] / params['sigmas'][:, None]
 
        loss_per_atom = 0.5 * torch.sum((target_cart_coord_diff - pred_cart_coord_diff)**2, dim=1) * params['sigmas']**2

        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()

        # target = -target_cart_coord_diff / (params['sigmas']** 2)[:, None]
        # pred_cart_coor_diff = params['pred_cart_coor_diff'] / params['sigmas'][:, None]
        # loss_per_atom = 0.5 * torch.sum(torch.square(pred_cart_coor_diff - target), dim=1) * params['sigmas'] ** 2
        # return scatter(loss_per_atom, batch.batch, reduce='mean').mean()

        # losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * params['sigmas'] ** 2
        # loss = torch.mean(losses)
        # return loss

    return forward_fn, loss_fn


def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
    """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
    assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch):
        # model_fn = mutils.get_model_fn(model, train=train)
        model_fn = model

        labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
        noise = torch.randn_like(batch)
        perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                         sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
        score = model_fn(perturbed_data, labels)
        losses = torch.square(score - noise)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = torch.mean(losses)
        return loss

    return loss_fn



class UNCOND_CDVAE_(CDVAE):

    def __init__(self, hparams):
        super().__init__(hparams)
        params = {k:v for k, v in self.hparams.decoder.items() if k!= '_target_'}
        params['lengths_and_angles'] = True
        self.decoder = GemNetTDecoder(**params)

        self.model_classifier = None

        # self.sde_loss_fn = None
        # self.sde_forward_fn = None
        # if self.hparams.algo == "sde":
            
        #     if self.hparams.continuous:
        #         self.sde_forward_fn, self.sde_loss_fn = get_sde_loss_fn(self.hparams.sde, self.hparams.train, reduce_mean=self.hparams.reduce_mean,
        #                                 continuous=True, likelihood_weighting=self.hparams.likelihood_weighting)
        #     else:
        #         assert not self.hparams.likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
        #         if isinstance(self.hparams.sde, VESDE):
        #             self.sde_forward_fn, self.sde_loss_fn = get_smld_loss_fn(self.hparams.sde, self.hparams.train, reduce_mean=self.hparams.reduce_mean)
        #         elif isinstance(self.hparams.sde, VPSDE):
        #             self.sde_forward_fn, self.sde_loss_fn  = get_ddpm_loss_fn(self.hparams.sde, self.hparams.train, reduce_mean=self.hparams.reduce_mean)
        #         else:
        #             raise ValueError(f"Discrete training for {self.hparams.sde.__class__.__name__} is not recommended.")        

        # lattice_type= 7
        # params = {k:v for k, v in self.hparams.encoder.items() if k!= '_target_'}
        # params['num_targets'] = self.hparams.latent_dim
        # self.lattice_type_encoder = DimeNetPlusPlusWrap(**params)
        # self.fc_lattice_type = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim, self.hparams.fc_num_layers, 7)


    # def predict_lattice_type(self, batch):
    #     hidden = self.lattice_type_encoder(batch)
    #     return self.fc_lattice_type(hidden)

    @torch.no_grad()
    def langevin_dynamics(self, z, ld_kwargs, gt_num_atoms=None, gt_atom_types=None, labels=None):

        if ld_kwargs.save_traj:
            all_frac_coords = []
            all_pred_cart_coord_diff = []
            all_noise_cart = []
            all_atom_types = []

        num_atoms, _, lengths, angles, composition_per_atom = self.decode_stats(z, gt_num_atoms)

        if gt_num_atoms is not None:
            num_atoms = gt_num_atoms

        # obtain atom types.
        composition_per_atom = F.softmax(composition_per_atom, dim=-1)
        if gt_atom_types is None:
            cur_atom_types = self.sample_composition(
                composition_per_atom, num_atoms)
        else:
            cur_atom_types = gt_atom_types

        # init coords.
        cur_frac_coords = torch.rand((num_atoms.sum(), 3), device=z.device)

        # init coords.
        cur_lengths = lengths
        cur_angles = angles

        if self.model_classifier is not None:
            def get_update_fn(model_classifier, labels):
                def update_fn(pred_cart_coord_diff, sigma, t, frac_coords, atom_types, num_atoms, lengths, angles):
                    # t คือ noise scale
                    with torch.enable_grad():
                        frac_coords = frac_coords.requires_grad_(True)
                        logits = model_classifier(t, frac_coords, atom_types, num_atoms, lengths, angles)
                        y = torch.nn.functional.log_softmax(logits, -1)[:, labels].sum()
                        grad,  = torch.autograd.grad(y, frac_coords)
                        # หรือ 
                        # y.backward()
                        # return pred_cart_coord_diff / sigma + cur_frac_coords.grad
                    return pred_cart_coord_diff / sigma + grad

                return update_fn

            update_fn = get_update_fn(self.model_classifier, labels)
        else:
            def update_fn(pred_cart_coord_diff, sigma, *args):
                return pred_cart_coord_diff / sigma 
            
    
        one_vec = torch.ones((num_atoms.size(0), 1), device=z.device)
        # annealed langevin dynamics.
        for t, sigma in tqdm(enumerate(self.sigmas), total=self.sigmas.size(0), disable=ld_kwargs.disable_bar):
        # for sigma in tqdm(self.sigmas, total=self.sigmas.size(0), disable=ld_kwargs.disable_bar):

            if sigma < ld_kwargs.min_sigma:
                break
            step_size = ld_kwargs.step_lr * (sigma / self.sigmas[-1]) ** 2

            for step in range(ld_kwargs.n_step_each):

                noise_cart = torch.randn_like(cur_frac_coords) * torch.sqrt(step_size * 2)
                pred_cart_coord_diff, pred_atom_types, pred_lengths_and_angles = self.decoder(z, cur_frac_coords, cur_atom_types, num_atoms, cur_lengths, cur_angles)

                # rescale
                self.lattice_scaler.match_device(z)
                scaled_preds = self.lattice_scaler.inverse_transform(pred_lengths_and_angles)
                cur_lengths = scaled_preds[:, :3]
                cur_angles = scaled_preds[:, 3:]
                if self.hparams.data.lattice_scale_method == 'scale_length':
                    cur_lengths = cur_lengths * num_atoms.view(-1, 1).float()**(1/3)
                
                if gt_atom_types is None:
                    cur_atom_types = torch.argmax(pred_atom_types, dim=1) + 1

                pred_cart_coord_diff = update_fn(pred_cart_coord_diff, sigma, one_vec*t, cur_frac_coords, cur_atom_types, num_atoms, cur_lengths, cur_angles)


                cur_cart_coords = frac_to_cart_coords(cur_frac_coords, cur_lengths, cur_angles, num_atoms)
                cur_cart_coords = cur_cart_coords + step_size * pred_cart_coord_diff + noise_cart
                cur_frac_coords = cart_to_frac_coords(cur_cart_coords, cur_lengths, cur_angles, num_atoms)

                if gt_atom_types is None:
                    cur_atom_types = torch.argmax(pred_atom_types, dim=1) + 1

                if ld_kwargs.save_traj:
                    all_frac_coords.append(cur_frac_coords)
                    all_pred_cart_coord_diff.append(
                        step_size * pred_cart_coord_diff)
                    all_noise_cart.append(noise_cart)
                    all_atom_types.append(cur_atom_types)

        output_dict = {'num_atoms': num_atoms, 'lengths': cur_lengths, 'angles': cur_angles,
                       'frac_coords': cur_frac_coords, 'atom_types': cur_atom_types,
                       'is_traj': False}

        if ld_kwargs.save_traj:
            output_dict.update(dict(
                all_frac_coords=torch.stack(all_frac_coords, dim=0),
                all_atom_types=torch.stack(all_atom_types, dim=0),
                all_pred_cart_coord_diff=torch.stack(
                    all_pred_cart_coord_diff, dim=0),
                all_noise_cart=torch.stack(all_noise_cart, dim=0),
                is_traj=True))

        return output_dict
 
    # def sde_decoder(self, z, noisy_frac_coords, rand_atom_types, num_atoms, pred_lengths, pred_angles):

    # def model(perturbed_data, t):
    #     # นำ t ไปใช้ด้วย?
    #     return self.decoder(z, noisy_frac_coords, rand_atom_types, num_atoms, pred_lengths, pred_angles)

    # return self.sde_loss_fn(model, noisy_frac_coords)


    def forward(self, batch, teacher_forcing, training):
        mu, log_var, z = self.encode(batch)

        (pred_num_atoms, pred_lengths_and_angles, pred_lengths, pred_angles, pred_composition_per_atom) = self.decode_stats(z, batch.num_atoms, batch.lengths, batch.angles, teacher_forcing)

        # sample noise levels.
        noise_level = torch.randint(0, self.sigmas.size(0), (batch.num_atoms.size(0),), device=self.device)
        used_sigmas_per_atom = self.sigmas[noise_level].repeat_interleave(batch.num_atoms, dim=0)

        type_noise_level = torch.randint(0, self.type_sigmas.size(0), (batch.num_atoms.size(0),), device=self.device)
        used_type_sigmas_per_atom = (self.type_sigmas[type_noise_level].repeat_interleave(batch.num_atoms, dim=0))

        # add noise to atom types and sample atom types.
        pred_composition_probs = F.softmax(pred_composition_per_atom.detach(), dim=-1)

        atom_type_probs = (F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM) + pred_composition_probs * used_type_sigmas_per_atom[:, None])

        rand_atom_types = torch.multinomial(atom_type_probs, num_samples=1).squeeze(1) + 1


        # # sde
        # if self.sde_forward_fn is not None:

        #     params = self.sde_forward_fn(
        #         lambda x, labels: self.sde_decoder(z, x, rand_atom_types, batch.num_atoms, pred_lengths, pred_angles), 
        #         batch.frac_coords, pred_lengths, pred_angles, batch.num_atoms)

        #     noisy_frac_coords = params['noisy_frac_coords']
        #     pred_cart_coord_diff = params['pred_cart_coord_diff']
        #     pred_atom_types = params['pred_atom_types']
        # else:
        #     # add noise to the cart coords
        #     cart_noises_per_atom = (
        #         torch.randn_like(batch.frac_coords) *
        #         used_sigmas_per_atom[:, None])

        #     cart_coords = frac_to_cart_coords(
        #         batch.frac_coords, pred_lengths, pred_angles, batch.num_atoms)

        #     cart_coords = cart_coords + cart_noises_per_atom
        #     noisy_frac_coords = cart_to_frac_coords(
        #         cart_coords, pred_lengths, pred_angles, batch.num_atoms)

        #     pred_cart_coord_diff, pred_atom_types = self.decoder(
        #         z, noisy_frac_coords, rand_atom_types, batch.num_atoms, pred_lengths, pred_angles)


        # add noise to the cart coords
        cart_noises_per_atom = (torch.randn_like(batch.frac_coords) * used_sigmas_per_atom[:, None])

        cart_coords = frac_to_cart_coords(batch.frac_coords, pred_lengths, pred_angles, batch.num_atoms)

        cart_coords = cart_coords + cart_noises_per_atom
        noisy_frac_coords = cart_to_frac_coords(cart_coords, pred_lengths, pred_angles, batch.num_atoms)

        # น่าจะไม่ต้องใช้ teacher forcing แล้ว?
        pred_cart_coord_diff, pred_atom_types, pred_lengths_and_angles2 = self.decoder(z, noisy_frac_coords, rand_atom_types, batch.num_atoms, pred_lengths, pred_angles)


        if self.model_classifier is not None:
            # ตอนนี้มองว่าสิ่งที่ออกมาจาก decoder เป็นผลจากการ perturbed coords, atom_types, lengths_and_angles (มองว่าออกจาก cdvae ถือว่ามี noise อยู่)
            # langevin เริ่มจาก noise ของ coordinates เป็นหลัก ใช้ค่า predict จาก decoder ซึ่งทำการ denoise ทั้งหมดออกจากกัน
            # ตอนหา loss ก็ทำให้ค่าเข้าใกล้จริง  หรือควรจะใช้ค่าจริง?  แต่ถ้าให้สอดคล้องกับ langevin ใช้ค่า predict จาก decoder เลือกเป็นใช้ค่าจาก decoder ยกเว้น 500 รอบแรกเป็น teacher forcing

            self.lattice_scaler.match_device(pred_lengths_and_angles2)
            scaled = self.lattice_scaler.inverse_transform(pred_lengths_and_angles2)
            lengths = scaled[:, :3]
            angles = scaled[:, 3:]

            if self.hparams.data.lattice_scale_method == 'scale_length':
                lengths = lengths * batch.num_atoms.view(-1, 1).float()**(1/3)

            return self.model_classifier.loss(pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, batch, 
                noise_level[:, None], torch.argmax(pred_atom_types, dim=1) + 1, batch.num_atoms, lengths, angles
            )

        # compute loss.
        num_atom_loss = self.num_atom_loss(pred_num_atoms, batch)
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, batch)
        composition_loss = self.composition_loss(pred_composition_per_atom, batch.atom_types, batch)


        # if self.sde_loss_fn is not None:
        #     self.sde_loss_fn(params, batch)
        #     pass
        # else:
        #     coord_loss = self.coord_loss(pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, batch)  


        # lattice_loss2 = self.lattice_loss2(pred_lengths_and_angles2, batch)
        lattice_loss2 = self.lattice_loss(pred_lengths_and_angles2, batch)        

        coord_loss = self.coord_loss(pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, batch)

        type_loss = self.type_loss(pred_atom_types, batch.atom_types,
                                   used_type_sigmas_per_atom, batch)

        kld_loss = self.kld_loss(mu, log_var)

        # lattice_type_loss = self.lattice_type_loss(batch)

        # if self.hparams.predict_property_class:
        #     property_class_loss = self.property_class_loss(z, batch)
        # else:
        #     property_class_loss = 0.

        # if self.hparams.predict_property:
        #     property_loss = self.property_loss(z, batch)
        # else:
        #     property_loss = 0.

        return {
            'num_atom_loss': num_atom_loss,
            'lattice_loss': lattice_loss,
            'lattice_loss2': lattice_loss2,
            # 'lattice_type_loss':lattice_type_loss,
            'composition_loss': composition_loss,
            'coord_loss': coord_loss,
            'type_loss': type_loss,
            'kld_loss': kld_loss,
            # 'property_loss': property_loss,
            # 'property_class_loss': property_class_loss,
            'pred_num_atoms': pred_num_atoms,
            'pred_lengths_and_angles': pred_lengths_and_angles,
            'pred_lengths_and_angles2': pred_lengths_and_angles2,
            'pred_lengths': pred_lengths,
            'pred_angles': pred_angles,
            'pred_cart_coord_diff': pred_cart_coord_diff,
            'pred_atom_types': pred_atom_types,
            'pred_composition_per_atom': pred_composition_per_atom,
            'target_frac_coords': batch.frac_coords,
            'target_atom_types': batch.atom_types,
            'rand_frac_coords': noisy_frac_coords,
            'rand_atom_types': rand_atom_types,
            'z': z,
        }

    # def lattice_type_loss(self, batch):
    #     return F.cross_entropy(self.predict_lattice_type(batch), batch.z[:, 0])

    # def lattice_loss2(self, pred_lengths_and_angles, batch):
    #     # ไม่มีการ scale ให้ทนายเป็นค่าจริงเลย เพราะเอาไปใช้จริง และเริ่มมาจากค่าจริง
    #     target_lengths_and_angles = torch.cat([batch.lengths, batch.angles], dim=-1)        
    #     return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    # def classifier_loss(self, pred_cart_coord_diff, noise_level, frac_coords, atom_types, num_atoms, lengths_and_angles):

    #     self.lattice_scaler.match_device(lengths_and_angles)
    #     scaled = self.lattice_scaler.inverse_transform(lengths_and_angles)
    #     lengths = scaled[:, :3]
    #     angles = scaled[:, 3:]

    #     if self.hparams.data.lattice_scale_method == 'scale_length':
    #         lengths = lengths * num_atoms.view(-1, 1).float()**(1/3)

    #     return self.model_classifier(pred_cart_coord_diff, noise_level, frac_coords, atom_types, num_atoms, lengths, angles)


    def compute_stats(self, batch, outputs, prefix):
        num_atom_loss = outputs['num_atom_loss']
        lattice_loss = outputs['lattice_loss']
        lattice_loss2 = outputs['lattice_loss2']
        coord_loss = outputs['coord_loss']
        type_loss = outputs['type_loss']
        kld_loss = outputs['kld_loss']
        composition_loss = outputs['composition_loss']
        # property_loss = outputs['property_loss']
        # property_class_loss = outputs['property_class_loss']
        # lattice_type_loss = outputs['lattice_type_loss']

        loss = (
            self.hparams.cost_natom * num_atom_loss +
            self.hparams.cost_lattice * lattice_loss +
            self.hparams.cost_lattice * lattice_loss2 +
            self.hparams.cost_coord * coord_loss +
            self.hparams.cost_type * type_loss +
            self.hparams.beta * kld_loss +
            self.hparams.cost_composition * composition_loss# +
            # self.hparams.cost_property * property_loss +
            # self.hparams.cost_property_class * property_class_loss +
            # self.hparams.cost_property_class * lattice_type_loss 
        )

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_natom_loss': num_atom_loss,
            f'{prefix}_lattice_loss': lattice_loss,
            f'{prefix}_coord_loss': coord_loss,
            f'{prefix}_type_loss': type_loss,
            f'{prefix}_kld_loss': kld_loss,
            f'{prefix}_composition_loss': composition_loss,
            f'{prefix}_lattice_loss2': lattice_loss2,
            # f'{prefix}_lattice_type_loss': lattice_type_loss,
            # f'{prefix}_property_loss': property_loss, 
            # f'{prefix}_property_class_loss': property_class_loss,
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
            scaled_preds = self.lattice_scaler.inverse_transform(
                pred_lengths_and_angles)
            pred_lengths = scaled_preds[:, :3]
            pred_angles = scaled_preds[:, 3:]

            if self.hparams.data.lattice_scale_method == 'scale_length':
                pred_lengths = pred_lengths * \
                    batch.num_atoms.view(-1, 1).float()**(1/3)
            lengths_mard = mard(batch.lengths, pred_lengths)
            angles_mae = torch.mean(torch.abs(pred_angles - batch.angles))

            pred_volumes = lengths_angles_to_volume(pred_lengths, pred_angles)
            true_volumes = lengths_angles_to_volume(
                batch.lengths, batch.angles)
            volumes_mard = mard(true_volumes, pred_volumes)


            # evalute lattice prediction2.
            pred_lengths_and_angles2 = outputs['pred_lengths_and_angles2']
            scaled_preds2 = self.lattice_scaler.inverse_transform(pred_lengths_and_angles2)
            pred_lengths2 = scaled_preds2[:, :3]
            pred_angles2 = scaled_preds2[:, 3:]

            if self.hparams.data.lattice_scale_method == 'scale_length':
                pred_lengths2 = pred_lengths2 * batch.num_atoms.view(-1, 1).float()**(1/3)
            lengths_mard2 = mard(batch.lengths, pred_lengths2)
            angles_mae2 = torch.mean(torch.abs(pred_angles2 - batch.angles))

            pred_volumes2 = lengths_angles_to_volume(pred_lengths2, pred_angles2)
            volumes_mard2 = mard(true_volumes, pred_volumes2)


            # evaluate atom type prediction.
            pred_atom_types = outputs['pred_atom_types']
            target_atom_types = outputs['target_atom_types']
            type_accuracy = pred_atom_types.argmax(
                dim=-1) == (target_atom_types - 1)
            type_accuracy = scatter(type_accuracy.float(
            ), batch.batch, dim=0, reduce='mean').mean()

            log_dict.update({
                f'{prefix}_loss': loss,
                # f'{prefix}_property_loss': property_loss, 
                # f'{prefix}_property_class_loss': property_class_loss,
                f'{prefix}_natom_accuracy': num_atom_accuracy,
                f'{prefix}_lengths_mard': lengths_mard,
                f'{prefix}_angles_mae': angles_mae,
                f'{prefix}_volumes_mard': volumes_mard,
                f'{prefix}_type_accuracy': type_accuracy,

                f'{prefix}_lengths_mard2': lengths_mard2,
                f'{prefix}_angles_mae2': angles_mae2,
                f'{prefix}_volumes_mard2': volumes_mard2,

            })

        return log_dict, loss


class CLASSIFIER_CDVAE(torch.nn.Module):

    def __init__(self, hparams, num_classes):
        super().__init__()
        self.hparams = hparams
        self.fc_class = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim, self.hparams.fc_num_layers, num_classes)

    def forward(self, z):
        return self.fc_class(z)    


class NOISE_BASED_CLASSIFIER_CDVAE(torch.nn.Module):

    def __init__(self, hparams, num_classes):
        super().__init__()

        self.hparams = hparams
        params = {k:v for k, v in self.hparams.decoder.items() if k!= '_target_'}
        params['classifier'] = True
        params['num_targets'] = self.hparams.latent_dim
        params['latent_dim'] = 1 # noise level
        self.decoder = GemNetTDecoder(**params)
        self.fc_class = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim, self.hparams.fc_num_layers, num_classes)


    def forward(self, noise_levels, frac_coords, atom_types, num_atoms, lengths, angles):
        return self.fc_class(self.decoder(noise_levels, frac_coords, atom_types, num_atoms, lengths, angles))

    @torch.enable_grad()
    def grad_log_p_and_logits(self, batch, noise_level, frac_coords, atom_types, num_atoms, lengths, angles):
#        with torch.enable_grad():
        frac_coords = frac_coords.requires_grad_(True)
        logits = self(noise_level, frac_coords, atom_types, num_atoms, lengths, angles)
        # เลือกเฉพาะ class ที่เกี่ยวข้องออกมา ?        
        y = torch.nn.functional.log_softmax(logits, -1)[batch.z[:, 0]].sum()
        grad_log_p,  = torch.autograd.grad(y, frac_coords, retain_graph = True)

        return grad_log_p, logits

    def loss(self, pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, batch, noise_level, atom_types, num_atoms, lengths, angles):

        noisy_cart_coords = frac_to_cart_coords(noisy_frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        target_cart_coords = frac_to_cart_coords(batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        _, target_cart_coord_diff = min_distance_sqr_pbc(target_cart_coords, noisy_cart_coords, batch.lengths, batch.angles, batch.num_atoms, self.device, return_vector=True)
        target_cart_coord_diff = target_cart_coord_diff / used_sigmas_per_atom[:, None]**2
        pred_cart_coord_diff = pred_cart_coord_diff / used_sigmas_per_atom[:, None]

        grad_log_p, logits = self.grad_log_p_and_logits(batch, noise_level, noisy_frac_coords, atom_types, num_atoms, lengths, angles)

        loss_per_atom = torch.sum((target_cart_coord_diff - pred_cart_coord_diff - grad_log_p)**2, dim=1)

        loss_per_atom = 0.5 * loss_per_atom * used_sigmas_per_atom**2
        
        loss_dlsm = scatter(loss_per_atom, batch.batch, reduce='mean').mean()
        
        loss_ce = torch.nn.functional.cross_entropy(logits, batch.z[:, 0])

        return loss_dlsm, loss_ce, logits

class UNCOND_CDVAE2(CDVAE):

    def __init__(self, hparams):
        super().__init__(hparams)
        params = {k:v for k, v in self.hparams.decoder.items() if k!= '_target_'}
        params['lengths_and_angles'] = True
        self.decoder = GemNetTDecoder(**params)

        self.model_classifier = None

    @torch.no_grad()
    def langevin_dynamics(self, z, ld_kwargs, gt_num_atoms=None, gt_atom_types=None, labels=None):

        if ld_kwargs.save_traj:
            all_frac_coords = []
            all_pred_cart_coord_diff = []
            all_noise_cart = []
            all_atom_types = []

        num_atoms, _, lengths, angles, composition_per_atom = self.decode_stats(z, gt_num_atoms)

        if gt_num_atoms is not None:
            num_atoms = gt_num_atoms

        # obtain atom types.
        composition_per_atom = F.softmax(composition_per_atom, dim=-1)
        if gt_atom_types is None:
            cur_atom_types = self.sample_composition(
                composition_per_atom, num_atoms)
        else:
            cur_atom_types = gt_atom_types

        # init coords.
        cur_frac_coords = torch.rand((num_atoms.sum(), 3), device=z.device)

        # init coords.
        cur_lengths = lengths
        cur_angles = angles

        if self.model_classifier is not None:
            def get_update_fn(model_classifier, labels):
                def update_fn(pred_cart_coord_diff, sigma, t, frac_coords, atom_types, num_atoms, lengths, angles):
                    # t คือ noise scale
                    with torch.enable_grad():
                        frac_coords = frac_coords.requires_grad_(True)
                        logits = model_classifier(t, frac_coords, atom_types, num_atoms, lengths, angles)
                        y = torch.nn.functional.log_softmax(logits, -1)[:, labels].sum()
                        grad,  = torch.autograd.grad(y, frac_coords)
                        # หรือ 
                        # y.backward()
                        # return pred_cart_coord_diff / sigma + cur_frac_coords.grad
                    return pred_cart_coord_diff / sigma + grad

                return update_fn

            update_fn = get_update_fn(self.model_classifier, labels)
        else:
            def update_fn(pred_cart_coord_diff, sigma, *args):
                return pred_cart_coord_diff / sigma 
            
    
        one_vec = torch.ones((num_atoms.size(0), 1), device=z.device)
        # annealed langevin dynamics.
        for t, sigma in tqdm(enumerate(self.sigmas), total=self.sigmas.size(0), disable=ld_kwargs.disable_bar):
        # for sigma in tqdm(self.sigmas, total=self.sigmas.size(0), disable=ld_kwargs.disable_bar):

            if sigma < ld_kwargs.min_sigma:
                break
            step_size = ld_kwargs.step_lr * (sigma / self.sigmas[-1]) ** 2

            for step in range(ld_kwargs.n_step_each):

                noise_cart = torch.randn_like(cur_frac_coords) * torch.sqrt(step_size * 2)
                pred_cart_coord_diff, pred_atom_types, pred_lengths_and_angles = self.decoder(z, cur_frac_coords, cur_atom_types, num_atoms, cur_lengths, cur_angles)

                # rescale
                self.lattice_scaler.match_device(z)
                scaled_preds = self.lattice_scaler.inverse_transform(pred_lengths_and_angles)
                cur_lengths = scaled_preds[:, :3]
                cur_angles = scaled_preds[:, 3:]
                if self.hparams.data.lattice_scale_method == 'scale_length':
                    cur_lengths = cur_lengths * num_atoms.view(-1, 1).float()**(1/3)
                
                if gt_atom_types is None:
                    cur_atom_types = torch.argmax(pred_atom_types, dim=1) + 1

                pred_cart_coord_diff = update_fn(pred_cart_coord_diff, sigma, one_vec*t, cur_frac_coords, cur_atom_types, num_atoms, cur_lengths, cur_angles)


                cur_cart_coords = frac_to_cart_coords(cur_frac_coords, cur_lengths, cur_angles, num_atoms)
                cur_cart_coords = cur_cart_coords + step_size * pred_cart_coord_diff + noise_cart
                cur_frac_coords = cart_to_frac_coords(cur_cart_coords, cur_lengths, cur_angles, num_atoms)

                if gt_atom_types is None:
                    cur_atom_types = torch.argmax(pred_atom_types, dim=1) + 1

                if ld_kwargs.save_traj:
                    all_frac_coords.append(cur_frac_coords)
                    all_pred_cart_coord_diff.append(
                        step_size * pred_cart_coord_diff)
                    all_noise_cart.append(noise_cart)
                    all_atom_types.append(cur_atom_types)

        output_dict = {'num_atoms': num_atoms, 'lengths': cur_lengths, 'angles': cur_angles,
                       'frac_coords': cur_frac_coords, 'atom_types': cur_atom_types,
                       'is_traj': False}

        if ld_kwargs.save_traj:
            output_dict.update(dict(
                all_frac_coords=torch.stack(all_frac_coords, dim=0),
                all_atom_types=torch.stack(all_atom_types, dim=0),
                all_pred_cart_coord_diff=torch.stack(
                    all_pred_cart_coord_diff, dim=0),
                all_noise_cart=torch.stack(all_noise_cart, dim=0),
                is_traj=True))

        return output_dict
 

    def forward(self, batch, teacher_forcing, training):
        mu, log_var, z = self.encode(batch)

        (pred_num_atoms, pred_lengths_and_angles, pred_lengths, pred_angles, pred_composition_per_atom) = self.decode_stats(z, batch.num_atoms, batch.lengths, batch.angles, teacher_forcing)

        # sample noise levels.
        noise_level = torch.randint(0, self.sigmas.size(0), (batch.num_atoms.size(0),), device=self.device)
        used_sigmas_per_atom = self.sigmas[noise_level].repeat_interleave(batch.num_atoms, dim=0)

        type_noise_level = torch.randint(0, self.type_sigmas.size(0), (batch.num_atoms.size(0),), device=self.device)
        used_type_sigmas_per_atom = (self.type_sigmas[type_noise_level].repeat_interleave(batch.num_atoms, dim=0))

        # add noise to atom types and sample atom types.
        pred_composition_probs = F.softmax(pred_composition_per_atom.detach(), dim=-1)

        atom_type_probs = (F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM) + pred_composition_probs * used_type_sigmas_per_atom[:, None])

        rand_atom_types = torch.multinomial(atom_type_probs, num_samples=1).squeeze(1) + 1


        lattice_noise_level = torch.randint(0, self.sigmas.size(0), (batch.num_atoms.size(0),), device=self.device)

        used_lattice_sigmas = self.sigmas[lattice_noise_level]
        # pred_lengths = pred_lenths + (torch.randn_like(batch.frac_coords) * used_lattice_sigmas[:, None])
        # lattice_params_to_matrix_torch
        # lattice_matrix_to_params_to_matrix_torch

        # pred_lengths = torch.tensor([a, b, c])

        # pred_lengths_and_angles = 

        lengths = batch.lengths + pred_lengths * used_lattice_sigmas[:, None]
        angles = batch.angles + pred_angles * used_lattice_sigmas[:, None]

        # add noise to the cart coords
        cart_noises_per_atom = (torch.randn_like(batch.frac_coords) * used_sigmas_per_atom[:, None])

        cart_coords = frac_to_cart_coords(batch.frac_coords, pred_lengths, pred_angles, batch.num_atoms)

        cart_coords = cart_coords + cart_noises_per_atom
        noisy_frac_coords = cart_to_frac_coords(cart_coords, pred_lengths, pred_angles, batch.num_atoms)

        # น่าจะไม่ต้องใช้ teacher forcing แล้ว?
        pred_cart_coord_diff, pred_atom_types, pred_lengths_and_angles2 = self.decoder(z, noisy_frac_coords, rand_atom_types, batch.num_atoms, pred_lengths, pred_angles)


        if self.model_classifier is not None:
            # ตอนนี้มองว่าสิ่งที่ออกมาจาก decoder เป็นผลจากการ perturbed coords, atom_types, lengths_and_angles (มองว่าออกจาก cdvae ถือว่ามี noise อยู่)
            # langevin เริ่มจาก noise ของ coordinates เป็นหลัก ใช้ค่า predict จาก decoder ซึ่งทำการ denoise ทั้งหมดออกจากกัน
            # ตอนหา loss ก็ทำให้ค่าเข้าใกล้จริง  หรือควรจะใช้ค่าจริง?  แต่ถ้าให้สอดคล้องกับ langevin ใช้ค่า predict จาก decoder เลือกเป็นใช้ค่าจาก decoder ยกเว้น 500 รอบแรกเป็น teacher forcing

            self.lattice_scaler.match_device(pred_lengths_and_angles2)
            scaled = self.lattice_scaler.inverse_transform(pred_lengths_and_angles2)
            lengths = scaled[:, :3]
            angles = scaled[:, 3:]

            if self.hparams.data.lattice_scale_method == 'scale_length':
                lengths = lengths * batch.num_atoms.view(-1, 1).float()**(1/3)

            return self.model_classifier.loss(pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, batch, 
                noise_level[:, None], torch.argmax(pred_atom_types, dim=1) + 1, batch.num_atoms, lengths, angles
            )

        # compute loss.
        num_atom_loss = self.num_atom_loss(pred_num_atoms, batch)
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, batch)
        composition_loss = self.composition_loss(pred_composition_per_atom, batch.atom_types, batch)


        # if self.sde_loss_fn is not None:
        #     self.sde_loss_fn(params, batch)
        #     pass
        # else:
        #     coord_loss = self.coord_loss(pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, batch)  


        # lattice_loss2 = self.lattice_loss2(pred_lengths_and_angles2, batch)
        lattice_loss2 = self.lattice_loss(pred_lengths_and_angles2, batch)        

        coord_loss = self.coord_loss(pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, batch)

        type_loss = self.type_loss(pred_atom_types, batch.atom_types,
                                   used_type_sigmas_per_atom, batch)

        kld_loss = self.kld_loss(mu, log_var)

        # lattice_type_loss = self.lattice_type_loss(batch)

        # if self.hparams.predict_property_class:
        #     property_class_loss = self.property_class_loss(z, batch)
        # else:
        #     property_class_loss = 0.

        # if self.hparams.predict_property:
        #     property_loss = self.property_loss(z, batch)
        # else:
        #     property_loss = 0.

        return {
            'num_atom_loss': num_atom_loss,
            'lattice_loss': lattice_loss,
            'lattice_loss2': lattice_loss2,
            # 'lattice_type_loss':lattice_type_loss,
            'composition_loss': composition_loss,
            'coord_loss': coord_loss,
            'type_loss': type_loss,
            'kld_loss': kld_loss,
            # 'property_loss': property_loss,
            # 'property_class_loss': property_class_loss,
            'pred_num_atoms': pred_num_atoms,
            'pred_lengths_and_angles': pred_lengths_and_angles,
            'pred_lengths_and_angles2': pred_lengths_and_angles2,
            'pred_lengths': pred_lengths,
            'pred_angles': pred_angles,
            'pred_cart_coord_diff': pred_cart_coord_diff,
            'pred_atom_types': pred_atom_types,
            'pred_composition_per_atom': pred_composition_per_atom,
            'target_frac_coords': batch.frac_coords,
            'target_atom_types': batch.atom_types,
            'rand_frac_coords': noisy_frac_coords,
            'rand_atom_types': rand_atom_types,
            'z': z,
        }

    # def lattice_type_loss(self, batch):
    #     return F.cross_entropy(self.predict_lattice_type(batch), batch.z[:, 0])

    # def lattice_loss2(self, pred_lengths_and_angles, batch):
    #     # ไม่มีการ scale ให้ทนายเป็นค่าจริงเลย เพราะเอาไปใช้จริง และเริ่มมาจากค่าจริง
    #     target_lengths_and_angles = torch.cat([batch.lengths, batch.angles], dim=-1)        
    #     return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    # def classifier_loss(self, pred_cart_coord_diff, noise_level, frac_coords, atom_types, num_atoms, lengths_and_angles):

    #     self.lattice_scaler.match_device(lengths_and_angles)
    #     scaled = self.lattice_scaler.inverse_transform(lengths_and_angles)
    #     lengths = scaled[:, :3]
    #     angles = scaled[:, 3:]

    #     if self.hparams.data.lattice_scale_method == 'scale_length':
    #         lengths = lengths * num_atoms.view(-1, 1).float()**(1/3)

    #     return self.model_classifier(pred_cart_coord_diff, noise_level, frac_coords, atom_types, num_atoms, lengths, angles)


    def compute_stats(self, batch, outputs, prefix):
        num_atom_loss = outputs['num_atom_loss']
        lattice_loss = outputs['lattice_loss']
        lattice_loss2 = outputs['lattice_loss2']
        coord_loss = outputs['coord_loss']
        type_loss = outputs['type_loss']
        kld_loss = outputs['kld_loss']
        composition_loss = outputs['composition_loss']
        # property_loss = outputs['property_loss']
        # property_class_loss = outputs['property_class_loss']
        # lattice_type_loss = outputs['lattice_type_loss']

        loss = (
            self.hparams.cost_natom * num_atom_loss +
            self.hparams.cost_lattice * lattice_loss +
            self.hparams.cost_lattice * lattice_loss2 +
            self.hparams.cost_coord * coord_loss +
            self.hparams.cost_type * type_loss +
            self.hparams.beta * kld_loss +
            self.hparams.cost_composition * composition_loss# +
            # self.hparams.cost_property * property_loss +
            # self.hparams.cost_property_class * property_class_loss +
            # self.hparams.cost_property_class * lattice_type_loss 
        )

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_natom_loss': num_atom_loss,
            f'{prefix}_lattice_loss': lattice_loss,
            f'{prefix}_coord_loss': coord_loss,
            f'{prefix}_type_loss': type_loss,
            f'{prefix}_kld_loss': kld_loss,
            f'{prefix}_composition_loss': composition_loss,
            f'{prefix}_lattice_loss2': lattice_loss2,
            # f'{prefix}_lattice_type_loss': lattice_type_loss,
            # f'{prefix}_property_loss': property_loss, 
            # f'{prefix}_property_class_loss': property_class_loss,
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
            scaled_preds = self.lattice_scaler.inverse_transform(
                pred_lengths_and_angles)
            pred_lengths = scaled_preds[:, :3]
            pred_angles = scaled_preds[:, 3:]

            if self.hparams.data.lattice_scale_method == 'scale_length':
                pred_lengths = pred_lengths * \
                    batch.num_atoms.view(-1, 1).float()**(1/3)
            lengths_mard = mard(batch.lengths, pred_lengths)
            angles_mae = torch.mean(torch.abs(pred_angles - batch.angles))

            pred_volumes = lengths_angles_to_volume(pred_lengths, pred_angles)
            true_volumes = lengths_angles_to_volume(
                batch.lengths, batch.angles)
            volumes_mard = mard(true_volumes, pred_volumes)


            # evalute lattice prediction2.
            pred_lengths_and_angles2 = outputs['pred_lengths_and_angles2']
            scaled_preds2 = self.lattice_scaler.inverse_transform(pred_lengths_and_angles2)
            pred_lengths2 = scaled_preds2[:, :3]
            pred_angles2 = scaled_preds2[:, 3:]

            if self.hparams.data.lattice_scale_method == 'scale_length':
                pred_lengths2 = pred_lengths2 * batch.num_atoms.view(-1, 1).float()**(1/3)
            lengths_mard2 = mard(batch.lengths, pred_lengths2)
            angles_mae2 = torch.mean(torch.abs(pred_angles2 - batch.angles))

            pred_volumes2 = lengths_angles_to_volume(pred_lengths2, pred_angles2)
            volumes_mard2 = mard(true_volumes, pred_volumes2)


            # evaluate atom type prediction.
            pred_atom_types = outputs['pred_atom_types']
            target_atom_types = outputs['target_atom_types']
            type_accuracy = pred_atom_types.argmax(
                dim=-1) == (target_atom_types - 1)
            type_accuracy = scatter(type_accuracy.float(
            ), batch.batch, dim=0, reduce='mean').mean()

            log_dict.update({
                f'{prefix}_loss': loss,
                # f'{prefix}_property_loss': property_loss, 
                # f'{prefix}_property_class_loss': property_class_loss,
                f'{prefix}_natom_accuracy': num_atom_accuracy,
                f'{prefix}_lengths_mard': lengths_mard,
                f'{prefix}_angles_mae': angles_mae,
                f'{prefix}_volumes_mard': volumes_mard,
                f'{prefix}_type_accuracy': type_accuracy,

                f'{prefix}_lengths_mard2': lengths_mard2,
                f'{prefix}_angles_mae2': angles_mae2,
                f'{prefix}_volumes_mard2': volumes_mard2,

            })

        return log_dict, loss
