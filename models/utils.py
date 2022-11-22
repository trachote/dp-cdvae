import os
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from .cdvae import CDVAE
from .decode_stats import MLPDecodeStats
from .noise_models import CDVAESDE, EDMSDE, get_noise_fn
from .encoders import DimeNetppEncoder, GemNetTEncoder
from .decoders import GemNetTDecoder 
from .sde_lib import VESDE, VPSDE

import torch
import math

# def get_model(cfg):
#     if cfg.model._target_=='CDVAE':
#         return CDVAE(cfg)
#     elif cfg.model._target_=='QNET':
#         return QNET(cfg)

def get_hparams(hparams):
    hparams = {k: v for k, v in hparams.items() if k != "_target_"}
    return hparams
    
def get_model(cfg):
    # Encoder
    if cfg.encoder._target_ == "DimeNetpp":
        encoder = DimeNetppEncoder(**get_hparams(cfg.encoder))
    elif cfg.encoder._target_ == "GemNetT":
        encoder = GemNetTEncoder(**get_hparams(cfg.encoder))
    
    # Decode stats
    if cfg.decode_stats._target_ == "MLP":
        decode_stats = MLPDecodeStats(**get_hparams(cfg.decode_stats))
    elif cfg.decode_stats._target_ == "DecodeStatsB":
        decode_stats = DecodeStatsB(**get_hparams(cfg.decode_stats))
    
    # Noise model
    if cfg.noise_model._target_ == "CDVAESDE":
        noise_model = CDVAESDE(**get_hparams(cfg.noise_model))
    elif cfg.noise_model._target_ == "EDMSDE":
        noise_model = EDMSDE(**get_hparams(cfg.noise_model))
    elif cfg.noise_model._target_ == "VESDE":
        noise_model = partial(get_noise_fn, VESDE)
    elif cfg.noise_model._target_ == "VPSDE":
        noise_model = partial(get_noise_fn, VPSDE)
    
    # Decoder
    if cfg.decoder._target_ == "GemNetT":
        decoder = GemNetTDecoder(**get_hparams(cfg.decoder))
    elif cfg.decoder._target_ == "DecoderB":
        decoder = DecoderB(**get_hparams(cfg.decoder))
        
    # Properties
    if cfg.prop_model._target_ == 'None':
        prop_model = None
    elif cfg.prop_model._target_ == "PropModel":
        prop_model = PropModel(**get_hparams(cfg.prop_model))
    
    return CDVAE(encoder, decode_stats, noise_model, decoder, cfg, prop_model)    
    

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    """
    From https://github.com/yang-song/score_sde_pytorch
    """
    #assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    #emb = timesteps.float()[:, None] * emb[None, :]
    emb = timesteps.float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

def default_init(scale=1.):
    """
    The same initialization used in DDPM.
    From https://github.com/yang-song/score_sde_pytorch
    """
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, 'fan_avg', 'uniform')

def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
    """Ported from JAX. """

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
            "invalid mode for variance scaling initializer: {}".format(mode))
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init
