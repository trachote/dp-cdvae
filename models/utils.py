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
        noise_model = CDVAESDE(cfg.noise_model)
    elif cfg.noise_model._target_ == "EDMSDE":
        noise_model = EDMSDE(cfg.noise_model)
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
    
