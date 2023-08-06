from omegaconf import DictConfig, OmegaConf

from dpcdvae import models, diffusion_modules, encoders, decoders
from dpcdvae.decoders import decode_stats

def get_hparams(hparams):
    hparams = {k: v for k, v in hparams.items() if k != "_target_"}
    return hparams
    
def get_model(cfg):
    # Encoder
    if cfg.encoder._target_ == "DimeNetpp":
        encoder = encoders.DimeNetppEncoder(**get_hparams(cfg.encoder))
    elif cfg.encoder._target_ == "GemNetT":
        encoder = encoders.GemNetTEncoder(**get_hparams(cfg.encoder))
    elif cfg.encoder._target_ == "GINEncoder":
        encoder = encoders.GINEncoder(**get_hparams(cfg.encoder))
    elif cfg.encoder._target_ == "DimeGINEncoder":
        encoder = encoders.DimeGINEncoder(cfg.encoder)
    elif cfg.encoder._target_ == "DimeNaEncoder":
        encoder = encoders.DimeNaEncoder(cfg.encoder)
    elif cfg.encoder._target_ == "DimeGINNaEncoder":
        encoder = encoders.DimeGINNaEncoder(cfg.encoder)
    
    # Decode stats
    if cfg.param_decoder._target_ == "MLP":
        param_decoder = decode_stats.MLPDecodeStats(**get_hparams(cfg.param_decoder))
    elif cfg.decode_stats._target_ == "GINDecodeStats":
        param_decoder = decode_stats.GINDecodeStats(**get_hparams(cfg.param_decoder))
    elif cfg.decode_stats._target_ == "DimeDecodeStats":
        param_decoder = decode_stats.DimeDecodeStats(**get_hparams(cfg.param_decoder))
    elif cfg.decode_stats._target_ == "MLPDecodeAtoms":
        param_decoder = decode_stats.MLPDecodeAtoms(**get_hparams(cfg.param_decoder))
    
    # Noise model
    if cfg.diffalgo._target_ == "SBM": # score-based models
        diffalgo = diffusion_modules.SBM(**get_hparams(cfg.diffalgo))
    elif cfg.diffalgo._target_ == "DDPM":
        diffalgo = diffusion_modules.DDPM(**get_hparams(cfg.diffalgo))
    
    # Decoder
    if cfg.diffnet._target_ == "GemNetT":
        diffnet = decoders.GemNetTDecoder(**get_hparams(cfg.diffnet))
    
    # Model
    if cfg.model._target_ == "DPCDVAE":
        return models.DPCDVAE(encoder, param_decoder, diffalgo, diffnet, cfg)
        