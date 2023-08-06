import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch_scatter import scatter_mean
from gnn.embeddings import MAX_ATOMIC_NUM

class SBM(nn.Module):
    def __init__(self, sigma_begin, sigma_end, type_sigma_begin, type_sigma_end, num_noise_level):  
        super().__init__()
        sigmas = torch.tensor(np.exp(np.linspace(
            np.log(sigma_begin),
            np.log(sigma_end),
            num_noise_level)), dtype=torch.float32)
        self.sigmas = nn.Parameter(sigmas, requires_grad=False)
        
        type_sigmas = torch.tensor(np.exp(np.linspace(
            np.log(type_sigma_begin),
            np.log(type_sigma_end),
            num_noise_level)), dtype=torch.float32)
        self.type_sigmas = nn.Parameter(type_sigmas, requires_grad=False)

        self.T = num_noise_level
                
    def perturb_sample(self, x, h, composition_probs, num_atoms, t=None):
        # Get noise level randomly -> equivalent to get time step randomly in DDPM
        if t is None:
            noise_level = torch.randint(0, self.sigmas.size(0),
                                        (num_atoms.size(0),),
                                        device=num_atoms.device)
            type_noise_level = torch.randint(0, self.type_sigmas.size(0),
                                             (num_atoms.size(0),),
                                             device=num_atoms.device)
        else:
            noise_level = torch.tensor([t]*num_atoms.size(0), device=num_atoms.device)
            type_noise_level = noise_level.clone()
        
        # Get fixed sigmas by noise level
        used_sigmas_per_atom = self.sigmas[noise_level].repeat_interleave(num_atoms, dim=0)       
        used_type_sigmas_per_atom = self.type_sigmas[type_noise_level].repeat_interleave(num_atoms, dim=0)
        
        self.t = None
        self.t_norm = None
        self.sigma_t  = used_sigmas_per_atom
        self.type_sigma_t = used_type_sigmas_per_atom
        
        # Add noise to coords
        x_noises = torch.randn_like(x) * used_sigmas_per_atom[:, None]
        noisy_x = x + x_noises
        
        # Add noise to atomic types
        atom_type_probs = F.one_hot(h - 1, num_classes=MAX_ATOMIC_NUM) +\
                          composition_probs * used_type_sigmas_per_atom[:, None]
        #rand_atom_types = torch.multinomial(atom_type_probs, num_samples=1).squeeze(1) + 1
        return (noisy_x, atom_type_probs), None, None

    def get_sample_noise(self, num_atoms, gt_atom_type):
        noise_cart = torch.rand((num_atoms.sum(), 3), device=num_atoms.device)
        return noise_cart, gt_atom_type
    
    def forward(self):
        return NotImplementedError

    
class DDPM(nn.Module):
    def __init__(self, in_node_nf, n_dims, timesteps, type_sigma_begin, type_sigma_end, 
                 noise_schedule, beta_start, beta_end, beta_init=-10, beta_final=6, 
                 same_time_step=False):
        super().__init__()
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.T = timesteps
        self.same_time_step = same_time_step
        
        type_sigmas = torch.tensor(np.exp(np.linspace(
            np.log(type_sigma_begin),
            np.log(type_sigma_end),
            timesteps)), dtype=torch.float32)
        self.type_sigmas = nn.Parameter(type_sigmas, requires_grad=False)
        
        # denoising diffusion
        ## betas
        betas = get_beta_schedule(
            beta_schedule=noise_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            num_diffusion_timesteps=timesteps,
            beta_init=beta_init,
            beta_final=beta_final,
        )
        betas = torch.from_numpy(betas).float()
        self.betas = nn.Parameter(betas, requires_grad=False)
        ## alphas_bar -> renamed to alphas
        alphas = 1. - betas
        alphas[1:] = alphas[1:].cumprod(dim=0)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.num_timesteps = self.betas.size(0)
        
        print("BETAS\n", self.betas[::20])
        print("\nALPHA_BARS\n", self.alphas[::20])

    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)
    
    def get_node_mask(self, num_atoms):
        node_mask = torch.arange(len(num_atoms), device=num_atoms.device)
        return node_mask.repeat_interleave(num_atoms).unsqueeze(-1)

    def sample_gaussian(self, size, device, num_atoms=None, remove_mean=False):
        x = torch.randn(size, device=device)
        if remove_mean:
            node_mask = self.get_node_mask(num_atoms)
            mean = scatter_mean(x, node_mask, dim=0)
            x = x - mean.repeat_interleave(num_atoms, dim=0)
        return x

    def perturb_sample(self, r, Z, composition_probs, num_atoms, t=None):
        # Sample a timestep t.                
        lowest_t = 1
        if t is None:
            t = torch.randint(
                lowest_t, self.T, size=(num_atoms.size(0),), device=r.device)
            if self.same_time_step:
                s = t
            else:
                s = torch.randint(
                    lowest_t, self.T, size=(num_atoms.size(0),), device=r.device)
        else:
            assert isinstance(t, int)
            t = torch.tensor([t]*num_atoms.size(0), device=r.device).long()
            s = t
        t_int = t.repeat_interleave(num_atoms, dim=0)
        s_int = s.repeat_interleave(num_atoms, dim=0)
        
        alpha_t = self.alphas[t_int][:, None]
        sigma_type_t = self.type_sigmas[s_int][:, None]
        
        eps_r = self.sample_gaussian(size=(r.size(0), self.n_dims), 
                                    device=num_atoms.device,
                                    num_atoms=num_atoms,
                                    remove_mean=True)
        
        r_t = torch.sqrt(alpha_t) * r + torch.sqrt(1 - alpha_t) * eps_r
        A_s = F.one_hot(Z - 1, num_classes=MAX_ATOMIC_NUM) \
                          + composition_probs * sigma_type_t
        return (r_t, A_s), eps_r, (t[:, None], s[:, None])

    def get_sample_noise(self, num_atoms, gt_atom_type):
        eps_r = self.sample_gaussian(size=(num_atoms.sum(), self.n_dims),
                                   device=num_atoms.device,
                                   num_atoms=num_atoms,
                                   remove_mean=True)
        #z_x = torch.randn((num_atoms.sum(), self.n_dims), device=num_atoms.device)
        return eps_r, gt_atom_type
    
    def forward(self):
        return NotImplementedError
    

# Defining some useful util functions.
def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod

class PositiveLinear(nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: int = -2):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)

class PredefinedNoiseSchedule(nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """
    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        #print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        #print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)
        self.show_schedule()
        
    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1).to(self.gamma.device)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


class GammaNetwork(nn.Module):
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper."""
    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-10.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([3.5]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1).to(self.gamma_0.device)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma
    
    
def get_noise_fn(sde, x, h, reduce_mean=True, get_loss=False, likelihood_weighting=True, eps=1e-5):
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
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    #score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
    
    def perturb_sample():
        return mean + std[:, None, None, None] * z
    
    def loss_fn(score_fn):
        perturbed_data = perturb_sample()
        score = score_fn(perturbed_data, t)
        if not likelihood_weighting:
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2
        loss = torch.mean(losses)
        return loss
    
    if get_loss:
        return loss_fn
    else:
        return perturb_sample

    
def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps, **kwargs):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        beta_init = kwargs['beta_init'] if 'beta_init' in kwargs.keys() else -10
        beta_final = kwargs['beta_final'] if 'beta_final' in kwargs.keys() else 6
        betas = np.linspace(-10, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == 'exp':
        betas = np.exp(np.linspace(np.log(beta_start), np.log(beta_end),
                       num_diffusion_timesteps), dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas
