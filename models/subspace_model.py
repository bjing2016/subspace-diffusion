from . import utils
import torch.nn as nn
import torch, copy, sde_lib, time
from models import utils as mutils
from absl import flags
FLAGS = flags.FLAGS
import numpy as np 

def downsample(x):
    return 2*torch.nn.AvgPool2d(2, stride=2, padding=0)(x)
def upsample(x):
    x = x.view(-1, *x.shape[-3:])
    B, _, R, _ = x.shape
    return x.reshape(B, 3, R, 1, R, 1).repeat(1, 1, 1, 2, 1, 2).reshape(B, 3, 2*R, 2*R) / 2
def repeat(func, x, n):
    for _ in range(n):
        x = func(x)
    return x

stds = {
    "cifar": {16: 0.07505646805106986, 8: 0.11017048700915723},
    "celeba": {128: 0.034147739617184675, 64: 0.05084107453389564, 32: 0.07347487958749131, 16: 0.1030002890867888, 8: 0.14072510083194403},
    "church": {128: 0.06969019929745306, 64: 0.0883778833712964, 32: 0.10887079373374633, 16: 0.13137195950968375, 8: 0.15799366823221628}
}

@utils.register_model(name='subspace')
class SubspaceScore(nn.Module):
    ## ONLY WORKS FOR SUB-VP ##
    def __init__(self, config):

        super().__init__()
        self.config = config
        self.full_size = config.data.image_size
        self.subspace_size = FLAGS.subspace
        self.t_boundary = FLAGS.time
        self.dataset = FLAGS.dataset
     
        config_full = config
        config_full.data.image_size = self.full_size
        self.full_score_model = mutils.create_model(config_full)
      
        config_subspace = copy.deepcopy(config)
        config_subspace.data.image_size = self.subspace_size
        self.subspace_score_model = mutils.create_model(config_subspace)
        
        self.full_score_model.eval()
        self.subspace_score_model.eval()
        self.num_full = self.num_subspace = 0

    def forward(self, x, time_cond):  # x: B, 3, 32, 32
        config = self.config
        if config.training.sde == 'subvpsde':
            t = time_cond / 999 
            beta_min, beta_max = config.model.beta_min, config.model.beta_max 
            sde = sde_lib.subVPSDE(beta_min, beta_max, N=config.model.num_scales)
            beta = beta_min + (beta_max - beta_min)*t
            Beta = 1/2*t**2*(beta_max-beta_min) + t*beta_min
            sigma = 1-torch.exp(-Beta)
            alpha = torch.exp(-Beta/2)
        
            if t[0] <= self.t_boundary:
                out = self.full_score_model(x, time_cond)
                self.num_full += 1
                return out
            # These adjustments are necessary because of the data scaler used to train DDPM++ models. Works for CIFAR-10 ONLY!
            if self.subspace_size == 16:
                times = 1
                x_subspace = repeat(downsample, x, times) + alpha.view(-1, 1, 1, 1)
            elif self.subspace_size == 8:
                times = 2
                x_subspace = repeat(downsample, x, times) + 3*alpha.view(-1, 1, 1, 1)
            
            score = self.subspace_score_model(x_subspace, time_cond)
            self.num_subspace += 1
            
            score = repeat(upsample, score, times)
            
            std = 2 * stds[self.dataset][self.subspace_size]
            gauss = repeat(upsample, repeat(downsample, x, times), times) - x
            gauss = gauss / ((alpha*std)**2 + sigma**2).view(-1, 1, 1, 1)
            
            std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            
            return score - gauss * std[:, None, None, None]
            
        elif config.training.sde == 'vesde':
            sigma_boundary = np.exp((1-self.t_boundary) * np.log(self.config.model.sigma_min) + self.t_boundary * np.log(self.config.model.sigma_max))
            sigma_min, sigma_max = config.model.sigma_min, config.model.sigma_max
            sde = sde_lib.VESDE(sigma_min, sigma_max, N=config.model.num_scales)
            
            if time_cond[0] <= sigma_boundary:
                
                out = self.full_score_model(x, time_cond)
                return out
            
            times = int(np.log2(x.shape[-1] / self.subspace_size))
            x_subspace = repeat(downsample, x, times)
            
            score = self.subspace_score_model(x_subspace, time_cond)
            score = repeat(upsample, score, times)
            
            std = stds[self.dataset][self.subspace_size]
            gauss = repeat(upsample, repeat(downsample, x, times), times) - x
            gauss = gauss / (std**2 + time_cond**2).view(-1, 1, 1, 1)
            
            #print(t[0], 'subspace', time.time()-start)
            return score + gauss
            