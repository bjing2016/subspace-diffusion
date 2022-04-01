import torch
import numpy as np
stds = {
    "cifar": {16: 0.07505646805106986, 8: 0.11017048700915723},
    "celeba": {128: 0.034147739617184675, 64: 0.05084107453389564, 32: 0.07347487958749131, 16: 0.1030002890867888, 8: 0.14072510083194403},
    "church": {128: 0.06969019929745306, 64: 0.0883778833712964, 32: 0.10887079373374633, 16: 0.13137195950968375, 8: 0.15799366823221628}
}
base_resolutions = {
    "cifar": 32,
    "celeba": 32,
    "church": 32
}
        
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

def upsampling_fn(x, alpha, sigma, dataset):
    base_resolution = base_resolutions[dataset]
    curr_resolution = x.shape[-1]
    n = int(np.log2(base_resolution / curr_resolution))
    x = repeat(upsample, x, n)
    std = np.sqrt(sigma**2 + alpha**2 * stds[dataset][curr_resolution]**2)
    noise = torch.normal(mean=0, std=std, size=x.shape, device=x.device)
    noise = noise - repeat(upsample, repeat(downsample, noise, n), n)
    return x + noise
