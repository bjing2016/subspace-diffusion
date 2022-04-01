from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os
import functools
import itertools
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tqdm
import io
import likelihood
import controllable_generation
from utils import restore_checkpoint
import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import sampling
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling import (ReverseDiffusionPredictor, 
                      LangevinCorrector, 
                      EulerMaruyamaPredictor, 
                      AncestralSamplingPredictor, 
                      NoneCorrector, 
                      NonePredictor,
                      AnnealedLangevinDynamics)
import datasets
from torchvision.utils import make_grid, save_image
from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_integer("subspace", 64, "")
flags.DEFINE_float("time", None, "")
flags.DEFINE_string("checkpoint_full", "workdir/church.256/checkpoint_126.pth", "")
flags.DEFINE_string("checkpoint_subspace", "workdir/church.64/checkpoints/checkpoint_22.pth", "")
flags.DEFINE_string("image_path", "inpainting.png", "")
flags.DEFINE_string("dataset", "church", "")
flags.DEFINE_integer("batch_size", 5, "")

from configs.ve import church_ncsnpp_continuous as configs
import losses

def main(argv):

    # set hyperparameters
    sde = 'VESDE'
    checkpoint_full = FLAGS.checkpoint_full
    checkpoint_subspace = FLAGS.checkpoint_subspace
    config = configs.get_config()  
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
    if FLAGS.time is not None:
        BOUNDARIES = [FLAGS.time]
    else:
        BOUNDARIES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    batch_size =   FLAGS.batch_size
    config.training.batch_size = batch_size
    config.eval.batch_size = batch_size
    random_seed = 0 

    sigmas = mutils.get_sigmas(config)
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # retrieve the model and restore the checkpoints
    score_model = mutils.create_model(config, subspace=True)

    optimizer_full = losses.get_optimizer(config, score_model.full_score_model.parameters())
    ema_full = ExponentialMovingAverage(score_model.full_score_model.parameters(), decay=config.model.ema_rate)
    state_full = dict(optimizer=optimizer_full, model=score_model.full_score_model, ema=ema_full, step=0)

    optimizer_subspace = losses.get_optimizer(config, score_model.subspace_score_model.parameters())
    ema_subspace = ExponentialMovingAverage(score_model.subspace_score_model.parameters(), decay=config.model.ema_rate)
    state_subspace = dict(optimizer=optimizer_subspace, model=score_model.subspace_score_model, ema=ema_subspace, step=0)

    state_full = restore_checkpoint(checkpoint_full, state_full, device=config.device)
    state_subspace = restore_checkpoint(checkpoint_subspace, state_subspace, device=config.device)
    ema_full.copy_to(score_model.full_score_model.parameters())
    ema_subspace.copy_to(score_model.subspace_score_model.parameters())

    # load the dataset
    train_ds, eval_ds, _ = datasets.get_dataset(config)
    eval_iter = iter(eval_ds)
    bpds = []

    # set up inpainter
    predictor = ReverseDiffusionPredictor 
    corrector = LangevinCorrector 
    snr = 0.075 
    n_steps = 1 
    probability_flow = False 

    pc_inpainter = controllable_generation.get_pc_inpainter(sde,
                                                            predictor, corrector,
                                                            inverse_scaler,
                                                            snr=snr,
                                                            n_steps=n_steps,
                                                            probability_flow=probability_flow,
                                                            continuous=config.training.continuous,
                                                            denoise=True)
    # get the first batch_size images
    batch = next(eval_iter)
    img = batch['image']._numpy()
    img = torch.from_numpy(img).permute(0, 3, 1, 2).to(config.device)

    # set the mask
    mask = torch.ones_like(img)
    mask[:, :, :, 128:] = 0.

    images = []
    images.append(img.cpu().unsqueeze(1))
    images.append((img * mask).cpu().unsqueeze(1))
    score_model.eval()

    # for each time boundary sample the inpainting
    for t in BOUNDARIES:
        score_model.t_boundary = t
        x = pc_inpainter(score_model, scaler(img), mask).detach().cpu()
        images.append(x.unsqueeze(1))

    # save the images generated in a grid
    images = torch.cat(images, dim=1).reshape(-1, 3, 256, 256)
    image_grid = make_grid(images, len(BOUNDARIES)+2, padding=2)
    save_image(image_grid, FLAGS.image_path)


if __name__ == "__main__":
  app.run(main)