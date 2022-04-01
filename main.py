# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training and evaluation"""

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
    
import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("save_name", None, "File name for generated samples.")
flags.DEFINE_enum("mode", None, ["train", "eval", "eval_subspace"], "Running mode: train or eval (or eval_subspace)")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.DEFINE_string("checkpoint_full", "/workdir_cifar10_ddpmpp_continuous_32/checkpoints/checkpoint_26.pth", "")
flags.DEFINE_string("checkpoint_subspace", "/workdir_cifar10_ddpmpp_continuous_16/checkpoints/checkpoint_26.pth", "")

flags.DEFINE_integer("subspace", 16, "")
flags.DEFINE_float("time", 0.5, "")

flags.mark_flags_as_required(["workdir", "config", "mode"])




def main(argv):
  FLAGS.config.training.snapshot_sampling = False
  if FLAGS.mode == "train":
    # Create the working directory
    tf.io.gfile.makedirs(FLAGS.workdir)
    # Set logger so that it outputs to both console and file
    # Make logging work for both disk and Google Cloud Storage
    gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    # Run the training pipeline
    run_lib.train(FLAGS.config, FLAGS.workdir)
  elif FLAGS.mode == "eval":
    # Run the evaluation pipeline
    run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
  elif FLAGS.mode == "eval_subspace":
    # Run the evaluation pipeline
    run_lib.evaluate_subspace(FLAGS.config, FLAGS.workdir, FLAGS.save_name,
                              FLAGS.checkpoint_full, FLAGS.checkpoint_subspace)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(main)
