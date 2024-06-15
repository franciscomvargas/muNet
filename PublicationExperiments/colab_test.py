# python 3.11

# Inspired in:
# - [Github]{https://github.com/google-research/google-research/tree/master/muNet}
# - [Google Colab]{https://colab.research.google.com/github/google-research/google-research/blob/master/muNet/mu2Net.ipynb#scrollTo=M93tll7z29rX}

######################################################
##  - CONFIGURATIONS
######################################################

# @title ~ EXPERIMENT PARAMETERS
EXPERIMENT_NAME = 'TestingWaters'  # @param { type: 'string', isTemplate: true }
# param @ BENCHMARK [
# -   'ViT tiny 0 layers / Cmaterdb benchmark',
# -   'ViT tiny 3 layers / Chars benchmark',
# -   'ViT base / VDD benchmark',
# -   'ViT large / ViT benchmark',
# -   'ViT large / VTAB-full benchmark',
# -   'ViT large / VDD benchmark',
# -   'ViT large / Chars benchmark',
# -   'ViT large / VTAB-1k benchmark'
# ] { type: 'string', isTemplate: true }
BENCHMARK = 'ViT tiny 0 layers / Cmaterdb benchmark' 
# param @ CONFIGURATION [
# -   'muNet', 
# -   'Size scale:98', 
# -   'Size scale:95', 
# -   'Size scale:90', 
# -   'Size scale:70', 
# -   'Size scale:30', 
# -   'Size scale:2', 
# -   'Finetune all', 
# -   'Freeze bottom layers:0', 
# -   'Freeze bottom layers:2', 
# -   'Freeze bottom layers:3', 
# -   'Freeze bottom layers:4', 
# -   'Freeze bottom layers:12', 
# -   'Adapters:8', 
# -   'Adapters:16', 
# -   'Adapters:32', 
# -   'Adapters:64', 
# -   'Adapters:128', 
# -   'Adapters:256', 
# -   'Adapters:512'
# ]  { type: 'string', isTemplate: true }
CONFIGURATION = 'muNet' 
# param @ AUTO_TUNE [
# -   True, 
# -   False
# ] { type: 'boolean', isTemplate: true }
AUTO_TUNE = True 
# param @ EXPERIMENTS_ROOT_DIR { type: 'string', isTemplate: true }
EXPERIMENTS_ROOT_DIR = '/tmp/' 
# param @ CONTINUE_FROM_STATE_DIR { type: 'string', isTemplate: true }
CONTINUE_FROM_STATE_DIR = '' 

if AUTO_TUNE:
  assert CONFIGURATION == 'muNet' or CONFIGURATION.startswith('Size scale:'), \
      f'Invalid configuration for auto-tune: {CONFIGURATION}'


# @title ~ ADDITIONAL PARAMETERS
# Set to true to continue interrupted experiment with matching EXPERIMENT_NAME
AUTO_CONTINUE = False  # @param [True, False] { type: 'boolean', isTemplate: true }
# Print debug statements.
VERBOSE = False  # @param [True, False] { type: 'boolean', isTemplate: true }
# Skip intermediate state save if last state was written within this time range.
SKIP_INTERMEDIATE_STATE_SECS = 3600  # @param { type: 'integer', isTemplate: true }

## Required REPOs
import sys
if './task_adaptation' not in sys.path:
  sys.path.append('./task_adaptation')
if './vision_transformer' not in sys.path:
  sys.path.append('./vision_transformer')

######################################################
##  - PACKAGES
######################################################

# Libraries Instanciation
import copy
import datetime
import gc
import jax
import jax.numpy as jnp
import json
import math
import matplotlib
import numpy as np
import os
import optax
import pandas as pd
import random
import re
import time
from collections import defaultdict
from functools import partial
from matplotlib import pyplot as plt
from threading import Thread
from typing import Optional

# Colab Configurations !?
# import jax.tools.colab_tpu # !?
# jax.tools.colab_tpu.setup_tpu() # !?

# FLAX: (https://pypi.org/project/flax/)
# Flax is a high-performance neural network library and ecosystem for JAX 
# that is designed for flexibility: Try new forms of training by forking 
# an example and by modifying the training loop, not by adding features to a framework.
import flax
import flax.linen as nn
from flax.training import checkpoints as flax_checkpoints

# TENSORFLOW
import tensorflow as tf
import tensorflow_datasets as tfds
tf.compat.v1.enable_eager_execution()

# vision_transformer (From git CLONE)
from ml_collections import ConfigDict, FrozenConfigDict
from vision_transformer.vit_jax import input_pipeline
from vision_transformer.vit_jax import checkpoint
from vision_transformer.vit_jax.configs import models as models_config  # Model configurations.
from vision_transformer.vit_jax import models_vit as models # Actual model code.

# task_adaptation (From git CLONE)
import task_adaptation.registry as task_adapt_registry
import task_adaptation.data.caltech
import task_adaptation.data.cifar
import task_adaptation.data.dtd
import task_adaptation.data.oxford_flowers102
import task_adaptation.data.oxford_iiit_pet
import task_adaptation.data.sun397
import task_adaptation.data.svhn
import task_adaptation.data.patch_camelyon
import task_adaptation.data.eurosat
import task_adaptation.data.resisc45
import task_adaptation.data.diabetic_retinopathy
import task_adaptation.data.clevr
import task_adaptation.data.dmlab
import task_adaptation.data.dsprites
import task_adaptation.data.kitti
import task_adaptation.data.smallnorb

######################################################
##  - DICTIONARIES
######################################################

# Ref TFDS catalog: https://www.tensorflow.org/datasets/catalog/beans
# These are usable task ids of image classificatiion tasks from the TFDS catalog.
TFDS_IMAGE_CLASSIFCATON_DATASETS = set([
    'beans',
    'binary_alpha_digits',
    'caltech_birds2010',
    'caltech_birds2011',
    'cars196',
    'cassava',
    'cats_vs_dogs',
    'cifar10',
    'cifar100',
    'cifar10_1',
    'citrus_leaves',
    'cmaterdb/bangla',
    'cmaterdb/devanagari',
    'cmaterdb/telugu',
    'colorectal_histology',
    'deep_weeds',
    'emnist/balanced',
    'emnist/byclass',
    'emnist/bymerge',
    'emnist/digits',
    'emnist/letters',
    'emnist/mnist',
    'fashion_mnist',
    'food101',
    'horses_or_humans',
    'imagenet2012',
    'imagenet2012_subset',
    'imagenet_lt',
    'imagenet_resized/8x8',
    'imagenet_resized/16x16',
    'imagenet_resized/32x32',
    'imagenet_resized/64x64',
    'imagenet_v2',
    'imagenette',
    'kmnist',
    'malaria',
    'mnist',
    'mnist_corrupted',
    'omniglot',
    'plant_village',
    'plantae_k',
    'quickdraw_bitmap',
    'rock_paper_scissors',
    'stanford_dogs',
    'stl10',
    'tf_flowers',
    'uc_merced',
    'visual_domain_decathlon/aircraft',
    'visual_domain_decathlon/cifar100',
    'visual_domain_decathlon/daimlerpedcls',
    'visual_domain_decathlon/dtd',
    'visual_domain_decathlon/gtsrb',
    'visual_domain_decathlon/imagenet12',
    'visual_domain_decathlon/omniglot',
    'visual_domain_decathlon/svhn',
    'visual_domain_decathlon/ucf101',
    'visual_domain_decathlon/vgg-flowers',
    ])

# ------------------------------------------------------ #

# Tasks ids of the VTAB tasks.
# Append suffix '/1k' to get the 1k version of each task.
VTAB_TASKS = [
              ## NATURAL TASKS
              'caltech101',
              # cifar100/10 were already added with slightly different val split but same test set.
              # So here is added only the 1k versions.
              'cifar100/1k',
              'cifar10/1k',
              'dtd',
              'oxford_flowers102',
              'oxford_iiit_pet',
              'sun397',
              'svhn_cropped',
              ## SPECIALIZED TASKS
              'patch_camelyon',
              'eurosat',
              'resisc45',
              'diabetic_retinopathy_detection/btgraham-300',
              ## STRUCTURED TASKS
              'clevr/count_cylinders',  # Not in results table.
              'clevr/count_all',  # Clevr-Count
              'clevr/closest_object_distance',  # Clevr-Dist
              'dmlab',
              'dsprites/label_x_position',  # dSpr-Loc
              'dsprites/label_orientation',  # dSpr-Ori
              'kitti/closest_object_distance',  # Not in results table.
              'kitti/count_vehicles',  # Not in results table.
              'kitti/closest_vehicle_distance',  # Kitti-dist
              'smallnorb/label_category',  # Not in results table.
              'smallnorb/label_lighting',  # Not in results table.
              'smallnorb/label_azimuth',  # Azim
              'smallnorb/label_elevation',  # Elev
              ]

for tn in VTAB_TASKS:
  assert tn not in TFDS_IMAGE_CLASSIFCATON_DATASETS, tn

# ------------------------------------------------------ #

######################################################
##  - FUNCTIONS
######################################################

# ------------------------------------------------------ #

TFDS_BUILDERS_CACHE = {}

def get_tfds_builder(tfds_name):
  global TFDS_BUILDERS_CACHE
  if tfds_name not in TFDS_BUILDERS_CACHE:
    TFDS_BUILDERS_CACHE[tfds_name] = tfds.builder(tfds_name)
    TFDS_BUILDERS_CACHE[tfds_name].download_and_prepare()
  return TFDS_BUILDERS_CACHE[tfds_name]

# ------------------------------------------------------ #

def ids_str2ints(ids_str):
  return [int(v) for v in ids_str.split('_')] if ids_str else []

def ids_ints2str(ids_ints):
  return '_'.join([str(v) for v in sorted(ids_ints)])

# ------------------------------------------------------ #

AddPositionEmbs = models.AddPositionEmbs
Encoder1DBlock = models.Encoder1DBlock
VisionTransformer = models.VisionTransformer

class ResidualAdapter(nn.Module):
  adapter_dim: int

  @nn.compact
  def __call__(self, x):
    hidden_dim = x.shape[-1]
    y = nn.LayerNorm()(x)
    y = nn.Dense(self.adapter_dim)(y)
    y = nn.gelu(y)
    # Default initalization.
    # y = nn.Dense(hidden_dim)(y)
    # Initialization from https://arxiv.org/pdf/1902.00751.pdf
    # y = nn.Dense(hidden_dim, kernel_init=nn.initializers.normal(stddev=1e-3))(y)
    # Zero Initialization so that added adapter does not change the representation.
    y = nn.Dense(hidden_dim, kernel_init=jax.nn.initializers.zeros)(y)
    return x + y  # Residual.

# Modified from vision_transformer/vit_jax/models Encoder to add residual adapters.
class Encoder(nn.Module):
  num_layers: int
  mlp_dim: int
  num_heads: int
  adapter_layers: str  # <MOD
  adapter_dim: int  # MOD>
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1

  @nn.compact
  def __call__(self, inputs, *, train):
    assert inputs.ndim == 3  # (batch, len, emb)

    x = AddPositionEmbs(
        posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
        name='posembed_input')(
            inputs)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    # Input Encoder
    adapter_layers_ids = ids_str2ints(self.adapter_layers)  # <MOD>
    for lyr in range(self.num_layers):
      if lyr in adapter_layers_ids:  # <MOD
        x = ResidualAdapter(
            adapter_dim=self.adapter_dim,
            name=f'residual_adapter_{lyr}'
            )(x)  # MOD>
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}',
          num_heads=self.num_heads)(
              x, deterministic=not train)
    encoded = nn.LayerNorm(name='encoder_norm')(x)
    return encoded

# ------------------------------------------------------ #

def get_vit_filename(query):
  df = checkpoint.get_augreg_df()
  res = df.query(query).filename.unique()
  assert len(res) == 1
  return res[0]

# ------------------------------------------------------ #

USE_DROPOUT = False
VIT_CONFIG_CACHE = {}

def get_vit_config(query):
  if query not in VIT_CONFIG_CACHE:
    filename = get_vit_filename(query)
    config = models_config.AUGREG_CONFIGS[filename.split('-')[0]].copy_and_resolve_references()
    # Overwrite with custom Encoder.
    config.unlock()
    config.encoder = Encoder
    config.transformer.adapter_layers = ''
    config.transformer.adapter_dim = -1
    if not USE_DROPOUT:
      config.transformer.dropout_rate = 0.0
      config.transformer.attention_dropout_rate = 0.0
    config.lock()
    VIT_CONFIG_CACHE[query] = config
  return VIT_CONFIG_CACHE[query].copy_and_resolve_references()

def get_max_num_layers(query):
  config = get_vit_config(query)
  return config.transformer.num_layers

# ------------------------------------------------------ #

# Benchmarks used in the paper.
VIT_BENCHMARK = [
  'imagenet2012',
  'cifar100',
  'cifar10',
  ]
VTAB_FULL_BENCHMARK = [
  'caltech101',
  # 'cifar100',  # Already added with VIT_BENCHMARK
  'dtd',
  'oxford_flowers102',
  'oxford_iiit_pet',
  'sun397',
  'svhn_cropped',
  'patch_camelyon',
  'eurosat',
  'resisc45',
  'diabetic_retinopathy_detection/btgraham-300',
  'clevr/count_cylinders',
  'clevr/count_all',
  'clevr/closest_object_distance',
  'dmlab',
  'dsprites/label_x_position',
  'dsprites/label_orientation',
  'kitti/closest_object_distance',
  'kitti/count_vehicles',
  'kitti/closest_vehicle_distance',
  'smallnorb/label_category',
  'smallnorb/label_lighting',
  'smallnorb/label_azimuth',
  'smallnorb/label_elevation',
]
CHARS_BENCHMARK = [
  'emnist/digits',
  'emnist/letters',
  'kmnist',
  'mnist',
  'omniglot',
  'cmaterdb/bangla',
  'cmaterdb/devanagari',
  'cmaterdb/telugu',
  ]
VDD_BENCHMARK = [
  'visual_domain_decathlon/imagenet12',
  'visual_domain_decathlon/svhn',
  'visual_domain_decathlon/cifar100',
  'visual_domain_decathlon/gtsrb',
  'visual_domain_decathlon/daimlerpedcls',
  'visual_domain_decathlon/omniglot',
  'visual_domain_decathlon/ucf101',
  'visual_domain_decathlon/aircraft',
  'visual_domain_decathlon/dtd',
  'visual_domain_decathlon/vgg-flowers',
  ]
VTAB_1K_BENCHMARK = [
  'caltech101/1k',
  'cifar100/1k',
  'cifar10/1k',
  'dtd/1k',
  'oxford_flowers102/1k',
  'oxford_iiit_pet/1k',
  'sun397/1k',
  'svhn_cropped/1k',
  'patch_camelyon/1k',
  'eurosat/1k',
  'resisc45/1k',
  'diabetic_retinopathy_detection/btgraham-300/1k',
  'clevr/count_cylinders/1k',
  'clevr/count_all/1k',
  'clevr/closest_object_distance/1k',
  'dmlab/1k',
  'dsprites/label_x_position/1k',
  'dsprites/label_orientation/1k',
  'kitti/closest_object_distance/1k',
  'kitti/count_vehicles/1k',
  'kitti/closest_vehicle_distance/1k',
  'smallnorb/label_category/1k',
  'smallnorb/label_lighting/1k',
  'smallnorb/label_azimuth/1',
  'smallnorb/label_elevation/1k',
]

# ------------------------------------------------------ #

def set_continue_configs(exp_config):
  if CONTINUE_FROM_STATE_DIR:
    exp_config.load_rand_init = False
    exp_config.load_vit_checkpoint = False
    exp_config.load_experiment = True
    exp_config.load_experiment_dir = CONTINUE_FROM_STATE_DIR

# ------------------------------------------------------ #

DATASET_HPARAMS_KEYS_PRERFIX = 'ds_'
OPTIMIZER_HPARAMS_KEYS_PRERFIX = 'opt_'

def get_exp_config_ti3_chars():
  exp_config = ConfigDict()
  exp_config.experiment_name = EXPERIMENT_NAME
  exp_config.experiments_root_dir = EXPERIMENTS_ROOT_DIR
  exp_config.num_train_examples_between_validations_max = 51200  # 100 batches.
  exp_config.num_validations_per_path_training = 5
  exp_config.num_validation_examples_max = 5120  # 10 batches.
  exp_config.batch_size = 512
  exp_config.num_task_iters = 2
  exp_config.num_samples_per_task = 8*8
  exp_config.mutate_adapters = True
  # Force finetune last layer norm that technically is part of the head.
  exp_config.force_finetune_components = ['encoder_norm']
  # Population policy params:
  exp_config.policy_class = 'PPDecay'
  exp_config.policy_kwargs = {}
  # Scorer params:
  exp_config.scorer_class = 'ScorerDecay'
  exp_config.scorer_kwargs = dict(
      base=1.0,
      num_params=1_484_162,  # params in Ti/16 with 3 layers.
      )

  # Seed models params:
  exp_config.load_rand_init = False
  exp_config.load_vit_checkpoint = True
  exp_config.load_vit_checkpoint_query = 'name=="Ti/16" and ds=="i21k" and aug=="light1" and wd==0.1 and sd==0.0'
  exp_config.load_experiment = False
  exp_config.load_experiment_dir = ''
  set_continue_configs(exp_config)

  # Hyperparameters:
  exp_config.models_default_hparams = {
      '_mu_': 0.1,
      # Default num_classes has no effect since it is always overwritten or used
      # for rand init models whose head is always replaced.
      'num_classes': 1,
      # Set to ids_ints2str(range(max_num_layers)) to activate all adapters.
      'adapter_layers': '',
      'num_layers': 3,
      'adapter_dim': 32,
      'opt_lr': 0.01,
      'opt_lr_schedule': 'cosine',
      'opt_lr_warmup_ratio': 0.1,
      'opt_momentum': 0.9,
      'opt_nesterov': False,
      'ds_image_size': 32,
      'ds_crop': True,
      'ds_area_range_min': 0.05,
      'ds_aspect_ratio_range_min': 0.75,
      'ds_flip_left_right': True,
      'ds_brightness_delta': 0.0,
      'ds_contrast_delta': 0.0,
      'ds_saturation_delta': 0.0,
      'ds_hue_delta': 0.0,
  }
  exp_config.models_mutation_ranges = {}
  exp_config.task_names = CHARS_BENCHMARK
  exp_config_validate(exp_config)
  return exp_config

def get_exp_config_base_deca():
  exp_config = ConfigDict()
  exp_config.experiment_name = EXPERIMENT_NAME
  exp_config.experiments_root_dir = EXPERIMENTS_ROOT_DIR
  exp_config.num_train_examples_between_validations_max = 51200  # 200 batches.
  exp_config.num_validations_per_path_training = 30
  exp_config.num_validation_examples_max = 5120  # 20 batches.
  exp_config.batch_size = 256
  exp_config.num_task_iters = 2
  exp_config.num_samples_per_task = 8*8
  exp_config.mutate_adapters = True
  exp_config.force_finetune_components = ['encoder_norm']
  # Population policy params:
  exp_config.policy_class = 'PPDecay'
  exp_config.policy_kwargs = {}
  # Scorer params:
  exp_config.scorer_class = 'ScorerDecay'
  exp_config.scorer_kwargs = dict(
      base=1.0,
      num_params=85_652_738,  # params in B/16
      )
  # Seed models params:
  exp_config.load_rand_init = False
  exp_config.load_vit_checkpoint = True
  exp_config.load_vit_checkpoint_query = 'name=="B/16" and ds=="i21k" and aug=="medium1" and wd==0.1 and sd==0'
  exp_config.load_experiment = False
  exp_config.load_experiment_dir = ''
  set_continue_configs(exp_config)

  # Hyperparameters:
  max_num_layers = get_max_num_layers(exp_config.load_vit_checkpoint_query)
  exp_config.models_default_hparams = {
      '_mu_': 0.1,
      'num_classes': 1,
      'adapter_layers': '',
      'num_layers': max_num_layers,
      'adapter_dim': 32,
      'opt_lr': 0.01,
      'opt_lr_schedule': 'cosine',
      'opt_lr_warmup_ratio': 0.1,
      'opt_momentum': 0.9,
      'opt_nesterov': False,
      'ds_image_size': 80,
      'ds_crop': True,
      'ds_area_range_min': 0.05,
      'ds_aspect_ratio_range_min': 0.75,
      'ds_flip_left_right': True,
      'ds_brightness_delta': 0.0,
      'ds_contrast_delta': 0.0,
      'ds_saturation_delta': 0.0,
      'ds_hue_delta': 0.0,
  }
  exp_config.models_mutation_ranges = {}
  exp_config.task_names = VDD_BENCHMARK
  exp_config_validate(exp_config)
  return exp_config

def get_exp_config_ti0_cmaterdb():
  exp_config = ConfigDict()
  exp_config.experiment_name = EXPERIMENT_NAME
  exp_config.experiments_root_dir = EXPERIMENTS_ROOT_DIR
  exp_config.num_train_examples_between_validations_max = 51200  # 100 batches.
  exp_config.num_validations_per_path_training = 4
  exp_config.num_validation_examples_max = 5120  # 10 batches.
  exp_config.batch_size = 512
  exp_config.num_task_iters = 2
  exp_config.num_samples_per_task = 8*4
  exp_config.mutate_adapters = False
  exp_config.force_finetune_components = ['encoder_norm']
  # Population policy params:
  exp_config.policy_class = 'PPDecay'
  exp_config.policy_kwargs = {}
  # Scorer params:
  exp_config.scorer_class = 'ScorerDecay'
  exp_config.scorer_kwargs = dict(
      base=1.0,
      num_params=1_484_162,  # params in Ti/16 with 3 layers.
      )

  # Seed models params:
  exp_config.load_rand_init = True
  exp_config.load_vit_checkpoint = False
  # The query is used to get the model configs even if the checkpoint is not loaded.
  exp_config.load_vit_checkpoint_query = 'name=="Ti/16" and ds=="i21k" and aug=="light1" and wd==0.1 and sd==0.0'
  exp_config.load_experiment = False
  exp_config.load_experiment_dir = ''
  set_continue_configs(exp_config)

  # Hyperparameters:
  exp_config.models_default_hparams = {
      '_mu_': 0.2,
      'num_classes': 1,
      'adapter_layers': '',
      'num_layers': 0,
      'adapter_dim': 32,
      'opt_lr': 0.01,
      'opt_lr_schedule': 'cosine',
      'opt_lr_warmup_ratio': 0.1,
      'opt_momentum': 0.9,
      'opt_nesterov': False,
      'ds_image_size': 32,
      'ds_crop': True,
      'ds_area_range_min': 0.05,
      'ds_aspect_ratio_range_min': 0.75,
      'ds_flip_left_right': True,
      'ds_brightness_delta': 0.0,
      'ds_contrast_delta': 0.0,
      'ds_saturation_delta': 0.0,
      'ds_hue_delta': 0.0,
  }
  exp_config.models_mutation_ranges = {
      'num_layers': list(range(0, 4)),
  }
  exp_config.task_names = [
    'cmaterdb/bangla',
    'cmaterdb/devanagari',
    'private:cmaterdb/telugu',
    ]
  exp_config_validate(exp_config)
  return exp_config

def get_exp_config_large(benchmark_string_id):
  exp_config = ConfigDict()
  exp_config.experiment_name = EXPERIMENT_NAME
  exp_config.experiments_root_dir = EXPERIMENTS_ROOT_DIR
  # Cap to 1/10th of imagenet train set size to have similar ratio of exps reported in:
  # https://arxiv.org/abs/2106.10270
  exp_config.num_train_examples_between_validations_max = 128_116
  exp_config.num_validations_per_path_training = 4
  exp_config.num_validation_examples_max = 10_000
  # Fit HBM memory: TPUv4 megacore=64, TPUv3=32.
  exp_config.batch_size = 64
  exp_config.num_task_iters = 1
  # Assuming TPUv4 32 cores * 4 generations.
  exp_config.num_samples_per_task = 32 * 4
  exp_config.mutate_adapters = False
  exp_config.force_finetune_components = ['encoder_norm']
  # Population policy params:
  exp_config.policy_class = 'PPDecay'
  exp_config.policy_kwargs = {}
  # Scorer params:
  exp_config.scorer_class = 'ScorerDecay'
  exp_config.scorer_kwargs = dict(
      base=1.0,
      num_params=303_303_682,  # Params in L/16
      )
  # Seed models params:
  exp_config.load_rand_init = False
  exp_config.load_vit_checkpoint = True
  exp_config.load_vit_checkpoint_query = 'name=="L/16" and ds=="i21k" and aug=="medium2" and wd==0.03 and sd==0.1'
  exp_config.load_experiment = False
  exp_config.load_experiment_dir = ''
  set_continue_configs(exp_config)

  # Hyperparameters:
  max_num_layers = get_max_num_layers(exp_config.load_vit_checkpoint_query)
  exp_config.models_default_hparams = {
      '_mu_': 0.2,
      'num_classes': 1,
      'adapter_layers': '',
      'num_layers': max_num_layers,
      'adapter_dim': 16,
      'opt_lr': 0.01,
      'opt_lr_schedule': 'cosine',
      'opt_lr_warmup_ratio': 0.05,
      'opt_momentum': 0.9,
      'opt_nesterov': False,
      'ds_image_size': 384,
      'ds_crop': True,
      'ds_area_range_min': 0.05,
      'ds_aspect_ratio_range_min': 0.75,
      'ds_flip_left_right': True,
      'ds_brightness_delta': 0.0,
      'ds_contrast_delta': 0.0,
      'ds_saturation_delta': 0.0,
      'ds_hue_delta': 0.0,
  }
  exp_config.models_mutation_ranges = {}
  if benchmark_string_id == 'ViT large / ViT benchmark':
    exp_config.task_names = VIT_BENCHMARK
  elif benchmark_string_id == 'ViT large / VTAB-full benchmark':
    exp_config.task_names = VTAB_FULL_BENCHMARK
  elif benchmark_string_id == 'ViT large / VDD benchmark':
    exp_config.task_names = VDD_BENCHMARK
  elif benchmark_string_id == 'ViT large / Chars benchmark':
    exp_config.task_names = CHARS_BENCHMARK
  elif benchmark_string_id == 'ViT large / VTAB-1k benchmark':
    exp_config.task_names = VTAB_1K_BENCHMARK
  else:
    assert False, f'Unknown benchmark: {benchmark_string_id}'
  exp_config_validate(exp_config)
  return exp_config

def exp_config_add_auto_tune(exp_config):
  exp_config.models_mutation_ranges['adapter_dim'] = [8, 16, 32, 64, 128]
  exp_config.models_mutation_ranges['opt_lr'] = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
  exp_config.models_mutation_ranges['opt_lr_schedule'] = ['constant', 'cosine', 'restarts']
  exp_config.models_mutation_ranges['opt_lr_warmup_ratio'] = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4]
  exp_config.models_mutation_ranges['opt_momentum'] = [None, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
  exp_config.models_mutation_ranges['opt_nesterov'] = [True, False]
  exp_config.models_mutation_ranges['ds_image_size'] = [ 16 * i for i in (range(1, 1+int(384/16))) ]
  exp_config.models_mutation_ranges['ds_area_range_min'] = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
  exp_config.models_mutation_ranges['ds_aspect_ratio_range_min'] = [0.25, 0.5, 0.75, 1.0]
  exp_config.models_mutation_ranges['ds_flip_left_right'] = [True, False]
  exp_config.models_mutation_ranges['ds_brightness_delta'] = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
  exp_config.models_mutation_ranges['ds_contrast_delta'] = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
  exp_config.models_mutation_ranges['ds_saturation_delta'] = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
  exp_config.models_mutation_ranges['ds_hue_delta'] = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
  return exp_config

def exp_config_add_auto_tune_v2(exp_config):
  exp_config.models_mutation_ranges['_mu_'] = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]
  exp_config.models_mutation_ranges['opt_lr'] = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
  exp_config.models_mutation_ranges['opt_lr_warmup_ratio'] = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4]
  exp_config.models_mutation_ranges['opt_momentum'] = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
  exp_config.models_mutation_ranges['opt_nesterov'] = [True, False]
  exp_config.models_mutation_ranges['ds_area_range_min'] = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
  exp_config.models_mutation_ranges['ds_crop'] = [True, False]
  exp_config.models_mutation_ranges['ds_aspect_ratio_range_min'] = [0.25, 0.5, 0.75, 1.0]
  exp_config.models_mutation_ranges['ds_flip_left_right'] = [True, False]
  exp_config.models_mutation_ranges['ds_brightness_delta'] = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
  exp_config.models_mutation_ranges['ds_contrast_delta'] = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
  exp_config.models_mutation_ranges['ds_saturation_delta'] = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
  exp_config.models_mutation_ranges['ds_hue_delta'] = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
  return exp_config

def exp_config_validate(exp_config):
  for khp in exp_config.models_default_hparams:
    if khp in exp_config.models_mutation_ranges:
      assert exp_config.models_default_hparams[khp] \
          in exp_config.models_mutation_ranges[khp]

def exp_config_set_size_scale(exp_config, base_percent:int):
  exp_config.scorer_kwargs['base'] = float(base_percent) / 100.0
  if 'num_layers' not in exp_config.models_mutation_ranges:
    exp_config.models_mutation_ranges['num_layers'] = list(
        range(1, exp_config.models_default_hparams['num_layers']+1))
  return exp_config

def exp_config_set_baseline_common(exp_config):
  parallelism = jax.local_device_count()
  assert (int(exp_config.num_samples_per_task / parallelism) ==
          exp_config.num_samples_per_task / parallelism)
  exp_config.num_validations_per_path_training *= \
      exp_config.num_task_iters \
      * int(exp_config.num_samples_per_task/parallelism)
  exp_config.num_task_iters = 1
  exp_config.num_samples_per_task = parallelism
  exp_config.models_mutation_ranges = {}
  exp_config.policy_class = 'PPBaseline'
  exp_config.policy_kwargs = {}
  exp_config_validate(exp_config)
  return exp_config

def exp_config_set_baseline_finetune_all(exp_config):
  exp_config = exp_config_set_baseline_common(exp_config)
  exp_config.mutate_adapters = False
  exp_config.models_default_hparams['_mu_'] = 1.0
  exp_config.models_default_hparams['adapter_layers'] = ''
  exp_config_validate(exp_config)
  return exp_config

def exp_config_set_baseline_freeze_bottom_layers(exp_config, num_layers:int):
  exp_config = exp_config_set_baseline_common(exp_config)
  max_num_layers = exp_config.models_default_hparams['num_layers']
  assert max_num_layers >= num_layers
  unfrozen_layers = [f'encoderblock_{id}' for id in range(num_layers, max_num_layers)]
  exp_config.force_finetune_components = ['encoder_norm'] + unfrozen_layers
  exp_config.mutate_adapters = False
  exp_config.models_default_hparams['_mu_'] = 0.0
  exp_config.models_default_hparams['adapter_layers'] = ''
  exp_config_validate(exp_config)
  return exp_config

def exp_config_set_baseline_adapters(exp_config, adapter_dim:int):
  exp_config = exp_config_set_baseline_common(exp_config)
  exp_config.force_finetune_components = ['encoder_norm']
  exp_config.mutate_adapters = True
  exp_config.models_default_hparams['_mu_'] = 0.0
  max_num_layers = exp_config.models_default_hparams['num_layers']
  exp_config.models_default_hparams['adapter_layers'] = ids_ints2str(
      range(max_num_layers))
  exp_config.models_default_hparams['adapter_dim'] = adapter_dim
  exp_config_validate(exp_config)
  return exp_config

# ------------------------------------------------------ #

def get_sample_images(image_size:int, batch_size:int):
  return np.zeros((batch_size, image_size, image_size, 3))

def get_sample_labels(batch_size:int):
  return np.zeros(batch_size, dtype=np.int32)

def get_sample_batch(image_size:int, batch_size:int):
  return {'image': get_sample_images(image_size, batch_size),
          'label': get_sample_labels(batch_size),}

# ------------------------------------------------------ #

def get_vit_checkpoint(image_size, query):
  filename = get_vit_filename(query)
  config = get_vit_config(query)
  model = VisionTransformer(**config, num_classes=2)  # num_classes unused.
  init_params = copy.deepcopy(jax.device_get(
      model.init(jax.random.PRNGKey(0),
                 get_sample_images(image_size=image_size,
                                   batch_size=1),
                 train=USE_DROPOUT)['params']))
  params = checkpoint.load_pretrained(
    pretrained_path=f'gs://vit_models/augreg/{filename}.npz',
    init_params=init_params,
    model_config=config)
  return params

def get_vit_checkpoint_mapped(image_size, query):
  params = get_vit_checkpoint(image_size, query)
  params = params_model_to_comps(params)
  return params

def get_reshaped_posembed_component(image_size, query):
  params = get_vit_checkpoint_mapped(image_size, query)['posembed_input']
  return Component(name='posembed_input',
                   params=params,
                   train_locks=[NOT_TRAINABLE])

# ------------------------------------------------------ #

# Parameter mapping.
TRANSFORMER_KEYS = set(
    ['encoder_norm', 'posembed_input'] + \
    [f'encoderblock_{k}' for k in range(24)])

def params_model_to_comps(params):
  global TRANSFORMER_KEYS
  TRANSFORMER_KEYS.update(params['Transformer'].keys())
  new_params = {}
  for k in params.keys():
    if k == 'Transformer':
      t_params = params[k]
      for t_k in t_params.keys():
        new_params[t_k] = t_params[t_k]
    else:
      new_params[k] = params[k]
  params = flax.core.freeze(new_params)
  return flax.core.freeze(params)

def params_comps_to_model(params):
  params = params.unfreeze()
  params['Transformer'] = {}
  keys = list(params.keys())
  assert len(TRANSFORMER_KEYS) != 0
  for k in keys:
    if k in TRANSFORMER_KEYS:
      params['Transformer'][k] = params.pop(k)
  return flax.core.freeze(params)

# ------------------------------------------------------ #

def get_model_kwargs(hparams, exp_config):
  # Validate adapters params.
  for v in ids_str2ints(hparams['adapter_layers']):
    assert v < hparams['num_layers']
  return {
        'num_classes': int(hparams['num_classes']),
        'num_layers': int(hparams['num_layers']),
        'image_size': int(hparams['ds_image_size']),
        'adapter_layers': str(hparams['adapter_layers']),
        'adapter_dim': int(hparams['adapter_dim']),
        'query': str(exp_config.load_vit_checkpoint_query),
    }

def get_vit_model(num_classes, num_layers, adapter_layers, adapter_dim, query):
  config = get_vit_config(query)
  config['transformer']['num_layers'] = num_layers
  config['transformer']['adapter_layers'] = adapter_layers
  config['transformer']['adapter_dim'] = adapter_dim
  config = FrozenConfigDict(config)
  model = VisionTransformer(**config, num_classes=num_classes)
  return model

def get_vit_model_and_params(
    num_classes, num_layers, image_size, adapter_layers, adapter_dim, query,
    rng_key=0):
  model = get_vit_model(
      num_classes, num_layers, adapter_layers, adapter_dim, query)
  init_params = copy.deepcopy(jax.device_get(
      model.init(
          jax.random.PRNGKey(rng_key),
          get_sample_images(image_size=image_size, batch_size=1),
          train=USE_DROPOUT)['params']))
  return model, init_params

def get_vit_params_mapped(**kwargs):
  model, init_params = get_vit_model_and_params(**kwargs)
  init_params = params_model_to_comps(init_params)
  return init_params

# ------------------------------------------------------ #

def format_params(a, b):
  params = a.copy(b)
  assert len(params) == len(a) + len(b)  # Dicts should not overlap.
  params = params_comps_to_model(params)
  return params

# ------------------------------------------------------ #

def get_optimizer(
    lr: float,
    lr_schedule: str,
    lr_warmup_ratio: float,
    momentum: float,
    nesterov: bool,
    num_train_batches_between_validations: int,
    num_validations_per_path_training: int,
    ):
  if lr_schedule == 'constant':
    # Divide by 2 so that average lr is the same as other types.
    learning_rate = 0.5 * lr
  elif lr_schedule == 'cosine':
    train_steps = int(num_train_batches_between_validations
                      * num_validations_per_path_training)
    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=lr/100.0,
        peak_value=lr,
        warmup_steps=int(lr_warmup_ratio * train_steps),
        decay_steps=train_steps)
  elif lr_schedule == 'restarts':
    train_steps = num_train_batches_between_validations
    repeats = num_validations_per_path_training
    kwargs = dict(
        init_value=lr/100.0,
        peak_value=lr,
        warmup_steps=int(lr_warmup_ratio * train_steps),
        decay_steps=train_steps,
    )
    kwargs = [kwargs] * repeats
    learning_rate = optax.sgdr_schedule(kwargs)
  else:
    assert False, f'Invalid lr schedule: {lr_schedule}'

  return optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.sgd(
          learning_rate=learning_rate,
          momentum=momentum,
          nesterov=nesterov,
          accumulator_dtype=jnp.bfloat16))

# ------------------------------------------------------ #

def get_default_splits(tfds_name):
  info = get_tfds_builder(tfds_name).info
  splits = list(info.splits.keys())
  assert 'train' in splits, splits
  splits.remove('train')
  used_percent = 0
  slice_percent = 5
  pp = {}
  for k in ['test', 'validation']:
    if k in splits:
      pp[k] = k
      splits.remove(k)
    else:
      pp[k] = f'train[{used_percent}%:{used_percent+slice_percent}%]'
      used_percent += slice_percent
  pp['train'] = f'train[{used_percent}%:]'
  return pp

def get_dataset_and_splits(tfds_name: str):
  vtab_class = None
  if tfds_name in ['imagenet_v2', 'cifar10_1']:
    assert False,  f"{tfds_name} used as validation set for other tasks."

  if tfds_name == 'imagenet2012':
    dataset = {
        'train':'imagenet2012', 'validation':'imagenet_v2', 'test':'imagenet2012'}
    splits = {
        'train':'train', 'validation':'test', 'test':'validation'}
  elif tfds_name == 'cifar100':
    dataset = tfds_name
    splits = {
        'train':'train[:98%]', 'validation':'train[98%:]', 'test':'test'}
  elif tfds_name == 'cifar10':
    dataset = {
        'train':'cifar10', 'validation':'cifar10_1', 'test':'cifar10'}
    splits = {
        'train':'train', 'validation':'test', 'test':'test'}
  elif tfds_name.startswith('visual_domain_decathlon/'):
    dataset = tfds_name
    # Test has no labels, split validation in half.
    splits =  {
        'train':'train', 'validation':'validation[:50%]', 'test':'validation[50%:]'}
  elif tfds_name.startswith('cmaterdb/'):
    dataset = tfds_name
    # Increase size of validation set due to small dataset size.
    splits =  {
        'train':'train[20%:]', 'validation':'train[:20%]', 'test':'test'}
  elif tfds_name == 'omniglot':
    # Test has no labels, and missing validation, use additional splits.
    dataset = tfds_name
    splits = {'train':'train', 'validation':'small1', 'test':'small2'}
  elif tfds_name in VTAB_TASKS or (
      tfds_name.endswith('/1k') and tfds_name.replace('/1k', '') in VTAB_TASKS):
    is_vtab_1k = tfds_name.endswith('/1k')
    tfds_name = tfds_name.replace('/1k', '')
    registry_name = {
        'diabetic_retinopathy_detection/btgraham-300': 'diabetic_retinopathy',
        'svhn_cropped': 'svhn',
        'cifar100': 'cifar',
        'cifar10': 'cifar',
    }.get(tfds_name, tfds_name.split('/')[0])
    args = {
        'clevr/count_all': ('count_all',),
        'clevr/count_cylinders': ('count_cylinders',),
        'clevr/closest_object_distance': ('closest_object_distance',),
        'dsprites/label_x_position': ('label_x_position',),
        'dsprites/label_orientation': ('label_orientation',),
        'kitti/closest_object_distance': ('closest_object_distance',),
        'kitti/count_vehicles': ('count_vehicles',),
        'kitti/closest_vehicle_distance': ('closest_vehicle_distance',),
        'smallnorb/label_category': ('label_category',),
        'smallnorb/label_lighting': ('label_lighting',),
        'smallnorb/label_azimuth': ('label_azimuth',),
        'smallnorb/label_elevation': ('label_elevatio',),
        'cifar100': (100,),
        'cifar10': (10,),
    }.get(tfds_name, ())
    vtab_class = task_adapt_registry.Registry.lookup(
        f'data.{registry_name}')(*args)
    vtab_splits = vtab_class._tfds_splits
    dataset = {
        'caltech101': 'caltech101:3.*.*',
        'dtd': 'dtd:3.*.*',
        'oxford_flowers102': 'oxford_flowers102:2.*.*',
        'oxford_iiit_pet': 'oxford_iiit_pet:3.*.*',
        'sun397': 'sun397/tfds:4.*.',
        'svhn': 'svhn_cropped:3.*.*',
        'patch_camelyon': 'patch_camelyon:2.*.*',
        'eurosat': 'eurosat/rgb:2.*.*',
        'resisc45': 'resisc45:3.*.*',
        'diabetic_retinopathy': 'diabetic_retinopathy_detection/btgraham-300:3.*.*',
        'clevr': 'clevr:3.*.*',
        'dmlab': 'dmlab:2.0.1',
        'dsprites': 'dsprites:2.*.*',
        'kitti': 'kitti:3.2.0',
        'smallnorb': 'smallnorb:2.*.*',
        'cifar' : 'cifar100:3.*.*' if tfds_name == 'cifar100' else 'cifar10:3.*.*',
    }[registry_name]
    if is_vtab_1k:
      splits =  {
          'train': str(vtab_splits['train800']),
          'validation': str(vtab_splits['val200']),
          'test': str(vtab_splits['test']),
          }
    else:
      splits =  {
          'train': str(vtab_splits['train']),
          'validation': str(vtab_splits['val']),
          'test': str(vtab_splits['test']),
          }
  else:
    dataset = tfds_name
    splits = get_default_splits(tfds_name)
  return dataset, splits, vtab_class

# ------------------------------------------------------ #

######################################################
##  - CLASSES
######################################################

# ------------------------------------------------------ #

class Task():
  def __init__(self, name, exp_config):
    self.exp_config = exp_config
    if name.startswith(NOT_TRAINABLE):
      self.name = name
      self.private = False
      return

    if name.startswith('private:'):
      _, name = name.split('private:')
      self.private = True
    else:
      self.private = False

    self.dataset, self.splits, self.vtab_class = get_dataset_and_splits(name)
    self.name = name
    if self.vtab_class:
      self.num_classes = self.vtab_class.get_num_classes()
    else:
      self.num_classes = self.get_builder('train').info.features['label'].num_classes
    num_train_examples = self.get_builder('train').info.splits[self.splits['train']].num_examples
    self.train_batch_size = exp_config.batch_size
    self.num_train_batches_between_validations = math.ceil(
        min(num_train_examples,
            exp_config.num_train_examples_between_validations_max)
        / self.train_batch_size)
    self.cache_train = num_train_examples < min(100_000, (
        exp_config.num_validations_per_path_training
        * self.num_train_batches_between_validations
        * self.train_batch_size))

    num_validation_examples_tot = self.get_builder('validation').info.splits[self.splits['validation']].num_examples
    if exp_config.num_validation_examples_max <= num_validation_examples_tot:
      self.validation_batch_size = exp_config.batch_size
      self.num_validation_batches = math.floor(
          exp_config.num_validation_examples_max / self.validation_batch_size)
    else:
      # Adjust batch_size and num_batches to cover the smaller validation sets.
      self.num_validation_batches = math.ceil(
          num_validation_examples_tot / exp_config.batch_size)
      self.validation_batch_size = math.floor(
          num_validation_examples_tot / self.num_validation_batches)
      assert num_validation_examples_tot >= (self.num_validation_batches*self.validation_batch_size)
    self.num_validation_examples = self.num_validation_batches * self.validation_batch_size

    print(f'Task: {self.name}')
    print(f'  Train batches between validations: {self.num_train_batches_between_validations}')
    print(f'  Validation batches: {self.num_validation_batches}')
    print(f'  Validation batch size: {self.validation_batch_size}')
    print(f'  Dataset {{\n{self.dataset}}}')
    print(f'  Splits {{\n{self.splits}}}')


  def get_builder(self, mode):
    if type(self.dataset) == str:
      return get_tfds_builder(self.dataset)
    return get_tfds_builder(self.dataset[mode])

  def __str__(self):
    return f'Task_{self.name}'

  def is_trainable(self):
    return not self.name.startswith(NOT_TRAINABLE)

  def is_private(self):
    return self.private

  def get_ds(self, mode, hparams):
    data = self.get_builder(mode).as_dataset(
        split=self.splits[mode],
        shuffle_files=mode=='train')

    def _pp(data):
      im = data['image']
      im = tf.cast(im, tf.float32)
      # Must have 3 channels.
      if im.shape[-1] == 1:
        im = tf.squeeze(tf.stack([im] * 3, -1), axis=-2)
      assert im.shape[-1] == 3
      # Values in range [-1 , 1]
      im = im / 127.5 - 1

      if mode == 'train':
        if hparams['ds_crop'] and hparams['ds_area_range_min'] < 1.0:
          channels = im.shape[-1]
          begin, size, _ = tf.image.sample_distorted_bounding_box(
              tf.shape(im),
              tf.zeros([0, 0, 4], tf.float32),
              aspect_ratio_range=[hparams['ds_aspect_ratio_range_min'],
                                  1.0/hparams['ds_aspect_ratio_range_min']],
              area_range=[hparams['ds_area_range_min'], 1.0],
              # Overlap with bounding box, the bounding box should anyway
              # default defaults to whole image in this case.
              min_object_covered=0,
              use_image_if_no_bounding_boxes=True)
          im = tf.slice(im, begin, size)
          # Restore the depth-dimension lost by the above operation.
          im.set_shape([None, None, channels])
        if hparams['ds_flip_left_right']:
          if tf.random.uniform(shape=[]) > 0.5:
            im = tf.image.flip_left_right(im)
        if hparams['ds_brightness_delta'] > 0.0:
          im = tf.image.random_brightness(
              im, max_delta=hparams['ds_brightness_delta'])
        if hparams['ds_contrast_delta'] > 0.0:
          im = tf.image.random_contrast(
              im, lower=1 - hparams['ds_contrast_delta'],
              upper=1 + hparams['ds_contrast_delta'])
        if hparams['ds_saturation_delta'] > 0.0:
          im = tf.image.random_saturation(
              im, lower=1 - hparams['ds_saturation_delta'],
              upper=1 + hparams['ds_saturation_delta'])
        if hparams['ds_hue_delta'] > 0.0:
          im = tf.image.random_hue(im, max_delta=hparams['ds_hue_delta'])

      im = tf.image.resize(im, [hparams['ds_image_size'],
                                hparams['ds_image_size']])
      im = tf.clip_by_value(im, -1, 1)

      return {'image': im, 'label': data['label']}

    if mode == 'validation':
      data = data.take(self.num_validation_examples)
    if mode == 'validation' or (mode == 'train' and self.cache_train):
      data = data.cache()
    if mode != 'test':
      data = data.repeat()
    if self.vtab_class and self.vtab_class._base_preprocess_fn:
      data = data.map(self.vtab_class._base_preprocess_fn, tf.data.AUTOTUNE)
    data = data.map(_pp, tf.data.AUTOTUNE)
    if mode == 'train':
      batch_size = self.train_batch_size
    else:
      batch_size = self.validation_batch_size
    data = data.batch(batch_size)
    if mode == 'train':
      data = data.shuffle(10)
    return tfds.as_numpy(data.prefetch(tf.data.AUTOTUNE))

def get_task_factory_fn(exp_config):
  def get_task(task_name):
    return Task(name=task_name, exp_config=exp_config)
  return get_task

NOT_TRAINABLE = 'NOT_TRAINABLE'
not_trainable = Task(NOT_TRAINABLE, None)

# ------------------------------------------------------ #

def get_num_params(params):
  return sum(jax.tree.flatten(
      jax.tree.map(lambda p: np.prod(p.shape), params)
      )[0])

# ------------------------------------------------------ #

# Convert frozend dict of params to a list of components.
def params2comps(params, train_locks , name=None):
  components = []
  for k in params:
    if name is None or name == k:
      c = Component(name=k, params=params[k], train_locks=train_locks)
      components.append(c)
  return components

def params2comp_names(params):
  return list(params.keys())

# ------------------------------------------------------ #

def fingerprint_params(params):
  return np.sum(np.array(jax.tree.leaves(jax.tree.map(jnp.sum, params))))

class Component():
  counter = 0

  def reset_globals():
    Component.counter = 0

  def __init__(self, name:str, params, train_locks:set, opt_state=None):
    self.name = name
    self.params = jax.device_get(params)
    self.opt_state = jax.device_get(opt_state)
    self.num_params = None
    self.train_locks = set(train_locks)
    self.id = Component.counter
    Component.counter += 1

  def __str__(self):
    rtn = f'Component: {self.id}\n  Name: {self.name}'
    rtn += f'\n  Train locks: {self.train_locks}'
    rtn += f'\n  Fingerprint: {self.fingerprint()}'
    rtn += f'\n  Num params: {self.get_num_params()}'
    return rtn

  def get_num_params(self):
    if self.num_params is None:
      self.num_params = get_num_params(self.params)
    return self.num_params

  def fingerprint(self):
    return fingerprint_params(self.params)

  def is_trainable(self):
    return len(self.train_locks) == 0

  def clone(self):
    return Component(name=self.name,
                     params=copy.deepcopy(jax.device_get(self.params)),
                     train_locks=set(),
                     opt_state=copy.deepcopy(self.opt_state))

# ------------------------------------------------------ #

class ObjectCache():
  def __init__(self, factory_fn, max_size=None):
    self.factory_fn = factory_fn
    self.cache = {}
    self.max_size = max_size

  def __call__(self, *args, **kwargs):
    assert not args
    key = json.dumps(kwargs, sort_keys=True)
    if key not in self.cache:
      if self.max_size and self.max_size <= len(self.cache):
        rm_key = random.choice(list(self.cache.keys()))
        print(f'Removed from cache: {self.factory_fn.__name__}({rm_key})  [cache size {len(self.cache)}]')
        rm_obj = self.cache.pop(rm_key)
        del rm_obj
      self.cache[key] = self.factory_fn(**kwargs)
      if VERBOSE:
        print(f'Added to cache: {self.factory_fn.__name__}({key})  [cache size {len(self.cache)}]')
    else:
      if VERBOSE:
        print(f'Cache hit: {self.factory_fn.__name__}({key})  [cache size {len(self.cache)}]')
    return self.cache[key]

# ------------------------------------------------------ #

def incremental_mutation(value, values_list:list):
  assert value in values_list, f'{value} not in {values_list}'
  idx = values_list.index(value)
  idx += 1 if np.random.uniform() < 0.5 else -1
  idx = max(0, min(len(values_list) - 1, idx))
  return values_list[idx]

# ------------------------------------------------------ #

class Path():
  def reset_globals(exp_config):
    Path.exp_config = exp_config
    Path.counter = 0
    Path.paths = []
    Path.scorer = None  # To be set to scorer of choice during init of exp.
    # Cache output of functions calls with same args.
    Path.tasks = ObjectCache(get_task_factory_fn(exp_config))
    Path.posembed_components = ObjectCache(get_reshaped_posembed_component)
    Path.optimizers = ObjectCache(get_optimizer)
    Path.models = ObjectCache(get_vit_model)
    Path.init_params = ObjectCache(get_vit_params_mapped, max_size=1)

  def __init__(self, hparams, components, parent, task:Task):
    self.components = components
    self.id = Path.counter
    Path.counter += 1
    self.task = task
    self.parent = parent
    self.hparams = hparams
    self.metrics = {
        'offsprings': 0,
        'offsprings_per_task': json.dumps({}),
        'reloads': 0,
        'generation': 0 if parent is None else parent.metrics['generation'] + 1,
        'private': task.is_private(),
    }
    self.model = Path.models(
        num_classes = int(hparams['num_classes']),
        num_layers = int(hparams['num_layers']),
        adapter_layers = str(hparams['adapter_layers']),
        adapter_dim = int(hparams['adapter_dim']),
        query = str(self.exp_config.load_vit_checkpoint_query))
    Path.paths.append(self)

  def __str__(self):
    rtn = f'Path: {self.id}'
    rtn += f'\n  Components: {[c.id for c in self.components]}'
    if self.parent:
      rtn += f'\n  Parent: {self.parent.id}'
    rtn += f'\n  Task: {self.task.name}'
    rtn += f'\n  Total Parameters: {get_num_params(self.get_all_params())}'
    rtn += f'\n  Accounted params: {self.accounted_num_params()}'
    for k,v in self.hparams.items():
      rtn += f'\n    {k}: {v}'
    for k,v in self.metrics.items():
      rtn += f'\n    {k}: {v}'
    rtn += f'\n  Score: {self.score()}'
    return rtn

  def is_trainable(self):
    return self.task.is_trainable()

  def is_private(self):
    return self.task.is_private()

  def score(self):
    return Path.scorer.score(self)

  def get_all_params(self):
    params = {}
    for c in self.components:
      params[c.name] = c.params
    return flax.core.freeze(params)

  def get_trainable_params(self):
    params = {}
    for c in self.components:
      if c.is_trainable():
        params[c.name] = c.params
    return flax.core.freeze(params)

  def get_fixed_params(self):
    params = {}
    for c in self.components:
      if not c.is_trainable():
        params[c.name] = c.params
    return flax.core.freeze(params)

  def update_trainable(self, trained_params, opt_state):
    assert len(trained_params.keys()) == len(opt_state.keys())
    trainable_count = 0
    for c in self.components:
      if c.is_trainable():
        trainable_count += 1
        assert c.name in trained_params.keys()
        assert c.name in opt_state.keys()
        c.params = trained_params[c.name]
        c.opt_state = opt_state[c.name]
    assert len(trained_params.keys()) == trainable_count, f'{len(trained_params.keys())} {trainable_count}'

  def accounted_num_params(self):
    rtn = 0
    for c in self.components:
      tl = copy.copy(c.train_locks)
      assert type(tl) is set
      tl.add(self.task.name)
      if NOT_TRAINABLE in tl:
        tl.remove(NOT_TRAINABLE)
      if len(tl) == 0:
        return np.nan
      rtn += c.get_num_params() / len(tl)
    return rtn

  def clone(
      self,
      task:Task,
      ds_hparams,
      policy):
    exp_config = Path.exp_config
    assert exp_config == task.exp_config
    comps = []
    new_hparams = copy.deepcopy(self.hparams)
    new_hparams['num_classes'] = task.num_classes
    # Overwrite dataset hparams with those sampled for the generation batch.
    new_hparams.update(ds_hparams)

    def get_component_ref(c, clone=False):
      if c.is_trainable() or clone:
        # Clone trainable component.
        return c.clone()
      # Refer to frozen component.
      return c

    for k in sorted(exp_config.models_mutation_ranges):
      if (policy.do_mutate(new_hparams['_mu_']) and
          (k in ['_mu_', 'num_layers', 'adapter_dim']
            or k.startswith(OPTIMIZER_HPARAMS_KEYS_PRERFIX))):
        new_hparams[k] = incremental_mutation(
            new_hparams[k],
            exp_config.models_mutation_ranges[k])
    new_hparams['adapter_layers'] = mutate_adapters(
        exp_config.mutate_adapters,
        adapter_layers_ids=new_hparams['adapter_layers'],
        num_layers=new_hparams['num_layers'],
        mutation_prob=new_hparams['_mu_'],
        policy=policy)

    init_params = Path.init_params(
        **get_model_kwargs(new_hparams, exp_config))
    new_comp_names = params2comp_names(init_params)
    for new_comp_name in new_comp_names:
      comp = None
      # Attept to reuse matching component from closer ancestor.
      ancestor = self
      while ancestor is not None:
        comps_lookup = {c.name:c for c in ancestor.components}
        if new_comp_name in comps_lookup:
          # Head must be trainable if no acestor is of same task will fall back
          # to random init of correct shape.
          if new_comp_name == 'head' and not comps_lookup[new_comp_name].is_trainable():
            assert task.name != ancestor.task.name, f"{task.name} != {ancestor.task.name}"
            ancestor = ancestor.parent
            continue

          # Check shapes match otherwise skip.
          if jax.tree.map(jnp.shape, init_params[new_comp_name]) != jax.tree.map(jnp.shape, comps_lookup[new_comp_name].params):
            if new_comp_name == 'posembed_input':
              # Change of image size changed shape of position embeddings,
              # this can happen if ds_image_size is tuned,
              # continue searching through ancestors for matching size.
              assert 'ds_image_size' in exp_config.models_mutation_ranges
              assert new_hparams['ds_image_size'] != ancestor.hparams['ds_image_size']
              ancestor = ancestor.parent
              continue
            if new_comp_name.startswith('residual_adapter_'):
              # Change of adapter inner dimension changed shape of dense layers,
              # this can happen if adapter_dim is tuned,
              # continue searching through ancestors for matching size.
              assert 'adapter_dim' in exp_config.models_mutation_ranges
              assert new_hparams['adapter_dim'] != ancestor.hparams['adapter_dim']
              ancestor = ancestor.parent
              continue

            print(f'WARNING: Shapes do not match for component: {new_comp_name}  {ancestor.task.name}->{task.name}')
            print(jax.tree.map(jnp.shape, init_params[new_comp_name]))
            print(jax.tree.map(jnp.shape, comps_lookup[new_comp_name].params))
            assert False  # Should not happen in current configuration.

          comp = get_component_ref(comps_lookup[new_comp_name],
                                   clone=policy.do_mutate(new_hparams['_mu_'],
                                                          new_comp_name))
          break
        ancestor = ancestor.parent

      # Get reshaped posembed_input.
      if comp is None and new_comp_name == 'posembed_input':
        pe_comp = Path.posembed_components(
            image_size=new_hparams['ds_image_size'],
            query=exp_config.load_vit_checkpoint_query)
        # Clone to make the component trainable.
        comp = get_component_ref(pe_comp, clone=True)

      # Otherwise create one from random init params.
      if comp is None:
        if VERBOSE:
          print('Init:', new_comp_name)
        # Possible rand init triggering combinations in current configurations.
        assert (
            new_comp_name == 'head'
            or new_comp_name.startswith('residual_adapter_')
            or (new_comp_name.startswith('encoderblock_') and \
                exp_config.models_default_hparams['num_layers'] < max(
                exp_config.models_mutation_ranges.get('num_layers', [-1])))
            )
        comp = params2comps(init_params, train_locks=[], name=new_comp_name)[0]
      assert comp is not None
      comps.append(comp)

    rtn = Path(new_hparams, comps, parent=self, task=task)
    if task == self.task:
      self.metrics['offsprings'] = self.metrics.get('offsprings', 0) + 1

    offsprings_per_task = json.loads(self.metrics['offsprings_per_task'])
    offsprings_per_task[task.name] = offsprings_per_task.get(task.name, 0) + 1
    self.metrics['offsprings_per_task'] = json.dumps(offsprings_per_task)

    return rtn

  def get_optimizer(self):
    return Path.optimizers(
        lr=float(self.hparams['opt_lr']),
        lr_schedule=str(self.hparams['opt_lr_schedule']),
        lr_warmup_ratio=float(self.hparams['opt_lr_warmup_ratio']),
        momentum=float(self.hparams['opt_momentum']),
        nesterov=bool(self.hparams['opt_nesterov']),
        num_train_batches_between_validations=int(
            self.task.num_train_batches_between_validations),
        num_validations_per_path_training=int(
            self.task.exp_config.num_validations_per_path_training),
    )

# ------------------------------------------------------ #

def mutate_adapters(mutate, adapter_layers_ids, num_layers, mutation_prob, policy, allow_removal=False):
  a_ids = set(ids_str2ints(adapter_layers_ids))
  if mutate:
    for a_id in range(num_layers):
      if policy.do_mutate(mutation_prob):
        if a_id in a_ids:
          if allow_removal:
            a_ids.remove(a_id)
        else:
          a_ids.add(a_id)
  # Drop adapters of layers dropped by a possible mutation in num_layers.
  a_ids = [a_id for a_id in a_ids if a_id < num_layers]
  return ids_ints2str(a_ids)

# ------------------------------------------------------ #

class Scorer():
  def score(self, path):
    assert False, 'Not implemented'

class ScorerQuality(Scorer):
  def score(self, path):
    if ('quality' not in path.metrics
        or math.isnan(path.metrics['quality'])):
      return None
    assert path.metrics['quality'] >= 0, \
        f'{path.task.name} {path.metrics["quality"]}'
    score = path.metrics['quality']
    assert score >= 0
    return score

class ScorerDecay(Scorer):
  def __init__(self, base, num_params):
    self.base = base
    assert self.base > 0.0
    assert self.base <= 1.0
    self.num_params = num_params
    assert self.num_params > 0
  def score(self, path):
    if ('quality' not in path.metrics
        or math.isnan(path.metrics['quality'])):
      return None
    assert path.metrics['quality'] >= 0, \
        f'{path.task.name} {path.metrics["quality"]}'
    score = path.metrics['quality'] * (self.base ** (path.accounted_num_params() / self.num_params))
    assert score >= 0
    return score

# ------------------------------------------------------ #

class PPDecay():
  def __init__(self, exp_config):
    self.exp_config = exp_config

  def do_mutate(self, mutation_prob, comp_name=None):
    if comp_name:
      if comp_name in exp_config.force_finetune_components:
        return True
    return mutation_prob > np.random.uniform()

  def sample_parent(self, paths, task_name):
    for path in paths:
      offsprings = json.loads(path.metrics['offsprings_per_task']).get(task_name, 0)
      print(' ', path.id, offsprings)
      assert not math.isnan(offsprings)
      if np.random.uniform() < 0.5 ** offsprings:
        return path
    return None

  def sample_path(self, pop, task:Task, ds_hparams):
    parent = self.sample_parent(
        sorted(pop.paths[task], key=lambda p: p.score(), reverse=True),
        task.name)
    if not parent:
      print('  seeds')
      parent = self.sample_parent(pop.seed_paths, task.name)
      if parent:  # Rotate seeds.
        pos = pop.seed_paths.index(parent) + 1
        pop.seed_paths = pop.seed_paths[pos:] + pop.seed_paths[:pos]
    if not parent:  # Random sample.
      parent = random.choice(pop.paths[task] + pop.seed_paths)
      print('  random', parent.id)

    child = parent.clone(task, ds_hparams, self)

    gc.collect()

    # Store record of mutations.
    mutations = {}
    for k in child.hparams:
      if parent.hparams.get(k) != child.hparams[k]:
        mutations[k] = (parent.hparams.get(k), child.hparams[k])
    child.metrics['mutations'] = json.dumps(mutations)
    print(child.id, child.metrics['mutations'])
    return child

  def sample_ds_hparams(self, pop, task:Task):
    assert pop.exp_config is self.exp_config
    ds_hparams = {}
    for key in self.exp_config.models_default_hparams:
      if key.startswith(DATASET_HPARAMS_KEYS_PRERFIX):
        ds_hparams[key] = self.exp_config.models_default_hparams[key]
    best_path = pop.get_best_path(task)
    if best_path:
      ds_hparams.update(
          {k : best_path.hparams[k] for k in ds_hparams if k in best_path.hparams})
    for k in ds_hparams:
      if (k in self.exp_config.models_mutation_ranges
          and pop.policy.do_mutate(self.exp_config.models_default_hparams['_mu_'])):
        ds_hparams[k] = incremental_mutation(
            ds_hparams[k],
            self.exp_config.models_mutation_ranges[k])
    return ds_hparams

# ------------------------------------------------------ #

class PPBaseline():
  def __init__(self, exp_config):
    self.exp_config = exp_config

  def sample_parent(self, paths):
    assert False, 'Baselines should not reach evolutionary codepath.'

  def do_mutate(self, mutation_prob, comp_name=None):
    if comp_name:
      if comp_name in exp_config.force_finetune_components:
        return True
    if mutation_prob == 0.0:
      return False
    elif mutation_prob == 1.0:
      return True
    else:
      assert False, mutation_prob

  def sample_path(self, pop, task:Task, ds_hparams):
    assert len(pop.paths[not_trainable]) == 1
    parent = pop.paths[not_trainable][0]
    child = parent.clone(task, ds_hparams, self)
    return child

  def sample_ds_hparams(self, pop, task:Task):
    ds_hparams = {}
    for key in self.exp_config.models_default_hparams:
      if key.startswith(DATASET_HPARAMS_KEYS_PRERFIX):
        ds_hparams[key] = self.exp_config.models_default_hparams[key]
    return ds_hparams

# ------------------------------------------------------ #

class Population():
  def __init__(self, exp_config):
    self.paths = defaultdict(list)
    self.exp_config = exp_config
    self.paths_df = pd.DataFrame()
    self.comps_df = pd.DataFrame()
    self.policy = globals()[exp_config.policy_class](
        **exp_config.policy_kwargs,
        exp_config=exp_config)

  def get_best_path(self, task:Task):
    if len(self.paths[task]) == 0:
      return None
    # Most recent path achieving max score.
    return max(sorted(self.paths[task], key=lambda p: p.id, reverse=True),
               key=lambda p: p.score())

  def sample_path(self, task:Task, ds_hparams):
    return self.policy.sample_path(pop=self, task=task, ds_hparams=ds_hparams)

  def sample_ds_hparams(self, task:Task):
    ds_hparams = self.policy.sample_ds_hparams(pop=self, task=task)
    return ds_hparams

  def add_train_locks(self, task:Task):
    # Check.
    for ps in self.paths.values():
      for p in ps:
        for c in p.components:
          assert task.name not in c.train_locks
    # Add locks.
    paths = self.paths[task]
    for p in paths:
      for c in p.components:
        c.train_locks.add(task.name)

  def rm_train_locks(self, task:Task):
    # Remove locks.
    paths = self.paths[task]
    for p in paths:
      for c in p.components:
        if task.name in c.train_locks:
          c.train_locks.remove(task.name)
    # Check.
    for ps in self.paths.values():
      for p in ps:
        for c in p.components:
          assert task.name not in c.train_locks

  def set_seed_paths(self, task:Task):
    self.seed_paths = []
    for paths in self.paths.values():
      for path in paths:
        if path.task is task:
          continue
        if path.task.is_private():
          continue
        self.seed_paths.append(path)
    random.shuffle(self.seed_paths)

  def start_task(self, task:Task):
    self.set_seed_paths(task)
    self.rm_train_locks(task)

  def end_task(self, task:Task):
    # Keep only best one.
    best_path = self.get_best_path(task)
    assert best_path is not None
    self.paths[task] = [best_path]

    self.add_train_locks(task)
    self.garbage_collect_paths()

  def garbage_collect_paths(self):
    # Store stats before dropping references to trigger garbage collection
    # of unused paths, components and parameters.
    self.paths_df = self.paths_df._append(paths_to_df(Path.paths),
                                         ignore_index=True)
    self.comps_df = self.comps_df._append(components_to_df(Path.paths),
                                         ignore_index=True)

    # Drop unused paths generated in this task iteration for garbage collection.
    Path.paths = []
    # Simplify ancestor tree to contain only live paths.
    live_paths_ids = [p.id for paths in self.paths.values() for p in paths]
    # Notice that the simplification is done also for paths of other tasks,
    # since they may be pointing to a path of this task that was just pruned.
    for path in [path for paths in self.paths.values() for path in paths]:
      ancestor = path.parent
      if ancestor is None:
        continue
      while True:
        if ancestor.id in live_paths_ids:
          path.parent = ancestor
          break
        ancestor = ancestor.parent

# ------------------------------------------------------ #

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

def pop_to_df(pop):
  return paths_to_df([p for paths in pop.paths.values() for p in paths])

def paths_to_df(paths):
  # Collect all metrics names.
  metrics_keys = set()
  hparams_keys = set()
  for path in paths:
    metrics_keys.update(path.metrics)
    hparams_keys.update(path.hparams)

  data = defaultdict(list)
  for path in paths:
    data['task_name'].append(path.task.name)
    data['id'].append(path.id)
    data['parent_id'].append(path.parent.id if path.parent else -1)
    data['parent_task_name'].append(path.parent.task.name if path.parent else None)
    data['final_accounted_params'].append(path.accounted_num_params())
    data['components'].append('_'.join([str(c.id) for c in path.components]))
    for k in hparams_keys:
      data[f'hparams.{k}'].append(path.hparams[k] if k in path.hparams else None)
    for k in metrics_keys:
      data[f'metrics.{k}'].append(path.metrics[k] if k in path.metrics else None)
    data['score'].append(path.score())
  return pd.DataFrame(data)

def components_to_df(paths):
  # Collect all components.
  comps = set()
  for p in paths:
    comps.update(p.components)

  data = defaultdict(list)
  for c in comps:
    data['id'].append(c.id)
    data['name'].append(c.name)
    data['num_params'].append(c.get_num_params())
    data['train_locks'].append(','.join(c.train_locks))
  return pd.DataFrame(data)

def print_df_segments(df, segment_length:int = 10):
  tot_length = df.shape[0]
  # Pad column title with spaces to keep alignment across segments.
  def prepend_spaces(original_str, pad_to_len):
    return ' ' * (pad_to_len-len(original_str)) + original_str
  pad_to_len = max([len(tn) for tn in set(df['task_name'].to_list())])+1
  df = df.rename(columns={
    'task_name': prepend_spaces('task_name', pad_to_len),
    'parent_task_name': prepend_spaces('parent_task_name', pad_to_len),
    })
  for x in range(0, tot_length, segment_length):
    print(df[x:min(x+segment_length, tot_length)])

def df_leaderboard(df):
  df = df.loc[df['task_name'] != NOT_TRAINABLE]
  # Place columns on the left for readability.
  all_keys = sorted(df.columns.tolist())
  first_keys = ['task_name','score', 'metrics.quality', 'metrics.test_quality',
                'id', 'parent_id','parent_task_name', 'final_accounted_params']
  first_keys = [k for k in first_keys if k in all_keys]
  sorted_keys = first_keys + [k for k in all_keys if k not in first_keys]
  df = df[sorted_keys]
  print_df_segments(df)
  print(f'Avg score:        {df["score"].mean():.6f}')
  print(f'Avg quality:      {df["metrics.quality"].mean():.6f}')
  if 'metrics.test_quality' in df:
    print(f'Avg test quality: {df["metrics.test_quality"].mean():.6f}')

# ------------------------------------------------------ #

# Print path.
def prp(path):
  rtn = []
  if VERBOSE:
    rtn.append(str(path))
    for c in path.components:
      rtn.append(str(c))
  else:
    rtn.append(str(path.id))
  return '\n'.join(rtn)

# ------------------------------------------------------ #

def df_write_to_file(df, dir_path, df_name):
  filename_df = os.path.join(dir_path, f'{df_name}.csv')
  with tf.io.gfile.GFile(filename_df, 'w') as outfile:
    df.to_csv(outfile, index=False)

def df_read_from_file(dir_path, df_name):
  filename_df = os.path.join(dir_path, f'{df_name}.csv')
  with tf.io.gfile.GFile(filename_df, 'r') as infile:
    df = pd.read_csv(infile)
  # Pandas read_csv() reads empty stings as NaNs. Set NaNs to empty strings in
  # columns with type strings/object.
  for c in df.columns:
    if df[c].dtype == np.object_:
        df[c].fillna('', inplace=True)
  return df

def get_comps_params(pop:Population):
  comps_params = {}
  for c in set([c for paths in pop.paths.values() for p in paths for c in p.components]):
    comps_params[f'{c.name}:{c.id}'] = c.params
    if c.opt_state is not None:
      comps_params[f'opt_state:{c.name}:{c.id}'] = c.opt_state
  return comps_params

LAST_CHECKPOINT_TIME = time.time()
def checkpoint_save(
    experiment_dir:str, comps_params, generation_id:int, loop_id:int):
  flax_checkpoints.save_checkpoint(
      ckpt_dir=experiment_dir,
      target=comps_params,
      step=loop_id,
      prefix=f"checkpoint_{generation_id}_",
      overwrite=True)
  global LAST_CHECKPOINT_TIME
  LAST_CHECKPOINT_TIME = time.time()

def save_state(
    pop:Population,
    generation_id:int,
    loop_id:int,
    exp_config:FrozenConfigDict):
  # Save data needed to resume exp.
  write_st = time.time()
  df_leaderboard(pop_to_df(pop))
  pop.garbage_collect_paths()
  print('WRITING CHECKPOINT:', loop_id, generation_id)
  if loop_id == 0:
    tf.io.gfile.makedirs(exp_config.experiment_dir)
    json.dump(exp_config.as_configdict().to_dict(),
              tf.io.gfile.GFile(os.path.join(exp_config.experiment_dir,
                                      'config.json'),
                         'wb'), indent=2)
  state_dir = os.path.join(
      exp_config.experiment_dir, f"state_{loop_id}_{generation_id}")
  tf.io.gfile.makedirs(state_dir)
  # Write state in background threads.
  write_threads = []
  write_threads.append(Thread(target=df_write_to_file, args=(pop_to_df(pop), state_dir, 'population')))
  write_threads.append(Thread(target=df_write_to_file, args=(pop.paths_df, state_dir, 'paths')))
  write_threads.append(Thread(target=df_write_to_file, args=(pop.comps_df, state_dir, 'components')))
  write_threads.append(Thread(target=checkpoint_save, args=(state_dir, get_comps_params(pop), loop_id, generation_id)))
  for t in write_threads:
    t.start()
  print(f'WRITE START TIME: {time.time() - write_st:.2f} s')
  return write_threads

# ------------------------------------------------------ #

def load_population_from_checkpoint(
    pop:Population,
    ckpt_dir:str,
    population_df):
  loaded_params = flax.core.freeze(
      flax_checkpoints.restore_checkpoint(
          ckpt_dir=ckpt_dir,
          target=None))
  id_2_comp = {}
  for k in loaded_params.keys():
    if k.startswith('opt_state:'):
      continue
    name, id = k.split(':')
    if 'opt_state:'+k in loaded_params.keys():
      opt_state = loaded_params['opt_state:' + k]
    else:
      opt_state = None
    c = Component(name=name, params=loaded_params[k], train_locks=[], opt_state=opt_state)
    c.id = int(id)
    assert c.id not in id_2_comp
    id_2_comp[c.id] = c
  # For parent assignemt.
  id_2_path = {}
  path_2_parent_id = {}
  for index, row in population_df.iterrows():
    comps_ids = row['components'].split('_')
    comps = []
    for id in comps_ids:
      comps.append(id_2_comp[int(id)])
    task_name = row['task_name']
    if task_name == NOT_TRAINABLE:
      task = not_trainable
    else:
      task = Path.tasks(task_name=task_name)
    # Retrieve hparams and metrics.
    hparams = {}
    metrics = {}
    for k in row.keys():
      if k.startswith('hparams.'):
        hparams[k[len('hparams.'):]] = row[k]
      if k.startswith('metrics.'):
        metrics[k[len('metrics.'):]] = row[k]
    if type(hparams['adapter_layers']) is float:
      if math.isnan(hparams['adapter_layers']):
        hparams['adapter_layers'] = ''
      else:
        hparams['adapter_layers'] = str(int(hparams['adapter_layers']))
    metrics['reloads'] = metrics['reloads'] + 1
    # Create path.
    path = Path(
        hparams=hparams,
        components=comps,
        parent=None,
        task=task,
        )
    path.metrics = metrics
    path.id = int(row['id'])
    # Add train locks.
    for c in path.components:
      c.train_locks.add(task_name)
    pop.paths[task].append(path)
    assert path.id not in id_2_path
    id_2_path[path.id] = path
    if task_name != NOT_TRAINABLE:
      path_2_parent_id[path] = int(row['parent_id'])

  # Set parents.
  for path, parent_id in path_2_parent_id.items():
    path.parent = id_2_path[parent_id]
  Path.counter = 1 + max([id for id in id_2_path])
  Component.counter = 1 + max([id for id in id_2_comp])
  Path.paths = []

# ------------------------------------------------------ #

@partial(jax.jit, static_argnames='model')
def eval_step(params, images, labels, model):
  logits = model.apply({'params': params}, images, train=USE_DROPOUT)
  # Avg accuracy on the batch.
  return (logits.argmax(axis=-1) == labels).mean()

# ------------------------------------------------------ #

@partial(jax.jit, static_argnames=['model', 'optimizer'], donate_argnums=[0, 2])
def train_step(params, fixed_params, opt_state, images, labels, model, optimizer):
  def loss_fn(params, fixed_params, images, labels):
    logits = model.apply({'params': format_params(params, fixed_params)},
                         images, train=USE_DROPOUT)
    labels = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.mean(jnp.sum(labels * nn.log_softmax(logits), axis=-1))
  grads = jax.grad(loss_fn)(params, fixed_params, images, labels)
  updates, opt_state = optimizer.update(grads, opt_state, params=params)
  params = optax.apply_updates(params, updates)
  return params, opt_state

# ------------------------------------------------------ #

LOOP_START = time.time()

def train_loop(paths, ds_train, ds_validation, devices, exp_config):
  global LOOP_START
  timing = {'start_time': time.time(),
            'start_time_loop': LOOP_START}
  task = paths[0].task
  # The following values should be shared by all paths in this generation batch.
  for path in paths:
    assert task == path.task
    assert paths[0].hparams['ds_image_size'] == path.hparams['ds_image_size']

  gc.collect()

  # Compile.
  compile_train_batches_arr = jax.device_put_replicated(
      get_sample_batch(
        paths[0].hparams['ds_image_size'],
        task.train_batch_size),
      devices)
  compile_eval_batches_arr = jax.device_put_replicated(
      get_sample_batch(
          paths[0].hparams['ds_image_size'],
          task.validation_batch_size),
      devices)

  for p_id, path in enumerate(paths):
    if VERBOSE:
      print('Parent')
      print(prp(path.parent))
      print(prp(path))
    path.device_id = p_id % len(devices)
    path.device = devices[path.device_id]
    path.optimizer = path.get_optimizer()
    path.optimizer_init_fn = jax.jit(path.optimizer.init, device=path.device)
    path.best_params_local = None
    path.best_opt_state_local = None
    path.best_quality = None
    path.best_score = path.parent.score() if path.task is path.parent.task else -np.inf
    path.evals = []

    # Launch parallel compilation of eval and train step functions.
    params_local = path.get_trainable_params()
    path.compile_params_device = jax.device_put(params_local, path.device)
    path.compile_fixed_params_device = jax.device_put(
        path.get_fixed_params(),
        path.device)
    path.compile_train = Thread(
        target=train_step,
        args=(path.compile_params_device,
              path.compile_fixed_params_device,
              path.optimizer_init_fn(params_local),
              compile_train_batches_arr['image'][path.device_id],
              compile_train_batches_arr['label'][path.device_id],
              path.model,
              path.optimizer))
    path.compile_eval = Thread(
        target=eval_step,
        args=(format_params(
                  path.compile_params_device,
                  path.compile_fixed_params_device),
              compile_eval_batches_arr['image'][path.device_id],
              compile_eval_batches_arr['label'][path.device_id],
              path.model))
    path.compile_eval.start()

  for path in paths:
    path.compile_eval.join()
    del path.compile_eval
    timing['end_compile_eval'] = time.time()
    path.compile_train.start()
  del compile_eval_batches_arr

  for path in paths:
    path.compile_train.join()
    del path.compile_train
    del path.compile_params_device
    del path.compile_fixed_params_device
    timing['end_compile'] = time.time()
  del compile_train_batches_arr

  gc.collect()

  # Parameter transfer.
  for path in paths:
    path.params_device = jax.device_put(
        path.get_trainable_params(),
        path.device)
    path.fixed_params_device = jax.device_put(
        path.get_fixed_params(),
        path.device)
    path.opt_state_device = path.optimizer_init_fn(path.params_device)
    # Set opt state.
    for c in path.components:
      if c.is_trainable():
        assert c.name in path.opt_state_device[1][0].trace.keys()
        if c.opt_state is not None:
          path.opt_state_device = (
              path.opt_state_device[0],
              (optax.TraceState(
                  trace=path.opt_state_device[1][0].trace.copy(
                      {c.name: jax.device_put(c.opt_state,
                                              path.device)})),
               path.opt_state_device[1][1]
               )
          )

  iter_ds_validation = iter(ds_validation)
  # TRAIN
  for t_step, train_batch in zip(
      range(exp_config.num_validations_per_path_training
            * task.num_train_batches_between_validations),
      ds_train,
  ):
    train_batch_arr = jax.device_put_replicated(train_batch, devices)
    for p_id, path in enumerate(paths):
      if t_step == 0:
        timing['end_prep'] = time.time()
        t_step_0_time = time.time()
      path.params_device, path.opt_state_device = train_step(
          path.params_device,
          path.fixed_params_device,
          path.opt_state_device,
          train_batch_arr['image'][path.device_id],
          train_batch_arr['label'][path.device_id],
          path.model,
          path.optimizer)
      if t_step == 0 and time.time() - t_step_0_time > 1:
        print(f'WARNING: First train step took: {time.time()-t_step_0_time:.2f} s')
    del train_batch, train_batch_arr

    # EVAL
    if (t_step+1) % task.num_train_batches_between_validations == 0:
      first_eval = ((t_step+1) == task.num_train_batches_between_validations)
      if first_eval:
        timing['start_eval'] = time.time()
      for path in paths:
        path.accs = []
      for e_step, eval_batch in zip(
          range(task.num_validation_batches),
          iter_ds_validation,
          ):
        eval_batch_arr = jax.device_put_replicated(eval_batch, devices)
        for p_id, path in enumerate(paths):
          if first_eval and e_step == 0:
            e_step_0_time = time.time()
          path.accs.append(
              eval_step(
                  format_params(path.params_device, path.fixed_params_device),
                  eval_batch_arr['image'][path.device_id],
                  eval_batch_arr['label'][path.device_id],
                  path.model))
          if first_eval and e_step == 0 and time.time() - e_step_0_time > 1:
            print(f'WARNING: First eval step took: {time.time()-e_step_0_time:.2f} s')
      del eval_batch, eval_batch_arr

      # Get params of best models.
      qs = []
      eval_idx = (t_step+1) // task.num_train_batches_between_validations
      for path in paths:
        quality = np.mean(path.accs)
        del path.accs
        qs.append(f'{quality:.4f}')
        path.evals.append(quality)
        # Set quality in metrics for current score computation.
        path.metrics['quality'] = quality
        path_score = path.score()
        if path_score >= path.best_score:
          path.best_params_local = jax.device_get(path.params_device)
          path.best_opt_state_local = jax.device_get(path.opt_state_device[1][0].trace)
          path.best_score = path_score
          path.best_quality = quality
          qs[-1] += '*'
      train_time = time.time() - timing['end_compile']
      avg_path_time = (train_time / eval_idx) / len(paths)
      print(('\t'.join(qs) + f'\t< Eval {eval_idx}').expandtabs(8),
            f'tot:{train_time:.1f}s', f'avg/path:{avg_path_time:.1f}s')

      if first_eval:
        timing['end_eval'] = time.time()

  for path in paths:
    del path.params_device
    del path.fixed_params_device
    del path.opt_state_device
    del path.optimizer
    del path.optimizer_init_fn
  gc.collect()

  timing['end_train'] = time.time()

  loop_time = timing['start_time'] - LOOP_START
  compile_time = timing['end_compile'] - timing['start_time']
  compile_eval_time = timing['end_compile_eval'] - timing['start_time']
  compile_train_time = timing['end_compile'] - timing['end_compile_eval']
  prep_time = timing['end_prep'] - timing['end_compile']
  train_time = timing['end_train'] - timing['end_prep']
  eval_time = timing['end_eval'] - timing['start_eval']
  LOOP_START = time.time()

  for path in paths:
    path.metrics['loop_time'] = loop_time
    path.metrics['compile_time'] = compile_time
    path.metrics['prep_time'] = prep_time
    path.metrics['train_time'] = train_time
    path.metrics['eval_time'] = eval_time
    path.metrics['start_time'] = timing['start_time']
    path.metrics['start_time_loop'] = timing['start_time_loop']
    path.metrics['end_time'] = time.time()
    num_all_params = get_num_params(path.get_all_params())
    num_trainable_params = get_num_params(path.get_trainable_params())
    path.metrics['trainable_params_ratio'] = num_trainable_params/num_all_params
    path.metrics['num_trainable_params'] = num_trainable_params
    path.metrics['quality'] = max(path.evals)
    path.metrics['evals'] = json.dumps([float(v) for v in path.evals])
    path.metrics['training_accounted_params'] = path.accounted_num_params()
    path.metrics['training_score'] = path.score()

    if path.best_params_local:
      path.metrics['improved'] = True
      path.update_trainable(path.best_params_local,
                            path.best_opt_state_local)
      assert path.best_quality == path.metrics['quality']
      assert path.best_score == path.metrics['training_score']
    else:
      path.metrics['improved'] = False
      # Path will be early pruned if not an improvement, so skip parameters update.
      assert path.best_params_local == None
      assert path.best_opt_state_local == None
      assert path.best_quality == None

    del path.best_params_local
    del path.best_opt_state_local
    del path.best_score
    del path.best_quality
    del path.evals

    if VERBOSE:
      print('UPDATED:')
      print(prp(path))

  pqs = []
  qs = []
  psc = []
  sc = []
  for path in paths:
    if path.task is path.parent.task:
      pqs.append(f'{path.parent.metrics["quality"]:.4f}')
      psc.append(f'{path.parent.score():.4f}')
    else:
      pqs.append('NEW')
      psc.append('NEW')
    qs.append(f'{path.metrics["quality"]:.4f}')
    sc.append(f'{path.score():.4f}')
    if path.metrics['improved']:
      sc[-1] += '+'

  print(('\t'.join([f'{path.parent.id}' for path in paths]) +
        '\t< Parent id').expandtabs(8))
  print(('\t'.join([f'{path.id}' for path in paths]) +
        '\t< Path id').expandtabs(8))
  print(('\t'.join(pqs) + '\t< Parent best quality').expandtabs(8))
  print(('\t'.join(qs) + '\t< Path best quality').expandtabs(8))
  print(('\t'.join(psc) + '\t< Parent score').expandtabs(8))
  print(('\t'.join(sc) + '\t< Path score').expandtabs(8))

  print('time\tINIT\tCOMPevl\tCOMPtrn\tPREP\tTRN+EVL\t1stEVAL'.expandtabs(8))
  print(f'(s)\t{loop_time:.1f}\t{compile_eval_time:.1f}\t{compile_train_time:.1f}\t{prep_time:.1f}\t{train_time:.1f}\t{eval_time:.1f}'.expandtabs(8))

# ------------------------------------------------------ #

# Run a full paths sampling iteration for a task.
def task_iter(
    task, devices, pop:Population, generation_id:int, loop_id:int,
    exp_config:FrozenConfigDict):
  num_devices = len(devices)
  # Track best path.
  best_path = pop.get_best_path(task)
  num_gen_batches = math.ceil(exp_config.num_samples_per_task/num_devices)
  for _ in range(num_gen_batches):
    if generation_id >= num_gen_batches:
      break
    print('----')
    print(f'GENERATION: [{generation_id+1}/{num_gen_batches}]')
    ds_hparams = pop.sample_ds_hparams(task)
    ds_train = task.get_ds('train', ds_hparams)
    ds_validation = task.get_ds('validation', ds_hparams)
    paths = []
    for i in range(num_devices):
      paths.append(pop.sample_path(task, ds_hparams))
    train_loop(paths, ds_train, ds_validation, devices, exp_config)
    for path in paths:
      if path.metrics['improved']:
        assert path not in pop.paths
        pop.paths[task].append(path)
    # Track best path.
    curr_best_path = pop.get_best_path(task)
    if curr_best_path != best_path:
      if best_path:
        assert curr_best_path.score() >= best_path.score()
      best_path = curr_best_path
      best_path.metrics['new_best'] = True
      print(f'Best id:{best_path.id}',
            f'score:{best_path.score():.4f}',
            f'quality:{best_path.metrics["quality"]:.4f}',
            f'gen:{generation_id}',
            f'\n{best_path.hparams}')
    generation_id += 1
    if generation_id < num_gen_batches:
      # Skip intermediate state save if last state was written recently.
      if (time.time() - LAST_CHECKPOINT_TIME) > SKIP_INTERMEDIATE_STATE_SECS:
        save_state(pop, generation_id, loop_id, exp_config)
      else:
        print('Skip checkpointing, seconds since last save:',
              f'{time.time() - LAST_CHECKPOINT_TIME:.0f}')
  assert best_path in pop.paths[task]

# ------------------------------------------------------ #

TEST_MODELS_IMMUTABILITY = False

# Run final eval on test set.
def run_test_eval(path, ds_test):
  # Running on same device should allow to reuse the fn compiled for validation
  # if batch size matches.
  params = path.get_all_params()
  if not hasattr(path, 'device'):
    path.device = random.choice(jax.local_devices())
  params_device = jax.device_put(params_comps_to_model(params), path.device)
  acc_sum = []
  tot_num_samples = 0
  # Warning: if repeat() is called on this dataset, then this loop never ends.
  for batch in ds_test:
    acc_avg = eval_step(
        params_device,
        batch['image'],
        batch['label'],
        path.model)
    batch_size = batch['image'].shape[0]
    # Need to recompute sum because last batch can have different size to allow
    # for exact eval on the test set.
    acc_sum.append(acc_avg * batch_size)
    tot_num_samples += batch_size
  del params_device
  acc_avg = np.sum(acc_sum) / tot_num_samples
  if 'test_quality' in path.metrics and not math.isnan(path.metrics['test_quality']):
    assert np.isclose(path.metrics['test_quality'], acc_avg), \
        f'{path.task.name} {path.metrics["test_quality"]} {acc_avg}'
  path.metrics['test_quality'] = acc_avg

def run_all_test_evals(pop):
  eval_st = time.time()
  for path in [path for paths in pop.paths.values() for path in paths if path.is_trainable()]:
    if 'test_quality' in path.metrics and not math.isnan(path.metrics['test_quality']) and not TEST_MODELS_IMMUTABILITY:
      continue
    ds_test = path.task.get_ds('test', path.hparams)
    run_test_eval(path, ds_test)
  print(f'TEST EVAL TIME: {time.time() - eval_st:.2f} s')

# ------------------------------------------------------ #

def reset_globals(exp_config):
  Path.reset_globals(exp_config)
  Component.reset_globals()

# ------------------------------------------------------ #

def init_population(exp_config:FrozenConfigDict, continue_exp_dir:str = ''):
  reset_globals(exp_config)

  Path.scorer = globals()[exp_config.scorer_class](**exp_config.scorer_kwargs)
  pop = Population(exp_config=exp_config)

  def reload_state(load_exp_dir):
    pop.paths_df = df_read_from_file(
        load_exp_dir,
        df_name='paths')
    pop.comps_df = df_read_from_file(
        load_exp_dir,
        df_name='components')
    df_reloaded_population = df_read_from_file(
        load_exp_dir,
        df_name='population')
    load_population_from_checkpoint(
        pop,
        load_exp_dir,
        df_reloaded_population)
    print('Loaded models from', load_exp_dir, ':')
    df_leaderboard(pop_to_df(pop))
    Path.counter = 1 + int(pop.paths_df['id'].max())
    Component.counter = 1 + int(pop.comps_df['id'].max())

  # Load population from previous experiment.
  if continue_exp_dir:
    reload_state(continue_exp_dir)
    return pop
  elif exp_config.load_experiment:
    reload_state(exp_config.load_experiment_dir)

  # Add new seed models.
  if exp_config.load_rand_init or exp_config.load_vit_checkpoint:
    hparams = exp_config.models_default_hparams.as_configdict()
    # Add a randomly initialized model.
    if exp_config.load_rand_init:
      path0_params = get_vit_params_mapped(
          **get_model_kwargs(hparams, exp_config))
      path = Path(
          hparams,
          params2comps(path0_params, train_locks=[NOT_TRAINABLE]),
          parent=None,
          task=not_trainable)
      pop.paths[not_trainable].append(path)
    # Add model loaded from checkpoint.
    if exp_config.load_vit_checkpoint:
      path_params = get_vit_checkpoint_mapped(
          hparams['ds_image_size'],
          exp_config.load_vit_checkpoint_query)
      path = Path(hparams, params2comps(
          path_params,
          train_locks=[NOT_TRAINABLE]),
          parent=None,
          task=not_trainable)
      pop.paths[not_trainable].append(path)

  return pop

# ------------------------------------------------------ #

def latest_checkpoint(ckpt_dir: str, prefix: str = 'checkpoint_'):
  ckpt_dir = os.fspath(ckpt_dir)
  glob_path = os.path.join(ckpt_dir, f'{prefix}*')
  checkpoint_files = flax_checkpoints.natural_sort(tf.io.gfile.glob(glob_path))
  checkpoint_files = [f for f in checkpoint_files if not f.endswith('_tmp')]
  return checkpoint_files[-1] if checkpoint_files else None

def continue_exp(exp_dir):
  # Load configs.
  print('CONTINUING EXISTING EXPERIMENT:', exp_dir)
  load_config_dict_file = os.path.join(exp_dir, 'config.json')
  exp_config = FrozenConfigDict(json.load(
      tf.io.gfile.GFile(load_config_dict_file, 'r')))
  # Get loop_id from checkpoint file name.
  checkpoint_path = latest_checkpoint(exp_dir+'/state_*/')
  matched = re.findall(r'checkpoint_([0-9]+)_([0-9]+)$', checkpoint_path)
  assert len(matched)==1
  generation_id = int(matched[0][1])
  loop_id = int(matched[0][0])
  pop = init_population(exp_config, continue_exp_dir=exp_dir+f'/state_{loop_id}_{generation_id}/')
  print('FROM CHECKPOINT:', loop_id, generation_id)
  assert exp_config.experiment_dir == exp_dir
  return pop, exp_config, generation_id, loop_id

def setup_new_experiment(exp_config):
  # Finalize and save config.
  exp_config.experiment_id = exp_config.experiment_name \
      + datetime.datetime.strftime(
          datetime.datetime.now(), ':%Y-%m-%d-%H-%M-%S')
  exp_config.experiment_dir = os.path.join(
      exp_config.experiments_root_dir, exp_config.experiment_id)
  exp_config = FrozenConfigDict(exp_config)
  pop = init_population(exp_config)
  print('NEW EXPERIMENT:', exp_config.experiment_dir)
  return pop, exp_config, 0, 0

# ------------------------------------------------------ #

def setup_exp():
  if BENCHMARK == 'ViT tiny 3 layers / Chars benchmark':
    exp_config = get_exp_config_ti3_chars()
    exp_config.experiment_name += ':t3-chars'
  elif BENCHMARK == 'ViT base / VDD benchmark':
    exp_config = get_exp_config_base_deca()
    exp_config.experiment_name += ':b-deca'
  elif BENCHMARK == 'ViT tiny 0 layers / Cmaterdb benchmark':
    exp_config = get_exp_config_ti0_cmaterdb()
    exp_config.experiment_name += ':t0-cmaterdb'
  elif BENCHMARK.startswith('ViT large / '):
    exp_config = get_exp_config_large(BENCHMARK)
    exp_config.experiment_name += ':large'
  else:
    assert False, BENCHMARK

  if AUTO_TUNE:
    assert CONFIGURATION == 'muNet' or CONFIGURATION.startswith('Size scale:')
    exp_config.experiment_name += ':autotune'
    if BENCHMARK in ['ViT tiny 3 layers / Chars benchmark',
                      'ViT base / VDD benchmark']:
      exp_config = exp_config_add_auto_tune(exp_config)
    elif BENCHMARK in ['ViT tiny 0 layers / Cmaterdb benchmark'] or BENCHMARK.startswith('ViT large / '):
      exp_config = exp_config_add_auto_tune_v2(exp_config)
    else:
      assert False, BENCHMARK

  if CONFIGURATION == 'Finetune all':
    exp_config = exp_config_set_baseline_finetune_all(exp_config)
    exp_config.experiment_name += ':finetune'
  elif CONFIGURATION.startswith('Freeze bottom layers'):
    num_layers = int(CONFIGURATION.split(':')[1])
    exp_config = exp_config_set_baseline_freeze_bottom_layers(
        exp_config, num_layers)
    exp_config.experiment_name += f':freeze{num_layers}'
  elif CONFIGURATION.startswith('Adapters:'):
    adapter_dim = int(CONFIGURATION.split(':')[1])
    exp_config = exp_config_set_baseline_adapters(exp_config, adapter_dim)
    exp_config.experiment_name += f':adapters{adapter_dim}'
  elif CONFIGURATION.startswith('Size scale:'):
    base_percent = int(CONFIGURATION.split(':')[1])
    exp_config = exp_config_set_size_scale(exp_config, base_percent)
    exp_config.experiment_name += f':size{base_percent}'
  elif CONFIGURATION == 'muNet':
    exp_config.experiment_name += f':munet'
  else:
    assert False, CONFIGURATION

  if AUTO_CONTINUE:
    exp_dir_prefix = os.path.join(
      exp_config.experiments_root_dir, exp_config.experiment_name)
    matching_dirs = tf.io.gfile.glob(exp_dir_prefix + '*')
    assert len(matching_dirs) < 2, \
        f'Multiple dirs matched for auto restart {matching_dirs}'
    if len(matching_dirs) == 1:
      print('AUTO CONTINE')
      return continue_exp(matching_dirs[0])

  return setup_new_experiment(exp_config)

# ------------------------------------------------------ #

######################################################
##  - MAIN LOOP
######################################################

# ------------------------------------------------------ #

# Main loop over tasks.
pop, exp_config, generation_id, loop_id = setup_exp()

devices = jax.local_devices()
print('DEVICE COUNT:', len(devices))
num_tasks = len(exp_config.task_names)
num_loops = exp_config.num_task_iters * num_tasks
write_threads = []
for _ in range(num_loops):
  if loop_id >= num_loops:
    break
  t_i = loop_id // num_tasks
  task_idx = loop_id % num_tasks
  task_name = exp_config.task_names[task_idx]
  print('\n\n====')
  print(f'LOOP: [{loop_id+1}/{exp_config.num_task_iters * num_tasks}]')
  print(f'TASK: {task_name}')
  task = Path.tasks(task_name=task_name)
  pop.start_task(task)
  task_iter(task, devices, pop, generation_id, loop_id, exp_config)
  pop.end_task(task)
  loop_id += 1
  generation_id = 0

  run_all_test_evals(pop)
  write_threads = save_state(pop, generation_id, loop_id, exp_config)
  # Display stats.
  avg_time_per_sample = (
      pop.paths_df['metrics.end_time'].mean() \
          - pop.paths_df['metrics.start_time_loop'].mean()
      ) / len(devices)
  print(f'Avg time per path: {avg_time_per_sample:.2f} s')

# Wait for last state write to complete.
for t in write_threads:
  t.join()

# ------------------------------------------------------ #
