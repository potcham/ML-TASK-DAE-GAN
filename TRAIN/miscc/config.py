from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# Dataset name: flowers, birds
__C.DATASET_NAME = 'coco'
__C.CONFIG_NAME = 'DAE-GAN'
__C.DATA_DIR = 'Your_path/DAE-GAN/data/coco'
__C.GPU_ID = 1
__C.CUDA = True
__C.WORKERS = 1
__C.MAX_ATTR_NUM = 3
__C.MAX_ATTR_LEN = 5

__C.RNN_TYPE = 'LSTM'   # 'GRU'
__C.B_VALIDATION = True

__C.TREE = edict()
__C.TREE.BRANCH_NUM = 4
__C.TREE.BASE_SIZE = 64


# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 32
__C.TRAIN.MAX_EPOCH = 120
__C.TRAIN.SNAPSHOT_INTERVAL = 5
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.ENCODER_LR = 2e-4
__C.TRAIN.RNN_GRAD_CLIP = 0.25
__C.TRAIN.FLAG = False
__C.TRAIN.NET_E = 'Your_path/DAE-GAN/DAMSMencoders/coco/text_encoder100.pth'
__C.TRAIN.NET_G = 'Your_path/DAE-GAN/output/coco_DAGAN_2021_02_09_00_42_09/Model/netG_epoch_120.pth'
__C.TRAIN.B_NET_D = False

__C.TRAIN.SMOOTH = edict()
__C.TRAIN.SMOOTH.GAMMA1 = 4.0
__C.TRAIN.SMOOTH.GAMMA3 = 10.0
__C.TRAIN.SMOOTH.GAMMA2 = 5.0
__C.TRAIN.SMOOTH.LAMBDA = 50.0


# Modal options
__C.GAN = edict()
__C.GAN.DF_DIM = 32
__C.GAN.GF_DIM = 64
__C.GAN.Z_DIM = 100
__C.GAN.CONDITION_DIM = 100
__C.GAN.R_NUM = 3
__C.GAN.B_ATTENTION = True
__C.GAN.B_DCGAN = False


__C.TEXT = edict()
__C.TEXT.CAPTIONS_PER_IMAGE = 5
__C.TEXT.EMBEDDING_DIM = 256
__C.TEXT.WORDS_NUM = 20


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.safe_load(f))

    _merge_a_into_b(yaml_cfg, __C)
