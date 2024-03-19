import os
import os.path as osp
from os.path import dirname, abspath

DEFAULT_SEED = 42
DS_SEED = 123  # uses this seed when splitting datasets

# -------------- Paths
CONFIG_PATH = abspath(__file__)
SRC_ROOT = dirname(CONFIG_PATH)
PROJECT_ROOT = dirname(SRC_ROOT)
MODEL_DIR = osp.join(PROJECT_ROOT, 'models')


