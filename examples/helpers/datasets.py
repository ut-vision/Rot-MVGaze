from functools import partial

from torch import Generator  # NOQA
from torch.utils.data import DataLoader  # NOQA
from torchvision.transforms import *  # NOQA

from gzcv.data_loader import *
from gzcv.data_loader.dataset import *  # NOQA
from gzcv.data_loader.preprocess import *  # NOQA
from gzcv.utils import build_from_config

types = globals()

build_from_config = partial(build_from_config, namespace=types)
