from functools import partial

from gzcv.models import *
from gzcv.utils import build_from_config

types = globals()

build_from_config = partial(build_from_config, namespace=types)
