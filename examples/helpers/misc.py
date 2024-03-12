from functools import partial

from gzcv.analyze import *
from gzcv.tools import *
from gzcv.utils import *

types = globals()

build_from_config = partial(build_from_config, namespace=types)
