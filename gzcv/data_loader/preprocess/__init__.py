from typing import List

from .augment import *
from .io import *
from .transform import *


class ComposePreprocess(object):
    def __init__(self, preprocesses: List):
        self.preprocesses = preprocesses

    def __call__(self, data):
        for preprocess in self.preprocesses:
            data = preprocess(data)
        return data
