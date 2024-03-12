import random

import numpy as np
import torch


def fix_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.initial_seed()
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
