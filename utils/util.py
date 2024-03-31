import os, importlib
from omegaconf import OmegaConf
import numpy as np
import random
import torch

def set_seed(seed=0):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		

def build_model_from_cfg(cfg_path):
	cfg = OmegaConf.load(cfg_path)
	"""
	cfg is like:
		type: networks.GazeResNet.GazeRes18
		params: {}
	"""
	module, cls = cfg['type'].rsplit(".", 1)
	MODEL = getattr(importlib.import_module(module, package=None), cls)
	model = MODEL(**cfg.get("params", dict()))
	return model
   