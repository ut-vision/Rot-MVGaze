import copy
from typing import Any, Dict, List, Union

import torch
import numpy as np


class AverageMeter(object):
	"""
	Computes and stores the average and
	current value.
	"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count



def recover_image( image_tensor, MEAN, STD):
	"""
	read a tensor and recover it to image in cv2 format
	args:
		image_tensor: [C, H, W] or [B, C, H, W]
	return:
		image_save: [B, H, W, C]
	"""
	if image_tensor.ndim == 3:
		image_tensor = image_tensor.unsqueeze(0)

	x = torch.mul(image_tensor, torch.FloatTensor(STD).view(3,1,1).to(image_tensor.device))
	x = torch.add(x, torch.FloatTensor(MEAN).view(3,1,1).to(image_tensor.device) )
	x = x.data.cpu().numpy()
	# [C, H, W] -> [H, W, C]
	image_rgb = np.transpose(x, (0, 2, 3, 1))
	# RGB -> BGR
	image_bgr = image_rgb[:, :, :, [2,1,0]]
	# float -> int
	image_save = np.clip(image_bgr*255, 0, 255).astype('uint8')

	return image_save