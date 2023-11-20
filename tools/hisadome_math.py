
import math

import os, sys, yaml, pickle, shutil, tarfile #, glob
import cv2
import h5py
import random
import logging
import copy
import json
import numpy as np
import albumentations
import PIL
from collections import OrderedDict
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from glob import glob

import torch
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

# sys.path.insert(0,'/home/jqin/a_iccv23/pose_inv_gaze/src/tools')
from label_transform import draw_head_and_gaze






def vector_to_pitchyaw(vec):
	unit_vec = vec / torch.norm(vec, dim=-1, keepdim=True)
	theta = torch.arcsin(unit_vec[..., 1])
	phi = torch.atan2(unit_vec[..., 0], unit_vec[..., 2])
	pitchyaw = torch.stack([theta, phi], dim=-1)
	return pitchyaw



def pitchyaw_to_vector(pitchyaws):
	"""
	Args:
		pitchyaw:   [..., 2]
	Returns:
		unit_vec3d: [..., 3]
	"""
	pitch = pitchyaws[..., :1].clone()
	yaw = pitchyaws[..., 1:].clone()
	cos_pitch = torch.cos(pitch)
	sin_pitch = torch.sin(pitch)
	cos_yaw = torch.cos(yaw)
	sin_yaw = torch.sin(yaw)
	unit_vec3d = torch.cat(
		[cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw], dim=-1
	)
	return unit_vec3d



def rotate_pitchyaw(rot, pitchyaw):
	"""
	Summary:
		returns `rot @ gaze` in pitchyaw format
	Args:
		rot:      [batch_size, ..., 3, 3]
		pitchyaw: [batch_size, ..., 2]
	Returns:
		rot_gaze: [batch_size, ..., 2]
	"""
	vec = pitchyaw_to_vector(pitchyaw)
	vec = rot @ vec.unsqueeze(-1)
	rot_gaze = vector_to_pitchyaw(vec.squeeze(-1))
	return rot_gaze




def pitchyaw_to_rotmat(pitchyaw):
	is_torch = isinstance(pitchyaw, torch.Tensor)
	# print('pitchyaw = ', pitchyaw)
	# print('fliplr', torch.flip(pitchyaw, dims=[0]))
	rotmat = rotate_from_euler("xy", pitchyaw)
	# rotmat = SciRotation.from_euler("yx", pitchyaw, degrees=False).as_matrix()
	# rotmat = SciRotation.from_euler("yx", torch.flip(pitchyaw, dims=[0]), degrees=False).as_matrix()

	return rotmat



def rotate_from_euler(order: str, degrees):
	"""
	order: str, sequenece of [x, y, z]
	degrees: torch.tensor, [..., 3]
	"""
	device = degrees.device
	ones = torch.ones(*degrees.shape[:-1], 1, device=device)
	zeros = torch.zeros(*degrees.shape[:-1], 1, device=device)
	rotations = {}
	if "x" in order:
		dim = order.index("x")
		degrees_x = degrees[..., dim : dim + 1]
		rotations["rx"] = torch.stack(  # NOQA
			[
				torch.cat([ones, zeros, zeros], dim=-1),
				torch.cat([zeros, degrees_x.cos(), -degrees_x.sin()], dim=-1),
				torch.cat([zeros, degrees_x.sin(), degrees_x.cos()], dim=-1),
			],
			dim=-1,
		)
	if "y" in order:
		dim = order.index("y")
		degrees_y = degrees[..., dim : dim + 1]
		rotations["ry"] = torch.stack(  # NOQA
			[
				torch.cat([degrees_y.cos(), zeros, degrees_y.sin()], dim=-1),
				torch.cat([zeros, ones, zeros], dim=-1),
				torch.cat([-degrees_y.sin(), zeros, degrees_y.cos()], dim=-1),
			],
			dim=-1,
		)
	if "z" in order:
		dim = order.index("z")
		degrees_z = degrees[..., dim : dim + 1]
		rotations["rz"] = torch.stack(  # NOQA
			[
				torch.cat([degrees_z.cos(), -degrees_z.sin(), zeros], dim=-1),
				torch.cat([degrees_z.sin(), degrees_z.cos(), zeros], dim=-1),
				torch.cat([zeros, zeros, ones], dim=-1),
			],
			dim=-1,
		)
	r = rotations[f"r{order[0]}"]
	for axis in order[1:]:
		r = r @ rotations[f"r{axis}"]
	return r.transpose(-1, -2).to(torch.float)



def to_canonical(rotation_mat, gaze):
	gaze_vector = pitchyaw_to_vector(gaze).unsqueeze(dim=1)
	
	gaze_vector_canonical = torch.matmul(gaze_vector, rotation_mat.transpose(1,2) ).squeeze(dim=1) # (N, 1, 3)

	gaze_canonical = vector_to_pitchyaw(gaze_vector_canonical) # (N, 2)
	return gaze_canonical

	
if __name__=="__main__":

	# head = torch.tensor( [-0.2, -0.3])[None,...]
	# gaze = torch.tensor( [0.5, 0.2])[None,...]

	rot = pitchyaw_to_rotmat(  torch.tensor( [0, 0]) )
	print( 'rot: ', rot )
	# gaze_ = rotate_pitchyaw(rot.transpose(1,2), gaze)
	# print('gaze_: ', gaze_)

	sample_dir = '/home/jqin/a_iccv23/toy_data/hisadome_samples'
	os.makedirs(sample_dir, exist_ok=True)
	with h5py.File('/work/jqin/xgaze_224/train/subject0000.h5', 'r') as f:
		for i in range(18):
			image = f['face_patch'][i]
			gaze = f['face_gaze'][i]
			head_pose = f['face_head_pose'][i]
			image_out = draw_head_and_gaze(image, head_pose, gaze)

			print(" gaze: ", gaze)

			rot = pitchyaw_to_rotmat( torch.from_numpy(head_pose *np.array([-1,1]))) 

			gaze_canon = rotate_pitchyaw( rot.T, torch.from_numpy(gaze).float() )

			print(" gaze_canon: ", gaze_canon)

			image_out_canon = draw_head_and_gaze(image, np.zeros(2,), gaze_canon)
			gaze_recover = rotate_pitchyaw( rot, gaze_canon )

			
			# print(" gaze_recover: ", gaze_recover)

			cv2.imwrite( sample_dir +'/image_{}.jpg'.format( i ),  cv2.hconcat([image_out, image_out_canon]) )