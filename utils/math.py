
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



def pitchyaw_to_vector(pitchyaws):
    r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.

    Args:
        pitchyaws: Input array of yaw and pitch angles, either numpy array or tensor.

    Returns:
        Output array of shape (n x 3) with 3D vectors per row, of the same type as the input.
    """
    if isinstance(pitchyaws, np.ndarray):
        return pitchyaw_to_vector_numpy(pitchyaws)
    elif isinstance(pitchyaws, torch.Tensor):
        return pitchyaw_to_vector_torch(pitchyaws)
    else:
        raise ValueError("Unsupported input type. Only numpy arrays and torch tensors are supported.")

def pitchyaw_to_vector_numpy(pitchyaws):
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out

def pitchyaw_to_vector_torch(pitchyaws):
    n = pitchyaws.size()[0]
    sin = torch.sin(pitchyaws)
    cos = torch.cos(pitchyaws)
    out = torch.empty((n, 3), device=pitchyaws.device)
    out[:, 0] = torch.mul(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = torch.mul(cos[:, 0], cos[:, 1])
    return out

def vector_to_pitchyaw(vectors):
    """Convert given gaze vectors to pitch (theta) and yaw (phi) angles.

    Args:
        vectors: Input array of gaze vectors, either numpy array or tensor.

    Returns:
        Output array of shape (n x 2) with pitch and yaw angles, of the same type as the input.
    """
    if isinstance(vectors, np.ndarray):
        return vector_to_pitchyaw_numpy(vectors)
    elif isinstance(vectors, torch.Tensor):
        return vector_to_pitchyaw_torch(vectors)
    else:
        raise ValueError("Unsupported input type. Only numpy arrays and torch tensors are supported.")

def vector_to_pitchyaw_numpy(vectors):
    n = vectors.shape[0]
    vectors = vectors / np.linalg.norm(vectors, axis=1).reshape(n, 1)
    out = np.empty((n, 2))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out

def vector_to_pitchyaw_torch(vectors):
    n = vectors.size()[0]
    vectors = vectors / torch.norm(vectors, dim=1).reshape(n, 1)
    out = torch.empty((n, 2), device=vectors.device)
    out[:, 0] = torch.asin(vectors[:, 1])  # theta
    out[:, 1] = torch.atan2(vectors[:, 0], vectors[:, 2])  # phi
    return out



def angular_error(a, b):
    """Calculate angular error (via cosine similarity)."""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return angular_error_numpy(a, b)
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return angular_error_torch(a, b)
    else:
        raise ValueError("Input type mismatch. Both inputs should be either numpy arrays or torch tensors.")

def angular_error_numpy(a, b):
    """Calculate angular error for numpy arrays."""
    a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
    b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))

    return np.arccos(similarity) * 180.0 / np.pi

def angular_error_torch(a, b):
    """Calculate angular error for torch tensors."""
    a = pitchyaw_to_vector(a) if a.size()[1] == 2 else a
    b = pitchyaw_to_vector(b) if b.size()[1] == 2 else b

    ab = torch.sum(a * b, dim=1)
    a_norm = torch.norm(a, dim=1)
    b_norm = torch.norm(b, dim=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = torch.clamp(a_norm, min=1e-7)
    b_norm = torch.clamp(b_norm, min=1e-7)

    similarity = ab / (a_norm * b_norm)

    return torch.acos(similarity) * 180.0 / np.pi



# def vector_to_pitchyaw(vec):
# 	unit_vec = vec / torch.norm(vec, dim=-1, keepdim=True)
# 	theta = torch.arcsin(unit_vec[..., 1])
# 	phi = torch.atan2(unit_vec[..., 0], unit_vec[..., 2])
# 	pitchyaw = torch.stack([theta, phi], dim=-1)
# 	return pitchyaw



# def pitchyaw_to_vector(pitchyaws):
# 	"""
# 	Args:
# 		pitchyaw:   [..., 2]
# 	Returns:
# 		unit_vec3d: [..., 3]
# 	"""
# 	pitch = pitchyaws[..., :1].clone()
# 	yaw = pitchyaws[..., 1:].clone()
# 	cos_pitch = torch.cos(pitch)
# 	sin_pitch = torch.sin(pitch)
# 	cos_yaw = torch.cos(yaw)
# 	sin_yaw = torch.sin(yaw)
# 	unit_vec3d = torch.cat(
# 		[cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw], dim=-1
# 	)
# 	return unit_vec3d



# def rotate_pitchyaw(rot, pitchyaw):
# 	"""
# 	Summary:
# 		returns `rot @ gaze` in pitchyaw format
# 	Args:
# 		rot:      [batch_size, ..., 3, 3]
# 		pitchyaw: [batch_size, ..., 2]
# 	Returns:
# 		rot_gaze: [batch_size, ..., 2]
# 	"""
# 	vec = pitchyaw_to_vector(pitchyaw)
# 	vec = rot @ vec.unsqueeze(-1)
# 	rot_gaze = vector_to_pitchyaw(vec.squeeze(-1))
# 	return rot_gaze

 


def rotation_matrix_2d( pitch_yaw, inverse=False):
	'''
	inverse = True if from label to canonical
	inverse = False if from canonical to label
	'''
	assert isinstance(pitch_yaw, torch.Tensor), 'make sure the pitchyaw here is torch.tensor'
	pitch_yaw_copy = pitch_yaw.clone()
	if pitch_yaw_copy.dim() == 1:
		pitch_yaw_copy = pitch_yaw_copy.unsqueeze(0)  ## (2,) -->  (1, 2)
	'''the definition of head pose requires -1'''
	# print('pitch_yaw_copy: ', pitch_yaw_copy.shape)
	pitch_yaw_copy = pitch_yaw_copy * torch.tensor([-1, 1], dtype=torch.float32, device=pitch_yaw.device)
	cos = torch.cos(pitch_yaw_copy)
	sin = torch.sin(pitch_yaw_copy)


	ones = torch.ones_like(cos[:, 0])
	zeros = torch.zeros_like(cos[:, 0])
	matrices_1 = torch.stack([ones, zeros, zeros,
								zeros, cos[:, 0], -sin[:, 0],
								zeros, sin[:, 0], cos[:, 0]
								], dim=1)
	matrices_2 = torch.stack([cos[:, 1], zeros, sin[:, 1],
								zeros, ones, zeros,
								-sin[:, 1], zeros, cos[:, 1]
								], dim=1)
	matrices_1 = matrices_1.view(-1, 3, 3)
	matrices_2 = matrices_2.view(-1, 3, 3)
	matrices = torch.matmul(matrices_2, matrices_1)
	if inverse:
		matrices = torch.transpose(matrices, 1, 2)
	return matrices



	
if __name__=="__main__":

	pass