
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

import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(__file__))
from visualization import plot_cameras, draw_gaze


def mean_eye_nose(landmarks):
	assert landmarks.shape[0]==6
	# get the face center
	two_eye_center = torch.mean(landmarks[0:4, :], dim=0).view(1,-1)
	nose_center = torch.mean(landmarks[4:6, :], dim=0).view(1,-1)
	face_center = torch.mean(torch.cat((two_eye_center, nose_center), dim=0), dim=0).view(1,-1)
	return face_center


def pitchyaw_to_vector(pitchyaws):
	if pitchyaws.dim() == 1:
		pitchyaws = pitchyaws.unsqueeze(0)  ## (2,) -->  (1, 2)
	sin = torch.sin(pitchyaws)
	cos = torch.cos(pitchyaws)
	return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], 1).float()


def vector_to_pitchyaw(vectors):
	'''vector shape [n,3]'''
	vectors = torch.div(vectors, torch.norm(vectors, dim=1).unsqueeze(dim=1) )
	return torch.stack( [torch.asin(vectors[:, 1]), torch.atan2(vectors[:, 0], vectors[:, 2])], dim=1).float() ## stack [ theta, phi]

def rotation_matrix_2d( pitch_yaw):
	'''
	inverse = True if from label to canonical
	inverse = False if from canonical to label
	'''
	assert isinstance(pitch_yaw, torch.Tensor), 'make sure the pitchyaw here is torch.tensor'
	pitch_yaw_copy = pitch_yaw.clone()
	if pitch_yaw_copy.dim() == 1:
		pitch_yaw_copy = pitch_yaw_copy.unsqueeze(0)  ## (2,) -->  (1, 2)
	'''the definition of head pose requires -1'''
	pitch_yaw_copy[:, 0] *= -1

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
	matrices_1 = matrices_1.reshape( [-1, 3, 3])
	matrices_2 = matrices_2.reshape([-1, 3, 3])
	matrices = torch.matmul(matrices_2, matrices_1) ## (N, 3, 3)

	return matrices.float()

def get_rotation(source_pose, target_pose):
	source_rotation1 = rotation_matrix_2d(source_pose ) ## canonical --> source
	source_rotation1_inv =  source_rotation1.transpose(1, 2) ##  source --> canonical
	target_rotation2 = rotation_matrix_2d(target_pose ) ## canonical --> target

	rotation = torch.matmul(target_rotation2, source_rotation1_inv)
	return rotation


def to_canonical(head, gaze):
	gaze_vector = pitchyaw_to_vector(gaze).unsqueeze(dim=1)
	## Physical rotation matrix from head to canonical
	'''the -1 on yaw is included in rotation_matrix_2d now'''
	rotation_mat = rotation_matrix_2d(pitch_yaw=head ) # (N, 3, 3) cononical --> head
	rotation_mat = rotation_mat.transpose(1, 2) # (N, 3, 3) head -->  cononical
	gaze_vector_canonical = torch.matmul(gaze_vector, rotation_mat.transpose(1,2) ).squeeze(dim=1) # (N, 1, 3)
	gaze_canonical = vector_to_pitchyaw(gaze_vector_canonical) # (N, 2)
	return gaze_canonical

def canonical_to_camera(head, gaze_canonical):
	gaze_vector_canonical = pitchyaw_to_vector(gaze_canonical).unsqueeze(dim=1)
	## Physical rotation matrix from canonical to head
	'''the -1 on yaw is included in rotation_matrix_2d now'''
	# rotation_mat = rotation_matrix_2d(pitch_yaw=head * torch.tensor([-1,1]) ) # (N, 3, 3)
	rotation_mat = rotation_matrix_2d(pitch_yaw=head ) # (N, 3, 3) cononical --> head
	gaze_vector = torch.matmul(gaze_vector_canonical, rotation_mat.transpose(1,2)).squeeze(dim=1) # (N, 1, 3)
	gaze = vector_to_pitchyaw(gaze_vector) # (N, 2)
	return gaze





def head_pose_to_camera_pose(hR, ht):
	""" transform the head pose hr, ht to camera pose wrt face center
	Args:	
		hR: rotation matrix of the head 
		ht --> translation from the camera to the head coordinate origin e.g. (0, 0, 600)
			face points (68 x 3) in its defined coordinate: x ( the face model)
			face points (68 x 3) in this camera coordinate:  hR @ x + ht
	Return:
		hR.T
		- hR.T @ ht
	"""
	cam_rotation = hR.transpose(-2,-1)
	cam_translation = -torch.matmul(  ht.view(-1, 1, 3), cam_rotation.transpose(-2,-1) )  # the transpose here is because of the array multiplication

	return cam_rotation, cam_translation



def visualize_and_compare(cam_rotation, cam_translation, visual_output_dir:str):
	fig = plt.figure(figsize=(15, 15))
	ax = fig.add_subplot(projection='3d')
	ax.view_init(elev=10, azim=-120)

	## (Optional) plot an "world" coordinate origin
	plot_cameras(ax, np.eye(3), np.array([ 0, 0, -0.6 ]), -1, color='r')

	for i in range(cam_rotation.shape[0]):
		cam_r = cam_rotation[i].numpy()
		cam_t = cam_translation[i].numpy()
		plot_cameras(ax, cam_r, cam_t/1000, i)
	# ax = Axes3D(fig)
	l = 0.6
	ax.set_xlim([-l, l])
	ax.set_ylim([-l, l])
	ax.set_zlim([-l, l])
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	# plt.show()
	plt.savefig(os.path.join(visual_output_dir, 'visual.jpg'))
	plt.close()

