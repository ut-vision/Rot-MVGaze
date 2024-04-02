import os, json, yaml, random
import os.path as osp
import h5py, json, copy
from glob import glob
from typing import List
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset


from omegaconf import OmegaConf, ListConfig, DictConfig
from typing import List, Dict, Any

from rich.progress import track

class GazeDataset(Dataset):
	def __init__(self, 
			  	dataset_name: str,
				dataset_path: str, 
				color_type,
				image_transform,
				keys_to_use: List[str] = None, 
				camera_tag='all',
				stereo=False,
				):
		self.dataset_name = dataset_name
		assert self.dataset_name in ['xgaze', 'mpiinv'], 'currently only support xgaze and mpiinv'
		self.stereo = stereo
		self.path = dataset_path
		self.hdfs = {}

		assert color_type in ['rgb', 'bgr']
		self.color_type = color_type

		self.camera_tag = camera_tag
		camera_tags = {
			'all': list(range(18)),
			'novel_train': [ x for x in range(18) if x not in list(range(2, 18, 3))],
			'novel_test': list(range(2, 18, 3)),

		}

		self.cameras_idx = camera_tags[self.camera_tag]

		#### -------------------------------------------------------- read the h5 files ------------------------------------------------------- 
		self.selected_keys = [k for k in keys_to_use]
		assert len(self.selected_keys) > 0
		self.file_paths = [os.path.join(self.path, k) for k in self.selected_keys]
		for num_i in range(0, len(self.selected_keys)):
			file_path = os.path.join(self.path, self.selected_keys[num_i]) # the subdirectories: train, test are not used in MPIIFaceGaze and MPII_Rotate
			self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
			print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
			assert self.hdfs[num_i].swmr_mode
		####----------------------------------------------------------------------------------------------------------------------------------- 

		
		self.idx_to_kv = []
		for num_i in track( range(0, len(self.selected_keys)) , description="Building stereo pair index"):
			n = self.hdfs[num_i]['face_patch'].shape[0]
			start_idx, end_idx = 0, n
			
			valid_indices = [i for i in range(start_idx, end_idx) if (i % 18) in self.cameras_idx]

			for idx in valid_indices:
				frame_number = idx // 18
				frame_start_idx = frame_number * 18
				frame_valid_indices = [i for i in range(frame_start_idx, frame_start_idx + 18) if i in valid_indices and i != idx]
				if frame_valid_indices:
					idx_b = random.choice(frame_valid_indices)
					self.idx_to_kv.append((num_i, idx, idx_b))
			

			
			
		for num_i in range(0, len(self.hdfs)):            
			if self.hdfs[num_i]:
				self.hdfs[num_i].close()
				self.hdfs[num_i] = None



		self.__hdfs = None
		self.hdf = None

		self.transform = image_transform

	def __len__(self):
		return len(self.idx_to_kv)

	def __del__(self):
		for num_i in range(0, len(self.hdfs)):
			if self.hdfs[num_i]:
				self.hdfs[num_i].close()
				self.hdfs[num_i] = None
 
	@property
	def archives(self):
		if self.__hdfs is None: # lazy loading here!
			self.__hdfs = [h5py.File(h5_path, "r", swmr=True) for h5_path in self.file_paths]
		return self.__hdfs


	def preprocess_image(self, image):
		image = image.astype(np.float32)
		if self.color_type == 'bgr':
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = self.transform(image.astype(np.uint8)		)
		return image

	def __getitem__(self, index):
		
		key, idx, idx_b = self.idx_to_kv[index]
		self.hdf = self.archives[key]
		# self.hdf = h5py.File(os.path.join(self.path, self.selected_keys[key]), 'r', swmr=True)
		assert self.hdf.swmr_mode

		image = self.hdf['face_patch'][idx, :]
		gaze = self.hdf['face_gaze'][idx].astype('float')
		head_pose = self.hdf['face_head_pose'][idx].astype('float')
		if self.dataset_name == 'mpiinv':
			head_pose[0] *= -1
		data = {
			'img_0': self.preprocess_image(image),
			'gt_gaze': gaze,
			'head_pose_0': head_pose,
			'idx_0': idx,
		}
		
		if self.stereo:
			gt_gaze_1 = self.hdf['face_gaze'][idx_b].astype('float')
			head_pose_1 = self.hdf['face_head_pose'][idx_b].astype('float')
			if self.dataset_name == 'mpiinv':
				head_pose_1[0] *= -1
			data.update({
				'img_1': self.preprocess_image(self.hdf['face_patch'][idx_b, :]),
				'gt_gaze_1': gt_gaze_1,
				'head_pose_1': head_pose_1,
				'idx_1': idx_b,
			})

		return data




if __name__ == "__main__":
	pass