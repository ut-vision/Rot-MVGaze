
import os
import sys
import os.path as osp
import argparse
import cv2
import numpy as np
from glob import glob
import random
import time
import csv
import h5py
import copy
import pandas as pd
import importlib
from functools import partial
from omegaconf import OmegaConf
from tqdm import tqdm
from rich.progress import track
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import torchvision
from torchvision import transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from typing import Optional




from losses.stereo_loss import IterationLoss, StereoL1Loss
from losses.gaze_loss import GazeLoss

from utils.helper import AverageMeter

def set_seed(seed_value=42):
	random.seed(seed_value)
	np.random.seed(seed_value)
	torch.manual_seed(seed_value)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed_value)
		torch.cuda.manual_seed_all(seed_value)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

from dataset.xgaze import XGazeDataset
from utils.augment import RandomMultiErasing
from dataset.mpiigaze import MPIIGazeDataset

from utils.math import rotation_matrix_2d, pitchyaw_to_vector, vector_to_pitchyaw, angular_error

class Trainer(nn.Module):

	def __init__(self, 
			  model, 
			  metrics,
			  train_loader, 
			  test_loader,
			  ckpt_pretrained=None,
			  output_dir=None):
		super().__init__()

		self.train_loader = train_loader
		self.test_loader = test_loader
		self.model = model
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		if ckpt_pretrained is not None:
			ckpt = torch.load(ckpt_pretrained)
			self.model.load_state_dict(ckpt, strict=True)
			print('load from ckpt: ', ckpt_pretrained)
		
		self.model.to(self.device)

		
		self.metrics = metrics 

		self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
		self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)

		self.start_epoch = 0
		self.epochs = 15
		self.train_iter = 0
		self.output_dir = output_dir
		os.makedirs(self.output_dir, exist_ok=True)
		self.ckpt_dir = osp.join(self.output_dir, 'ckpt')
		os.makedirs(self.ckpt_dir, exist_ok=True)

		self.tensorboard_dir = osp.join(self.output_dir, 'tensorboard')
		os.makedirs(self.tensorboard_dir, exist_ok=True)
		self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

	def train(self):
		for epoch in range(self.start_epoch, self.epochs):
			self.train_one_epoch(epoch)
			error = self.test(epoch)
			
			if (epoch + 1) % (self.epochs//3) == 0:
				add_file_name = 'epoch_' + str(epoch+1).zfill(2) + '_error=' + str(round(error, 2))

				self.save_checkpoint(
					state=self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict(), 
					add=add_file_name
				)
	
	def prepare_dual_input(self, batch):
		img_0 = batch['img_0'].float().to(self.device)
		gt_gaze = batch['gt_gaze'].float().to(self.device)
		head_pose_0 = batch['head_pose_0'].float().to(self.device)  

		img_1 = batch['img_1'].float().to(self.device)
		gt_gaze_1 = batch['gt_gaze_1'].float().to(self.device)
		head_pose_1 = batch['head_pose_1'].float().to(self.device)

		rot_0 = rotation_matrix_2d(head_pose_0)  ## from canonical to head_0
		rot_1 = rotation_matrix_2d(head_pose_1)  ## from canonical to head_1

		data = {"img_0": img_0, "rot_0": rot_0, "gt_gaze": gt_gaze,
				"img_1":img_1,  "rot_1": rot_1, "gt_gaze_1": gt_gaze_1}
		return data
	


	def train_one_epoch(self, epoch):
		print(f'Epoch: {epoch + 1} / {self.epochs}')
		self.model.train()
		for i, data in enumerate(track(self.train_loader, description='Training', transient=True)):

			data = self.prepare_dual_input(data)
			data = self.model(data) 
			loss_gaze = self.metrics(data)


			pred_gaze = data["pred_gaze"]
			gaze_var = data["gt_gaze"]
			input_var = data["img_0"]

			error_gaze = np.mean(angular_error(pred_gaze.cpu().data.numpy(), gaze_var.cpu().data.numpy()))

			if self.train_iter!=0 and self.train_iter % 10 == 0:
				self.writer.add_scalar( 'train/loss_gaze', loss_gaze.item(), self.train_iter)
				self.writer.add_scalar( 'train/error_gaze', error_gaze.item(), self.train_iter)
				log_img = torchvision.utils.make_grid(input_var[:8], nrow=4, normalize=True)   
				self.writer.add_image( 'train/images', log_img, self.train_iter)

			
			self.optimizer.zero_grad()
			loss_gaze.backward()
			self.optimizer.step()


			self.train_iter += 1

		self.scheduler.step()
		

	def test(self, epoch):
		errors_gaze = AverageMeter()

		self.model.eval()
		for i, data in enumerate(track(self.test_loader, description='Testing', transient=True)):

			data = self.prepare_dual_input(data)
			data = self.model(data) 
			pred_gaze = data["pred_gaze"]
			gaze_var = data["gt_gaze"]
			input_var = data["img_0"]

			error_gaze = np.mean(angular_error(pred_gaze.cpu().data.numpy(), gaze_var.cpu().data.numpy()))

			if self.train_iter!=0 and self.train_iter % 10 == 0:
				log_img = torchvision.utils.make_grid(input_var[:8], nrow=4, normalize=True)   
				self.writer.add_image( 'test/images', log_img, self.train_iter)

			errors_gaze.update(error_gaze.item(),  data['image'].size(0))

		print(f'Epoch: {epoch + 1}, Error gaze: {errors_gaze.avg}')
		self.model.train()

		self.writer.add_scalar('test/epoch_error_gaze', errors_gaze.avg, epoch + 1)

		with open(osp.join(self.output_dir, 'test_results.txt'), 'a') as f:
			f.write('test on epoch {}, error: {}\n'.format(epoch + 1, errors_gaze.avg))
		return errors_gaze.avg

	def save_checkpoint(self, state, add=None):
		"""
		Save a copy of the model
		"""
		if add is not None:
			filename = add + '.pth.tar'
		else:
			filename = 'ckpt.pth.tar'
		ckpt_path = os.path.join(self.ckpt_dir, filename)
		torch.save(state, ckpt_path)
		print('save file to: ', ckpt_path)






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
   



def get_parser(**parser_kwargs):
	def str2bool(v):
		if isinstance(v, bool):
			return v
		if v.lower() in ("yes", "true", "t", "y", "1"):
			return True
		elif v.lower() in ("no", "false", "f", "n", "0"):
			return False
		else:
			raise argparse.ArgumentTypeError("Boolean value expected.")

	parser = argparse.ArgumentParser(**parser_kwargs)
	
	parser.add_argument(
		"-mode", "--mode", type=str, 
		# choices=["train", "test",],
		help=" train or test",
		default=False,
	)
	parser.add_argument(
		"--num_workers", type=int, help="num_workers", default=8,
	)

	parser.add_argument(
		"--batch_size", type=int, help="batch size", default=50,
	)

	parser.add_argument(
		"--test_batch_size", type=int, help="test_batch_size ", default=50,
	)
	parser.add_argument(
		"--epochs", type=int, help="number of epochs", default=25,
	)
	parser.add_argument(
		"--valid_epoch", type=int, help="frequency (num of epochs) of validating ", default=1,
	)
	parser.add_argument(
		"--eval_epoch", type=int, help="frequency (num of epochs) of evaluating ", default=10,
	)

	parser.add_argument(
		"--save_epoch", type=int, help="frequency (num of epochs) of saving ckpt ", default=10,
	)

	parser.add_argument(
		"-out", "--output_dir", help="path of the output", # default=False,
	)

	parser.add_argument(
		"--ckpt_resume", help="resume from checkpoint", default=None, type=str,
	)
	parser.add_argument(
		"--print_freq", help="loss print frequency", default=50, type=int,
	)

	parser.add_argument(
		"--model_cfg_path", help="path to the configuration file of the model", default=50, type=int,
	)
	
	return parser

def set_seed(seed):
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	# ensure reproducibility
	os.environ["PYTHONHASHSEED"] = str(seed)
	
if __name__ == '__main__':
	this_dir = os.path.dirname(os.path.realpath(__file__))
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	parser = get_parser()
	args, unknown = parser.parse_known_args()

	set_seed(0)
	now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
	now_day = datetime.datetime.now().strftime("%Y-%m-%d")
	now_time = datetime.datetime.now().strftime("%H-%M-%S")
	output_dir = osp.join(args.output_dir, now_day, now_time)

	created_subjects = ['subject0003.h5', 'subject0004.h5', 'subject0008.h5', 'subject0033.h5', 'subject0035.h5', 
						'subject0040.h5', 'subject0041.h5', 'subject0080.h5', 'subject0083.h5', 'subject0106.h5']
	


	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	image_size = 224
	augment_transform = transforms.Compose([
		transforms.ToPILImage(),
		transforms.ColorJitter(brightness=1.0, contrast=0.1, saturation=0.1),
		transforms.RandomAffine(degrees=0.0, scale=[0.99, 1.01], translate=[0.01, 0.01]),
		transforms.ToTensor(),
		transforms.Normalize(mean=mean, std=std),
		transforms.Resize((image_size, image_size), antialias=True),
		RandomMultiErasing(p=0.5, proportion=[0.5, 0.6], dot_size=[0.05, 0.3]),
	])

	test_transform = transforms.Compose([
					transforms.ToPILImage(),
					transforms.ToTensor(),
					transforms.Resize((image_size, image_size), antialias=True),
					transforms.Normalize(mean=mean, std=std)
				])

	xgaze_dataset = XGazeDataset(dataset_path="/home/jqin/wk/Datasets/xgaze_v2_128",
									color_type='bgr',
									image_transform=augment_transform,
									keys_to_use=['subject0000.h5', 'subject0003.h5', 'subject0005.h5'],
									camera_tag='novel_train',
									stereo=True,
									)

	# print( xgaze_dataset.idx_to_kv)
	# print( xgaze_dataset.key_idx_dict)

	print('xgaze_dataset num of samples: ', len(xgaze_dataset))
	train_loader = DataLoader(xgaze_dataset, batch_size=64, shuffle=True, num_workers=8)


	mpii_dataset = XGazeDataset(dataset_path="/home/jqin/wk/Datasets/xgaze_v2_128",
									color_type='bgr',
									image_transform=augment_transform,
									keys_to_use=['subject0000.h5', 'subject0003.h5', 'subject0005.h5'],
									camera_tag='novel_test',
									stereo=True,
									)
	
	print('mpii_dataset num of samples: ', len(mpii_dataset))
	test_loader = DataLoader(mpii_dataset, batch_size=128, shuffle=False, num_workers=8)
	

	from models.rot_mv import AblationFeatRotationSymm


	model = AblationFeatRotationSymm(backbone_depth=50, num_iter=3, 
								  share_weights=False, 
								  encode_rotmat=False, 
								  share_feature=False, 
								  ignore_rotmat=False)

	# model = build_model_from_cfg(args.model_cfg_path)

	summary(model)

	stereo_l1_loss = StereoL1Loss(rel_weight=0.01, reference_decay=1.0, distance_metric='angular_error', pred_gaze_key='pred_gaze')
	metrics = IterationLoss(loss=stereo_l1_loss, iter_decay=0.5)


	ckpt_rotmv_xgaze2mpiinv_novel = '/home/jqin/wk/Rot-MVGaze/checkpoints/xgaze2mpiinv/symm-iter3-xgaze2mpii-novel--1/ckpt/train-model-last.pth'

	trainer = Trainer(
		model=model,
		metrics=metrics,
		train_loader=train_loader,
		test_loader=test_loader,
		ckpt_pretrained=ckpt_rotmv_xgaze2mpiinv_novel,
		output_dir=output_dir, 
	)
	trainer.train()
   

