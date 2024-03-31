
import os
import sys
import os.path as osp
import cv2
import datetime
import numpy as np
from functools import partial
from omegaconf import OmegaConf
from tqdm import tqdm
from rich.progress import track
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from torchsummary import summary
from losses.stereo_loss import IterationLoss, StereoL1Loss
from losses.gaze_loss import GazeLoss

from utils.helper import AverageMeter
from utils.math import rotation_matrix_2d, pitchyaw_to_vector, vector_to_pitchyaw, angular_error

class Trainer(nn.Module):

	def __init__(self, 
			  config,
			  model, 
			  metrics,
			  train_loader, 
			  test_loader,
			  print_freq=50,
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
		summary(model)
		
		self.metrics = metrics 

		self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
		self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)

		self.start_epoch = 0
		self.epochs = 15
		self.train_iter = 0
		self.output_dir = output_dir
		os.makedirs(self.output_dir, exist_ok=True)
		
		OmegaConf.save(config, osp.join(self.output_dir, 'config.yaml'))
	
		self.ckpt_dir = osp.join(self.output_dir, 'ckpt')
		os.makedirs(self.ckpt_dir, exist_ok=True)

		self.tensorboard_dir = osp.join(self.output_dir, 'tensorboard')
		os.makedirs(self.tensorboard_dir, exist_ok=True)
		self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

		self.print_freq = print_freq

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

			if self.train_iter!=0 and self.train_iter % self.print_freq == 0:
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
		avg_error_gaze = AverageMeter()
		self.model.eval()
		for i, data in enumerate(track(self.test_loader, description='Testing', transient=True)):
			data = self.prepare_dual_input(data)
			data = self.model(data) 
			pred_gaze = data["pred_gaze"]
			gaze_var = data["gt_gaze"]
			input_var = data["img_0"]

			error_gaze = np.mean(angular_error(pred_gaze.cpu().data.numpy(), gaze_var.cpu().data.numpy()))

			if i !=0 and i % self.print_freq == 0:
				log_img = torchvision.utils.make_grid(input_var[:8], nrow=4, normalize=True)   
				self.writer.add_image( 'test/images', log_img, self.train_iter)
			avg_error_gaze.update(error_gaze,  gaze_var.size(0))
		
		msg = 'test on epoch {}, error: {}\n'.format(epoch, avg_error_gaze.avg)
		print( msg )
		self.writer.add_scalar('test/epoch_error_gaze', avg_error_gaze.avg, epoch)
		with open(osp.join(self.output_dir, 'test_results.txt'), 'a') as f:
			f.write(msg)
		return avg_error_gaze.avg

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


