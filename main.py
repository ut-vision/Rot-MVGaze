
import os, sys
import os.path as osp
import argparse
import cv2
import numpy as np
from glob import glob
import random

import h5py, copy
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
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from losses.stereo_loss import IterationLoss, StereoL1Loss
from losses.gaze_loss import GazeLoss

from utils.helper import AverageMeter

from dataset.xgaze import XGazeDataset
from utils.augment import RandomMultiErasing

from utils.math import rotation_matrix_2d, pitchyaw_to_vector, vector_to_pitchyaw, angular_error

from models.rot_mv import FeatRotationSymm
from trainer import Trainer

from utils.util import set_seed, build_model_from_cfg

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

proj_dir = os.path.dirname(os.path.realpath(__file__))
data_path_dict = OmegaConf.load( osp.join(proj_dir, 'data_path.yaml'))
xgaze_subject = OmegaConf.load(osp.join(proj_dir, 'configs/subject/xgaze.yaml'))['subject']
mpiinv_subject = OmegaConf.load(osp.join(proj_dir, 'configs/subject/mpiinv.yaml'))['subject']



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
		"--mode", type=str, choices=["train", "test",], default="train",
	)

	parser.add_argument(
		"--seed", type=int, help="random seed", default=0,
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
		"--epochs", type=int, help="number of epochs", default=15,
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


def configure_dataset(exp_name):
	dataset_setting = exp_name.split('_')[0]
	headpose_setting = exp_name.split('_')[1]
	if headpose_setting == 'known':
		camera_type_train, camera_type_test = 'all', 'all'
	elif headpose_setting == 'novel':
		camera_type_train, camera_type_test = 'novel_train', 'novel_test'

	if dataset_setting == 'xgaze2mpiinv':
		train_dataset = XGazeDataset(dataset_path=data_path_dict['xgaze'],
								color_type='bgr',
								image_transform=augment_transform,
								keys_to_use=xgaze_subject,
								camera_tag=camera_type_train,
								stereo=True,
								)
		test_dataset = XGazeDataset(dataset_path=data_path_dict['mpiinv'],
									color_type='bgr',
									image_transform=test_transform,
									keys_to_use=mpiinv_subject,
									camera_tag=camera_type_test,
									stereo=True,
									)

	elif dataset_setting == 'mpiinv2xgaze':
		train_dataset = XGazeDataset(dataset_path=data_path_dict['mpiinv'],
								color_type='bgr',
								image_transform=augment_transform,
								keys_to_use=mpiinv_subject,
								camera_tag=camera_type_train,
								stereo=True,
								)
		test_dataset = XGazeDataset(dataset_path=data_path_dict['xgaze'],
									color_type='bgr',
									image_transform=test_transform,
									keys_to_use=xgaze_subject,
									camera_tag=camera_type_test,
									stereo=True,
									)
	
	elif dataset_setting == 'xgaze':
		train_dataset = XGazeDataset(dataset_path=data_path_dict['xgaze'],
								color_type='bgr',
								image_transform=augment_transform,
								keys_to_use=xgaze_subject[:60],
								camera_tag='all',
								stereo=True,
								)
		test_dataset = XGazeDataset(dataset_path=data_path_dict['xgaze'],
									color_type='bgr',
									image_transform=test_transform,
									keys_to_use=xgaze_subject[60:],
									camera_tag='all',
									stereo=True,
									)
	elif dataset_setting == 'mpiinv':
		train_dataset = XGazeDataset(dataset_path=data_path_dict['mpiinv'],
								color_type='bgr',
								image_transform=augment_transform,
								keys_to_use=['p00.h5', 'p01.h5', 'p02.h5', 'p03.h5', 'p04.h5', 'p05.h5', 'p06.h5', 'p07.h5', 'p08.h5', 'p09.h5', 'p10.h5', 'p11.h5', 'p12.h5', 'p13.h5', 'p14.h5',],
								camera_tag='all',
								stereo=True,
								)
		test_dataset = XGazeDataset(dataset_path=data_path_dict['mpiinv'],
									color_type='bgr',
									image_transform=test_transform,
									keys_to_use=['p15.h5'],
									camera_tag='all',
									stereo=True,
									)

	else:
		raise NotImplementedError
	return train_dataset, test_dataset
if __name__ == '__main__':
	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	parser = get_parser()
	args, unknown = parser.parse_known_args()

	set_seed(args.seed)
	now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
	now_day = datetime.datetime.now().strftime("%Y-%m-%d")
	now_time = datetime.datetime.now().strftime("%H-%M-%S")
	output_dir = osp.join(args.output_dir, now_day, now_time)

	config = OmegaConf.create(vars(args))
	

	train_dataset, test_dataset = configure_dataset('xgaze_novel')	

	train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
	test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False, num_workers=config.num_workers)
	


	model = FeatRotationSymm(backbone_depth=50, num_iter=3, 
								  share_weights=False, 
								  encode_rotmat=False, 
								  share_feature=False, 
								  ignore_rotmat=False)


	

	stereo_l1_loss = StereoL1Loss(rel_weight=0.01, reference_decay=1.0, distance_metric='angular_error', pred_gaze_key='pred_gaze')
	metrics = IterationLoss(loss=stereo_l1_loss, iter_decay=0.5)

	ckpt_rotmv_xgaze2mpiinv_novel = '/home/jqin/wk/Rot-MVGaze/checkpoints/xgaze2mpiinv/symm-iter3-xgaze2mpii-novel--1/ckpt/train-model-last.pth'

	trainer = Trainer(
		config=config,
		model=model,
		metrics=metrics,
		train_loader=train_loader,
		test_loader=test_loader,
		ckpt_pretrained=ckpt_rotmv_xgaze2mpiinv_novel,
		output_dir=output_dir, 
	)

	if args.mode == 'train':
		trainer.train()
	else:
		trainer.test(-1)
   

