import numpy as np
from typing import Any, Callable, List, Optional, Type, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torchvision.models as models
from models.resnet import resnet18, resnet50
from models.backbones.blocks import Mlp


class IntensityBatchNorm(nn.Module):
	def __init__(self, n_channels: int, momentum: float = 0.05, eps: float = 1e-4) -> None:
		super().__init__()
		self.register_buffer("running_mean", torch.ones(1, 1, n_channels))
		self._momentum = momentum
		self._eps = eps
		self._tracking_stats = True

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		x: [batch_size, 3, feat_dim]
		"""
		intensity = torch.norm(x, dim=-2, keepdim=True).detach() # [batchsize, 1, feat_dim]

		var = torch.var(intensity, unbiased=False, dim=0, keepdim=True)
		std = torch.sqrt(var.clamp_min(self._eps)) # avoid nan during backward
		if self.training:
			self.running_mean = self.running_mean * (1 - self._momentum) + std * self._momentum
		
		return x / (self.running_mean + self._eps)


class ImageFeatFuser(nn.Module):
	def __init__(self, img_feat_dim: int, num_feat_vec: int) -> None:
		super().__init__()
		self.in_channel = img_feat_dim + num_feat_vec * 3
		self.num_feat_vec = num_feat_vec
		self._fuser = Mlp(
			self.in_channel, [self.in_channel, num_feat_vec * 3]
			# self.in_channel, [self.in_channel, self.in_channel, num_feat_vec * 3]
		)
	def forward(
		self, img_feat: torch.Tensor, rotatable_feat: torch.Tensor
	) -> torch.Tensor:
		# rotatable_feat = self._bn_in(rotatable_feat)
		in_feat = torch.cat([img_feat, rotatable_feat.flatten(-2, -1)], dim=-1)
		out_feat = self._fuser(in_feat)
		return out_feat

	
class ImageRotmatFeatFuser(nn.Module):
	def __init__(self, img_feat_dim: int, num_feat_vec: int) -> None:
		super().__init__()
		self.in_channel = img_feat_dim + num_feat_vec * 3 + 9
		self.num_feat_vec = num_feat_vec
		self._fuser = Mlp(
			self.in_channel, [self.in_channel, self.in_channel, num_feat_vec * 3]
		)

	def forward(
		self, img_feat: torch.Tensor, rotatable_feat: torch.Tensor, rot: torch.Tensor
	) -> torch.Tensor:
		in_feat = torch.cat([img_feat, rotatable_feat.flatten(-2, -1), rot.flatten(-2, -1)], dim=-1)
		out_feat = self._fuser(in_feat)
		return out_feat


class RotFeatFuser(nn.Module):
	def __init__(self, num_feat_vec: int) -> None:
		super().__init__()
		self.num_feat_vec = num_feat_vec
		self.in_channel = num_feat_vec * 6
		self._fuser = Mlp(
			self.in_channel, [self.in_channel, self.in_channel, 3 * num_feat_vec]
		)
		self._batchnorm = IntensityBatchNorm(num_feat_vec)

	def forward(self, feat_0: torch.Tensor, feat_1: torch.Tensor) -> torch.Tensor:
		feat_norm_0 = self._batchnorm(feat_0)
		feat_norm_1 = self._batchnorm(feat_1)
		in_feat = torch.cat([feat_norm_0, feat_norm_1], dim=-1).flatten(-2, -1)
		out_feat = self._fuser(in_feat)
		return out_feat.reshape(-1, 3, self.num_feat_vec)





class Feat3dLifter(nn.Module):
	def __init__(self, in_feat_dim: int, num_feat_vec: int) -> None:
		super().__init__()
		self.num_feat_vec = num_feat_vec
		self._lifter = Mlp(in_feat_dim, [num_feat_vec * 3, num_feat_vec * 3])

	def forward(self, in_feat: torch.Tensor) -> torch.Tensor:
		return self._lifter(in_feat).reshape(-1, 3, self.num_feat_vec)



class FeatRotationSymm(nn.Module):
	def __init__(
		self,
		backbone_depth: int = 50,
		num_iter: Optional[int] = None,
		share_weights: bool = False,
		encode_rotmat: bool = False,
		share_feature: bool = False,
		ignore_rotmat: bool = False,
	) -> None:
		super().__init__()
		
		self._num_iter = num_iter
		self._output_index = num_iter - 1

		self._num_feat_vec = 512
		
		if backbone_depth ==50:
			resnet = resnet50(pretrained=True)
		elif backbone_depth==18:
			resnet = resnet18(pretrained=True)

		self._feat_extractor = nn.Sequential(
			resnet,
			resnet.avgpool, ### added, because my resnet forward do not pass the avgpool
			nn.Flatten(start_dim=-3, end_dim=-1),
		)
		self._fc_dim = resnet.fc.in_features

		self._lifter = Feat3dLifter(self._fc_dim, self._num_feat_vec)

		assert not (ignore_rotmat and encode_rotmat)
		self._ignore_rotmat = ignore_rotmat
		self._encode_rotmat = encode_rotmat
		if self._encode_rotmat:
			fuser_type = ImageRotmatFeatFuser
		else:
			fuser_type = ImageFeatFuser

		if self._ignore_rotmat:
			fuser_type = ImageFeatFuser
		
		self._share_feature = share_feature

		# self._logger = logging.getLogger(self.__class__.__name__)
		# self._logger.info(
		# 	f"encode_rotmat: {self._encode_rotmat}, ignore_rotmat: {self._ignore_rotmat}, share_weights: {share_weights}, share feat: {self._share_feature}"
		# )
		
		if share_weights:
			self._img_fusers = nn.ModuleList(
				[fuser_type(self._fc_dim, self._num_feat_vec)] * num_iter
			)
			self._gaze_estimators = nn.ModuleList(
				[Mlp(self._num_feat_vec * 3 + self._fc_dim, out_channels=[512, 2])]
				* num_iter
			)
		else:
			if share_feature:
				self._img_fusers = nn.ModuleList(
					[
						RotFeatFuser(self._num_feat_vec) for _ in range(num_iter)
					]
				)
				self._gaze_estimators = nn.ModuleList(
					[
						Mlp(self._num_feat_vec * 6, out_channels=[512, 2])
						for _ in range(num_iter)
					]
				)
			else:
				self._img_fusers = nn.ModuleList(
					[
						fuser_type(self._fc_dim, self._num_feat_vec)
						for _ in range(num_iter)
					]
				)
				self._gaze_estimators = nn.ModuleList(
					[
						Mlp(self._num_feat_vec * 3 + self._fc_dim, out_channels=[512, 2])
						for _ in range(num_iter)
					]
				)


	def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
		img_0: torch.Tensor = data["img_0"]
		img_1: torch.Tensor = data["img_1"]
		rot_0: torch.Tensor = data["rot_0"]
		rot_1: torch.Tensor = data["rot_1"]

		rot_10 = rot_0 @ rot_1.transpose(-1, -2) ## [batch, 3, 3]
		rot_01 = rot_1 @ rot_0.transpose(-1, -2) ## [batch, 3, 3]

		img_feat_0 = self._feat_extractor(img_0)
		img_feat_1 = self._feat_extractor(img_1) ## [batch, self._fc_dim]
		rotatable_feat_0 = self._lifter(img_feat_0)
		rotatable_feat_1 = self._lifter(img_feat_1) ## [batch, 3, self._num_feat_vec] 

		if self._share_feature:
			img_feat_0 = rotatable_feat_0
			img_feat_1 = rotatable_feat_1

		pred = {
			"num_iter": self._num_iter,
			"img_feat_0": img_feat_0,
			"img_feat_1": img_feat_1,
			"initial_rot_feat_0": rotatable_feat_0,
			"initial_rot_feat_1": rotatable_feat_1,
		}

		for f_i, (img_fuser, gaze_estimator) in enumerate(
			zip(self._img_fusers, self._gaze_estimators)
		):
			pred_iter = {}
			rotatable_feat_0_swap = rotatable_feat_0

			if self._encode_rotmat:
				rotatable_feat_0 = img_fuser(
					img_feat_0, rotatable_feat_1, rot_10
				).reshape(-1, 3, self._num_feat_vec)
				rotatable_feat_1 = img_fuser(
					img_feat_1, rotatable_feat_0_swap, rot_01
				).reshape(-1, 3, self._num_feat_vec)
			elif self._ignore_rotmat:
				rotatable_feat_0 = img_fuser(img_feat_0, rotatable_feat_1).reshape(
					-1, 3, self._num_feat_vec
				)
				rotatable_feat_1 = img_fuser(
					img_feat_1, rotatable_feat_0_swap
				).reshape(-1, 3, self._num_feat_vec)
			else:
				rotatable_feat_0 = img_fuser(img_feat_0, rot_10 @ rotatable_feat_1).reshape(
					-1, 3, self._num_feat_vec
				)
				rotatable_feat_1 = img_fuser(
					img_feat_1, rot_01 @ rotatable_feat_0_swap
				).reshape(-1, 3, self._num_feat_vec)
				
			if self._share_feature:
				pred_gaze_0 = gaze_estimator(
					torch.cat([img_feat_0, rotatable_feat_0], dim=-1).flatten(1, -1)
				)
				pred_gaze_1 = gaze_estimator(
					torch.cat([img_feat_1, rotatable_feat_1], dim=-1).flatten(1, -1)
				)
			else:
				pred_gaze_0 = gaze_estimator(
					torch.cat([img_feat_0, rotatable_feat_0.flatten(1, -1)], dim=-1)
				)
				pred_gaze_1 = gaze_estimator(
					torch.cat([img_feat_1, rotatable_feat_1.flatten(1, -1)], dim=-1)
				)

			pred_iter = {
				"feat_0": rotatable_feat_0,
				"feat_1": rotatable_feat_1,
				"pred_gaze_0": pred_gaze_0,
				"pred_gaze_1": pred_gaze_1,
			}

			pred[f"iter_{f_i}"] = pred_iter

		pred["pred_gaze"] = pred[f"iter_{self._output_index}"]["pred_gaze_0"]
		data.update(pred)


		return data
