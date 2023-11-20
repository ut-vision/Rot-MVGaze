import abc
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


from .gaze_loss import GazeLoss

class AbstractLoss(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        if hasattr(self, "_name") and self._name is not None:
            return self._name
        else:
            return self.__class__.__name__

    @abc.abstractmethod
    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        pass


class StereoL1Loss(AbstractLoss):
    def __init__(
        self,
        rel_weight: float = 1,
        reference_decay: float = 1.0,
        distance_metric: str = "angular_error",
        pred_gaze_key: str = "pred_gaze",
        name: Optional[str] = None,
    ):
        super().__init__()
        self._rel_weight = rel_weight

        self._distance_metric = GazeLoss( 
                            gaze_weight=1.0,  ## meaningless
                            loss_type='angular') #globals()[distance_metric]

        self._reference_decay = reference_decay
        self._pred_gaze_key = pred_gaze_key
        if name is not None:
            self._name = name

    def forward(self, data: Dict[str, Any]):
        pred_gaze_0 = data[f"{self._pred_gaze_key}_0"]
        pred_gaze_1 = data[f"{self._pred_gaze_key}_1"]
        gt_gaze_0 = data["gt_gaze"]
        gt_gaze_1 = data["gt_gaze_1"]
        loss = self._distance_metric(pred_gaze_0, gt_gaze_0).mean()
        loss_aux = self._distance_metric(pred_gaze_1, gt_gaze_1).mean()

        return (loss + loss_aux * self._reference_decay) * self._rel_weight


class IterationLoss(AbstractLoss):
    def __init__(self, loss: AbstractLoss, iter_decay: float = 1.0, additional_decay: Optional[float] = None) -> None:
        super().__init__()
        self._name = "Iter" + loss.name
        self._loss = loss
        self._iter_decay = iter_decay
        self._addtional_decay = additional_decay

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        num_iter: int = data["num_iter"]
        total_loss = 0
        commmon: Dict[str, Any] = {
            key: value for key, value in data.items() if not key.startswith("iter_")
        }
        if self._addtional_decay is not None:
            num_iter -= 1
        
        for i in range(num_iter):
            iter_data = data[f"iter_{i}"]
            iter_data.update(commmon)
            total_loss = total_loss * self._iter_decay + self._loss(iter_data)

        if self._addtional_decay is not None:
            last_iter_data = data[f"iter_{num_iter}"]
            last_iter_data.update(commmon)
            total_loss += self._loss(last_iter_data) * self._addtional_decay
        
        return total_loss


# class SingleGazeLoss(AbstractLoss):
#     """
#     This loss is only for StereoAwareForSingle model.
#     """
#     def __init__(self, rel_weight: float) -> None:
#         super().__init__()
#         self._loss = L1Loss(rel_weight=rel_weight)

#     def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
#         gt_gaze = data["gt_gaze"]
#         pred_gaze = data["gaze_from_single"]
#         return self._loss(dict(gt_gaze=gt_gaze, pred_gaze=pred_gaze))



# class RandomFeatureRotationLoss(AbstractLoss):
#     def __init__(
#         self,
#         distance_metric="angular_error",
#         rel_weight=1.0,
#     ):
#         super().__init__()
#         self._distance_metric = globals()[distance_metric]
#         self._rel_weight = rel_weight

#     def forward(self, data: Dict[str, Any]) -> torch.Tensor:
#         pred_gaze_random_0 = data["pred_gaze_random_0"]
#         pred_gaze_random_1 = data["pred_gaze_random_1"]
#         gt_gaze_random_0 = data["gt_gaze_random_0"]
#         gt_gaze_random_1 = data["gt_gaze_random_1"]

#         loss_0 = self._distance_metric(pred_gaze_random_0, gt_gaze_random_0)
#         loss_1 = self._distance_metric(pred_gaze_random_1, gt_gaze_random_1)
#         return (loss_0 + loss_1).mean() * self._rel_weight


# # For simple single image random rotation model
# class SingleRandomFeatureRotationLoss(AbstractLoss):
#     def __init__(self, distance_metric="angular_error", rel_weight=1.0, alpha=0.5):
#         super().__init__()
#         self._distance_metric = globals()[distance_metric]
#         self._rel_weight = rel_weight
#         self._alpha = alpha

#     def forward(self, data: Dict[str, Any]) -> torch.Tensor:
#         pred_gaze_random = data["pred_gaze_random"]
#         pred_gaze = data["pred_gaze"]
#         gt_gaze_random = data["gt_gaze_random"]
#         gt_gaze = data["gt_gaze"]

#         loss_0 = self._distance_metric(pred_gaze, gt_gaze)
#         loss_1 = self._distance_metric(pred_gaze_random, gt_gaze_random)
#         return (loss_0 + self._alpha * loss_1).mean() * self._rel_weight



# class ImplicitRotationLoss(AbstractLoss):
#     def __init__(self, rel_weight: float) -> None:
#         super().__init__()
#         self._rel_weight = rel_weight

#     def forward(self, data: Dict[str, Any]) -> torch.Tensor:
#         gt_gaze_0 = data["gt_gaze_0"]
#         gt_gaze_1 = data["gt_gaze_1"]
#         pred_rot_01 = data["pred_rot_01"]

#         rotation_loss = angular_error(
#             rotate_pitchyaw(pred_rot_01, gt_gaze_0), gt_gaze_1
#         )
#         # print("rotation_loss = ", rotation_loss[:3])
#         return rotation_loss.mean() * self._rel_weight


# class RotationLoss(AbstractLoss):
#     def __init__(self, rel_weight: float) -> None:
#         super().__init__()
#         self._rel_weight = rel_weight

#     def forward(self, data: Dict[str, Any]) -> torch.Tensor:
#         rot = data["pred_rot_01"]
#         gt_rot = data["calib_rot_1"] @ data["calib_rot_0"].transpose(-1, -2)

#         loss = torch.square(rot - gt_rot).mean()
#         return loss * self._rel_weight


# class ConsistencyStereoL1Loss(AbstractLoss):
#     def __init__(
#         self,
#         rel_weight: float = 1,
#         reference_decay: float = 1.0,
#         distance_metric: str = "angular_error",
#         pred_gaze_key: str = "pred_gaze",
#         name: Optional[str] = None,
#     ):
#         super().__init__()
#         self._rel_weight = rel_weight
#         self._distance_metric = globals()[distance_metric]
#         self._reference_decay = reference_decay
#         self._pred_gaze_key = pred_gaze_key
#         if name is not None:
#             self._name = name

#     def forward(self, data: Dict[str, Any]):
#         pred_gaze_0 = data[f"{self._pred_gaze_key}_0"]
#         pred_gaze_1 = data[f"{self._pred_gaze_key}_1"]
#         gt_gaze_0 = data["gt_gaze"]
#         gt_gaze_1 = gt_gaze_0
#         loss = self._distance_metric(pred_gaze_0, gt_gaze_0).mean()
#         loss_aux = self._distance_metric(pred_gaze_1, gt_gaze_1).mean()

#         return (loss + loss_aux * self._reference_decay) * self._rel_weight


# class Feat3dAlignLoss(AbstractLoss):
#     def __init__(
#         self,
#         rel_weight: float = 1,
#         name: Optional[str] = None,
#     ):
#         super().__init__()
#         self._rel_weight = rel_weight
#         if name is not None:
#             self._name = name

#     def forward(self, data: Dict[str, Any]) -> torch.Tensor:
#         feat_0: torch.Tensor = data["feat_0"]
#         feat_1: torch.Tensor = data["feat_1"]
#         intensity_0 = torch.norm(feat_0, dim=-2, keepdim=True)
#         intensity_1 = torch.norm(feat_1, dim=-2, keepdim=True)
#         # HACK
#         # feat_1 = data["calib_rot_1"] @ data["calib_rot_0"].transpose(-1, -2) @ feat_0 * intensity_1
#         rot_01 = self._estimate_rotation(feat_0, feat_1)

#         feat_norm_0 = feat_0 / intensity_0
#         feat_norm_1 = feat_1 / intensity_1
#         norm_intensity_0 = intensity_0 / torch.norm(intensity_0, dim=-1, keepdim=True)
#         norm_intensity_1 = intensity_1 / torch.norm(intensity_1, dim=-1, keepdim=True)

#         feat_align_loss = (
#             torch.norm(feat_norm_1 - rot_01 @ feat_norm_0, dim=-2, keepdim=True)
#             * norm_intensity_0
#             * norm_intensity_1
#         )
#         return feat_align_loss.mean() * self._rel_weight

#     @staticmethod
#     def _estimate_rotation(feat_0, feat_1):
#         cov = feat_0 @ feat_1.transpose(-1, -2)
#         u, _, v = torch.svd(cov.float(), some=False, compute_uv=True)
#         rot_pos = v @ u.transpose(-1, -2)
#         return rot_pos


# class FeatureRotatabilityLoss(AbstractLoss):
#     def __init__(self, rel_weight: float, flooding_thres: float = 0) -> None:
#         super().__init__()
#         self._rel_weight = rel_weight
#         self._flooding_thres = flooding_thres

#     def forward(self, data: Dict[str, Any]) -> torch.Tensor:
#         feat_0 = data["feat_0"]
#         feat_1 = data["feat_1"]

#         pred_rot = self._estimate_rotation(feat_0, feat_1)
#         gt_rot = data["calib_rot_1"] @ data["calib_rot_0"].transpose(-1, -2)

#         loss = (pred_rot - gt_rot).norm(dim=-1).mean(dim=-1)
#         # print('loss.shape = ', loss.shape)
#         # print('loss before = ', loss[:3])
#         loss = self._flooding(loss)
#         # print('loss_after = ', loss[:3])
#         # print('pred = ', pred_rot[0, 0].data)
#         # print('gt = ', gt_rot[0, 0].data)
#         # print()

#         # return loss.detach() * self._rel_weight # HACK
#         return loss.mean() * self._rel_weight

#     def _flooding(self, loss):
#         return torch.clamp_min(loss, self._flooding_thres)

#     @staticmethod
#     def _estimate_rotation(feat_0, feat_1):
#         cov = feat_0 @ feat_1.transpose(-1, -2)
#         u, _, v = torch.svd(cov.float(), some=False, compute_uv=True)
#         rot_pos = v @ u.transpose(-1, -2)
#         return rot_pos
