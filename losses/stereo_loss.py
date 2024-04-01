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
                            gaze_weight=1.0, 
                            loss_type='angular') 

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
