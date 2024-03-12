import copy
from typing import Any, Dict, List, Union

import torch

from .evaluation import *
from .loss import *


class AverageMeter:
    def __init__(self) -> None:
        self.avg = {}
        self.sum = {}
        self.cnt = 0

    def reset(self) -> None:
        for metric_name in self.avg.keys():
            self.avg[metric_name] = 0
            self.sum[metric_name] = 0
        self.cnt = 0

    @torch.no_grad()
    def update(self, metric_resutls: Dict[str, torch.Tensor]) -> None:
        """
        [Args]: metric_results: dict[str, Any]
        """
        for metric, result in metric_resutls.items():
            self.sum.setdefault(metric, 0)
            if hasattr(result, "mean"):
                result = result.mean()
            self.sum[metric] += result
        self.cnt += 1

    def compute(self) -> Dict[str, float]:
        """
        [Returns]: average values in dict format
        """
        for metric, total in self.sum.items():
            self.avg[metric] = total / (self.cnt + 1e-6)
        return copy.deepcopy(self.avg)


class Metrics(object):
    def __init__(self, metrics: List[Union[AbstractLoss, AbstructEvaluator]]):
        super().__init__()
        self.metrics = metrics

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        total_loss = 0
        for metric in self.metrics:
            value = metric(data)
            name = metric.name
            results[name] = value
            if isinstance(metric, AbstractLoss):
                total_loss += value
        results["total_loss"] = total_loss
        return results


class SwitchMetrics(object):
    def __init__(
        self, metrics: List[Union[AbstractLoss, AbstructEvaluator]], switch_step: int
    ):
        super().__init__()
        self.metrics = metrics
        self.switch_step = switch_step
        self.cnt = 0

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.cnt += 1

        results = {}
        total_loss = 0
        for metric in self.metrics:
            value = metric(data)
            name = metric.name
            results[name] = value

            if isinstance(metric, AbstractLoss):
                if self.cnt < self.switch_step:
                    if name == "RandomFeatureRotationLoss":
                        total_loss += value
                else:
                    if name != "RandomFeatureRotationLoss":
                        total_loss += value
        results["total_loss"] = total_loss
        return results
