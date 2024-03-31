import copy
from typing import Any, Dict, List, Union

import torch

# class AverageMeter:
#     def __init__(self) -> None:
#         self.avg = {}
#         self.sum = {}
#         self.count = 0

#     def reset(self) -> None:
#         for metric_name in self.avg.keys():
#             self.avg[metric_name] = 0
#             self.sum[metric_name] = 0
#         self.count = 0

#     @torch.no_grad()
#     def update(self, metric_resutls: Dict[str, torch.Tensor]) -> None:
#         """
#         [Args]: metric_results: dict[str, Any]
#         """
#         for metric, result in metric_resutls.items():
#             self.sum.setdefault(metric, 0)
#             if hasattr(result, "mean"):
#                 result = result.mean()
#             self.sum[metric] += result
#         self.count += 1

#     def compute(self) -> Dict[str, float]:
#         """
#         [Returns]: average values in dict format
#         """
#         for metric, total in self.sum.items():
#             self.avg[metric] = total / (self.count + 1e-6)
#         return copy.deepcopy(self.avg)
    


class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count