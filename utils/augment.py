import os, json, yaml, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from typing import List, Dict, Any

class RandomMultiErasing(object):
    def __init__(self, proportion, p, dot_size):
        self.proportion = proportion
        self.p = p
        self.dot_size = dot_size

    def generate_mask(self, img_size: List[int], dot_size: int, proportion: int):
        h, w = img_size
        # hs, ws = int(h * self.unit_ratio), int(w * self.unit_ratio)
        hs = int(1 / dot_size)
        mask = (torch.rand(hs, hs) > proportion).to(torch.float32)
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), img_size).squeeze()
        return mask

    # def __call__(self, data):
    #     if random.random() > self.p:
    #         return data
    #     img = data["img"]
    #     size = img.shape[-2:]

    #     dot_size = np.random.uniform(*self.dot_size)
    #     proportion = np.random.uniform(*self.proportion)
    #     mask = self.generate_mask(size, dot_size, proportion)
    #     img *= mask

    #     data["img"] = img
    #     return data

    def __call__(self, img):
        if random.random() > self.p:
            return img
        size = img.shape[-2:]

        dot_size = np.random.uniform(*self.dot_size)
        proportion = np.random.uniform(*self.proportion)
        mask = self.generate_mask(size, dot_size, proportion)
        img *= mask
        return img