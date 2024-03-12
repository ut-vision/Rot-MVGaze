import logging
import random
from typing import List

import numpy as np
import scipy.spatial.transform as scit
import torch
import torch.nn.functional as nf
import torchvision.transforms.functional as F


class RandomStereoFlip(object):
    def __init__(self, num_imgs=18):
        raise RuntimeError("Flip augmentation may harm unknown head pose settings.")
        self.num_imgs = num_imgs

    def __call__(self, data):
        if random.random() < 0.5:
            img = data["img"]
            gaze = data["gt_gaze"]
            rot = data["rot"]

            img = F.hflip(img)
            gaze[1] *= -1
            rot[..., 0] *= -1
            rot[..., 0, :] *= -1

            data["img"] = img
            data["gt_gaze"] = gaze
            data["rot"] = rot
        return data


class RandomStereoFlip(object):
    def __init__(self, num_imgs=18):
        raise RuntimeError("Flip augmentation may harm unknown head pose settings.")
        self.num_imgs = num_imgs

    def __call__(self, data):
        do_flip = random.random() < 0.5
        if do_flip:
            for i in range(self.num_imgs):
                if f"img_{i}" in data:
                    img = data[f"img_{i}"]
                    gaze = data[f"gt_gaze_{i}"]
                    rot = data[f"rot_{i}"]

                    img = F.hflip(img)
                    gaze[1] *= -1
                    rot[..., 0] *= -1
                    rot[..., 0, :] *= -1

                    data[f"img_{i}"] = img
                    data[f"gt_gaze_{i}"] = gaze
                    data[f"rot_{i}"] = rot
                else:
                    break
        return data


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
        mask = nf.interpolate(mask.unsqueeze(0).unsqueeze(0), img_size).squeeze()
        return mask

    def __call__(self, data):
        if random.random() > self.p:
            return data
        img = data["img"]
        size = img.shape[-2:]

        dot_size = np.random.uniform(*self.dot_size)
        proportion = np.random.uniform(*self.proportion)
        mask = self.generate_mask(size, dot_size, proportion)
        img *= mask

        data["img"] = img
        return data


class RandomRotationJitter(object):
    def __init__(self, dr, p=0.5):
        """
        Args:
            dr: list of rotation degrees.
        """
        self.dr = dr
        self.p = p
        self.logger = logging.getLogger("RandomRotationJitter")
        self.logger.warning(
            "Jittering rotation will harm the model inference without rotation correction."
        )

    def __call__(self, data):
        if random.random() > self.p:
            return data
        rot = data["rot"]
        jitter_rot = self.generate_random_rotation()
        rot = jitter_rot @ rot
        data["rot"] = rot
        return data

    def generate_random_rotation(self):
        dr = (np.random.rand(3) - 0.5) * np.array(self.dr)
        r = scit.Rotation.from_euler("xyz", dr, degrees=True)
        rot = r.as_matrix()
        return rot
