import random
import re
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
import torchvision

from gzcv.utils.math import pitchyaw_to_rotmat


class FixSeed(object):
    def __init__(self, seed_shift: int) -> None:
        self.seed_shift = seed_shift

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            m = re.findall(r"\d+", data["subject_index"])
            subject_idx = int("".join(m))
        except TypeError:
            subject_idx = data["subject_index"]
        img_idx = data["img_index"]
        seed = (subject_idx * img_idx + self.seed_shift) % (1 << 31)

        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        return data


class Rename(object):
    def __init__(self, name_map: Dict[str, str]) -> None:
        super().__init__()
        self.name_map = name_map

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for cur_name, new_name in self.name_map.items():
            if cur_name in data:
                data[new_name] = data[cur_name]
        for old_name in self.name_map.keys():
            data.pop(old_name)
        return data


class PixelTransform(object):
    def __init__(self, img_key: str, transforms: List) -> None:
        super().__init__()
        self.img_key = img_key
        self.transforms = torchvision.transforms.Compose(transforms)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        img = data[self.img_key]
        data[self.img_key] = self.transforms(img)
        return data


class BGR2RGB:
    def __init__(self, img_key: str) -> None:
        self.img_key = img_key

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        img = data[self.img_key]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data["img"] = img
        return data


class NumpyToTensor(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        picked = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                picked[key] = value.to(torch.float32)
            elif isinstance(value, np.ndarray):
                picked[key] = torch.from_numpy(value).to(torch.float32)
            else:
                picked[key] = value
        return picked


class RenameRef(object):  # TODO poor naming sense
    def __init__(self) -> None:
        pass

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        master = {}
        for key, item in data.items():
            if key.endswith("_0"):
                master[key.strip("_0")] = item
        data.update(master)
        return data


class UnnormalizeRot(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        rot = data["rot"]
        face_mat_norm = data["face_mat_norm"]
        data["rot"] = face_mat_norm @ rot
        return data


class CalibFromHeadpose(object):
    def __init__(self) -> None:
        pass

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        head_pose = data["face_head_pose"]
        head_pose[0] *= -1
        head_mat = pitchyaw_to_rotmat(head_pose)
        data["rot"] = head_mat
        return data


# HACK This should be removed A.S.A if we fix the all the headposes in MPIISynthetic dataset.
class MPIISyntheticBarePatch:
    def __init__(self) -> None:
        pass

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["face_head_pose"][0] *= -1
        return data


class DoNothing:
    def __init__(self) -> None:
        pass

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data
