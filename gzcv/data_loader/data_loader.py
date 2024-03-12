import logging

import numpy as np
from rich.progress import track
from torch.utils.data import Dataset


class BaseGazeDataset(Dataset):
    def __init__(self, indices, preprocess):
        super().__init__()
        self.indices = indices
        self.logger = logging.getLogger(self.__class__.__name__)
        self.preprocess = preprocess

    def __getitem__(self, idx):
        idx_data = self.indices[idx]
        data = self.preprocess(idx_data)
        return data

    def __len__(self):
        return len(self.indices)


class StereoGazeDataset(BaseGazeDataset):
    def __init__(
        self,
        indices,
        preprocess,
        img_per_frame,
        stereo_preprocess=None,
        cam_pairs=None,
    ):
        super().__init__(indices, preprocess)
        self.frame_pool = self.build_pair_pool()
        self.stereo_preprocess = stereo_preprocess
        self.cam_pairs = cam_pairs
        self.img_per_frame = img_per_frame

    def __getitem__(self, idx):
        idx_data = self.indices[idx]
        idx_data = self.generate_stereo_pair(idx_data)
        stereo_data = {}

        for i, data in enumerate(idx_data):
            data = self.preprocess(data)
            for key in data.keys():
                new_key = f"{key}_{i}"
                stereo_data[new_key] = data[key]

        stereo_data = self.stereo_preprocess(stereo_data)
        return stereo_data

    def generate_stereo_pair(self, idx_data):
        stereo_data = [idx_data]

        subject_idx = idx_data["subject_index"]
        frame_idx = idx_data["frame_index"]
        same_scene = self.frame_pool[subject_idx][frame_idx]
        others = []
        src_candidates = self.cam_pairs[idx_data["cam_index"]]
        for data in same_scene:
            if src_candidates[data["cam_index"]]:
                others.append(data)

        others = np.random.choice(others, self.img_per_frame - 1, replace=False)
        stereo_data += others.tolist()
        return stereo_data

    def build_pair_pool(self):
        pool = {}
        for idx_data in track(
            self.indices, total=len(self.indices), description="Building frame pool"
        ):
            subject_idx = idx_data["subject_index"]
            frame_idx = idx_data["frame_index"]

            if subject_idx not in pool:
                pool[subject_idx] = {}

            if frame_idx not in pool[subject_idx]:
                pool[subject_idx][frame_idx] = []

            pool[subject_idx][frame_idx].append(idx_data)
        return pool
