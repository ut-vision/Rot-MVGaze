import glob
import logging
import os
from typing import List, Optional

import h5py
from rich.progress import track

from .abstract import AbstractIndex


class XGazeIndex(AbstractIndex):
    def __init__(
        self,
        root: str,
        subject_keys: Optional[List[str]] = None,
        cam_indices: Optional[List[int]] = None,
    ) -> None:
        self.root = root

        if subject_keys is None:
            subject_paths = glob.glob(os.path.join(self.root, "*.h5"))
            subject_keys = [os.path.basename(subj_path) for subj_path in subject_paths]
            subject_keys.sort()
        self.subject_keys = subject_keys

        if cam_indices is None:
            cam_indices = list(range(18))
        self.cam_indices = cam_indices

        logger = logging.getLogger(__class__.__name__)
        logger.info(f"Number of subjects = {len(self.subject_keys)}")
        logger.info(f"Camera index to use = {self.cam_indices}")

        super().__init__()

    def build_index(self):
        indices = []
        for subject in track(self.subject_keys, description="Building xgaze index"):
            with h5py.File(os.path.join(self.root, subject), swmr=True) as subject_hdf:
                for img_idx, (frame_idx, cam_idx) in enumerate(
                    zip(
                        subject_hdf["frame_index"][:, 0], subject_hdf["cam_index"][:, 0]
                    )
                ):
                    cam_idx -= 1
                    if cam_idx in self.cam_indices:
                        data = {
                            "hdf_path": os.path.join(self.root, subject),
                            "subject_index": subject,
                            "frame_index": frame_idx,
                            "cam_index": cam_idx,
                            "img_index": img_idx,
                            "id": f"{subject}-{frame_idx}-{cam_idx}",
                        }
                        indices.append(data)
        return indices
