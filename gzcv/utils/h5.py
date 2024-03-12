import os
from typing import Any, Dict, List

import h5py
import numpy as np


def save_as_h5file(dataset: List[Dict[str, Any]], save_path: str):
    dir_name = os.path.dirname(save_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    sample = dataset[0]
    buffer = {}
    for key in sample.keys():
        buffer[key] = []

    for data in dataset:
        for key, item in data.items():
            buffer[key].append(item)

    with h5py.File(save_path, "w") as h5f:
        for key, items in buffer.items():
            h5f.create_dataset(key, data=np.stack(items))


class SequentialH5Writer:
    def __init__(self, path, keys, shapes):
        assert len(keys) == len(shape)
        self.path = path
        self.cur_idx = 0

        with h5py.File(self.path, "w") as h5f:
            for key, shape in zip(keys, shapes):
                h5f.create_dataset(key, shape=shape)

    def __call__(self, data):
        with h5py.File(self.path, "a") as h5f:
            for key, value in data.items():
                h5f[key][self.cur_idx] = value
        self.cur_idx += 1
