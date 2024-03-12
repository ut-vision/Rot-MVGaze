import json
from typing import Any, Dict, List

import numpy as np
import torch


def to_buildin_types(data):
    if isinstance(data, list):
        return [to_buildin_types(l) for l in data]
    if isinstance(data, dict):
        return {key: to_buildin_types(value) for key, value in data.items()}
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().tolist()
    return data


def save_as_json(data: List[Dict[str, Any]], path: int, save_keys: List[str]):
    data = to_buildin_types(data)
    data_to_save = []
    print("hoge")
    for batch in data:
        batch_to_save = {key: value for key, value in batch.items() if key in save_keys}
        data_to_save.append(batch_to_save)
    print("for done")
    with open(path, "w") as jf:
        json.dump(data_to_save, jf, indent=4)
    print("dump done")
