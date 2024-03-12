from typing import Any, Dict, List

import torch


def to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    for key, item in batch.items():
        if isinstance(item, torch.Tensor):
            batch[key] = item.to(device)
    return batch


def to_numpy(batch: Dict[str, Any]) -> Dict[str, Any]:
    for key, item in batch.items():
        if isinstance(item, torch.Tensor):
            item = item.cpu().numpy()
        batch[key] = item
    return batch


def format_results(results: Dict[str, Any]) -> str:
    msg = ""
    for metric, value in results.items():
        msg += f"{metric}: {value:.2f}, "
    return msg


def devide_batch_to_samples(preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    samples = {}
    for batch in preds:
        batch_size = len(batch["id"])
        for b_i in range(batch_size):
            img_id = batch["id"][b_i]
            sample = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and len(value.shape) > 0:
                    sample[key] = value[b_i]
                elif isinstance(value, list):
                    sample[key] = value[b_i]
            samples[img_id] = sample
    return samples
