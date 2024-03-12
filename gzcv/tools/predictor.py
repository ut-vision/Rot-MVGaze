import re
from typing import Any, List

import torch
from rich.live import Live
from rich.progress import Progress

from gzcv.tools.utils import to_device


class Predictor:
    def __init__(self, device: str, img_key_pattern: str) -> None:
        self.device = device
        self.img_key_pattern = img_key_pattern

    @torch.no_grad()
    def __call__(self, model, loader, reporter=None) -> None:
        if reporter is None:
            progress = Progress()
            with Live(progress):
                preds = self.predict(model, loader, progress)
        else:
            progress = reporter.progress
            preds = self.predict(model, loader, progress)
        return preds

    def predict(self, model, loader, progress):
        task_id = progress.add_task("Inference", total=len(loader))
        model.eval()
        preds = []
        for data in loader:
            data = to_device(data, self.device)
            data = model(data)

            data_wo_img = {
                key: value
                for key, value in data.items()
                if re.fullmatch(self.img_key_pattern, key) is None
                and "recon_img" not in key
            }
            data_wo_img = self._to_cpu_recursive(data_wo_img)

            preds.append(data_wo_img)
            progress.advance(task_id)
        progress.remove_task(task_id)
        return preds

    def _to_cpu_recursive(self, data: Any) -> Any:
        if isinstance(data, torch.Tensor):
            return data.cpu()
        if isinstance(data, list):
            return [self._to_cpu_recursive(elem) for elem in data]
        if isinstance(data, dict):
            return {key: self._to_cpu_recursive(value) for key, value in data.items()}
        return data
