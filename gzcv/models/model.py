from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class MultiTaskModel(nn.Module):
    def __init__(self, models: List[nn.Module]) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for model in self.models:
            data = model(data)
        return data


def setup_model(model: nn.Module, resume: Optional[str] = None) -> nn.Module:
    if resume is not None:
        state_dict = torch.load(resume)
        model.load_state_dict(state_dict)
    return model
