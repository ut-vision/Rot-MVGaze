import logging

import torch


class AutoParameter(object):
    def __init__(
        self, cfg, model=None, loader_length=None, mem_margin=0.25, batch_size_step=16
    ):
        self.cfg = cfg
        self.model = model
        self.loader_length = loader_length
        self.mem_margin = mem_margin
        self.batch_size_step = batch_size_step
        self.logger = logging.getLogger("AutoParameter")

    def __call__(self):
        regists = ["register_step_size"]
        for reg in regists:
            if "register" in reg:
                getattr(self, reg)()
        return self.cfg

    def register_step_size(self) -> None:
        """
        step_size should be calcuralted after getting batch_size.
        """
        num_batches = self.loader_length // self.cfg.batch_size
        step_size_up = int(num_batches // 2)
        step_size_down = num_batches - step_size_up
        self.cfg.step_size_up = step_size_up
        self.cfg.step_size_down = step_size_down
        self.logger.info(
            f"Scheduler's step_size_up and down is automatically tuend to {step_size_up} / {step_size_down}."
        )

    def get_device_capacity(self):
        """
        Return: GPU memory capacity in GB.
        """
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = gpu_memory / (1 << 30)
        available_memory_gb = gpu_memory_gb * (1 - self.mem_margin)
        print("available memory = ", available_memory_gb, "GB")
        return available_memory_gb
