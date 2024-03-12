import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch

from gzcv.metrics import AverageMeter
from gzcv.tools.utils import format_results, to_device
from gzcv.utils import to_buildin_types


class Trainer(object):
    def __init__(
        self,
        model,
        train_loader=None,
        val_loader=None,
        optimizer=None,
        scheduler=None,
        visualizer=None,
        device: str = None,
        metrics=None,
        epochs: int = 0,
        val_freq: Optional[int] = None,
        main_metric: str = None,
        use_amp: bool = False,
        reporter=None,
        evaluator=None,
        predictor=None,
        log_dir=None,
        experiment_name: str = None,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logging.getLogger(__class__.__name__)
        self.log_dir = log_dir
        self.ckpt_dir = os.path.join(self.log_dir, "ckpt")
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self.epochs = epochs
        self.val_freq = val_freq

        self.best_score = 1e9

        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics
        self.main_metric = main_metric
        self.meter = AverageMeter()

        self.evaluator = evaluator
        self.predictor = predictor
        self.visualizer = visualizer
        self.reporter = reporter
        self.experiment_name = experiment_name

        self.use_amp = use_amp
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def train(self):
        with self.reporter.display:
            all_results: List[Dict[str, Any]] = []
            for epoch in range(self.epochs):
                self.logger.info(f"Start epoch {epoch}")
                self.train_one_epoch()

                if self.val_freq > 0 and (epoch + 1) % self.val_freq == 0:
                    preds = self.predictor(self.model, self.val_loader, self.reporter)
                    average, preds = self.evaluator(preds, self.reporter)
                    self.update_and_save_best(average)
                    all_results.append(average)
            self.write_results(all_results)

        pth_name = f"{self.experiment_name}-model-last.pth"
        save_path = os.path.join(self.ckpt_dir, pth_name)
        self.save_model(save_path)
        return self.model

    def train_one_epoch(self):
        self.model.train()
        self.meter.reset()

        for data in self.train_loader:
            self.optimizer.zero_grad()
            data = to_device(data, self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                data = self.model(data)
                metrics = self.metrics(data)
                loss = metrics["total_loss"]
                if torch.any(torch.isnan(loss)):
                    continue

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.meter.update(metrics)
            self.reporter.update(metrics)
            self.scheduler.step()

        self.logger.info(format_results(self.meter.compute()))

    def update_and_save_best(self, eval_res):
        if eval_res[self.main_metric] < self.best_score:
            self.best_score = eval_res[self.main_metric]

            # HACK tightly coupled
            pth_name = f"{self.experiment_name}-model-best.pth"
            save_path = os.path.join(self.ckpt_dir, pth_name)
            self.save_model(save_path)
            self.logger.info(f"Best score updated and save it to {save_path}")
            return True
        return False

    def save_model(self, path: str):
        if isinstance(self.model, torch.nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        torch.save(model_state, path)

    def write_results(self, results: List[Dict[str, Any]]) -> None:
        results_path = os.path.join(
            self.log_dir, f"{self.experiment_name}-results.json"
        )
        with open(results_path, "w") as jf:
            json.dump(to_buildin_types(results), jf, indent=4)
