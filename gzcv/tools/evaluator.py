import logging
from typing import Any, Dict, List

from rich.live import Live
from rich.progress import Progress

from gzcv.metrics import AverageMeter
from gzcv.tools.utils import format_results


class Evaluator:
    def __init__(self, device, metrics):
        self.device = device
        self.metrics = metrics
        self.avg_meter = AverageMeter()
        self.logger = logging.getLogger(__class__.__name__)

    def __call__(self, preds: List[Dict[str, Any]], reporter=None):
        self.avg_meter.reset()
        if reporter is None:
            progress = Progress()
            with Live(progress):
                average, preds = self.evaluate(preds, progress)
        else:
            progress = reporter.progress
            average, preds = self.evaluate(preds, progress)

        self.logger.info(format_results(average))
        return average, preds

    def evaluate(self, preds, progress):
        task_id = progress.add_task("Evaluating", total=len(preds))
        for pred in preds:
            result = self.metrics(pred)
            pred.update(result)
            self.avg_meter.update(result)
            progress.advance(task_id)
        progress.remove_task(task_id)
        average = self.avg_meter.compute()
        return average, preds
