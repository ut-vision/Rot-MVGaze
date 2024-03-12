from collections import deque

from rich import box
from rich.console import Group
from rich.live import Live
from rich.progress import Progress
from rich.table import Table

from gzcv.metrics import AverageMeter


class Reporter:
    def __init__(self, num_epochs, num_iter, max_height=10, freq=10):
        self.meter = AverageMeter()
        self.progress = Progress()
        self.one_epoch = self.progress.add_task("Epoch", total=num_iter)
        self.total = self.progress.add_task("Total", total=num_epochs * num_iter)

        self.group = Group(self.progress)
        self.live = Live(self.group)
        self.buffer = deque()
        self.max_height = max_height
        self.freq = freq
        self.cnt = 0
        self.num_iter = num_iter

    @property
    def display(self):
        return self.live

    def update(self, results):
        if self.cnt % self.num_iter == 0:
            self.progress.reset(self.one_epoch)
            self.meter.reset()

        self.meter.update(results)
        if self.cnt % self.freq == 0:
            table = self.stream_results()
            new_group = Group(table, self.progress)
            self.live.update(new_group)

        self.progress.advance(self.one_epoch)
        self.progress.advance(self.total)

        self.cnt += 1

    def stream_results(self):
        average = self.meter.compute()
        row = [f"[dim]{self.cnt}", *[f"{value:.2f}" for value in average.values()]]
        self.buffer.append(row.copy())
        if len(self.buffer) > self.max_height:
            self.buffer.popleft()
        headers = list(average.keys())
        table = Table("[dim]iter", *headers, box=box.HORIZONTALS, show_edge=False)

        for past_row in self.buffer:
            table.add_row(*past_row)
        return table

    def stop(self):
        self.live.stop()

    def start(self):
        self.live.start()
