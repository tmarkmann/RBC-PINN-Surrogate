import os
import time

import psutil
from lightning.pytorch.callbacks import Callback


class TimerCallback(Callback):
    def __init__(self):
        self.sequences = 0

    def on_test_start(self, trainer, pl_module) -> None:
        self.start = time.time()

    def on_test_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self.sequences += len(batch)

    def on_test_end(self, trainer, pl_module):
        self.end = time.time()
        elapsed_time = self.end - self.start
        time_per_sequence = elapsed_time / self.sequences

        print(f"Test time: {elapsed_time:.4f} s")
        print(f"Time per sequence: {time_per_sequence:.4f} s")

    def get_process_memory(self):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / 1024**2
