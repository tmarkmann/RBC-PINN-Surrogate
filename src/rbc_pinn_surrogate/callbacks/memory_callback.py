import os

import psutil
from lightning.pytorch.callbacks import Callback


class MemoryCallback(Callback):
    def __init__(self, start_memory: float):
        self.start = start_memory

    def on_test_start(self, trainer, pl_module) -> None:
        # model size
        param_size = 0
        buffer_size = 0
        for param in pl_module.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in pl_module.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        print(f"Model Size: {size_all_mb:.2f} MB")

        # memory
        print(f"Memory: {self.get_process_memory() - self.start:.2f} MB")

    def on_test_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        print(f"Memory: {self.get_process_memory() - self.start:.2f} MB")

    def on_test_end(self, trainer, pl_module):
        print(f"Memory: {self.get_process_memory() - self.start:.2f} MB")

    def get_process_memory(self):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / 1024**2
