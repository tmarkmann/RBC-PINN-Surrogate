from abc import ABC, abstractmethod
import time
from typing import Any, List

import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
import numpy as np


class Visualizer(ABC):
    def __init__(
        self,
        size: List[int],
        cmap: str,
        display: bool,
        fps: int = 10,
    ) -> None:
        self.size = size
        self.H, self.W = size
        self.cmap = cmap
        self.show = display
        self.setup = False
        self.fps = fps

    def _setup_display(self) -> None:
        if self.show:
            # Do not raise/steal focus on macOS when figures draw
            mpl.rcParams["figure.raise_window"] = False
            plt.ion()
            plt.show()
        else:
            matplotlib.use("Agg")

        self.setup = True

    def display(self) -> None:
        # setup display
        if not self.setup:
            self._setup_display()

        # control framerate
        dt = 1.0 / self.fps if self.fps > 0 else 0.0
        t0 = time.time()

        # redraw all open figures
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            fig.canvas.draw()
            fig.canvas.flush_events()

            # let GUI update briefly
            elapsed = time.time() - t0
            sleep_time = max(0.0, dt - elapsed)
            try:
                if self.show:
                    fig.canvas.start_event_loop(sleep_time)
                else:
                    time.sleep(sleep_time)
            except Exception:
                time.sleep(sleep_time)

    @abstractmethod
    def update(self, data: Any) -> Figure:
        pass


class TemperatureVisualizer(Visualizer):
    def __init__(
        self,
        size: List[int] = [64, 96],
        vmin: float = None,
        vmax: float = None,
        display: bool = False,
        fps: int = 10,
    ) -> None:
        # params
        super().__init__(size=size, cmap="coolwarm", display=display, fps=fps)
        self.vmin = vmin
        self.vmax = vmax

        # Temperature field
        self.fig, self.ax = plt.subplots()
        self.image = self.ax.imshow(
            np.zeros(self.size),
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            origin="lower",
        )

        # Axes settings
        self.ax.set_axis_off()

    def update(self, data: Any) -> Figure:
        self.image.set_array(data)
        self.display()
        return self.fig
