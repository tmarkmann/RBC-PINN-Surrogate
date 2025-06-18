from abc import ABC
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure


class RBCFieldVisualizer(ABC):
    def __init__(
        self,
        size: List[int] = [64, 96],
        vmin: float = None,
        vmax: float = None,
    ) -> None:
        # Matplotlib settings
        matplotlib.use("Agg")

        # Temperature field
        self.fig_t, self.ax_t = plt.subplots()
        self.image_t = self.ax_t.imshow(
            np.zeros(size),
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )
        self.ax_t.set_axis_off()

        # Velocity field
        self.fig_v, self.ax_v = plt.subplots()
        self.image_v = self.ax_v.imshow(
            np.zeros(size),
            cmap="coolwarm",
            vmin=None,
            vmax=None,
        )
        self.ax_v.set_axis_off()

    def draw(self, field: npt.NDArray[np.float32], type: str) -> Figure:
        """
        Show an image or update the image being shown
        """
        # Update T image
        if type == "T":
            self.image_t.set_array(field)
            self.fig_t.canvas.draw()
            self.fig_t.canvas.flush_events()
            return self.fig_t
        elif type == "U":
            self.image_v.set_array(field)
            self.fig_v.canvas.draw()
            self.fig_v.canvas.flush_events()
            return self.fig_v
        else:
            raise Exception(f"Invalid field type: {type}")
