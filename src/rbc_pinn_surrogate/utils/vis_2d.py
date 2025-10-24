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
            cmap="rainbow",
            vmin=vmin,
            vmax=vmax,
        )
        self.ax_t.set_axis_off()

        # Velocity field
        self.fig_v, self.ax_v = plt.subplots()
        self.image_v = self.ax_v.imshow(
            np.zeros(size),
            cmap="rainbow",
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


# ----------------------------- RBCLiveVisualizer --------------------------------
class RBCLiveVisualizer(ABC):
    """
    Live visualizer for 2D RBC states with an action vector that sets the lower boundary temperature.

    Features
    --------
    - Single figure with two vertically stacked panels:
        (1) Temperature field (HxW)
        (2) Action panel: a 1xW strip showing the 12-bin action expanded across the width
    - .update(T, action) to update the display live
    - .iterate(dataset, state_getter, action_getter, interval_ms=50, limit=None) convenience loop
    - Optional color limits for both temperature and action
    - Optional saving of frames

    Notes
    -----
    - This class does NOT set a Matplotlib backend (unlike RBCFieldVisualizer). Use your preferred backend.
    - For interactive live updates, call plt.ion() once in your app, or pass show=True here.
    """

    def __init__(
        self,
        size: List[int] = [64, 96],
        vmin: float | None = None,
        vmax: float | None = None,
        action_vmin: float | None = None,
        action_vmax: float | None = None,
        bins: int = 12,
        cmap: str = "coolwarm",
        show: bool = True,
    ) -> None:
        self.size = size
        self.H, self.W = size
        self.bins = bins
        self.cmap = cmap

        # Figure with 2 rows: Temperature and Action strip
        self.fig, (self.ax_T, self.ax_a) = plt.subplots(
            2, 1, figsize=(6, 6), height_ratios=[8, 1], constrained_layout=True
        )

        # Temperature image
        self.im_T = self.ax_T.imshow(
            np.zeros(size, dtype=np.float32),
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            origin="lower",
            aspect="auto",
        )
        self.ax_T.set_title("Temperature field T")
        self.ax_T.set_axis_off()

        # Action strip (1 x W, later repeated vertically for visibility)
        self.im_a = self.ax_a.imshow(
            np.zeros((1, self.W), dtype=np.float32),
            cmap=self.cmap,
            vmin=action_vmin if action_vmin is not None else vmin,
            vmax=action_vmax if action_vmax is not None else vmax,
            interpolation="nearest",
            origin="upper",
            aspect="auto",
        )
        self.ax_a.set_yticks([])
        self.ax_a.set_xlabel("Lower-boundary temperature (12 segments)")

        if show:
            plt.ion()
            self.fig.show()

    def _expand_action_to_width(
        self, action: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Expand a length-`bins` action vector into a width-W array by repeating values per segment.
        Returns shape (1, W).
        """
        assert action.ndim == 1, "action must be a 1D array of length `bins`"
        assert action.shape[0] == self.bins, (
            f"expected action of length {self.bins}, got {action.shape[0]}"
        )

        seg_w = self.W // self.bins
        if seg_w == 0:
            # fallback: resize via linear interpolation if W < bins (unlikely)
            x_old = np.linspace(0, 1, self.bins, dtype=np.float32)
            x_new = np.linspace(0, 1, self.W, dtype=np.float32)
            a_resized = np.interp(x_new, x_old, action.astype(np.float32))
            return a_resized.reshape(1, self.W).astype(np.float32)

        # Repeat each bin value seg_w times
        expanded = np.repeat(action.astype(np.float32), seg_w)
        # Pad the tail if W is not divisible by bins
        if expanded.shape[0] < self.W:
            pad = np.full((self.W - expanded.shape[0],), action[-1], dtype=np.float32)
            expanded = np.concatenate([expanded, pad], axis=0)
        elif expanded.shape[0] > self.W:
            expanded = expanded[: self.W]
        return expanded.reshape(1, self.W)

    def update(
        self,
        T: npt.NDArray[np.float32],
        action: npt.NDArray[np.float32] | list[float] | tuple[float, ...],
    ):
        """
        Update the temperature image and action strip.

        Parameters
        ----------
        T : (H, W) arraylike
            Temperature field.
        action : (bins,) arraylike
            Action vector (length `bins`) controlling the lower boundary temperature.
        """
        if isinstance(action, (list, tuple)):
            action = np.asarray(action, dtype=np.float32)
        else:
            action = action.astype(np.float32, copy=False)

        # Ensure shapes
        T = np.asarray(T, dtype=np.float32)
        assert T.shape == (self.H, self.W), (
            f"expected T shape {(self.H, self.W)}, got {T.shape}"
        )

        # Update images
        self.im_T.set_array(T)
        self.im_a.set_array(self._expand_action_to_width(action))

        # Draw/flush
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save_frame(self, path: str):
        """Save the current figure as an image (e.g., to make a video later)."""
        self.fig.savefig(path, bbox_inches="tight")

    def iterate(
        self,
        dataset,
        state_getter=lambda sample: sample["T"],  # adapt to your dataset API
        action_getter=lambda sample: sample["action"],  # adapt to your dataset API
        interval_ms: int = 50,
        limit: int | None = None,
    ):
        """
        Convenience loop to step through a dataset and update the visualization.

        Parameters
        ----------
        dataset : Iterable
            Yields samples (e.g., dicts or tuples). Customize the getters.
        state_getter : callable
            Function mapping a dataset sample -> (H, W) temperature array.
        action_getter : callable
            Function mapping a dataset sample -> (bins,) action vector.
        interval_ms : int
            Delay between frames (uses plt.pause).
        limit : Optional[int]
            Maximum number of samples to display.
        """
        plt.ion()
        count = 0
        for sample in dataset:
            T = state_getter(sample)
            a = action_getter(sample)
            self.update(T, a)
            plt.pause(interval_ms / 1000.0)

            count += 1
            if limit is not None and count >= limit:
                break
