import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Literal
import os
from tqdm.auto import tqdm


def animation_3d(
    gt,
    pred,
    anim_dir: str,
    anim_name: str,
    feature: Literal["T", "u", "v", "w"],
    fps: int = 10,
    rotate: bool = False,
    angle_per_sec: int = 10,
    contour_levels: int = 50,
):
    """data as [channel, time, height, width, depth]"""

    channel = ["T", "u", "v", "w"].index(feature)
    diff = pred - gt

    if channel == 0:
        vmin, vmax = 1, 2
    else:
        vmin, vmax = -1, 1

    frame_idx = 0
    time_length = gt.shape[1]

    fig = plt.figure()
    ax1 = plt.subplot(1, 3, 1, projection="3d")
    (plt.axis("off"),)
    ax2 = plt.subplot(1, 3, 2, projection="3d")
    plt.axis("off")
    ax3 = plt.subplot(1, 3, 3, projection="3d")
    plt.axis("off")

    orig_faces = plot_cube_faces(
        gt[channel, 0, :, :, :],
        ax1,
        contour_levels=contour_levels,
        show_back_faces=rotate,
        vmin=vmin,
        vmax=vmax,
    )
    pred_faces = plot_cube_faces(
        pred[channel, 0, :, :, :],
        ax2,
        contour_levels=contour_levels,
        show_back_faces=rotate,
        vmin=vmin,
        vmax=vmax,
    )
    diff_faces = plot_cube_faces(
        diff[channel, 0, :, :, :],
        ax3,
        cmap="RdBu_r",
        contour_levels=contour_levels,
        show_back_faces=rotate,
        vmin=-1,
        vmax=1,
    )

    ax1.set_aspect("equal", adjustable="box")
    ax2.set_aspect("equal", adjustable="box")
    ax3.set_aspect("equal", adjustable="box")

    ax1.set_title("input")
    ax2.set_title("output")
    ax3.set_title("difference")

    elev = 15
    ax1.view_init(elev=elev)
    ax2.view_init(elev=elev)
    ax3.view_init(elev=elev)

    # divider = make_axes_locatable(ax3)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # cbar = fig.colorbar(diff_faces[1], cax=cax, ticks=[0, 0.5, 1], extend="both", orientation='vertical')

    last_frame = 0

    def frame_updater(frame, pbar):
        """Computes the next frame of the animation."""
        nonlocal gt, pred, diff, frame_idx, last_frame
        nonlocal orig_faces, pred_faces, diff_faces
        if frame <= last_frame:
            # no new frame
            return orig_faces + pred_faces + diff_faces

        frame_idx += 1
        last_frame = frame_idx

        # update frames
        reset_contours(orig_faces + pred_faces + diff_faces)
        orig_faces = plot_cube_faces(
            gt[channel, frame_idx, :, :, :],
            ax1,
            contour_levels=contour_levels,
            show_back_faces=rotate,
            vmin=vmin,
            vmax=vmax,
        )
        pred_faces = plot_cube_faces(
            pred[channel, frame_idx, :, :, :],
            ax2,
            contour_levels=contour_levels,
            show_back_faces=rotate,
            vmin=vmin,
            vmax=vmax,
        )
        diff_faces = plot_cube_faces(
            diff[channel, frame_idx, :, :, :],
            ax3,
            cmap="RdBu_r",
            contour_levels=contour_levels,
            show_back_faces=rotate,
            vmin=-1,
            vmax=1,
        )

        # cbar.remove()
        # cbar = fig.colorbar(diff_faces[1], ax=ax3, orientation='vertical')

        plt.suptitle(f"Simulation {frame_idx} / {time_length}")

        # update color map limits
        set_clims(orig_faces + pred_faces, vmin, vmax)
        set_clims(diff_faces, vmin=-1, vmax=1)

        if rotate:
            azim = -50 + angle_per_sec * frame / fps
            ax1.view_init(elev=elev, azim=azim)
            ax2.view_init(elev=elev, azim=azim)
            ax3.view_init(elev=elev, azim=azim)

        pbar.update(1)

        return orig_faces + pred_faces + diff_faces

    with tqdm(total=time_length, desc=f"animating {feature}", unit="frames") as pbar:
        anim = animation.FuncAnimation(
            fig,
            frame_updater,
            frames=time_length,
            interval=1000 / fps,
            blit=True,
            fargs=(pbar,),
        )

        os.makedirs(anim_dir, exist_ok=True)
        out = os.path.join(anim_dir, anim_name)
        anim.save(out, dpi=500)
    plt.close()
    return out


def plot_cube_faces(
    arr,
    ax,
    dims=(2 * np.pi, 2 * np.pi, 2),
    contour_levels=100,
    cmap="rainbow",
    show_back_faces=False,
    vmin=None,
    vmax=None,
    **contour_kwargs,
) -> list:
    # transpose back to julia order to reuse code TODO
    arr = np.transpose(arr, (2, 1, 0))

    z0 = np.linspace(0, dims[2], arr.shape[2])
    x0 = np.linspace(0, dims[0], arr.shape[0])
    y0 = np.linspace(0, dims[1], arr.shape[1])
    x, y, z = np.meshgrid(x0, y0, z0)

    xmax, ymax, zmax = max(x0), max(y0), max(z0)
    _vmin, _vmax = np.min(arr), np.max(arr)
    if not vmin:
        vmin = _vmin
    if not vmax:
        vmax = _vmax
    levels = np.linspace(vmin, vmax, contour_levels)

    z1 = ax.contourf(
        x[:, :, 0],
        y[:, :, 0],
        arr[:, :, -1].T,
        zdir="z",
        offset=zmax,
        vmin=vmin,
        vmax=vmax,
        levels=levels,
        cmap=cmap,
        **contour_kwargs,
    )
    if show_back_faces:
        z2 = ax.contourf(
            x[:, :, 0],
            y[:, :, 0],
            arr[:, :, 0].T,
            zdir="z",
            offset=0,
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            cmap=cmap,
        )
    y1 = ax.contourf(
        x[0, :, :].T,
        arr[:, 0, :].T,
        z[0, :, :].T,
        zdir="y",
        offset=0,
        vmin=vmin,
        vmax=vmax,
        levels=levels,
        cmap=cmap,
        **contour_kwargs,
    )
    if show_back_faces:
        y2 = ax.contourf(
            x[0, :, :].T,
            arr[:, -1, :].T,
            z[0, :, :].T,
            zdir="y",
            offset=ymax,
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            cmap=cmap,
            **contour_kwargs,
        )
    x1 = ax.contourf(
        arr[-1, :, :].T,
        y[:, 0, :].T,
        z[:, 0, :].T,
        zdir="x",
        offset=xmax,
        vmin=vmin,
        vmax=vmax,
        levels=levels,
        cmap=cmap,
        **contour_kwargs,
    )
    if show_back_faces:
        x2 = ax.contourf(
            arr[0, :, :].T,
            y[:, 0, :].T,
            z[:, 0, :].T,
            zdir="x",
            offset=0,
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            cmap=cmap,
            **contour_kwargs,
        )

    return [z1, z2, y1, y2, x1, x2] if show_back_faces else [z1, y1, x1]


def reset_contours(contours):
    for cont in contours:
        cont.remove()


def set_clims(contours, vmin, vmax):
    for cont in contours:
        cont.set_clim(vmin, vmax)
