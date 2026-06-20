# utils/visualization.py

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Rectangle
from PIL import Image
from tqdm import tqdm

# Default color used for any RGB channel that isn't tied to a chemical species.
_DEFAULT_CHANNEL_VALUE = 0.3
_DEFAULT_FILENAME = "simulation.gif"


def _capsule_outline(position, orientation, length, radius, n_points=12):
    """
    Return the (x, y) outline of a rod-shaped (capsule) cell: a cylinder of
    the given length and radius, centered at position and aligned with
    orientation, with hemispherical caps at each end.
    """
    theta = np.arctan2(orientation[1], orientation[0])
    half_length = length / 2.0

    right_angles = np.linspace(-np.pi / 2, np.pi / 2, n_points)
    left_angles = np.linspace(np.pi / 2, 3 * np.pi / 2, n_points)

    right_cap = np.column_stack(
        [half_length + radius * np.cos(right_angles), radius * np.sin(right_angles)]
    )
    left_cap = np.column_stack(
        [-half_length + radius * np.cos(left_angles), radius * np.sin(left_angles)]
    )
    local_points = np.vstack([right_cap, left_cap])

    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return local_points @ rotation.T + np.asarray(position)


def _channel_value(row, species, scale):
    """Concentration of `species` in `row`, normalized to [0, 1] by `scale`."""
    if species is None:
        return _DEFAULT_CHANNEL_VALUE
    value = row.get(species, 0.0)
    if scale <= 0:
        return 0.0
    return float(np.clip(value / scale, 0.0, 1.0))


def _render_frames(df, times, environment, red, green, blue, scales, show_progress):
    """
    Render every animation frame to an in-memory RGBA image up front.

    Drawing many individual cell patches is what makes this visualization
    slow; doing that once per frame here, rather than live during playback,
    means displaying (or saving) the animation afterward is just fast image
    blitting, regardless of how large the colony grows.
    """
    width, height = environment.bounds
    pad = max(width, height) * 0.1
    x_min = min(0.0, df["position_x"].min()) - pad
    x_max = max(width, df["position_x"].max()) + pad
    y_min = min(0.0, df["position_y"].min()) - pad
    y_max = max(height, df["position_y"].max()) + pad

    fig, ax = plt.subplots()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")

    # Tint everything red, then paint the in-bounds region back to white so
    # only the out-of-bounds margin remains tinted.
    ax.add_patch(
        Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            facecolor="red",
            alpha=0.2,
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle((0, 0), width, height, facecolor="white", edgecolor="none", zorder=1)
    )

    def cell_color(row):
        return (
            _channel_value(row, red, scales.get(red, 1.0)),
            _channel_value(row, green, scales.get(green, 1.0)),
            _channel_value(row, blue, scales.get(blue, 1.0)),
        )

    # Built via an explicit comprehension, not dict(groupby_obj): GroupBy
    # exposes a `.keys` attribute (the grouping column name, a plain string)
    # that the dict() constructor mistakes for a mapping's keys() method.
    frames_by_time = {t: group for t, group in df[df["alive"]].groupby("time")}

    frames = []
    cell_polygons = []
    for t in tqdm(times, disable=not show_progress, desc="Rendering frames"):
        frame_df = frames_by_time.get(t)
        if frame_df is not None:
            for _, row in frame_df.iterrows():
                outline = _capsule_outline(
                    (row["position_x"], row["position_y"]),
                    (row["orientation_x"], row["orientation_y"]),
                    row["length"],
                    row["radius"],
                )
                patch = Polygon(
                    outline,
                    closed=True,
                    facecolor=cell_color(row),
                    edgecolor="black",
                    zorder=2,
                )
                ax.add_patch(patch)
                cell_polygons.append(patch)

        ax.set_title(f"t = {t:.2f}")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())

        for patch in cell_polygons:
            patch.remove()
        cell_polygons.clear()

    plt.close(fig)
    return frames


def _save_frames(frames, save_path, filename, fps):
    """Save pre-rendered RGBA frames as an animated GIF under save_path."""
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, filename)

    duration_ms = max(round(1000.0 / fps), 1)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        filepath,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    return filepath


def visualize(
    simulation,
    red=None,
    green=None,
    blue=None,
    interval=200,
    save_path=None,
    filename=_DEFAULT_FILENAME,
    show_progress=True,
):
    """
    Show a 2D animation of a Simulation's cells over time in a pop-up window.

    Every frame is rendered to an in-memory image up front (see
    `_render_frames`) rather than drawing cell patches live during
    interactive playback, so playback stays smooth no matter how many cells
    the colony grows to.

    Args:
        simulation: A `Simulation` instance that has already been `run()`.
        red, green, blue: Optional chemical species names. Each cell's color
            in that channel is its concentration of the given species,
            normalized by the species' maximum value over the simulation.
            Channels left as None default to a constant mid-gray value.
        interval: Delay between animation frames, in milliseconds.
        save_path: Optional directory to save the animation into, as an
            animated GIF. Created if it doesn't already exist. If None
            (default), the animation is only shown, not saved.
        filename: File name to save under, within `save_path`. Defaults to
            "simulation.gif".
        show_progress: Whether to show a progress bar while rendering frames.

    Returns:
        The `matplotlib.animation.FuncAnimation` driving the pop-up window.
    """
    df = simulation.to_dataframe()
    environment = simulation.colony.environment

    scales = {}
    for species in (red, green, blue):
        if species is not None and species in df.columns:
            scales[species] = df[species].max()

    times = sorted(df["time"].unique())

    frames = _render_frames(
        df, times, environment, red, green, blue, scales, show_progress
    )

    if save_path is not None:
        _save_frames(frames, save_path, filename, fps=1000.0 / interval)

    fig, ax = plt.subplots()
    ax.axis("off")
    fig.tight_layout(pad=0)
    image_artist = ax.imshow(frames[0])

    def update(frame_index):
        image_artist.set_data(frames[frame_index])
        return [image_artist]

    anim = FuncAnimation(
        fig, update, frames=len(frames), interval=interval, blit=True, repeat=True
    )
    plt.show()
    return anim


def animate_colony():
    pass


def color_cells():
    pass


def plot_field():
    pass
