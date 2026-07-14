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

# Transparency and stacking order for the optional field heatmap drawn
# behind cells in `_render_frames`: below the cell patches (zorder=2) but
# above the wall_map background (zorder=1), so it reads as a subtle
# backdrop rather than competing with the cells for attention.
_FIELD_OVERLAY_ALPHA = 0.45
_FIELD_OVERLAY_ZORDER = 1.5

# wall_map entry -> RGB, used to paint the environment background: media (0)
# is left white/uncolored so it doesn't compete with the field overlay, wall
# (1) is a solid neutral gray, out-of-bounds (-1) is the same red tint as the
# margin beyond the environment's physical extent.
_WALL_MAP_COLORS = {
    -1: (0.86, 0.24, 0.24),
    0: (1.0, 1.0, 1.0),
    1: (0.35, 0.35, 0.35),
}


def _wall_map_rgba(wall_map):
    """Render a wall_map matrix as an RGBA image per `_WALL_MAP_COLORS`."""
    rgba = np.ones(wall_map.shape + (4,))
    for value, color in _WALL_MAP_COLORS.items():
        rgba[wall_map == value, 0:3] = color
    return rgba


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


def _add_field_overlay(ax, field_name, values, width, height, cmap, vmin, vmax):
    """
    Draw a Field's values as a light, semi-transparent heatmap layer behind
    the cells, and add a colorbar labeled with the field's name.

    The overlay's extent is exactly the environment rectangle (0 to width, 0
    to height) that `_render_frames` paints its wall_map background onto —
    the same region the red out-of-bounds tint is painted *outside* of — so
    the overlay sits on top of that background within bounds and never
    bleeds into the tinted margin. `zorder` places it above the wall_map
    background but below the cell patches (see `_FIELD_OVERLAY_ZORDER`).
    """
    image = ax.imshow(
        values,
        origin="lower",
        extent=[0, width, 0, height],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=_FIELD_OVERLAY_ALPHA,
        zorder=_FIELD_OVERLAY_ZORDER,
    )
    ax.figure.colorbar(image, ax=ax, label=field_name)
    return image


def _render_frames(
    df,
    times,
    env_by_time,
    red,
    green,
    blue,
    scales,
    show_progress,
    field_name=None,
    field_snapshots=None,
    field_cmap="YlOrRd",
    field_vmin=0.0,
    field_vmax=None,
):
    """
    Render every animation frame to an in-memory RGBA image up front.

    Drawing many individual cell patches is what makes this visualization
    slow; doing that once per frame here, rather than live during playback,
    means displaying (or saving) the animation afterward is just fast image
    blitting, regardless of how large the colony grows.
    """
    first_env = next(iter(env_by_time.values()))
    width, height = first_env.size
    pad = max(width, height) * 0.1
    x_min = min(0.0, df["position_x"].min()) - pad
    x_max = max(width, df["position_x"].max()) + pad
    y_min = min(0.0, df["position_y"].min()) - pad
    y_max = max(height, df["position_y"].max()) + pad

    fig, ax = plt.subplots()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")

    # Tint the margin beyond the environment's physical extent red (out of
    # bounds), then paint the environment itself from its wall_map: media
    # white, walls gray, and any interior out-of-bounds (-1) cells the same
    # red as the margin.
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
    ax.imshow(
        _wall_map_rgba(first_env.wall_map),
        origin="lower",
        extent=[0, width, 0, height],
        zorder=1,
        interpolation="nearest",
    )

    field_image = None
    if field_name is not None:
        field_image = _add_field_overlay(
            ax,
            field_name,
            field_snapshots[times[0]],
            width,
            height,
            field_cmap,
            field_vmin,
            field_vmax,
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
        if field_image is not None:
            field_image.set_data(field_snapshots[t])

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

        env = env_by_time.get(t, first_env)
        ax.set_title(env.name, loc="left")
        ax.set_title(f"t = {t:.2f}", loc="right")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())

        for patch in cell_polygons:
            patch.remove()
        cell_polygons.clear()

    plt.close(fig)
    return frames


def _render_field_frames(
    times, env_by_time, snapshots, field_name, cmap, vmin, vmax, show_progress
):
    """
    Render every field-animation frame to an in-memory RGBA image up front,
    the same "render ahead of time" strategy as `_render_frames` uses for
    cell colonies.
    """
    first_env = next(iter(env_by_time.values()))
    width, height = first_env.size

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    image = ax.imshow(
        snapshots[times[0]],
        origin="lower",
        extent=[0, width, 0, height],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(image, ax=ax, label=field_name)

    frames = []
    for t in tqdm(times, disable=not show_progress, desc="Rendering frames"):
        image.set_data(snapshots[t])
        env = env_by_time.get(t, first_env)
        ax.set_title(env.name, loc="left")
        ax.set_title(f"t = {t:.2f}", loc="right")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())

    plt.close(fig)
    return frames


def _render_field_plot(field_names, time, env, values_by_field, cmap, vmin, vmax):
    """
    Render one static heatmap per field, side by side in one figure.

    Uses the same imshow/colorbar/title convention as `_render_field_frames`
    (one panel per field, colorbar labeled with the field's name, env name
    top-left / time top-right), but for a single point in time rather than
    a whole animation.

    Unlike `_display_and_save`, this does not call `plt.show()` itself: the
    figure is returned so the caller can add further annotations (e.g.
    `fig.axes[0].axvline(...)`) before displaying it — calling `plt.show()`
    early would, under Jupyter's inline backend, capture the figure
    immediately and miss anything added afterward.
    """
    width, height = env.size
    fig, axes = plt.subplots(
        1, len(field_names), figsize=(4.5 * len(field_names), 4), squeeze=False
    )

    for ax, field_name in zip(axes[0], field_names):
        image = ax.imshow(
            values_by_field[field_name],
            origin="lower",
            extent=[0, width, 0, height],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax, label=field_name)
        ax.set_title(env.name, loc="left")
        ax.set_title(f"t = {time:.2f}", loc="right")

    fig.tight_layout()
    return fig


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


def _display_and_save(frames, interval, save_path, filename):
    """
    Save (optionally) and interactively play back pre-rendered RGBA frames.

    Shared by `Simulation.visualize_colony` and `Simulation.visualize_field`:
    once a list of frames has been rendered ahead of time, showing (and
    optionally saving) them works identically regardless of what the frames
    depict.
    """
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
