# utils/visualization.py

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Rectangle

# Default color used for any RGB channel that isn't tied to a chemical species.
_DEFAULT_CHANNEL_VALUE = 0.3


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


def visualize(simulation, red=None, green=None, blue=None, interval=200):
    """
    Show a 2D animation of a Simulation's cells over time in a pop-up window.

    Args:
        simulation: A `Simulation` instance that has already been `run()`.
        red, green, blue: Optional chemical species names. Each cell's color
            in that channel is its concentration of the given species,
            normalized by the species' maximum value over the simulation.
            Channels left as None default to a constant mid-gray value.
        interval: Delay between animation frames, in milliseconds.

    Returns:
        The `matplotlib.animation.FuncAnimation` driving the pop-up window.
    """
    df = simulation.to_dataframe()
    environment = simulation.colony.environment
    width, height = environment.BOUNDS

    scales = {}
    for species in (red, green, blue):
        if species is not None and species in df.columns:
            scales[species] = df[species].max()

    times = sorted(df["time"].unique())

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

    cell_patches = {}

    def _cell_color(row):
        return (
            _channel_value(row, red, scales.get(red, 1.0)),
            _channel_value(row, green, scales.get(green, 1.0)),
            _channel_value(row, blue, scales.get(blue, 1.0)),
        )

    def update(frame_index):
        t = times[frame_index]
        frame_df = df[(df["time"] == t) & df["alive"]]
        current_ids = set(frame_df["cell_id"])

        for cell_id in list(cell_patches):
            if cell_id not in current_ids:
                cell_patches.pop(cell_id).remove()

        for _, row in frame_df.iterrows():
            cell_id = row["cell_id"]
            position = (row["position_x"], row["position_y"])
            orientation = (row["orientation_x"], row["orientation_y"])
            outline = _capsule_outline(
                position, orientation, row["length"], row["radius"]
            )
            color = _cell_color(row)

            if cell_id in cell_patches:
                cell_patches[cell_id].set_xy(outline)
                cell_patches[cell_id].set_facecolor(color)
            else:
                patch = Polygon(
                    outline, closed=True, facecolor=color, edgecolor="black", zorder=2
                )
                ax.add_patch(patch)
                cell_patches[cell_id] = patch

        ax.set_title(f"t = {t:.2f}")
        return list(cell_patches.values())

    anim = FuncAnimation(
        fig, update, frames=len(times), interval=interval, blit=False, repeat=True
    )
    plt.show()
    return anim


def animate_colony():
    pass


def color_cells():
    pass


def plot_field():
    pass
