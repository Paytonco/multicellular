# core/simulation.py

import pandas as pd
from tqdm import tqdm

_DEFAULT_FILENAME = "simulation.gif"


class Simulation:
    """
    Steps a Colony forward in time, recording the state of every cell
    (position, orientation, and internal chemical concentrations) at
    every timestep.
    """

    def __init__(self, colony, dt, t_max, simulation_method="ODE"):
        """
        Args:
            simulation_method: how every cell's reaction network is advanced
                each step: "ODE" (forward Euler, default), "SSA" (Gillespie),
                or "CLE" (chemical Langevin equation). Case-insensitive.
        """
        self.colony = colony
        self.dt = dt
        self.t_max = t_max
        self.simulation_method = simulation_method.upper()
        self.time = 0.0
        self.history = []
        self.env_history = []
        self.field_history = []

    def record(self):
        """Record the current state of every cell and the active environment."""
        self.env_history.append((self.time, self.colony.environment))
        # Copied rather than referenced: a Field's `values` array is mutated
        # in place by diffusion/export between recorded steps, so without a
        # copy every entry would end up pointing at the same, ever-changing
        # array instead of a snapshot of its value at `self.time`.
        self.field_history.append(
            (
                self.time,
                {
                    name: field.values.copy()
                    for name, field in self.colony.environment.fields.items()
                },
            )
        )
        for cell in self.colony.cells:
            record = {
                "time": self.time,
                "cell_id": cell.id,
                "alive": cell.alive,
                "position_x": cell.position[0],
                "position_y": cell.position[1],
                "orientation_x": cell.orientation[0],
                "orientation_y": cell.orientation[1],
                "length": cell.length,
                "radius": cell.radius,
            }
            record.update(cell.concentrations)
            self.history.append(record)

    def run(self, show_progress=True, t_max=None):
        """
        Step the colony forward, recording state at every step.

        On the first call, steps from t=0 to `t_max` (or `self.t_max` if
        `t_max` is not given), recording the initial state first. Calling
        `run` again continues from the current time instead of resetting
        it, appending to the existing history — pass a new, larger `t_max`
        to extend the simulated window. This supports e.g. switching the
        colony's environment (via `colony.switch_environment`) partway
        through a simulation and continuing it.
        """
        if t_max is not None:
            self.t_max = t_max

        if not self.history:
            self.time = 0.0
            self.record()

        start_step = round(self.time / self.dt)
        n_steps = round(self.t_max / self.dt)
        for step_index in tqdm(
            range(start_step + 1, n_steps + 1), disable=not show_progress
        ):
            self.colony.step(self.dt, self.simulation_method)
            self.time = step_index * self.dt
            self.record()

        return self.to_dataframe()

    def to_dataframe(self):
        """Return the recorded history as a pandas DataFrame."""
        return pd.DataFrame(self.history)

    def visualize_colony(
        self,
        red=None,
        green=None,
        blue=None,
        interval=200,
        save_path=None,
        filename=_DEFAULT_FILENAME,
        show_progress=True,
        stride=1,
    ):
        """
        Show a 2D animation of this Simulation's cells over time in a pop-up window.

        Every frame is rendered to an in-memory image up front (see
        `_render_frames`) rather than drawing cell patches live during
        interactive playback, so playback stays smooth no matter how many cells
        the colony grows to.

        Args:
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
            stride: Only render every `stride`-th recorded time step (default 1 =
                every step). Use stride > 1 to produce a shorter GIF without
                re-running the simulation.

        Returns:
            The `matplotlib.animation.FuncAnimation` driving the pop-up window.
        """
        from ..utils.visualization import _display_and_save, _render_frames

        df = self.to_dataframe()
        env_by_time = {t: env for t, env in self.env_history}

        scales = {}
        for species in (red, green, blue):
            if species is not None and species in df.columns:
                scales[species] = df[species].max()

        times = sorted(df["time"].unique())[::stride]

        frames = _render_frames(
            df, times, env_by_time, red, green, blue, scales, show_progress
        )

        return _display_and_save(frames, interval, save_path, filename)

    def visualize_field(
        self,
        field_name,
        cmap="viridis",
        vmin=0.0,
        vmax=None,
        interval=200,
        save_path=None,
        filename=_DEFAULT_FILENAME,
        show_progress=True,
        stride=1,
    ):
        """
        Show a 2D animation of one Field's values over time in a pop-up window.

        Works the same way as `visualize_colony`: every frame (a heatmap of
        the field's grid, plus the environment name/time labels) is rendered
        to an in-memory image up front, so playback is just fast image
        blitting no matter how many timesteps were recorded.

        Args:
            field_name: Name of the `Field` to animate, e.g. "AHL" or "dye".
            cmap: Matplotlib colormap name used for the heatmap.
            vmin: Lower bound of the color scale.
            vmax: Upper bound of the color scale. If None (default), uses
                the field's maximum recorded value over the whole simulation.
            interval: Delay between animation frames, in milliseconds.
            save_path: Optional directory to save the animation into, as an
                animated GIF. Created if it doesn't already exist. If None
                (default), the animation is only shown, not saved.
            filename: File name to save under, within `save_path`. Defaults to
                "simulation.gif".
            show_progress: Whether to show a progress bar while rendering frames.
            stride: Only render every `stride`-th recorded time step (default 1 =
                every step). Use stride > 1 to produce a shorter GIF without
                re-running the simulation.

        Returns:
            The `matplotlib.animation.FuncAnimation` driving the pop-up window.
        """
        from ..utils.visualization import _display_and_save, _render_field_frames

        env_by_time = {t: env for t, env in self.env_history}
        snapshots = {t: fields[field_name] for t, fields in self.field_history}
        times = sorted(snapshots)[::stride]

        if vmax is None:
            vmax = max(snapshots[t].max() for t in times)

        frames = _render_field_frames(
            times, env_by_time, snapshots, field_name, cmap, vmin, vmax, show_progress
        )

        return _display_and_save(frames, interval, save_path, filename)

    def plot_field(self, field_names, time=None, cmap="viridis", vmin=None, vmax=None):
        """
        Plot one or more Fields as static heatmaps at a single point in time.

        Uses the same imshow/colorbar/title convention as `visualize_field`
        (env name top-left, `t = ...` top-right, a colorbar labeled with the
        field's name), but renders one static figure instead of an
        animation. Passing several `field_names` draws them as side-by-side
        panels in one figure — handy for fields on very different scales
        (e.g. viscosity and diffusivity), which wouldn't share a sensible
        color scale.

        Args:
            field_names: Name of the `Field` to plot, or a list of names to
                plot side by side in one figure.
            time: Simulation time to plot. Snaps to the closest timestep
                actually recorded in `field_history`. Defaults to the most
                recently recorded time (i.e. the final state).
            cmap: Matplotlib colormap name used for every panel.
            vmin, vmax: Color-scale bounds, applied to every panel. Default
                to None, so each panel auto-scales to its own field's value
                range at the plotted time (plain `imshow` behavior) — pass
                both explicitly to force a shared/fixed scale, e.g. to keep
                two plots of the same field at different times comparable.

        Returns:
            The `matplotlib.figure.Figure` containing one panel per field.
            Unlike `visualize_colony`/`visualize_field`, this does not call
            `plt.show()` itself, so further annotations (e.g.
            `fig.axes[0].axvline(...)`) can be added before displaying it —
            it auto-displays as a Jupyter cell's return value, or call
            `plt.show()` explicitly once done customizing it.
        """
        from ..utils.visualization import _render_field_plot

        if isinstance(field_names, str):
            field_names = [field_names]

        recorded_times = [t for t, _ in self.field_history]
        target_time = (
            recorded_times[-1]
            if time is None
            else min(recorded_times, key=lambda t: abs(t - time))
        )
        env = dict(self.env_history)[target_time]
        values_by_field = dict(self.field_history)[target_time]

        return _render_field_plot(
            field_names, target_time, env, values_by_field, cmap, vmin, vmax
        )
