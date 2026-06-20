# core/simulation.py

import pandas as pd
from tqdm import tqdm


class Simulation:
    """
    Steps a Colony forward in time, recording the state of every cell
    (position, orientation, and internal chemical concentrations) at
    every timestep.
    """

    def __init__(self, colony, dt, t_max):
        self.colony = colony
        self.dt = dt
        self.t_max = t_max
        self.time = 0.0
        self.history = []

    def record(self):
        """Record the current state of every cell in the colony."""
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
            self.colony.step(self.dt)
            self.time = step_index * self.dt
            self.record()

        return self.to_dataframe()

    def to_dataframe(self):
        """Return the recorded history as a pandas DataFrame."""
        return pd.DataFrame(self.history)
