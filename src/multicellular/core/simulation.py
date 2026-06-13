# core/simulation.py

import pandas as pd


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
            }
            record.update(cell.concentrations)
            self.history.append(record)

    def run(self):
        """Step the colony from t=0 to t=t_max, recording state at every step."""
        self.time = 0.0
        self.history = []
        self.record()

        n_steps = round(self.t_max / self.dt)
        for step_index in range(1, n_steps + 1):
            self.colony.step(self.dt)
            self.time = step_index * self.dt
            self.record()

        return self.to_dataframe()

    def to_dataframe(self):
        """Return the recorded history as a pandas DataFrame."""
        return pd.DataFrame(self.history)
