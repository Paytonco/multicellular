# tests/test_simulation.py

import numpy as np
import pandas as pd
import pytest

from multicellular.core.cell import Cell
from multicellular.core.colony import Colony
from multicellular.core.environment import Environment
from multicellular.core.reactions import Reaction, ReactionNetwork
from multicellular.core.simulation import Simulation


def test_simulation_records_growth_and_division():
    env = Environment(shape=(10, 10))
    rng = np.random.default_rng(0)
    cell = Cell(
        id=0, position=[50.0, 50.0], orientation=[1.0, 0.0], length=2.0, rng=rng
    )
    # Capture the division target before any steps so we can check daughter lengths.
    division_target = cell._division_target
    colony = Colony([cell], env)

    # dt=10 guarantees the cell grows well past its target (L grows ~1000×) in one step.
    sim = Simulation(colony, dt=10.0, t_max=10.0)
    df = sim.run(show_progress=False)

    assert isinstance(df, pd.DataFrame)

    # Parent appears at t=0 with its initial length.
    parent_at_0 = df[(df["cell_id"] == 0) & (df["time"] == 0.0)]
    assert len(parent_at_0) == 1
    assert parent_at_0["length"].values[0] == pytest.approx(2.0)

    # After one step the parent has divided; exactly two daughters are recorded.
    at_t10 = df[df["time"] == 10.0]
    assert len(at_t10) == 2
    assert set(at_t10["cell_id"]) == {1, 2}
    assert all(at_t10["alive"])
    # Daughters are created at f * L_d = 0.5 * division_target (no growth yet).
    for length in at_t10["length"]:
        assert length == pytest.approx(division_target / 2)


def test_simulation_records_chemical_concentrations():
    rxn = Reaction(
        {"A": 1}, {"B": 1}, rate_law_type="mass_action", rate_params={"k": 1.0}
    )
    network = ReactionNetwork("linear", {"R": rxn})

    env = Environment(shape=(10, 10))
    cell = Cell(id=0, position=[50.0, 50.0], orientation=[1.0, 0.0], network=network)
    cell.set_concentration("A", 1.0)
    cell.set_concentration("B", 0.0)

    colony = Colony([cell], env)
    sim = Simulation(colony, dt=0.1, t_max=0.5)
    df = sim.run(show_progress=False)

    assert "A" in df.columns
    assert "B" in df.columns
    assert len(df) == 6  # t = 0, 0.1, ..., 0.5

    # A decreases and B increases as the reaction proceeds.
    assert df["A"].iloc[0] > df["A"].iloc[-1]
    assert df["B"].iloc[0] < df["B"].iloc[-1]
