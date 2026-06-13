# tests/test_simulation.py

import pandas as pd
import pytest

from multicellular.core.cell import Cell
from multicellular.core.colony import Colony
from multicellular.core.environment import Environment
from multicellular.core.reactions import Reaction, ReactionNetwork
from multicellular.core.simulation import Simulation


def test_simulation_records_growth_and_division():
    env = Environment(shape=(10, 10))
    cell = Cell(id=0, position=[50.0, 50.0], orientation=[1.0, 0.0], length=2.0)
    colony = Colony([cell], env)

    sim = Simulation(colony, dt=2.0, t_max=4.0)
    df = sim.run()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4  # 1 cell at t=0, 1 cell at t=2, 2 daughters at t=4

    # The parent cell grows for two steps before dividing.
    parent_rows = df[df["cell_id"] == 0]
    assert list(parent_rows["time"]) == [0.0, 2.0]
    assert list(parent_rows["length"]) == [2.0, 3.0]

    # Daughters appear at t=4 with new ids and conserve copy number / position.
    daughter_rows = df[df["time"] == 4.0]
    assert len(daughter_rows) == 2
    assert set(daughter_rows["cell_id"]) == {1, 2}
    assert daughter_rows["length"].tolist() == pytest.approx([2.0, 2.0])
    assert all(daughter_rows["alive"])


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
    df = sim.run()

    assert "A" in df.columns
    assert "B" in df.columns
    assert len(df) == 6  # t = 0, 0.1, ..., 0.5

    # A decreases and B increases as the reaction proceeds.
    assert df["A"].iloc[0] > df["A"].iloc[-1]
    assert df["B"].iloc[0] < df["B"].iloc[-1]
