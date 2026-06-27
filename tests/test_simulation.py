# tests/test_simulation.py

import numpy as np
import pandas as pd
import pytest

from multicellular.core.cell import Cell
from multicellular.core.colony import Colony
from multicellular.core.environment import Environment, Field
from multicellular.core.reactions import Reaction, ReactionNetwork
from multicellular.core.simulation import Simulation


def test_simulation_records_growth_and_division():
    env = Environment("env", shape=(10, 10))
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

    env = Environment("env", shape=(10, 10))
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


def test_run_again_with_larger_t_max_continues_and_appends_history():
    env = Environment("env", shape=(10, 10))
    cell = Cell(id=0, position=[50.0, 50.0], orientation=[1.0, 0.0])
    colony = Colony([cell], env)

    sim = Simulation(colony, dt=0.1, t_max=0.5)
    sim.run(show_progress=False)
    assert sim.time == pytest.approx(0.5)
    n_records_first_run = len(sim.history)

    sim.run(show_progress=False, t_max=1.0)

    assert sim.time == pytest.approx(1.0)
    assert len(sim.history) > n_records_first_run
    times = sorted({record["time"] for record in sim.history})
    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(1.0)
    # No duplicate record for the time the first run already ended at.
    assert sum(1 for record in sim.history if record["time"] == pytest.approx(0.5)) == 1


def test_run_again_with_same_t_max_is_a_no_op():
    env = Environment("env", shape=(10, 10))
    cell = Cell(id=0, position=[50.0, 50.0], orientation=[1.0, 0.0])
    colony = Colony([cell], env)

    sim = Simulation(colony, dt=0.1, t_max=0.5)
    sim.run(show_progress=False)
    history_after_first_run = list(sim.history)

    sim.run(show_progress=False)

    assert sim.history == history_after_first_run


def test_run_continues_after_switching_environment():
    field_before = Field("A", np.full((10, 10), 1.0), is_chemical=True)
    env_before = Environment("before", shape=(10, 10), fields=[field_before])
    # growth_rate=0.0 isolates this test from growth-driven dilution (see
    # test_cell.py), so the recorded "A" reflects only the chemical field.
    cell = Cell(id=0, position=[50.0, 50.0], orientation=[1.0, 0.0], growth_rate=0.0)
    colony = Colony([cell], env_before)

    sim = Simulation(colony, dt=0.1, t_max=0.5)
    sim.run(show_progress=False)

    field_after = Field("A", np.full((10, 10), 9.0), is_chemical=True)
    env_after = Environment("after", shape=(10, 10), fields=[field_after])
    colony.switch_environment(env_after)

    df = sim.run(show_progress=False, t_max=1.0)

    # t=0's "A" is NaN: it's recorded before the first step ever applies a
    # chemical field, so exclude it and compare the rest.
    before_switch = df[(df["time"] > 0.0) & (df["time"] <= 0.5)]
    after_switch = df[df["time"] > 0.5]
    assert np.allclose(before_switch["A"], 1.0)
    assert np.allclose(after_switch["A"], 9.0)


def test_env_history_tracks_environment_per_timestep():
    env = Environment("env", shape=(10, 10))
    cell = Cell(id=0, position=[50.0, 50.0], orientation=[1.0, 0.0])
    colony = Colony([cell], env)

    sim = Simulation(colony, dt=0.1, t_max=0.3)
    sim.run(show_progress=False)

    # One env_history entry per recorded timestep (t=0, 0.1, 0.2, 0.3).
    assert len(sim.env_history) == 4
    times = [t for t, _ in sim.env_history]
    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(0.3)
    for _, recorded_env in sim.env_history:
        assert recorded_env is env


def test_env_history_reflects_environment_switch():
    env_before = Environment("before", shape=(10, 10))
    cell = Cell(id=0, position=[50.0, 50.0], orientation=[1.0, 0.0])
    colony = Colony([cell], env_before)

    sim = Simulation(colony, dt=0.1, t_max=0.2)
    sim.run(show_progress=False)

    env_after = Environment("after", shape=(10, 10))
    colony.switch_environment(env_after)
    sim.run(show_progress=False, t_max=0.4)

    # First three entries (t=0, 0.1, 0.2) use env_before.
    for t, recorded_env in sim.env_history[:3]:
        assert recorded_env is env_before
    # Last two entries (t=0.3, 0.4) use env_after.
    for t, recorded_env in sim.env_history[3:]:
        assert recorded_env is env_after
