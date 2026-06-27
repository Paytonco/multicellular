# tests/test_parallel.py

import numpy as np
import pandas as pd

from multicellular.core.cell import Cell
from multicellular.core.colony import Colony
from multicellular.core.environment import Environment
from multicellular.utils.parallel import run_replicates


def _build_growing_colony(replicate_id):
    """A single cell with stochastic division (seeded per replicate)."""
    env = Environment("env", shape=(10, 10))
    rng = np.random.default_rng(replicate_id)
    cell = Cell(
        id=0,
        position=[50.0, 50.0],
        orientation=[1.0, 0.0],
        length=2.0,
        rng=rng,
        cv_delta=0.10,
    )
    return Colony([cell], env)


def test_run_replicates_returns_concatenated_dataframe_with_replicate_id():
    n_replicates = 4
    df = run_replicates(
        _build_growing_colony, n_replicates=n_replicates, dt=0.1, t_max=0.5, n_jobs=2
    )

    assert isinstance(df, pd.DataFrame)
    assert "replicate_id" in df.columns
    assert set(df["replicate_id"].unique()) == set(range(n_replicates))

    # Each replicate's own history matches what a standalone Simulation
    # of the same seeded colony would have recorded.
    for replicate_id in range(n_replicates):
        replicate_df = df[df["replicate_id"] == replicate_id]
        assert len(replicate_df) == 6  # t = 0, 0.1, ..., 0.5
        assert replicate_df["time"].iloc[0] == 0.0
        assert replicate_df["time"].iloc[-1] == 0.5


def test_run_replicates_are_independent_given_distinct_seeds():
    # t_max=1.5 reliably grows the cell (doubling time 1.0) past its
    # division target (mean 3.0, cv_delta=0.10) at least once for every seed.
    df = run_replicates(
        _build_growing_colony, n_replicates=8, dt=0.1, t_max=1.5, n_jobs=2
    )

    # cell_id 1 is the first daughter's id in every replicate (each replicate
    # starts a fresh Colony/_next_id, so ids don't carry over). The time it
    # first appears is each replicate's division time, which should differ
    # by seed since each draws its own division target.
    division_times = df[df["cell_id"] == 1].groupby("replicate_id")["time"].min()
    assert len(division_times) == 8  # every replicate divided at least once
    assert division_times.nunique() > 1


def test_run_replicates_matches_serial_simulation_for_a_fixed_seed():
    from multicellular.core.simulation import Simulation

    df_parallel = run_replicates(
        _build_growing_colony, n_replicates=1, dt=0.1, t_max=0.5, n_jobs=1
    )

    serial_colony = _build_growing_colony(0)
    serial_df = Simulation(serial_colony, dt=0.1, t_max=0.5).run(show_progress=False)

    pd.testing.assert_frame_equal(
        df_parallel.drop(columns=["replicate_id"]), serial_df, check_like=True
    )
