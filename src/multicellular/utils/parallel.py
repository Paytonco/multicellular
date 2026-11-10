# utils/parallel.py

from typing import Callable

import pandas as pd
from joblib import Parallel, delayed

from multicellular.core.colony import Colony
from multicellular.core.reactions import DEFAULT_TAU_EPSILON
from multicellular.core.simulation import Simulation


def _run_one(
    build_colony,
    replicate_id,
    dt,
    t_max,
    simulation_method,
    ode_method,
    show_progress,
    ssa_method="direct",
    tau_epsilon=DEFAULT_TAU_EPSILON,
):
    colony = build_colony(replicate_id)
    sim = Simulation(
        colony,
        dt,
        t_max,
        simulation_method=simulation_method,
        ode_method=ode_method,
        ssa_method=ssa_method,
        tau_epsilon=tau_epsilon,
    )
    df = sim.run(show_progress=show_progress)
    df = df.copy()
    df["replicate_id"] = replicate_id
    return df


def run_replicates(
    build_colony: Callable[[int], Colony],
    n_replicates: int,
    dt: float,
    t_max: float,
    simulation_method: str = "ODE",
    ode_method: str = "RK45",
    n_jobs: int = -1,
    show_progress: bool = False,
    ssa_method: str = "direct",
    tau_epsilon: float = DEFAULT_TAU_EPSILON,
) -> pd.DataFrame:
    """
    Run independent Simulations in parallel worker processes.

    Each replicate gets its own fresh Colony (built from scratch by
    `build_colony`) and Simulation, so there is no shared mutable state
    across replicates and no risk of cross-talk between them.

    Args:
        build_colony: Callable(replicate_id) -> Colony. Called once per
            replicate (inside the worker process) to construct an
            independent Colony. To get independent random streams, build the
            colony's Cells with a distinct `rng`, e.g.
            `Cell(..., rng=np.random.default_rng(replicate_id))`.
        n_replicates: Number of independent replicates to run.
        dt, t_max: Forwarded to each replicate's `Simulation`.
        simulation_method: Forwarded to each replicate's `Simulation`
            ("ODE", "SSA", or "CLE").
        ode_method: Forwarded to each replicate's `Simulation`; the
            `scipy.integrate.solve_ivp` method used when `simulation_method`
            is "ODE" (default "RK45").
        n_jobs: Forwarded to `joblib.Parallel` (-1 uses all available cores).
        show_progress: Whether each individual `Simulation.run` shows its
            own tqdm bar. Off by default since N bars interleaving across
            processes is unreadable; the replicates themselves are not
            individually progress-tracked here.
        ssa_method: Forwarded to each replicate's `Simulation`; the
            stochastic algorithm used when `simulation_method` is "SSA"
            ("direct", "next_reaction", or "tau_leaping").
        tau_epsilon: Forwarded to each replicate's `Simulation`;
            tau-leaping error control (default 0.03).

    Returns:
        A single DataFrame, the concatenation of every replicate's
        `Simulation.run()` output, with an added `replicate_id` column
        identifying which run each row came from.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_one)(
            build_colony,
            i,
            dt,
            t_max,
            simulation_method,
            ode_method,
            show_progress,
            ssa_method,
            tau_epsilon,
        )
        for i in range(n_replicates)
    )
    return pd.concat(results, ignore_index=True)
