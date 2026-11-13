"""
Benchmark one Simulation's per-step cost, broken down by Colony.step phase.

Times every phase of `Colony.step` separately, so a speedup can be
attributed to the phase it actually affected:

    diffuse      Environment.diffuse (FTCS on every diffusive Field)
    react        Environment.react (per-voxel environment reactions)
    step_cells   Colony._step_cells (every cell's reaction network + growth)
    fields       export_chemical_fields + sample_chemical_fields
    bounds       enforce_bounds + enforce_survival_conditions
    brownian     Brownian noise draws + displacements
    contact      cell-cell and cell-wall contact forces (on the "numba"
                 backend, Brownian motion too: both run in one pass)
    divisions    handle_divisions

Run it before and after a change, with the same arguments, and compare:

    python examples/bench_simulation.py --n-cells 300 --method ODE --field
    python examples/bench_simulation.py --n-cells 3000 --method SSA --field
    python examples/bench_simulation.py --n-cells 300 --method SSA \\
        --ssa-method tau_leaping --walls --anchored

The colony starts with `--n-cells` cells scattered over the box, each
carrying a small quorum-sensing network. With `--field`, the network's
"AHL" species is a diffusing chemical Field, so every cell is coupled to
its voxel.
"""

import argparse
import time
from collections import defaultdict

import numpy as np

from multicellular.core.cell import Cell
from multicellular.core.colony import Colony
from multicellular.core.environment import Environment, Field
from multicellular.core.reactions import Reaction, ReactionNetwork

GRID = 50
SIZE = 200.0

# Colony.step phases, keyed by the method that implements each one.
PHASES = {
    "_step_cells": "step_cells",
    "export_chemical_fields": "fields",
    "sample_chemical_fields": "fields",
    "enforce_bounds": "bounds",
    "enforce_survival_conditions": "bounds",
    "_draw_brownian_noise": "brownian",
    "_apply_brownian_motion": "brownian",
    "_apply_contact_forces": "contact",
    # The numba backend runs Brownian motion and contacts as one pass.
    "_move_cells_numba": "contact",
    "handle_divisions": "divisions",
}
ENV_PHASES = {"diffuse": "diffuse", "react": "react"}


def build_network(with_field):
    """A small quorum-sensing circuit: A is made, secreted as AHL, and taken back up."""
    reactions = {
        "make_A": Reaction({}, {"A": 1}, rate_params={"k": 10.0}),
        "deg_A": Reaction({"A": 1}, {}, rate_params={"k": 0.1}),
        "make_G": Reaction(
            {},
            {"G": 1},
            catalysts=["P", "A"],
            rate_law_type="hill_langmuir",
            rate_params={"alpha": 50, "beta": 1.0, "C": 0.01, "n": 2},
        ),
        "deg_G": Reaction({"G": 1}, {}, rate_params={"k": 0.1}),
    }
    if with_field:
        reactions["secrete"] = Reaction({"A": 1}, {"AHL": 1}, rate_params={"k": 0.5})
        reactions["uptake"] = Reaction({"AHL": 1}, {"A": 1}, rate_params={"k": 0.2})
    return ReactionNetwork("qs", reactions)


def build_colony(n_cells, seed, with_field, walls, anchored):
    rng = np.random.default_rng(seed)

    wall_map = np.zeros((GRID, GRID))
    if walls:
        # A ring of wall pixels a few voxels inside the edge.
        wall_map[3, 3:-3] = wall_map[-4, 3:-3] = 1
        wall_map[3:-3, 3] = wall_map[3:-3, -4] = 1

    fields = None
    if with_field:
        fields = [
            Field(
                "AHL",
                np.zeros((GRID, GRID)),
                is_chemical=True,
                diffuses=True,
                diffusivity=1e-11,
            )
        ]
    env = Environment(
        "bench", wall_map=wall_map, size=(SIZE, SIZE), fields=fields, rng=rng
    )

    lo, hi = 0.2 * SIZE, 0.8 * SIZE
    cells = []
    for _ in range(n_cells):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        cell = Cell(
            id=None,
            position=rng.uniform(lo, hi, size=2),
            orientation=[np.cos(angle), np.sin(angle)],
            length=rng.uniform(1.5, 3.0),
            rng=rng,
            network=build_network(with_field),
        )
        cell.concentrations = {"A": 1.0, "G": 0.0, "P": 1.0}
        cells.append(cell)
    if anchored:
        for x in np.linspace(lo, hi, 5):
            cells.append(
                Cell(
                    id=None,
                    position=[x, SIZE / 2.0],
                    orientation=[1.0, 0.0],
                    length=0.0,
                    radius=3.0,
                    shape="sphere",
                    anchored=True,
                    rng=rng,
                )
            )
    return Colony(cells, env)


def instrument(colony, timings):
    """Wrap each phase method on `colony` (and its environment) to accumulate wall time."""

    def wrap(obj, name, phase):
        original = getattr(obj, name)

        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                timings[phase] += time.perf_counter() - t0

        setattr(obj, name, timed)

    for name, phase in PHASES.items():
        wrap(colony, name, phase)
    for name, phase in ENV_PHASES.items():
        wrap(colony.environment, name, phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--n-cells", type=int, default=300)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--method", default="ODE", help="ODE, SSA, or CLE")
    parser.add_argument("--ssa-method", default="direct")
    parser.add_argument("--ode-method", default="RK45")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--field", action="store_true")
    parser.add_argument("--walls", action="store_true")
    parser.add_argument("--anchored", action="store_true")
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--backend", default=None, help="Colony backend, if supported")
    args = parser.parse_args()

    if args.n_threads is not None:
        import numba

        numba.set_num_threads(args.n_threads)

    colony = build_colony(
        args.n_cells, args.seed, args.field, args.walls, args.anchored
    )
    if args.backend is not None:
        colony.backend = args.backend

    def step():
        colony.step(
            args.dt,
            args.method,
            args.ode_method,
            ssa_method=args.ssa_method,
        )

    # Warm-up steps absorb one-time costs (e.g. JIT compilation) so they
    # don't pollute the per-step timings.
    for _ in range(args.warmup):
        step()

    timings = defaultdict(float)
    instrument(colony, timings)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        step()
    total = time.perf_counter() - t0

    print(
        f"method={args.method} ssa_method={args.ssa_method} "
        f"cells={len(colony.cells)} field={args.field} walls={args.walls} "
        f"anchored={args.anchored} steps={args.steps}"
    )
    print(f"{'phase':<12}{'ms/step':>10}{'share':>8}")
    order = ["diffuse", "react", "step_cells", "fields", "bounds"]
    order += ["brownian", "contact", "divisions"]
    for phase in order:
        ms = 1e3 * timings[phase] / args.steps
        print(f"{phase:<12}{ms:>10.2f}{timings[phase] / total:>8.1%}")
    print(f"{'total':<12}{1e3 * total / args.steps:>10.2f}")


if __name__ == "__main__":
    main()
