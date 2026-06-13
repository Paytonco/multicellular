# multicellular

An agent-based simulation framework for bacterial cell colonies. Each cell is
modeled as a rod-shaped body (cylinder with hemispherical caps) carrying an
internal chemical reaction network (e.g. a gene-regulatory circuit), and is
intended to grow, divide, move, and interact within a shared environment.

## Status

This package is under active development. Core building blocks for individual
cells, chemical reaction networks, a basic environment/field container, a
basic colony, and a basic simulation loop are implemented and tested; SBML
import and visualization are not yet implemented (see
[Unimplemented / stubs](#unimplemented--stubs)).

## Installation

Requires Python >= 3.11.

```bash
git clone https://github.com/Paytonco/multicellular.git
cd multicellular
pip install -e .
```

This installs the `multicellular` package (from `src/multicellular`) in
editable mode, along with its only current dependency, `numpy`.

## Usage

### `Cell`

```python
from multicellular import Cell

cell = Cell(
    id=1,
    position=[0.0, 0.0],
    orientation=[1.0, 0.0],
    length=2.0,
    radius=0.5,
)

print(cell.compute_volume())

# Grow over time and check for division
dt = 0.1
while not cell.ready_to_divide(threshold_length=4.0):
    cell.grow(dt, growth_rate=1.0)

daughter1, daughter2 = cell.divide()
```

### `Reaction` and `ReactionNetwork`

`Reaction` supports `mass_action`, `michaelis_menten`, `hill_langmuir`, and
`custom` rate laws. `ReactionNetwork` collects reactions, builds a
stoichiometry matrix, and advances concentrations with a forward-Euler ODE
step.

```python
from multicellular import ReactionNetwork
from multicellular.core.reactions import Reaction

# A -> B at rate k * [A]
rxn = Reaction(
    reactants={"A": 1},
    products={"B": 1},
    rate_law_type="mass_action",
    rate_params={"k": 1.0},
)

network = ReactionNetwork(name="linear", reactions={"R": rxn})

state = {"A": 1.0, "B": 0.0}
new_state = network.simulate_step(state, dt=0.1, volume=1.0)
```

### Cell with an internal reaction network

A `Cell` can be given a `ReactionNetwork`; calling `cell.step(dt)` advances
both its internal chemistry (via the network's ODE step) and its growth:

```python
from multicellular import Cell, ReactionNetwork
from multicellular.core.reactions import Reaction

rxn = Reaction({"A": 1}, {"B": 1}, rate_law_type="mass_action", rate_params={"k": 1.0})
network = ReactionNetwork("linear", {"R": rxn})

cell = Cell(id=0, position=[0.0, 0.0], orientation=[1.0, 0.0], network=network)
cell.concentrations = {"A": 1.0, "B": 0.0}

cell.step(dt=0.1)
print(cell.concentrations)
```

When a cell divides, each daughter receives its own cloned copy of the
reaction network (`network.clone()`). For most species, concentration is
conserved across division (each daughter inherits the parent's
concentration, which conserves total copy number since each daughter has
half the parent's volume).

### Low-copy species

Some species — e.g. low-copy plasmids — should not simply have their
concentration copied to both daughters; instead, their *copy number* should
be split stochastically between daughters. Use `set_concentration` with
`low_copy=True` to designate a species this way when initializing it:

```python
cell.set_concentration("plasmid", value, low_copy=True)
```

At division, the parent's copy number `n` for that species (`concentration *
volume`, rounded to the nearest whole number) is split so that one daughter
gets `x` copies and the other gets `n - x`, conserving total copy number
exactly:

- If `n <= Cell.LOW_COPY_GAUSSIAN_THRESHOLD` (35), `x ~ Binomial(n, 1/2)`.
- If `n > 35`, `x` is instead drawn from the Gaussian approximation to that
  binomial (CLT), `x ~ Normal(n/2, sqrt(n)/2)`, then rounded and clamped to
  `[0, n]`.

Each daughter's concentration for that species is then `x / daughter_volume`
(or `(n - x) / daughter_volume`). The `low_copy` designation itself is
inherited by both daughters.

### `Environment` and `Field`

`Environment` represents the shared extracellular space as a collection of
named `Field`s - matrices of values over a shared grid (e.g. chemical
concentrations, temperature, surface roughness). Reaction-diffusion dynamics
are not implemented yet; for now `Environment` is just a validated container.

```python
import numpy as np
from multicellular import Environment, Field

glucose = Field("glucose", np.ones((10, 10)))
env = Environment(shape=(10, 10), fields=[glucose])

env.get_field("glucose")
```

Every `Field` added to an `Environment` must have a `values` matrix matching
the environment's `shape`; adding a mismatched field raises `ValueError`.

`Environment.BOUNDS` is currently hardcoded to `(100.0, 100.0)`, representing
a 100um x 100um square that cells will interact within (this will be made
configurable later). A field's grid `shape` is independent of `BOUNDS` and
may cover an area larger than the simulation bounds. `env.in_bounds(position)`
checks whether a 2D position (e.g. a cell's center of mass) falls within
`[0, BOUNDS[0]] x [0, BOUNDS[1]]`.

### `Colony`

A `Colony` is a collection of `Cell`s living in an `Environment`:

```python
from multicellular import Cell, Colony, Environment

env = Environment(shape=(10, 10))
cells = [Cell(id=0, position=[10.0, 10.0], orientation=[1.0, 0.0])]

colony = Colony(cells, env)
colony.step(dt=0.1)  # steps every cell, then enforces environment bounds
```

`colony.step(dt)` calls `cell.step(dt)` for every cell, then checks each
living cell's center of mass against `environment.in_bounds(...)` (any cell
that has left the bounds is killed via `cell.kill()`), and finally replaces
any cell with `cell.ready_to_divide() == True` with its two daughter cells
(each daughter is assigned a new, unique `id`). `colony.living_cells` returns
the cells that are still alive.

Dead cells (`cell.alive == False`) are inert: `Cell.step` and
`Cell.apply_force` are no-ops for them, so dead cells no longer grow, run
their reaction network, or move in response to forces. Dead cells also never
divide.

### `Simulation`

`Simulation` steps a `Colony` from `t=0` to `t=t_max` in increments of `dt`,
recording every cell's position, orientation, length, alive status, and
chemical concentrations at every timestep:

```python
from multicellular import Cell, Colony, Environment, Simulation

env = Environment(shape=(10, 10))
cell = Cell(id=0, position=[50.0, 50.0], orientation=[1.0, 0.0])
colony = Colony([cell], env)

sim = Simulation(colony, dt=0.1, t_max=10.0)
df = sim.run()  # pandas DataFrame, one row per cell per recorded timestep
```

`sim.run()` records the initial state, then repeatedly calls `colony.step(dt)`
and records the new state of every cell remaining in the colony - including
any new daughter cells produced by division - until `t_max` is reached. The
returned DataFrame has columns `time`, `cell_id`, `alive`, `position_x`,
`position_y`, `orientation_x`, `orientation_y`, `length`, plus one column per
chemical species seen in any cell's `concentrations` (missing values are
`NaN` for cells/species that don't have that entry). The same data is also
available via `sim.history` (a list of per-cell-per-timestep dicts) and
`sim.to_dataframe()`.

## Unimplemented / stubs

These pieces exist as placeholders (empty classes/functions or
`NotImplementedError`) and are not yet usable:

- **`ReactionNetwork.simulate_step`** with `simulation_method="SSA"` or
  `"CLE"` - raises `NotImplementedError`. Only `"ODE"` (forward Euler) is
  implemented.
- **`ReactionNetwork.from_sbml`** and **`multicellular.parse_sbml`**
  (`utils/sbml_parser.py`) - raise `NotImplementedError` / are empty. Intended
  to construct a `ReactionNetwork` from an SBML model file.
- **`multicellular.animate_colony`**, **`multicellular.color_cells`**, and
  **`plot_field`** (`utils/visualization.py`) - empty functions. Intended for
  animating a `Colony` over time, coloring cells (e.g. by species or
  concentration), and plotting `Environment` fields.
- **`Cell.interact_with_environment`** - placeholder method (no-op), intended
  to let a cell read from / write to an `Environment`.
- **`examples/toggle_switch_demo.py`** and
  **`examples/quorum_sensing_demo.py`** - empty placeholder scripts, intended
  as end-to-end demos of a genetic toggle switch and a quorum-sensing circuit.

## Running tests

Run the full test suite from the project root:

```bash
pytest
```

Run a specific test file:

```bash
pytest tests/test_cell.py        # Cell geometry, growth, and division
pytest tests/test_reactions.py   # Reaction rate laws, stoichiometry, ODE stepping, Cell + ReactionNetwork integration
pytest tests/test_environment.py # Environment/Field construction and validation
pytest tests/test_colony.py      # Colony bounds enforcement, division, and dead-cell behavior
pytest tests/test_simulation.py  # Simulation loop and DataFrame export
pytest tests/test_dummy.py       # trivial sanity check
```

`tests/test_rections.py` is currently an empty placeholder (no tests to run
yet).
