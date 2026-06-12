# multicellular

An agent-based simulation framework for bacterial cell colonies. Each cell is
modeled as a rod-shaped body (cylinder with hemispherical caps) carrying an
internal chemical reaction network (e.g. a gene-regulatory circuit), and is
intended to grow, divide, move, and interact within a shared environment.

## Status

This package is under active development. Core building blocks for individual
cells and chemical reaction networks are implemented and tested; the
multi-cell colony, spatial environment, simulation loop, SBML import, and
visualization are not yet implemented (see [Unimplemented / stubs](#unimplemented--stubs)).

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
reaction network (`network.clone()`) along with a copy of the parent's
concentrations.

## Unimplemented / stubs

These pieces exist as placeholders (empty classes/functions or
`NotImplementedError`) and are not yet usable:

- **`multicellular.Colony`** (`core/colony.py`) - intended to manage a
  collection of `Cell` objects (spatial layout, neighbor interactions,
  population-level bookkeeping). Currently an empty class.
- **`multicellular.Environment`** (`core/environment.py`) - intended to model
  the shared extracellular space (e.g. diffusible chemical fields cells can
  read from / secrete into). Currently an empty class.
- **`multicellular.Simulation`** (`core/simulation.py`) - intended to
  orchestrate a `Colony` and `Environment` over time (the main simulation
  loop). Currently an empty class.
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
  as end-to-end demos of a genetic toggle switch and a quorum-sensing circuit
  once `Colony`/`Environment`/`Simulation` exist.

## Running tests

Run the full test suite from the project root:

```bash
pytest
```

Run a specific test file:

```bash
pytest tests/test_cell.py        # Cell geometry, growth, and division
pytest tests/test_reactions.py   # Reaction rate laws, stoichiometry, ODE stepping, Cell + ReactionNetwork integration
pytest tests/test_dummy.py       # trivial sanity check
```

`tests/test_colony.py` and `tests/test_rections.py` are currently empty
placeholders (no tests to run yet).
