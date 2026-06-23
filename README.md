# multicellular

An agent-based simulation framework for bacterial cell colonies. Each cell is
modeled as a rod-shaped body (cylinder with hemispherical caps) carrying an
internal chemical reaction network (e.g. a gene-regulatory circuit), and grows,
divides, diffuses, and pushes against its neighbors within a shared environment.

## Status

This package is under active development. The following are implemented and
tested:

- Individual cells with geometry, growth (with continuous dilution of
  concentrations as volume increases), division, and internal reaction
  networks
- Chemical reaction networks (mass-action, Michaelis-Menten, Hill-Langmuir,
  custom rate laws; forward-Euler ODE integration)
- Environment with spatially-varying diffusivity and viscosity fields
- Colony with overdamped-Langevin Brownian motion, Hookean cell-cell contact
  forces and torques, and optional chemical survival conditions
- Simulation loop with full history recording
- 2D animation via `visualize()`
- Running independent simulation replicates across multiple CPU cores via
  `run_replicates()`

Not yet implemented (see [Unimplemented / stubs](#unimplemented--stubs)): SBML
import, reaction-diffusion field dynamics, and several visualization utilities.

## Installation

Requires Python >= 3.11.

```bash
git clone https://github.com/Paytonco/multicellular.git
cd multicellular
pip install -e .
```

This installs the `multicellular` package (from `src/multicellular`) in
editable mode, along with its dependencies: `numpy`, `pandas`, `matplotlib`,
`tqdm`, `pillow`, and `joblib`.

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

`Cell` also exposes two methods for directly applying external motion:

- `cell.apply_force(velocity, dt)` — translates the cell by `velocity * dt`.
  Treats the argument as a velocity (effective force / drag), consistent with
  the overdamped regime. No-op if the cell is dead.
- `cell.apply_torque(omega, dt)` — rotates the orientation unit vector by
  `omega * dt` radians via a 2D rotation matrix. No-op if the cell is dead.

Both are also called internally by `Colony` when applying Brownian motion and
contact forces.

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

### Dilution by growth

Growth changes a cell's volume but not its molecule counts, so `cell.grow(dt)`
holds each species' copy number (`concentration * volume`) fixed across the
volume change: it computes every species' copy number at the pre-growth
volume, grows `length` (and so volume), then divides those copy numbers by
the new volume to get diluted concentrations. Since `cell.step(dt)` runs the
reaction network's ODE step before calling `grow`, copy number changes from
reactions and from growth-driven dilution are applied in two clearly
separated stages each timestep — reactions are the only thing that changes
copy number, growth only dilutes it:

```python
cell = Cell(id=0, position=[0.0, 0.0], orientation=[1.0, 0.0], length=2.0)
cell.set_concentration("A", 1.0)

volume_before = cell.compute_volume()
cell.grow(dt=1.0)
volume_after = cell.compute_volume()

# Copy number is conserved; concentration drops as volume grows.
assert cell.concentrations["A"] * volume_after == pytest.approx(1.0 * volume_before)
```

This is currently implemented with explicit multiply-then-divide arithmetic
since concentrations are stored as continuous values (this is what the ODE
and CLE simulation methods will need too); a future SSA implementation that
tracks copy numbers directly would not need this conversion, since dilution
would fall directly out of resampling propensities against the new volume.

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
named `Field`s — matrices of values over a shared grid (e.g. chemical
concentrations, temperature, surface roughness). Reaction-diffusion dynamics
are not implemented yet; for now `Environment` is just a validated container
for fields.

```python
import numpy as np
from multicellular import Environment, Field

glucose = Field("glucose", np.ones((10, 10)))
env = Environment(shape=(10, 10), fields=[glucose])

env.get_field("glucose")
```

Every `Field` added to an `Environment` must have a `values` matrix matching
the environment's `shape`; adding a mismatched field raises `ValueError`.

A `Field` also carries an `is_chemical` flag (default `False`). Fields marked
`is_chemical=True` are treated as diffusible chemical species: at each
timestep, `Colony` samples the field's local value at every living cell's
position and writes it into that cell's `concentrations` under the field's
`name`, overwriting any existing value for that species (see
[`Colony`](#colony) below). This sampling happens both before and after each
cell's internal step, so a chemical field's value is treated as an
externally-imposed boundary condition: the cell's own growth (see
[Dilution by growth](#dilution-by-growth) below) never dilutes it.

```python
glucose = Field("glucose", np.ones((10, 10)), is_chemical=True)
```

`Environment` also holds two spatial fields that describe the physical
properties of the medium, each a 2D numpy array of the same `shape` as the
grid:

- **`diffusivity`** (m²/s): local solute diffusivity. Defaults to a constant
  matrix filled with the diffusivity of water at 37°C (3.0 × 10⁻⁹ m²/s).
- **`eta`** (Pa·s): local dynamic viscosity. Defaults to a constant matrix
  filled with the viscosity of water at 37°C (6.9 × 10⁻⁴ Pa·s).

Both can be provided as spatially-varying arrays to model heterogeneous media.
`Colony` reads these fields at each cell's position to set the local drag for
Brownian motion (see [`Colony`](#colony) below).

```python
import numpy as np
from multicellular import Environment

# Defaults: uniform water at 37°C
env = Environment(shape=(10, 10))

# Spatially-varying viscosity (e.g. a gel region in the upper half)
eta = np.full((10, 10), 6.9e-4)
eta[:5, :] = 5e-3  # more viscous upper half
env = Environment(shape=(10, 10), eta=eta)

# Both fields can be set independently
env = Environment(shape=(10, 10), diffusivity=D, eta=eta)
```

`Environment.BOUNDS` is currently hardcoded to `(100.0, 100.0)`, representing
a 100 μm × 100 μm square that cells will interact within (this will be made
configurable later). A field's grid `shape` is independent of `BOUNDS` and
may cover an area larger than the simulation bounds. `env.in_bounds(position)`
checks whether a 2D position (e.g. a cell's center of mass) falls within
`[0, BOUNDS[0]] × [0, BOUNDS[1]]`.

### `Colony`

A `Colony` is a collection of `Cell`s living in an `Environment`:

```python
from multicellular import Cell, Colony, Environment

env = Environment(shape=(10, 10))
cells = [Cell(id=0, position=[10.0, 10.0], orientation=[1.0, 0.0])]

colony = Colony(cells, env)
colony.step(dt=0.1)
```

`Colony` accepts two mechanical parameters:

- **`k`** (default `10.0`): Hookean contact stiffness (force / length). Controls
  how hard cells push against one another when they overlap. Must be tuned
  alongside `dt`; the explicit integrator is stable when
  `dt < 2 * drag * length / k`.
- **`drag`** (default `1.0`): isotropic drag constant for contact dynamics
  (force · time / length²). Sets the translational drag `ζ_t = drag * length`
  and rotational drag `ζ_r = (drag / 12) * length³`.

It also accepts an optional **`survival_conditions`** parameter: a list of
`(species, operator, threshold)` tuples, e.g. `[("A", ">", 0)]`. Every step,
each living cell's concentration of `species` is compared against
`threshold` using `operator` — one of `">"`, `">="`, `"<"`, `"<="`, `"=="`,
`"!="` — and the cell dies as soon as any condition is violated. A species
missing from a cell's concentrations is treated as `0.0`. With multiple
conditions, a cell dies if *any* of them is violated:

```python
# Cells die once species "A" is depleted, or if "B" ever exceeds 10.
colony = Colony(cells, env, survival_conditions=[("A", ">", 0), ("B", "<=", 10)])
```

`colony.step(dt)` performs the following operations in order:

1. **Chemical fields** — for every `Field` on the environment with
   `is_chemical=True`, samples its local value at each living cell's
   position and sets that cell's concentration of the same-named species
   (no-op if there are no chemical fields).
2. **Internal step** — calls `cell.step(dt)` for every cell (chemistry +
   growth). Growth dilutes every species in `concentrations`, including
   ones just set from a chemical field in step 1.
3. **Chemical fields (re-applied)** — repeats step 1, so each chemical
   field's value overwrites whatever growth diluted it to in step 2. This
   keeps the field's value pinned to the environment between steps; reactions
   in step 2 already saw the correct, freshly-sampled value from step 1.
4. **Bounds enforcement** — kills any living cell whose center of mass lies
   outside `environment.BOUNDS`.
5. **Survival conditions** — kills any living cell whose concentrations
   violate a `survival_conditions` entry (no-op if none were given).
6. **Brownian motion** — applies an overdamped-Langevin random kick to every
   surviving living cell (see [Brownian motion](#brownian-motion) below).
7. **Contact forces** — applies pairwise Hookean repulsion and torques between
   all overlapping living cells (see [Contact forces](#contact-forces) below).
8. **Division** — replaces any cell with `cell.ready_to_divide() == True` with
   its two daughter cells, each assigned a new unique `id`.

`colony.living_cells` returns the cells that are still alive.

Dead cells (`cell.alive == False`) are inert: `Cell.step`, `Cell.apply_force`,
and `Cell.apply_torque` are all no-ops for them, and `Colony` skips both
Brownian motion and contact forces for dead cells. Dead cells also never
divide.

#### Switching environments

`colony.switch_environment(environment)` replaces the colony's `Environment`
in place, without touching its cells. This is the mechanism for simulating a
discrete event partway through a run — e.g. an inducer being added to the
medium — by swapping in a new `Environment` whose chemical `Field`s (see
[`Environment` and `Field`](#environment-and-field) above) represent the
post-induction state:

```python
import numpy as np
from multicellular import Cell, Colony, Environment, Field, Simulation

env = Environment(shape=(10, 10))
cell = Cell(id=0, position=[50.0, 50.0], orientation=[1.0, 0.0])
colony = Colony([cell], env)

sim = Simulation(colony, dt=0.1, t_max=5.0)
sim.run()  # simulate before induction

# At t=5.0, add an inducer (e.g. IPTG) uniformly across the medium.
iptg = Field("IPTG", np.full((10, 10), 1.0), is_chemical=True)
induced_env = Environment(shape=(10, 10), fields=[iptg])
colony.switch_environment(induced_env)

sim.run(t_max=10.0)  # continue simulating, with IPTG now present
```

See [`Simulation`](#simulation) below for how `sim.run()` continues from the
current time and appends to the existing history rather than restarting.

#### Brownian motion

Each timestep, `Colony` applies anisotropic Brownian displacements to every
living cell, consistent with the overdamped Langevin SDE. The local viscosity
`η` and diffusivity `D` are sampled from `environment.eta` and
`environment.diffusivity` at the cell's grid position (nearest-grid-point
lookup). The effective local viscosity is then:

```
η_local = η × (D_water_37C / D_field)
```

This Stokes-Einstein scaling means that a region with lower diffusivity is
treated as more viscous, and vice versa.

Translational and rotational drag coefficients are computed from `η_local` and
the cell's geometry using slender-body theory for a thin rod of total length
`L_eff = length + 2 × radius`:

```
γ_∥  = 2π η_local L_eff / ln(L_eff / radius)      (parallel to long axis)
γ_⊥  = 4π η_local L_eff / ln(L_eff / radius)      (perpendicular)
γ_rot = π η_local L_eff³ / (3 ln(L_eff / radius)) (rotation)
```

Diffusion coefficients follow from the Einstein relation (`D = k_BT / γ`),
and the Brownian displacements are drawn as:

```
Δr = √(2 D_∥ dt) ξ_∥ û  +  √(2 D_⊥ dt) ξ_⊥ û_⊥     (position, μm)
Δθ = √(2 D_rot dt) ξ_rot                               (orientation, rad)
```

where `û` is the cell's orientation unit vector, `û_⊥` is perpendicular to
it, and `ξ_∥`, `ξ_⊥`, `ξ_rot` are independent standard-normal draws from the
cell's own RNG. The position update is in μm and `dt` is in seconds.

#### Contact forces

Each timestep, `Colony` computes pairwise Hookean repulsion between every pair
of living cells whose surfaces overlap. The cell surface is modeled as a
spherocylinder: every point within distance `R` (the cell's `radius`) of its
axis segment. Contact between cells `i` and `j` is detected by computing the
minimum distance `d` between their axis segments; the overlap is:

```
δ = R_i + R_j − d
```

Cells are in contact when `δ > 0`. The repulsive force on cell `i` is:

```
F_ij = k · δ · N
```

where `N` is the unit contact normal pointing from `j` toward `i`, and `k` is
the stiffness set on the `Colony`. By Newton's third law, `F_ji = −F_ij`.

Because the contact point `p_c = (p_i + p_j) / 2` is generally off-center,
the force also generates a torque on each cell. In 2D the torque is the scalar
cross product of the lever arm and the force:

```
τ_i = (p_c − c_i) × F_ij   (r_x F_y − r_y F_x)
τ_j = (p_c − c_j) × F_ji
```

Forces and torques from all contacting neighbors are summed, then applied with
overdamped dynamics using the `drag` parameter:

```
Δc_i = (Σ F_ij) / ζ_t · dt,   ζ_t = drag · length
Δθ_i = (Σ τ_i)  / ζ_r · dt,   ζ_r = (drag / 12) · length³
```

When cell axes are nearly parallel the closest-point solution is non-unique;
`Colony` handles this explicitly by using the midpoint of the overlapping axial
projection rather than a degenerate endpoint, which gives the physically
correct contact location for densely-packed, aligned cells.

Candidate contact pairs are found with a uniform spatial grid (linked-cell
algorithm) rather than checking every pair of living cells: cells are binned
by center position into a grid sized to the largest possible center-to-center
overlap distance, and only the 3×3 neighborhood of bins around each cell is
searched. This produces identical force/torque results to an all-pairs check
but scales roughly linearly with colony size instead of quadratically, which
matters since colonies grow exponentially via division.

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
and records the new state of every cell remaining in the colony — including
any new daughter cells produced by division — until `t_max` is reached. The
returned DataFrame has columns `time`, `cell_id`, `alive`, `position_x`,
`position_y`, `orientation_x`, `orientation_y`, `length`, `radius`, plus one
column per chemical species seen in any cell's `concentrations` (missing
values are `NaN` for cells/species that don't have that entry). The same data
is also available via `sim.history` (a list of per-cell-per-timestep dicts)
and `sim.to_dataframe()`.

Calling `sim.run()` again continues the simulation instead of restarting it:
it picks up from the current `sim.time` and appends new records to the
existing `sim.history` rather than clearing it. Pass a new, larger `t_max` to
extend how far it runs (`sim.run(t_max=...)`), e.g. to simulate further after
calling `colony.switch_environment(...)` mid-simulation — see
[Switching environments](#switching-environments) above. Calling `run()` a
second time with the same `t_max` it already reached is a no-op.

### `run_replicates`

Stochasticity (division noise, Brownian motion, low-copy partitioning) means
a single `Simulation` run is one sample from a distribution; studying that
distribution means running many independent replicates. `run_replicates` runs
each replicate's full simulation in its own OS process (via `joblib`), using
all available CPU cores by default, and returns one combined DataFrame:

```python
from multicellular import Cell, Colony, Environment, run_replicates

def build_colony(replicate_id):
    import numpy as np
    env = Environment(shape=(10, 10))
    cell = Cell(
        id=0,
        position=[50.0, 50.0],
        orientation=[1.0, 0.0],
        rng=np.random.default_rng(replicate_id),  # distinct stream per replicate
    )
    return Colony([cell], env)

df = run_replicates(build_colony, n_replicates=50, dt=0.01, t_max=10.0, n_jobs=-1)
```

- `build_colony(replicate_id)` is called once per replicate (inside its
  worker process) and must return a fresh `Colony`. Give each cell's `rng`
  a distinct seed (e.g. derived from `replicate_id`) so replicates don't
  share — or accidentally duplicate — a random stream.
- `n_jobs` is forwarded to `joblib.Parallel`; `-1` (the default) uses every
  available core, and a smaller positive number caps how many to use.
- The returned DataFrame is the concatenation of every replicate's
  `Simulation.run()` output, with an added `replicate_id` column so rows
  from different replicates can be told apart (each replicate also numbers
  its own cells from `cell_id` 0 independently, so always group/filter by
  `replicate_id` first when comparing across replicates).
- This parallelizes *across* independent simulations only — it does not
  speed up the internal step loop of any single simulation. A single
  `Simulation.run()` call still runs single-process, exactly as before;
  parallelism is strictly opt-in via `run_replicates`.
- `show_progress` (default `False`) controls whether each individual
  replicate's own `Simulation.run()` shows its tqdm bar; left on, bars from
  different worker processes interleave and are hard to read.

### `visualize`

```python
visualize(
    simulation,
    red=None, green=None, blue=None,
    interval=200,
    save_path=None, filename="simulation.gif",
    show_progress=True,
)
```

shows a 2D animation of an already-`run()` `Simulation` in a pop-up matplotlib
window:

```python
from multicellular import visualize

visualize(sim, red="A", green="B", interval=200)
```

- Each cell is drawn as its rod shape (cylinder + hemispherical caps) using
  its recorded `position`, `orientation`, `length`, and `radius`.
- `red`/`green`/`blue` optionally name chemical species; a cell's color in
  that channel is its concentration of that species, normalized by the
  species' maximum value over the whole simulation. Channels left as `None`
  default to a constant mid-gray value.
- Dead cells stop appearing in the frame after they die.
- The region outside `environment.BOUNDS` is tinted red.
- `interval` is the delay between frames in milliseconds.

Every frame (all cell shapes, colors, and the `t = ...` title) is rendered to
an in-memory image up front, before anything is shown — drawing many
individual cell patches is the slow part of this visualization, so doing it
once per frame ahead of time (with a progress bar; set `show_progress=False`
to silence it) keeps playback smooth no matter how large the colony grows.
Display afterward is just fast image blitting through the pre-rendered
frames.

To save the animation, pass a directory via `save_path`; it's created if it
doesn't already exist, and the animation is written there as an animated GIF
(via Pillow — no external dependencies like `ffmpeg` required):

```python
visualize(sim, red="A", green="B", save_path="./out", filename="colony.gif")
```

`filename` defaults to `"simulation.gif"`. The animation is still shown
interactively afterward; pass `show_progress=False` and close the window
yourself (or run headlessly) if you only want the saved file.

## Unimplemented / stubs

These pieces exist as placeholders (empty classes/functions or
`NotImplementedError`) and are not yet usable:

- **`ReactionNetwork.simulate_step`** with `simulation_method="SSA"` or
  `"CLE"` — raises `NotImplementedError`. Only `"ODE"` (forward Euler) is
  implemented.
- **`ReactionNetwork.from_sbml`** and **`multicellular.parse_sbml`**
  (`utils/sbml_parser.py`) — raise `NotImplementedError` / are empty. Intended
  to construct a `ReactionNetwork` from an SBML model file.
- **`multicellular.animate_colony`**, **`multicellular.color_cells`**, and
  **`multicellular.plot_field`** (`utils/visualization.py`) — empty functions.
  Intended for animating a `Colony` directly, coloring cells by species or
  concentration, and plotting `Environment` fields.
- **`Cell.interact_with_environment`** — placeholder method (no-op), intended
  to let a cell read from / write to an `Environment`.
- **`examples/toggle_switch_demo.py`** and
  **`examples/quorum_sensing_demo.py`** — empty placeholder scripts, intended
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
pytest tests/test_parallel.py    # run_replicates: parallel execution, independence, correctness
pytest tests/test_dummy.py       # trivial sanity check
```
