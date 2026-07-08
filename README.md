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
  custom rate laws), simulated via forward-Euler ODE, the chemical Langevin
  equation (CLE), or Gillespie SSA
- Environment with spatially-varying diffusivity and viscosity fields, plus
  per-field diffusion (a mass-conserving finite-difference solver) for any
  `Field` marked `diffuses=True`
- Colony with overdamped-Langevin Brownian motion, Hookean cell-cell contact
  forces and torques, chemical field sensing and export (secretion), and
  optional chemical survival conditions
- Simulation loop with full history recording
- 2D animation of a colony via `Simulation.visualize_colony()`, or of a
  chemical `Field` over time via `Simulation.visualize_field()`; static
  heatmaps of one or more `Field`s at a point in time via
  `Simulation.plot_field()`
- Running independent simulation replicates across multiple CPU cores via
  `run_replicates()`

Not yet implemented (see [Unimplemented / stubs](#unimplemented--stubs)): SBML
import.

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
stoichiometry matrix, and advances concentrations with one of three
simulation methods: `"ODE"` (forward Euler; the default), `"CLE"` (chemical
Langevin equation), or `"SSA"` (Gillespie stochastic simulation algorithm).
The method is chosen per call to `simulate_step` (or, when running a full
`Colony`/`Simulation`, once via `Simulation`'s `simulation_method` argument —
see [`Simulation`](#simulation) below) rather than being fixed on the
`ReactionNetwork` itself, so the same network can be advanced with different
methods.

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
new_state = network.simulate_step(state, dt=0.1, volume=1.0)  # method="ODE" by default
```

`simulate_step`'s signature is the same for every method: a concentration
dict in, a concentration dict out (`{species: value}`, keyed by
`network.species`). `volume` is the cell's current volume (held fixed across
the call). `method` (`"ODE"`, `"CLE"`, or `"SSA"`, case-insensitive) selects
the simulation algorithm for that call. `rng` (an optional
`np.random.Generator`, used by `"CLE"` and `"SSA"`) defaults to a fresh
`np.random.default_rng()` if omitted; `Cell.step` passes the cell's own `rng`
so that one seed per `Cell` controls its entire stochastic trajectory —
division and Brownian-motion noise as well as chemistry.

```python
new_state = network.simulate_step(state, dt=0.1, volume=1.0, method="SSA", rng=rng)  # or "CLE"
```

Every rate law (`mass_action`, `michaelis_menten`, `hill_langmuir`, `custom`)
is defined in concentration space, the same as the ODE step. `"CLE"` and
`"SSA"` both bridge from that to a per-reaction propensity (molecules/time)
as `a_j = volume * v_j(concentration)`, reusing each reaction's existing
`rate()` rather than separate count-based rate laws. For an elementary
self-reaction like `2A -> B` this is a standard mean-field simplification of
the textbook combinatorial SSA propensity (`k * n*(n-1)/volume`) — the same
kind of approximation already implicit in non-elementary rate laws like
Michaelis-Menten and Hill-Langmuir.

- **`"CLE"`** stays in concentration space throughout: it takes one
  Euler-Maruyama step,
  ```
  C_new = C + dt * (S @ v(C)) + sqrt(dt / volume) * (S @ (sqrt(v(C)) * ξ))
  ```
  where `S` is the stoichiometry matrix, `v(C)` is the same rate vector the
  ODE step computes (clamped ≥ 0 before the square root), and `ξ` is one
  standard-normal draw per reaction. The `1/sqrt(volume)` scaling means
  smaller cells see proportionally larger relative noise. Like the ODE step,
  the result is clipped at zero.
- **`"SSA"`** is the only method that needs molecule counts. Internally, it
  converts the incoming concentrations to integer counts (`round(concentration
  * volume)`), runs the Gillespie direct method — repeatedly drawing an
  exponential waiting time and a reaction choice weighted by propensity,
  applying that reaction's stoichiometry, and advancing its own internal
  clock — until the elapsed time would exceed `dt` (the pending reaction is
  not fired; time simply advances to `dt` with the current state, per the
  standard partial-step convention), then converts counts back to
  concentrations before returning. This conversion is invisible to callers:
  `Cell.step` and `Cell.grow`'s dilution logic (see
  [Dilution by growth](#dilution-by-growth) above) work identically
  regardless of `simulation_method`.

#### Exporting species (secretion)

A reaction can also export species out of the cell entirely, via an
`exports` dict — the export-side counterpart of `products`: stoichiometry in
`exports` leaves the cell instead of being added back to its
`concentrations`. The recommended pattern is to export a species that's also
the reaction's sole reactant, so the cell's own pool is depleted by exactly
the amount that leaves:

```python
# X leaves the cell at rate k * [X] (mass-conserving efflux).
efflux = Reaction(
    reactants={"X": 1},
    products={},
    exports={"X": 1},
    rate_law_type="mass_action",
    rate_params={"k": 0.5},
)
```

`simulate_step`'s signature/return value (just the concentration dict) is
unchanged by this. The molecule count exported by the most recent call is
available as `network.last_exported` (a dict keyed by exported species
name), which `Cell.step` copies into `cell.pending_export` for `Colony` to
deposit into a `Field` — see
[Secretion (chemical export)](#secretion-chemical-export) below.

All three simulation methods keep the exported amount exactly consistent
with what's actually removed from the cell: for `"ODE"`/`"CLE"`, each export
reaction's per-step "firing extent" is clamped to `[0, available reactant]`
once, and *both* the intracellular delta and the exported delta are derived
from that same clamped value — clipping the two results independently
afterward instead could let CLE noise (or a forward-Euler overshoot at large
`dt`) silently create or destroy mass. For `"SSA"`, exported counts are
tracked exactly during the same firing loop that updates intracellular
counts, so they're exact by construction.

### Cell with an internal reaction network

A `Cell` can be given a `ReactionNetwork`; calling `cell.step(dt)` advances
both its internal chemistry and its growth. `cell.step` takes an optional
`method` argument (`"ODE"` by default, or `"CLE"`/`"SSA"`), forwarded to the
network's `simulate_step` — this is what `Colony.step` and `Simulation` use
to apply the simulation method chosen when the `Simulation` was created (see
[`Simulation`](#simulation) below):

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

This is implemented with explicit multiply-then-divide arithmetic, since
`Cell.concentrations` is always a continuous concentration dict regardless of
which simulation method is used. SSA tracks molecule counts internally during its own
step (see [`Reaction` and `ReactionNetwork`](#reaction-and-reactionnetwork)
below) but still hands back concentrations at the end of that step, so
`grow`'s dilution logic is identical across `"ODE"`, `"CLE"`, and `"SSA"`.

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
concentrations, temperature, surface roughness), validating that every field
added matches its grid `shape`.

Every `Environment` requires a `name` string as its first argument. The name
identifies the medium or experimental condition and is displayed on the
top-left of every animation frame produced by `Simulation.visualize_colony()`.

```python
import numpy as np
from multicellular import Environment, Field

glucose = Field("glucose", np.ones((10, 10)))
env = Environment("LB medium", shape=(10, 10), fields=[glucose])

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
[Dilution by growth](#dilution-by-growth) below) never dilutes it. Every living
cell's position is looked up in the field grid in one batched (vectorized)
operation per field, rather than one Python-level lookup per cell.

```python
glucose = Field("glucose", np.ones((10, 10)), is_chemical=True)
```

#### Field diffusion

A `Field` can also be marked `diffuses=True`, with a required per-field
`diffusivity` (m²/s — the diffusion coefficient of that specific chemical
species; see the note below on how this differs from `Environment`'s own
`diffusivity` grid):

```python
glucose = Field("glucose", np.ones((10, 10)), diffuses=True, diffusivity=3e-9)
```

Each `colony.step(dt)` calls `environment.diffuse(dt)`, which advances every
diffusive field under the 2D diffusion equation `∂C/∂t = D∇²C` using an
explicit centered-difference (FTCS) scheme with no-flux (Neumann)
boundaries, so each field's total amount is conserved rather than leaking
out at the grid edges. FTCS is only numerically stable for a sufficiently
small `dt` relative to the grid spacing and `D`; rather than require the
caller to pick a stable `dt`, `diffuse` automatically subdivides it into
however many equal sub-steps the stability bound
`dt <= dx²dy² / (2D(dx²+dy²))` requires, so it stays numerically stable
regardless of the timestep used elsewhere in the simulation.

```python
env = Environment("LB medium", shape=(10, 10), fields=[glucose])
env.diffuse(dt=0.1)  # also called automatically by Colony.step
```

`Environment` also holds two spatial fields that describe the physical
properties of the medium, each a 2D numpy array of the same `shape` as the
grid:

- **`diffusivity`** (m²/s): local solute diffusivity *of the medium itself*
  (used to scale viscosity for Brownian motion — a different quantity from a
  `Field`'s own `diffusivity`, which is the diffusion coefficient of one
  specific chemical species; see [Field diffusion](#field-diffusion) above).
  Defaults to a constant matrix filled with the diffusivity of water at 37°C
  (3.0 × 10⁻⁹ m²/s).
- **`eta`** (Pa·s): local dynamic viscosity. Defaults to a constant matrix
  filled with the viscosity of water at 37°C (6.9 × 10⁻⁴ Pa·s).

Both can be provided as spatially-varying arrays to model heterogeneous media.
`Colony` reads these fields at each cell's position to set the local drag for
Brownian motion (see [`Colony`](#colony) below).

```python
import numpy as np
from multicellular import Environment

# Defaults: uniform water at 37°C
env = Environment("M9 minimal", shape=(10, 10))

# Spatially-varying viscosity (e.g. a gel region in the upper half)
eta = np.full((10, 10), 6.9e-4)
eta[:5, :] = 5e-3  # more viscous upper half
env = Environment("gel slab", shape=(10, 10), eta=eta)

# Both fields can be set independently
env = Environment("custom medium", shape=(10, 10), diffusivity=D, eta=eta)
```

`Environment.bounds` defaults to `(100.0, 100.0)`, representing a 100 μm ×
100 μm square that cells interact within, but is configurable via the
`bounds=` constructor argument. A field's grid `shape` is independent of
`bounds` and may cover an area larger than the simulation bounds.
`env.in_bounds(position)` checks whether a 2D position (e.g. a cell's center
of mass) falls within `[0, bounds[0]] × [0, bounds[1]]`.

`Environment` also takes a `depth` (μm, default `1.0` — twice the default
`Cell` radius of 0.5 μm, i.e. sized for a single-cell-thick microfluidics
monolayer). Together with the grid spacing implied by `shape` and `bounds`,
this gives every grid cell a well-defined physical volume via the
`grid_cell_volume` property (`dx * dy * depth`, in μm³, matching
`Cell.compute_volume()`'s units) — used to convert molecule counts secreted
by cells into a field concentration change; see
[Secretion (chemical export)](#secretion-chemical-export) below.

```python
env = Environment("LB medium", shape=(10, 10), bounds=(50.0, 50.0), depth=1.0)
env.grid_cell_volume  # dx * dy * depth = 5.0 * 5.0 * 1.0 = 25.0 μm³
```

#### Secretion (chemical export)

Cells can export (secrete) a chemical species into the matching-named
`Field` by including an export reaction in their `ReactionNetwork` — see
[Exporting species (secretion)](#exporting-species-secretion) above for how
to define one. Every `colony.step(dt)`, after each cell's internal step,
`colony.export_chemical_fields()` deposits whatever each cell exported into
the `Field` sharing that species' name, at the cell's nearest grid point
(summing contributions from multiple cells that map to the same grid cell),
converting the exported molecule count to a Δconcentration via
`environment.grid_cell_volume`:

```python
import numpy as np
from multicellular import Cell, Colony, Environment, Field, ReactionNetwork
from multicellular.core.reactions import Reaction

efflux = Reaction(
    reactants={"X": 1}, products={}, exports={"X": 1},
    rate_law_type="mass_action", rate_params={"k": 0.5},
)
network = ReactionNetwork("efflux", {"R": efflux})

cell = Cell(id=0, position=[50.0, 50.0], orientation=[1.0, 0.0], network=network)
cell.set_concentration("X", 1.0)

field = Field("X", np.zeros((10, 10)))  # secretion sink
env = Environment("LB medium", shape=(10, 10), fields=[field])
colony = Colony([cell], env)

colony.step(dt=0.1)
print(cell.concentrations["X"], field.values[5, 5])  # X left the cell, appeared in the field
```

If a cell exports a species with no matching `Field` in the environment,
`export_chemical_fields` raises `ValueError` — validated for every exporting
cell before any field is mutated, so a missing field never causes a partial
deposit. The target field doesn't need `is_chemical=True` (that flag only
controls whether `Colony` *senses* the field back into cells — an orthogonal
concern from whether it can receive secretion) or `diffuses=True`, though
combining both lets secreted material spread spatially on subsequent steps.

### `Colony`

A `Colony` is a collection of `Cell`s living in an `Environment`:

```python
from multicellular import Cell, Colony, Environment

env = Environment("LB medium", shape=(10, 10))
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

`colony.step(dt)` performs the following operations in order (an optional
second argument, `method`, is forwarded to every cell's `Cell.step` — see
[Cell with an internal reaction network](#cell-with-an-internal-reaction-network)
above — and defaults to `"ODE"`, same as `Cell.step`):

1. **Diffusion** — calls `environment.diffuse(dt)`, advancing every field
   with `diffuses=True` (no-op if there are none; see
   [Field diffusion](#field-diffusion) above).
2. **Chemical fields** — for every `Field` on the environment with
   `is_chemical=True`, samples its local value at each living cell's
   position and sets that cell's concentration of the same-named species
   (no-op if there are no chemical fields).
3. **Internal step** — calls `cell.step(dt)` for every cell (chemistry +
   growth). Growth dilutes every species in `concentrations`, including
   ones just set from a chemical field in step 2.
4. **Chemical export** — calls `export_chemical_fields()`, depositing
   whatever each cell exported this step into the matching `Field` (no-op if
   no cell exported anything; see
   [Secretion (chemical export)](#secretion-chemical-export) above).
5. **Chemical fields (re-applied)** — repeats step 2, so each chemical
   field's value overwrites whatever growth diluted it to in step 3. This
   keeps the field's value pinned to the environment between steps; reactions
   in step 3 already saw the correct, freshly-sampled value from step 2.
   Running after step 4 means a cell sensing the same field it just exported
   into sees this step's freshly-deposited mass.
6. **Bounds enforcement** — kills any living cell whose center of mass lies
   outside `environment.bounds`.
7. **Survival conditions** — kills any living cell whose concentrations
   violate a `survival_conditions` entry (no-op if none were given).
8. **Brownian motion** — applies an overdamped-Langevin random kick to every
   surviving living cell (see [Brownian motion](#brownian-motion) below).
9. **Contact forces** — applies pairwise Hookean repulsion and torques between
   all overlapping living cells (see [Contact forces](#contact-forces) below).
10. **Division** — replaces any cell with `cell.ready_to_divide() == True` with
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

env = Environment("uninduced", shape=(10, 10))
cell = Cell(id=0, position=[50.0, 50.0], orientation=[1.0, 0.0])
colony = Colony([cell], env)

sim = Simulation(colony, dt=0.1, t_max=5.0)
sim.run()  # simulate before induction

# At t=5.0, add an inducer (e.g. IPTG) uniformly across the medium.
iptg = Field("IPTG", np.full((10, 10), 1.0), is_chemical=True)
induced_env = Environment("+ IPTG", shape=(10, 10), fields=[iptg])
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

This whole pipeline — field sampling, drag/diffusion coefficients, and the
displacement itself — runs as batched numpy array ops over every living cell
at once, rather than a Python loop per cell; only the final position/torque
write-back to each `Cell` is a (cheap) per-cell loop.

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
chemical concentrations at every timestep. It also owns the simulation
method used to advance every cell's `ReactionNetwork` each step — pass
`simulation_method="ODE"` (the default), `"SSA"`, or `"CLE"` when
constructing it:

```python
from multicellular import Cell, Colony, Environment, Simulation

env = Environment("LB medium", shape=(10, 10))
cell = Cell(id=0, position=[50.0, 50.0], orientation=[1.0, 0.0])
colony = Colony([cell], env)

sim = Simulation(colony, dt=0.1, t_max=10.0, simulation_method="ODE")  # or "SSA"/"CLE"
df = sim.run()  # pandas DataFrame, one row per cell per recorded timestep
```

`sim.run()` records the initial state, then repeatedly calls
`colony.step(dt, sim.simulation_method)` and records the new state of every
cell remaining in the colony — including any new daughter cells produced by
division — until `t_max` is reached. The
returned DataFrame has columns `time`, `cell_id`, `alive`, `position_x`,
`position_y`, `orientation_x`, `orientation_y`, `length`, `radius`, plus one
column per chemical species seen in any cell's `concentrations` (missing
values are `NaN` for cells/species that don't have that entry). The same data
is also available via `sim.history` (a list of per-cell-per-timestep dicts)
and `sim.to_dataframe()`.

`Simulation` also keeps a parallel record of which `Environment` was active at
each recorded timestep in `sim.env_history` — a list of `(time, environment)`
pairs, one per call to `record()`. This is used by `visualize_colony()` to
display the correct environment name on each frame, and is useful when
`colony.switch_environment(...)` is called mid-simulation.

Similarly, `sim.field_history` is a list of `(time, {field_name: values})`
pairs, one per call to `record()`, holding a *copy* of every `Field`'s
`values` grid on the active `Environment` at that timestep. A copy is
necessary because a `Field`'s `values` array is mutated in place by
diffusion and secretion between recorded steps — without copying, every
entry would end up pointing at the same, ever-changing array rather than a
snapshot of its value at that time. This is what `visualize_field()` (see
[`Simulation.visualize_field`](#simulationvisualize_field) below) animates,
and what `plot_field()` (see
[`Simulation.plot_field`](#simulationplot_field) below) draws a single
snapshot of.

Calling `sim.run()` again continues the simulation instead of restarting it:
it picks up from the current `sim.time` and appends new records to the
existing `sim.history` (`sim.env_history` and `sim.field_history` too)
rather than clearing them. Pass a new, larger `t_max` to extend how far it
runs (`sim.run(t_max=...)`), e.g. to simulate further after calling
`colony.switch_environment(...)` — see
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
    env = Environment("LB medium", shape=(10, 10))
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
- `simulation_method` (`"ODE"` by default, or `"SSA"`/`"CLE"`) is forwarded
  to every replicate's `Simulation`.
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

### `Simulation.visualize_colony`

```python
sim.visualize_colony(
    red=None, green=None, blue=None,
    field=None, field_cmap="YlOrRd", field_vmin=0.0, field_vmax=None,
    interval=200,
    save_path=None, filename="simulation.gif",
    show_progress=True,
)
```

shows a 2D animation of an already-`run()` `Simulation` in a pop-up matplotlib
window:

```python
sim.visualize_colony(red="A", green="B", interval=200)
```

- Each cell is drawn as its rod shape (cylinder + hemispherical caps) using
  its recorded `position`, `orientation`, `length`, and `radius`.
- `red`/`green`/`blue` optionally name chemical species; a cell's color in
  that channel is its concentration of that species, normalized by the
  species' maximum value over the whole simulation. Channels left as `None`
  default to a constant mid-gray value.
- Dead cells stop appearing in the frame after they die.
- The region outside `environment.bounds` is tinted red.
- Each frame shows the active environment's `name` on the top-left and the
  current time (`t = ...`) on the top-right. When `colony.switch_environment`
  is called mid-simulation, the name updates automatically at the frame where
  the switch occurred.
- `interval` is the delay between frames in milliseconds.

`field` optionally names a `Field` (e.g. `"temperature"`) to draw as a light,
semi-transparent heatmap behind the cells, with a colorbar labeled with the
field's name — handy for showing spatial structure (like the extent of a
chemical gradient) that individual cell coloring alone doesn't make clear,
especially where cells are sparse:

```python
sim.visualize_colony(red="temperature", field="temperature", field_cmap="RdYlBu_r")
```

- The heatmap's extent exactly matches `environment.bounds`, so it's never
  drawn over the red out-of-bounds tint — it replaces the white in-bounds
  backdrop instead, staying behind the cells.
- `field_cmap` is any Matplotlib colormap name.
- `field_vmin`/`field_vmax` set the heatmap's color scale, fixed across the
  whole animation. `field_vmax` defaults to the field's maximum recorded
  value; `field_vmin` defaults to `0.0`. For a field whose values don't
  start near zero (e.g. temperature), the `0.0` default can push a
  *diverging* `field_cmap` (like `"RdYlBu_r"`, blue-yellow-red) toward its
  pale midpoint for the whole plotted range — pass `field_vmin` explicitly
  (e.g. the field's actual minimum) to spread the full range across the
  colormap's endpoints.

Every frame (all cell shapes, colors, and the frame labels) is rendered to
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
sim.visualize_colony(red="A", green="B", save_path="./out", filename="colony.gif")
```

`filename` defaults to `"simulation.gif"`. The animation is still shown
interactively afterward; pass `show_progress=False` and close the window
yourself (or run headlessly) if you only want the saved file.

### `Simulation.visualize_field`

```python
sim.visualize_field(
    field_name,
    cmap="viridis",
    vmin=0.0, vmax=None,
    interval=200,
    save_path=None, filename="simulation.gif",
    show_progress=True,
)
```

shows a 2D animation of one `Field`'s values over time — e.g. watching a
chemical spread by diffusion, or an `Environment.diffuse()`-driven secretion
field build up as a colony grows:

```python
sim.visualize_field("AHL", cmap="YlOrRd", interval=100)
```

- Works the same way as `visualize_colony`: every frame (a heatmap of the
  field's grid at that timestep, plus the environment `name`/time labels) is
  rendered to an in-memory image up front, from `sim.field_history` (see
  [`Simulation`](#simulation) above), so playback is just fast image
  blitting regardless of how many timesteps were recorded.
- `field_name` must be present in `sim.field_history`; a `Field` that never
  existed in any recorded `Environment` raises `KeyError`.
- `cmap` is any Matplotlib colormap name.
- `vmin`/`vmax` set the color scale. `vmax` defaults to the field's maximum
  recorded value over the whole simulation (`vmin` defaults to `0.0`, e.g.
  a zero-floor for chemical concentrations); pass both explicitly to keep
  the scale fixed when comparing separate animations.
- Each frame shows the active environment's `name` on the top-left and the
  current time (`t = ...`) on the top-right, exactly as in
  `visualize_colony`.
- `interval`, `save_path`, `filename`, `show_progress`, and `stride` all
  behave identically to `visualize_colony` (see above).

```python
sim.visualize_field("dye", save_path="./out", filename="dye.gif")
```

### `Simulation.plot_field`

```python
sim.plot_field(
    field_names,
    time=None,
    cmap="viridis",
    vmin=None, vmax=None,
)
```

plots one or more `Field`s as static heatmaps at a single point in time —
e.g. a spatially-varying medium property, or one frame of a diffusing
chemical picked out for closer inspection:

```python
sim.plot_field("temperature", cmap="RdYlBu_r", vmin=0, vmax=45)
sim.plot_field(["eta", "diffusivity"])  # several fields, side by side
```

- Uses the same imshow/colorbar/title convention as `visualize_field` (env
  `name` top-left, `t = ...` top-right, a colorbar labeled with the field's
  name) but renders one static figure instead of an animation.
- `field_names` can be a single name or a list of names; each is drawn as
  its own panel in one figure — handy for fields on very different scales
  (e.g. viscosity and diffusivity), which wouldn't share a sensible color
  scale.
- `time` snaps to the closest timestep actually recorded in
  `sim.field_history`; it defaults to the most recently recorded time (the
  final state).
- `vmin`/`vmax` default to `None`, so each panel auto-scales to its own
  field's value range at the plotted time (plain `imshow` behavior) —
  unlike `visualize_field`, which fixes the scale across an entire
  animation. Pass both explicitly for a shared/fixed scale, e.g. to keep
  two separate `plot_field` calls comparable.
- Returns the `matplotlib.figure.Figure`, so callers can add further
  annotations (e.g. `fig.axes[0].axvline(...)`) before displaying it.

## Unimplemented / stubs

These pieces exist as placeholders (empty classes/functions or
`NotImplementedError`) and are not yet usable:

- **`ReactionNetwork.from_sbml`** and **`multicellular.parse_sbml`**
  (`utils/sbml_parser.py`) — raise `NotImplementedError` / are empty. Intended
  to construct a `ReactionNetwork` from an SBML model file.

## Running tests

Run the full test suite from the project root:

```bash
pytest
```

Run a specific test file:

```bash
pytest tests/test_cell.py        # Cell geometry, growth, division, and pending chemical export
pytest tests/test_reactions.py   # Reaction rate laws, stoichiometry, ODE/SSA/CLE stepping, export-reaction mass conservation
pytest tests/test_environment.py # Environment/Field construction, validation, diffusion, and grid cell volume
pytest tests/test_colony.py      # Colony bounds enforcement, division, dead-cell behavior, field sensing/diffusion/export
pytest tests/test_simulation.py  # Simulation loop and DataFrame export
pytest tests/test_visualization.py # visualize_colony/visualize_field/plot_field: animation output, GIF export, stride, static plots
pytest tests/test_parallel.py    # run_replicates: parallel execution, independence, correctness
pytest tests/test_dummy.py       # trivial sanity check
```
