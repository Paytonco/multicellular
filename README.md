# multicellular

An agent-based simulation framework for bacterial cell colonies. Each cell is
modeled as a rod-shaped body (cylinder with hemispherical caps), or as a
sphere for coccoid cells (see [Spherical cells](#spherical-cells)), carrying an
internal chemical reaction network (e.g. a gene-regulatory circuit), and grows,
divides, diffuses, and pushes against its neighbors within a shared environment.

## Status

This package is under active development. The following are implemented and
tested:

- Individual cells with geometry, growth (with continuous dilution of
  concentrations as volume increases), division, and internal reaction
  networks, as rods (bacilli) or spheres (cocci), which can be mixed in one
  colony
- Anchored (fixed) cells that keep their chemistry but never grow, divide, or
  move, e.g. eukaryotic host cells in a tissue colonized by bacteria
- Chemical reaction networks (mass-action, Michaelis-Menten, Hill-Langmuir,
  custom rate laws), simulated via ODE (`scipy.integrate.solve_ivp`, with
  explicit or stiff solvers), the chemical Langevin equation (CLE), or
  stochastic simulation (Gillespie's direct method, Gibson-Bruck's Next
  Reaction Method, or adaptive tau-leaping), either inside cells or on their own via
  `ReactionNetwork.simulate()`
- Environment with spatially-varying diffusivity and viscosity fields, plus
  per-field diffusion (a mass-conserving finite-difference solver) for any
  `Field` marked `diffuses=True`
- Reaction-diffusion: an `Environment`'s `Field`s can be coupled to each
  other by a `ReactionNetwork`, stepped forward independently at every
  non-wall grid cell (ODE, CLE, or SSA, same as inside a `Cell`)
- Colony with overdamped-Langevin Brownian motion, Hookean cell-cell and
  cell-wall contact forces and torques, mass-conserving exchange between
  cells and chemical `Field`s (sensing, uptake, and membrane-bound
  extracellular chemistry all draw on the real local pool), export
  (secretion), optional chemical survival conditions, and lysis-based
  delivery (a bulk one-time intracellular-to-`Field` deposit when a cell
  dies from a violated survival condition)
- Trap walls: an `Environment`'s `wall_map` marks any grid cell as media,
  wall, or out-of-bounds, and `Colony` pushes cells off wall geometry with
  the same Hookean contact model used between cells
- Simulation loop with full history recording
- 2D animation of a colony via `Simulation.visualize_colony()`, or of a
  chemical `Field` over time via `Simulation.visualize_field()`, saved as an
  animated GIF or an MP4 video; static heatmaps of one or more `Field`s at a
  point in time via `Simulation.plot_field()`
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
editable mode, along with its dependencies: `numpy`, `scipy`, `pandas`,
`matplotlib`, `tqdm`, `pillow`, `joblib`, `imageio`, and `imageio-ffmpeg`.

The last two are what let animations be saved as MP4 video; `imageio-ffmpeg`
bundles its own copy of `ffmpeg`, so there is nothing to install separately.

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

print(cell.compute_volume())  # μm³

# Grow over time until the cell reaches its division target
dt = 0.1
while not cell.ready_to_divide():
    cell.grow(dt)

daughter1, daughter2 = cell.divide()
```

#### Growth and division (adder model)

A cell grows exponentially: a rod's length evolves as
`L(t) = L_b * exp(growth_rate * t)`. It divides according to a stochastic
**adder** model. At birth, each cell samples a division target

```
S_d = a * S_b + Δ,    Δ ~ Normal(delta_bar, cv_delta * delta_bar)   (resampled until Δ > 0)
```

where `S_b` is its size at birth (`cell.size_at_birth`). The size is length
for rods and volume for spheres; see [Spherical cells](#spherical-cells).
`cell.ready_to_divide()` returns `True` once the size reaches `S_d`. At that
point `cell.divide()` returns two daughters with sizes `f * S_d` and
`(1 - f) * S_d`, placed so that their surfaces just touch. If the cell isn't
ready yet, `divide()` returns `None`. Each daughter samples its own target,
so the lineage keeps adding about `delta_bar` per generation.

The growth and division constructor arguments are:

| Argument | Default | Meaning |
|---|---|---|
| `growth_rate` | `ln 2` | Exponential growth rate λ (a mean doubling time of 1 time unit). |
| `growth_rate_law` | `None` | Optional `f(concentrations, extracellular) -> growth_rate`, called by `Colony` every step with the cell's concentrations and the local value of every `Field` (e.g. temperature- or nutrient-dependent growth; see `examples/ratkowsky_temperature_demo.ipynb`). |
| `delta_bar` | `1.0` μm (rods); initial volume (spheres) | Mean added size per generation. At steady state, mean birth size is `delta_bar` and mean division size is `2 * delta_bar`. |
| `cv_delta` | `0.10` | Coefficient of variation of the added increment Δ, which is the main source of division noise. |
| `a` | `1.0` | Size-control strategy: `1` is a pure adder, `0` a sizer, and `2` roughly a timer. |
| `cv_f` | `0.0` | CV of the division fraction `f ~ Normal(0.5, cv_f * 0.5)`. The default `0.0` gives a symmetric split; about `0.03` adds partition noise. |
| `rng` | fresh `default_rng()` | `np.random.Generator` behind all of this cell's randomness: division, Brownian motion, and SSA/CLE chemistry. Daughters share it. |
| `shape` | `"rod"` | `"rod"` or `"sphere"`; see [Spherical cells](#spherical-cells). |
| `lysis_targets` | `None` | See [Lysis-based delivery](#lysis-based-delivery). |
| `anchored` | `False` | Fix the cell in place; see [Anchored cells](#anchored-cells). |

`cell.step(dt)` runs the cell's reaction network (if it has one) and then
calls `grow(dt)`. `cell.to_dict()` returns a snapshot of the cell's state.

`Cell` also exposes two methods for directly applying external motion:

- `cell.apply_force(velocity, dt)` — translates the cell by `velocity * dt`.
  Treats the argument as a velocity (effective force / drag), consistent with
  the overdamped regime. No-op if the cell is dead or anchored.
- `cell.apply_torque(omega, dt)` — rotates the orientation unit vector by
  `omega * dt` radians via a 2D rotation matrix. No-op if the cell is dead or
  anchored.

Both are also called internally by `Colony` when applying Brownian motion and
contact forces.

#### Spherical cells

Pass `shape="sphere"` for a coccoid cell (a circle in the 2D plane):

```python
coccus = Cell(id=2, position=[5.0, 0.0], orientation=[1.0, 0.0],
              radius=0.5, shape="sphere")
```

A sphere is the zero-length case of the rod geometry (`length` is fixed at
0), so volume, contact detection, and rendering all use the same code as rods,
and rods and spheres can be mixed in one colony. What differs:

- **Growth**: the radius grows as `r *= exp(growth_rate * dt / 3)`, so volume
  grows at `growth_rate`, the same as for rods.
- **Division (adder)**: the size variable is volume (μm³), not length.
  `delta_bar` defaults to the initial volume `(4/3)πr³`. A cell splits into
  two spheres with volumes `f·V_d` and `(1−f)·V_d`, placed along its
  orientation so that they touch.
- **Mechanics**: the contact drag is `ζ_t = 3 * drag * radius`, and contacts
  never rotate a sphere, because every contact force passes through its
  center. Brownian motion uses Stokes drag (`6πηr`, and `8πηr³` for
  rotation). That rotational diffusion is what randomizes the axis the
  sphere divides along.

See section 14 of `examples/feature_tour.ipynb` for a worked example: a
single coccus growing and dividing, and a mixed rod/coccus colony.

#### Anchored cells

Pass `anchored=True` for a cell that is fixed in place, such as a eukaryotic
cell in a tissue being colonized by bacteria:

```python
host = Cell(id=0, position=[10.0, 6.0], orientation=[1.0, 0.0],
            radius=6.0, shape="sphere", anchored=True, network=host_network)
```

An anchored cell's body is frozen, but its biology is not:

- **Frozen**: it never grows (`step` only advances `age`, so its size and
  concentrations are not diluted), never divides, and never moves. `Colony`
  gives it no Brownian motion, no contact or wall forces, and does not call its
  `growth_rate_law`.
- **Still active**: its reaction network runs every step, it exchanges mass
  with chemical `Field`s (sensing, uptake, secretion) like any other cell, and
  it can die from a survival condition and deliver its `lysis_targets`.
- **Contacts**: a mobile cell overlapping an anchored cell receives the full
  Hookean contact force `k·δ` (and the matching torque), so an anchored cell
  behaves like a stiff cell that cannot move. Overlapping anchored cells
  ignore each other, so tissue cells can be packed or overlap freely.
  Anchored cells are kept out of the neighbor-search grid, so a few large
  host cells don't slow contact detection for a colony of small bacteria.
- **Chemistry is coupled at one voxel**: like every cell, an anchored cell
  exchanges mass only with the `Field` voxel containing its center, even if
  its body covers several voxels.

`anchored` is a plain attribute, so a cell can be pinned or released partway
through a simulation (`cell.anchored = False` resumes growth from its current
size). Daughters of a released cell are mobile. `Simulation` records an
`anchored` column, and `visualize_colony` draws anchored cells with a heavier
slate-blue outline.

See section 18 of `examples/feature_tour.ipynb` for a worked example: bacteria
colonizing a row of anchored host cells, killing the nearest hosts through a
secreted AHL signal.

### `Reaction` and `ReactionNetwork`

`Reaction` supports `mass_action`, `michaelis_menten`, `hill_langmuir`, and
`custom` rate laws. `ReactionNetwork` collects reactions, builds a
stoichiometry matrix, and advances concentrations with one of three
simulation methods: `"ODE"` (numerical integration via `scipy.integrate.solve_ivp`; the default, with `RK45` as the default solver), `"CLE"` (chemical
Langevin equation), or `"SSA"` (Gillespie stochastic simulation algorithm).
The method is chosen per call to `simulate_step` or `simulate` (or, when running a full
`Colony`/`Simulation`, once via `Simulation`'s `simulation_method` argument —
see [`Simulation`](#simulation) below) rather than being fixed on the
`ReactionNetwork` itself, so the same network can be advanced with different
methods.

Each rate law reads its parameters from `rate_params`:

| `rate_law_type` | Rate | Parameters |
|---|---|---|
| `"mass_action"` (default) | `k * Π [reactant]^stoich` | `k` (default `1.0`) |
| `"michaelis_menten"` | `Vmax * [E] * [S] / (Km + [S])`, where `S` is the first reactant and `E` the first catalyst (`[E] = 1` if there are no catalysts) | `Vmax`, `Km` (default `1.0` each) |
| `"hill_langmuir"` | `beta * [E] * (1 + alpha * C [z]^n) / (1 + C [z]^n)`, where `E` is `catalysts[0]` (e.g. RNAP) and `z` is `catalysts[1]` (the activator or inducer) | `alpha`, `beta`, `C`, `n` (all required) |
| `"custom"` | `custom_rate_law(reactant_conc, product_conc, catalyst_conc, rate_params)` | anything your function reads |

Catalysts appear in the rate but are neither consumed nor produced. For
repression, use a `custom` rate law (e.g. `beta / (1 + ([R]/K)^n)`).

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
the simulation algorithm for that call. `ode_method` picks the
`solve_ivp` integrator when `method="ODE"`. It is one of `"RK45"` (default),
`"RK23"`, `"DOP853"`, or a stiff solver: `"Radau"`, `"BDF"`, or `"LSODA"`.
Use a stiff solver for networks with widely separated time scales, e.g.
fast binding next to slow expression, where explicit solvers crawl. The
integration uses `rtol=1e-6` and `atol=1e-9`, and the result is clipped at
zero. `rng` (an optional
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
  regardless of `simulation_method`. A firing that would drive any count
  negative (possible only with custom rate laws, or with a fractional
  extracellular pool; see below) is rejected rather than clipped, so SSA
  never creates mass.

#### Choosing an SSA algorithm (`ssa_method`)

When `method="SSA"`, `ssa_method` picks the stochastic algorithm, the same
way `ode_method` picks the integrator for `"ODE"`. All three share the
count conversion, compartment handling, export tracking, and
reject-instead-of-clip rule described above.

| `ssa_method` | Exact? | Use it for |
|---|---|---|
| `"direct"` (default) | yes | Small networks: Gillespie's direct method, as above. |
| `"next_reaction"` | yes | Networks with many reactions, each touching few species. Gibson & Bruck's Next Reaction Method keeps each reaction's next firing time in an indexed priority queue. It uses a dependency graph to recompute only the propensities a firing affects, and rescales their unused waiting times instead of redrawing them. |
| `"tau_leaping"` | no | High copy numbers, where exact methods spend most of their time firing reactions one by one. This is Cao, Gillespie & Petzold's (2006) adaptive tau-leaping. Each leap fires a Poisson number of every reaction, with the leap size chosen so no propensity changes by more than about `tau_epsilon` (default `0.03`). |

Tau-leaping details:
- Reactions within 10 firings of exhausting a reactant are fired one at a
  time rather than leaped over.
- When a leap would be shorter than about 10 expected firings, the method
  falls back to exact direct-method firings, so at low copy numbers it
  behaves like `"direct"`.
- A leap that would drive a count negative is retried with half the size.
- Each species' leap bound uses the sensitivity of the rate laws that read
  it: the reaction order for `mass_action`, `1 + n` for `hill_langmuir`, and
  `1`–`2` for `michaelis_menten`. For `custom` rate laws, which can't be
  inspected, it assumes a conservative `3`. Lower `tau_epsilon` if a custom
  law is steeper than that.
- The drift bound uses each species' gross flux rather than its net flux.
  This keeps leaps short near equilibria, where explicit leaping would
  otherwise overshoot and inflate the variance.
- Smaller `tau_epsilon` is more accurate and slower.

Putative firing times (NRM) are not carried between steps. Growth, division,
diffusion and field exchange change the state between steps, and because
waiting times are memoryless, redrawing them at each step is exact.

```python
new_state = network.simulate_step(
    state, dt=0.1, volume=100.0, method="SSA", rng=rng, ssa_method="tau_leaping"
)
```

#### Simulating a network on its own (`simulate`)

To run a network over time without building a `Cell`, `Colony`, or
`Simulation`, use `simulate`. It calls `simulate_step` repeatedly from `t=0`
to `t_max` and returns a pandas DataFrame with a `time` column and one
column per species (in `network.species` order), including the initial
state:

```python
df = network.simulate({"A": 1.0}, t_max=10.0, dt=0.1)  # method="ODE" by default
df.plot(x="time")

df = network.simulate({"A": 1.0}, t_max=10.0, dt=0.1, volume=100.0, method="SSA", seed=0)
```

- Species missing from the initial state start at 0. A name that isn't a
  network species raises `ValueError`.
- The run takes `round(t_max / dt)` steps, and row `k` is at `time = k * dt`.
- `method`, `ode_method`, `ssa_method`, `tau_epsilon`, `volume`, and `compartment_volumes` mean the same
  as in `simulate_step`. `volume` only matters for `"SSA"`/`"CLE"`, where it
  sets the molecule count per unit concentration (and so the noise level).
- One random generator is used for the whole run: pass `seed` for a
  reproducible `"SSA"`/`"CLE"` trajectory, or your own `rng`.
- If the network exports species (see
  [Exporting species](#exporting-species-secretion) below), each gets a
  cumulative `exported_<species>` column, in molecule counts, so
  `[X] * volume + exported_X` is conserved for a pure efflux reaction.
- `show_progress=True` shows a tqdm progress bar.

#### Species in another compartment (`compartment_volumes`)

`simulate_step` takes an optional `compartment_volumes` dict mapping a
species to the volume of the compartment it lives in, for species that
don't live in `volume`, such as an extracellular species living in an
environment voxel. Rate laws are unchanged: they're still evaluated
against concentrations and scaled by `volume`, so reaction `j` moves
`v_j * volume` molecules per unit time. Each species' concentration,
though, changes by those molecules divided by *its own* compartment's
volume:

```
d[intracellular]/dt = S @ v
d[extracellular]/dt = (volume / V_compartment) * (S @ v)
```

Molecule counts are therefore conserved across compartments. Rates also
respond to depletion within the step (e.g. a transporter slows as it
drains its voxel). Per method:

- **`"ODE"`** integrates the weighted system directly.
- **`"SSA"`** counts each species in its own compartment's volume. It
  doesn't round other-compartment counts to whole molecules, so the
  owning pool's fractional amount is never created or destroyed by
  rounding; each firing still moves whole molecules.
- **`"CLE"`** clamps each cross-compartment reaction's extent so that it
  can't drive any species it touches negative. Clipping afterwards would
  create mass in the other pool.

```python
uptake = Reaction({"A_ext": 1}, {"A_int": 1}, rate_law_type="mass_action", rate_params={"k": 1.0})
network = ReactionNetwork("uptake", {"R": uptake})
new_state = network.simulate_step(
    {"A_ext": 2.0, "A_int": 0.0}, dt=0.1, volume=2.0,
    compartment_volumes={"A_ext": 20.0},   # A_ext lives in a 20 um^3 voxel
)
```

You rarely call this directly. `Colony` passes it automatically for every
network species backed by a chemical `Field`; see
[Extracellular species](#extracellular-species-uptake-and-membrane-bound-enzymes)
below.

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
with what's actually removed from the cell:

- **`"ODE"`** integrates the cumulative exported amount alongside the
  intracellular state. Each export reaction's rate is clamped to be `≥ 0`,
  and to `0` once any of its reactants is depleted.
- **`"CLE"`** clamps each export reaction's per-step firing extent to
  `[0, available reactant]` once. Both the intracellular change and the
  exported amount are then derived from that single clamped value.
  Clipping the two results independently could let noise create or
  destroy mass.
- **`"SSA"`** tracks exported counts in the same firing loop that updates
  intracellular counts, so they're exact by construction.

If the destination is a chemical (`is_chemical=True`) `Field`, an ordinary
reaction producing the field-backed species (e.g. `X -> X_ext`, with a
`Field` named `X_ext`) is equivalent to an export; see
[Extracellular species](#extracellular-species-uptake-and-membrane-bound-enzymes)
below. Exports remain the way to deposit into a `Field` that isn't
chemical.

### Cell with an internal reaction network

A `Cell` can be given a `ReactionNetwork`; calling `cell.step(dt)` advances
both its internal chemistry and its growth. `cell.step` takes optional
`method` (`"ODE"` by default, or `"CLE"`/`"SSA"`), `ode_method`
(default `"RK45"`), `ssa_method` (default `"direct"`), and `tau_epsilon`
(default `0.03`) arguments, which it forwards to the network's
`simulate_step`. `Colony.step` and `Simulation` use these to apply the
simulation method chosen when the `Simulation` was created (see
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
volume, grows the cell (and so its volume), then divides those copy numbers by
the new volume to get diluted concentrations. A rod grows its `length` and a
sphere grows its `radius`. Since `cell.step(dt)` runs the
reaction network's step before calling `grow`, copy number changes from
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
conserved across division. Each daughter inherits the parent's
concentration, which conserves total copy number because the daughters'
volumes split the parent's.

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

Every `Environment` requires a `name` string as its first argument, and a
`wall_map` — a 2D matrix of `-1`/`0`/`1` entries (see
[Walls and out-of-bounds](#walls-and-out-of-bounds-wall_map) below) whose
shape becomes the environment's grid `shape`, shared by every `Field`. The
name identifies the medium or experimental condition and is displayed on the
top-left of every animation frame produced by `Simulation.visualize_colony()`.

```python
import numpy as np
from multicellular import Environment, Field

glucose = Field("glucose", np.ones((10, 10)))
env = Environment("LB medium", wall_map=np.zeros((10, 10)), fields=[glucose])

env.get_field("glucose")
```

Every `Field` added to an `Environment` must have a `values` matrix matching
the environment's `shape`; adding a mismatched field raises `ValueError`.

A `Field` also carries an `is_chemical` flag (default `False`). A field
marked `is_chemical=True` holds an extracellular chemical species: any
species in a cell's `ReactionNetwork` that shares the field's `name` *is*
that extracellular species, and it exchanges real mass with the field (see
[Extracellular species](#extracellular-species-uptake-and-membrane-bound-enzymes)
below). After every step, each living cell's local value of every chemical
field is also recorded in `cell.extracellular` (a read-only snapshot,
keyed by field name). `Simulation` records it into the history DataFrame
under the field's name, and survival conditions can test it.

```python
glucose = Field("glucose", np.ones((10, 10)), is_chemical=True)
```

#### Extracellular species: uptake and membrane-bound enzymes

Intracellular and extracellular forms of a molecule are distinct species
with distinct names: `glucose` (backed by a chemical `Field`) versus
`glucose_int` (an ordinary intracellular species). Moving between them, or
reacting outside the cell, is just ordinary reactions. There's no special
"import" mechanism. Each step, for every living cell, `Colony`:

1. reads each field-backed species' concentration at the voxel containing
   the cell's center;
2. steps the cell's network with those species living in the voxel
   (`compartment_volumes={species: environment.grid_cell_volume}`; see
   [Species in another compartment](#species-in-another-compartment-compartment_volumes)
   above);
3. writes the voxel's updated concentrations straight back into the fields.

So any reaction that consumes or produces a field-backed species draws
from or adds to the local pool, and molecules are conserved between cells
and medium. A reaction moving `n` molecules changes the cell's own species
by `n / V_cell`, and the voxel's by `n / V_voxel`.

```python
import numpy as np
from multicellular import Cell, Colony, Environment, Field, ReactionNetwork
from multicellular.core.reactions import Reaction

# Transporter: extracellular glucose -> intracellular glucose_int.
transport = Reaction({"glucose": 1}, {"glucose_int": 1}, catalysts=["PtsG"],
                     rate_law_type="michaelis_menten", rate_params={"Vmax": 4.0, "Km": 0.5})
# Membrane-bound enzyme acting outside the cell: glucose -> gluconate, both
# extracellular, catalysed by the cell's own (intracellular) Gdh.
oxidation = Reaction({"glucose": 1}, {"gluconate": 1}, catalysts=["Gdh"],
                     rate_law_type="michaelis_menten", rate_params={"Vmax": 2.0, "Km": 0.5})

cell = Cell(id=0, position=[12.0, 12.0], orientation=[1.0, 0.0],
            network=ReactionNetwork("glucose", {"transport": transport, "oxidation": oxidation}))
cell.set_concentration("PtsG", 1.0)
cell.set_concentration("Gdh", 1.0)

env = Environment("closed chamber", wall_map=np.zeros((12, 12)), size=(24.0, 24.0), fields=[
    Field("glucose", np.ones((12, 12)), is_chemical=True, diffuses=True, diffusivity=1e-12),
    Field("gluconate", np.zeros((12, 12)), is_chemical=True, diffuses=True, diffusivity=1e-12),
])
colony = Colony([cell], env)
for _ in range(100):
    colony.step(dt=0.1)
# glucose field + gluconate field + glucose_int in the cell is conserved exactly.
```

Details:

- **Catalyst-only sensing is free.** A species that only ever appears as a
  catalyst (an inducer like IPTG, or a constant `RNAP` field) is read but
  never consumed, so sensing it doesn't change the field.
- **Not part of the cell.** Field-backed species are kept out of
  `cell.concentrations` (the zero entries a new cell seeds for every
  network species are dropped on its first step), so growth never dilutes
  them and division never partitions them. A species is field-backed only
  while the current environment has a chemical `Field` of that name. After
  `switch_environment` to an environment without it, the name is an
  ordinary intracellular species again.
- **Shared voxels.** Coupled cells are stepped one at a time, in an order
  shuffled by `environment.rng` each step, and each reads its voxel after
  the previous cells' writes. Cells sharing a voxel therefore share one
  pool and can't jointly overdraw it, and no cell gets systematic first
  access.
- **Voxel volume matters.** `environment.grid_cell_volume` (`dx * dy *
  depth`) is the extracellular compartment volume. A voxel smaller than
  the cell drains (and refills) faster than the cell's own species
  change; one much larger barely notices a single cell.
  `examples/danino_oscillator_demo.ipynb` uses this deliberately, sizing
  each voxel to one cell's share of free medium so that
  `V_cell / V_voxel = d/(1-d)`, the density factor in the Danino et al.
  model.
- **Don't export into a field-backed reactant's own field.** An export
  reaction whose reactant `X` is itself field-backed (because a chemical
  `Field` named `X` exists) consumes extracellular `X` and deposits it right
  back. Use a distinct intracellular name (`X_int`) for the cell's pool.

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

Diffusion also respects `wall_map` (see
[Walls and out-of-bounds](#walls-and-out-of-bounds-wall_map) below): a wall
(`1`) cell is a no-flux boundary internal to the grid, exactly like the
domain edge already is — flux across a media/wall face is zeroed instead of
letting mass flow into a cell that has no medium in it, and wall cells are
excluded from the update entirely, so they never accumulate mass. This keeps
the discrete flux exactly antisymmetric between every interacting pair of
cells, so total mass stays conserved even with walls present; mass that
would have crossed a wall instead piles up against it (or routes around it,
if the wall doesn't fully enclose that region). Out-of-bounds (`-1`) cells
are unaffected and remain ordinary diffusive medium — only walls block
diffusion.

```python
env = Environment("LB medium", wall_map=np.zeros((10, 10)), fields=[glucose])
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
env = Environment("M9 minimal", wall_map=np.zeros((10, 10)))

# Spatially-varying viscosity (e.g. a gel region in the upper half)
eta = np.full((10, 10), 6.9e-4)
eta[:5, :] = 5e-3  # more viscous upper half
env = Environment("gel slab", wall_map=np.zeros((10, 10)), eta=eta)

# Both fields can be set independently
env = Environment("custom medium", wall_map=np.zeros((10, 10)), diffusivity=D, eta=eta)
```

`Environment.size` defaults to `(100.0, 100.0)`, representing a 100 μm ×
100 μm square that cells interact within, but is configurable via the
`size=` constructor argument. A field's grid `shape` (taken from `wall_map`)
is independent of `size` and may cover an area larger than the simulation
extent. `env.in_bounds(position)` checks whether a 2D position (e.g. a
cell's center of mass) falls within `[0, size[0]] × [0, size[1]]` *and* does
not land on a `wall_map` out-of-bounds (`-1`) cell (see next section).

`Environment` also takes an `rng` (`np.random.Generator`, default a fresh
`default_rng()`). It drives `react()`'s SSA/CLE noise and the order in which
`Colony` steps cells that share chemical fields.

`Environment` also takes a `depth` (μm, default `1.0` — twice the default
`Cell` radius of 0.5 μm, i.e. sized for a single-cell-thick microfluidics
monolayer). Together with the grid spacing implied by `wall_map`'s shape and
`size`, this gives every grid cell a well-defined physical volume via the
`grid_cell_volume` property (`dx * dy * depth`, in μm³, matching
`Cell.compute_volume()`'s units) — used to convert molecule counts secreted
by cells into a field concentration change; see
[Secretion (chemical export)](#secretion-chemical-export) below.

```python
env = Environment("LB medium", wall_map=np.zeros((10, 10)), size=(50.0, 50.0), depth=1.0)
env.grid_cell_volume  # dx * dy * depth = 5.0 * 5.0 * 1.0 = 25.0 μm³
```

#### Field reactions (reaction-diffusion)

An `Environment` can also carry a `reactions` `ReactionNetwork` that couples
its `Field`s to each other, giving `∂C/∂t = D∇²C + R(C)` at every grid cell
(voxel). This reuses the exact same `ReactionNetwork.simulate_step`
machinery `Cell.step` uses internally (`"ODE"`, `"SSA"`, or `"CLE"`) — the
only new code is stepping it independently at every non-wall voxel, with
each species' value read from and written back to the matching-named
`Field`. With `"ODE"`, every voxel is integrated in one batched `solve_ivp`
call (`ReactionNetwork.simulate_ode_batch`), which is much faster than one
call per voxel. A `custom` rate law that only accepts scalars makes `react()`
fall back to stepping voxels one at a time, as `"SSA"` and `"CLE"` always do.

```python
import numpy as np
from multicellular import Environment, Field, ReactionNetwork
from multicellular.core.reactions import Reaction

decay = Reaction(reactants={"A": 1}, products={}, rate_params={"k": 0.1})
network = ReactionNetwork("decay", {"decay": decay})

field = Field("A", np.full((10, 10), 5.0), diffuses=True, diffusivity=1e-9)
env = Environment("env", wall_map=np.zeros((10, 10)), fields=[field], reactions=network)

env.react(dt=1.0)  # also called automatically by Colony.step, right after diffuse()
```

Every species in `reactions.species` must have a matching `Field` in the
environment — `react()` raises `ValueError` otherwise, checked before any
field is mutated. A `reactions` network with any `exports` is rejected at
construction: exports are `Cell`→`Field` secretion semantics (see
[Secretion (chemical export)](#secretion-chemical-export) below), and don't
apply to a Field-local reaction (there's no "outside" for a `Field` to
export to).

Just like `Cell.step`, `SSA`'s (and `CLE`'s) concentration↔copy-number
bridge is handled by the same `volume` argument `simulate_step` already
takes — here it's `environment.grid_cell_volume` (a voxel's physical
volume) rather than a cell's. No separate conversion logic is needed.
`react()` excludes wall (`wall_map == 1`) cells from the update, the same as
`diffuse()`: there's no medium there to react in.

`Colony.step(dt, method, ode_method)` calls `environment.react(dt, method, ode_method)` right after
`environment.diffuse(dt)` (see [`Colony`](#colony) below), so reaction-diffusion
runs every step even with an empty `Colony` (no cells at all) — see
`examples/feature_tour.ipynb`'s Turing pattern demo (section 9).

Each step diffuses every field over `dt` and then reacts over `dt`
(operator splitting), and the splitting error grows with `dt`. Pattern-forming
systems need `dt` well below their fastest reaction time scale. For example,
the Schnakenberg demo, whose kinetics run on a time scale of about 1, matches
a fully coupled solution at `dt = 0.02`, but at `dt = 0.5` it forms a
different, lower-amplitude pattern.

#### Walls and out-of-bounds (`wall_map`)

`wall_map` is a matrix of `-1`, `0`, and `1` entries, scaled to fit `size`,
that gives the environment its terrain:

- **`0` — media.** Cells move and react freely.
- **`1` — wall.** Cells feel a Hookean repulsion from wall geometry, the same
  contact model used between cells (see
  [Wall forces](#wall-forces) below) — they are pushed off rather than
  killed.
- **`-1` — out of bounds.** A cell whose center of mass lands here is killed
  by `enforce_bounds` (see [`Colony`](#colony) below), the same as leaving
  `size`'s physical extent entirely.

```python
import numpy as np
from multicellular import Environment

# A 100x100um chamber with a 20x20um solid wall block at its center, and
# the top row marked as a death zone.
wall_map = np.zeros((10, 10))
wall_map[4:6, 4:6] = 1   # wall block: x in [40, 60], y in [40, 60]
wall_map[9, :] = -1      # y in [90, 100] is out of bounds

env = Environment("trap", wall_map=wall_map, size=(100.0, 100.0))
```

At construction, `Environment` rasterizes `wall_map` into the flat faces and
corner points `Colony` needs for contact forces (`env.wall_faces`,
`env.wall_corners`): every wall-cell edge bordering a *media* cell is an
exposed edge, collinear runs of exposed edges merge into one flat face (so a
straight wall reads as one long face, not one tiny face per pixel), and each
face's own endpoints become a corner point. A wall cell's edge is **not**
exposed against another wall cell (fully interior to a solid block, so cells
can never reach it), against an out-of-bounds (`-1`) cell, or against the
grid's outer boundary — a `-1` cell already kills any cell whose center
reaches it, and there's no medium beyond the grid edge to push a cell back
into, so generating a repelling face in either case would be moot at best.
For a wall cell that borders out-of-bounds on one side and media on another
(e.g. a trap wall next to an open, unwalled outflow — see
`examples/mother_machine_demo.ipynb`), it would also be actively wrong: it
would push cells along the unrelated out-of-bounds-facing side instead of
leaving that direction untouched.

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
env = Environment("LB medium", wall_map=np.zeros((10, 10)), fields=[field])
colony = Colony([cell], env)

colony.step(dt=0.1)
print(cell.concentrations["X"], field.values[5, 5])  # X left the cell, appeared in the field
```

If a cell exports a species with no matching `Field` in the environment,
`export_chemical_fields` raises `ValueError` — validated for every exporting
cell before any field is mutated, so a missing field never causes a partial
deposit. The target field doesn't need `is_chemical=True` (that flag makes
the field's name an extracellular species of any network that references
it, an orthogonal concern from whether it can receive secretion) or
`diffuses=True`, though combining both lets secreted material spread
spatially and be sensed or taken up by other cells.

#### Lysis-based delivery

Where secretion above is a continuous, reaction-driven leak, lysis-based
delivery is a one-time bulk dump: a cell can be configured to deliver an
entire intracellular species' molecule count into a `Field` at the instant
it dies — modeling, e.g., a reporter or payload released when an engineered
kill-switch circuit lyses the cell.

Pass `lysis_targets` when constructing a `Cell`: a dict mapping an
intracellular species name to the name of the `Field` species it should be
delivered to.

```python
import numpy as np
from multicellular import Cell, Colony, Environment, Field

cell = Cell(
    id=0, position=[50.0, 50.0], orientation=[1.0, 0.0],
    lysis_targets={"reporter_internal": "reporter_field"},
)
cell.set_concentration("reporter_internal", 5.0)
cell.set_concentration("A", 0.0)  # will violate the survival condition below

field = Field("reporter_field", np.zeros((10, 10)))
env = Environment("LB medium", wall_map=np.zeros((10, 10)), fields=[field])
colony = Colony([cell], env, survival_conditions=[("A", ">", 0)])

colony.step(dt=0.1)
print(cell.alive, field.values[5, 5])  # False, reporter_internal's molecule count arrived here
```

Delivery fires **only** when `Colony.enforce_survival_conditions` kills a
cell for violating a `survival_conditions` entry (see
[`Colony`](#colony) below) — never when `Colony.enforce_bounds` kills a cell
for leaving the environment or entering a `wall_map` out-of-bounds (`-1`)
death zone (see
[Walls and out-of-bounds](#walls-and-out-of-bounds-wall_map) above), since
that represents outflow (e.g. off the edge of a microfluidic chip), not
lysis. At the moment of death, the cell's current molecule count for each
source species (`concentration * cell.compute_volume()`) is added to the
matching target `Field` at the cell's nearest grid point (summing
contributions from multiple cells that lyse at the same grid cell in the
same step), converting to a Δconcentration via `environment.grid_cell_volume`
— the same mechanics and the same `ValueError`-before-any-mutation guarantee
as `export_chemical_fields` above.

A source species must map to a **different** target species name —
`Cell.__init__` raises `ValueError` if any source equals its own target.
This isn't just a naming nicety. A species named after a chemical `Field`
*is* that field's extracellular pool (see
[Extracellular species](#extracellular-species-uptake-and-membrane-bound-enzymes)
above): the cell reads it and exchanges mass with it, but holds no copies
of it itself, so there's nothing intracellular to release. Deliver an
intracellular species into a distinct, field-backed one instead, the same
intracellular/extracellular naming discipline used for uptake.

`lysis_targets` is inherited by both daughters at division, the same way
`species` and `growth_rate_law` are.

### `Colony`

A `Colony` is a collection of `Cell`s living in an `Environment`:

```python
import numpy as np
from multicellular import Cell, Colony, Environment

env = Environment("LB medium", wall_map=np.zeros((10, 10)))
cells = [Cell(id=0, position=[10.0, 10.0], orientation=[1.0, 0.0])]

colony = Colony(cells, env)
colony.step(dt=0.1)
```

`Colony` accepts three mechanical parameters:

- **`k`** (default `10.0`): Hookean contact stiffness (force / length). Controls
  how hard cells push against one another when they overlap. Must be tuned
  alongside `dt`; the explicit integrator is stable when `dt < 2 * ζ_t / k`.
  That is `dt < 2 * drag * length / k` for rods and
  `dt < 6 * drag * radius / k` for spheres.
- **`k_wall`** (default: `10 * k`): Hookean stiffness for cell-wall contacts
  (see [Wall forces](#wall-forces) below). Walls default to much stiffer than
  cells, so they behave close to a rigid boundary rather than yielding like
  another cell would; pass an explicit value to override. Like `k`, it must
  be tuned alongside `dt` — the explicit integrator is stable when
  `dt < 2 * ζ_t / k_wall`, so a much stiffer `k_wall` needs a
  correspondingly smaller `dt`.
- **`drag`** (default `1.0`): isotropic drag constant for contact dynamics
  (force · time / length²). Sets the translational drag `ζ_t = drag * length`
  and rotational drag `ζ_r = (drag / 12) * length³` for rods, and
  `ζ_t = 3 * drag * radius` for spheres (which contact forces never rotate).

It also accepts **`brownian_motion`** (default `True`): whether `step`
applies an overdamped-Langevin Brownian kick to living cells each step (see
[Brownian motion](#brownian-motion) below). Set `brownian_motion=False` to
disable thermal motion entirely, e.g. for a tightly-confined geometry (like
a mother-machine trap; see `examples/mother_machine_demo.ipynb`) where only
contact forces should move cells.

It also accepts an optional **`survival_conditions`** parameter: a list of
`(species, operator, threshold)` tuples, e.g. `[("A", ">", 0)]`. Every step,
each living cell's concentration of `species` is compared against
`threshold` using `operator` — one of `">"`, `">="`, `"<"`, `"<="`, `"=="`,
`"!="` — and the cell dies as soon as any condition is violated. `species`
may be intracellular or the name of a chemical `Field` (read from the
cell's local `extracellular` snapshot, e.g. a toxin in the medium); a
species missing from both is treated as `0.0`. With multiple
conditions, a cell dies if *any* of them is violated. A cell killed this way
that also has `lysis_targets` set delivers its lysis payload to the
environment — see [Lysis-based delivery](#lysis-based-delivery) above:

```python
# Cells die once species "A" is depleted, or if "B" ever exceeds 10.
colony = Colony(cells, env, survival_conditions=[("A", ">", 0), ("B", "<=", 10)])
```

`colony.step(dt)` performs the following operations in order. Its optional
`method` (default `"ODE"`), `ode_method` (default `"RK45"`), `ssa_method`
(default `"direct"`), and `tau_epsilon` (default `0.03`) arguments are forwarded to every cell's `Cell.step` (see
[Cell with an internal reaction network](#cell-with-an-internal-reaction-network)
above) and to `environment.react`:

1. **Diffusion** — calls `environment.diffuse(dt)`, advancing every field
   with `diffuses=True` (no-op if there are none; see
   [Field diffusion](#field-diffusion) above).
2. **Field reactions** — calls `environment.react(dt, method, ode_method)`, advancing the
   environment's `reactions` network (if any) independently at every
   non-wall grid cell (no-op if `reactions` is `None`; see
   [Field reactions (reaction-diffusion)](#field-reactions-reaction-diffusion)
   above).
3. **Growth rates** — calls each living cell's `growth_rate_law` (if any)
   with its concentrations and the local value of every field.
4. **Internal step** — calls `cell.step` for every cell (chemistry +
   growth). Cells whose network references a chemical `Field` exchange
   mass with their voxel as part of this step, one cell at a time in a
   shuffled order (see
   [Extracellular species](#extracellular-species-uptake-and-membrane-bound-enzymes)
   above). Growth dilutes only intracellular species.
5. **Chemical export** — calls `export_chemical_fields()`, depositing
   whatever each cell exported this step into the matching `Field` (no-op if
   no cell exported anything; see
   [Secretion (chemical export)](#secretion-chemical-export) above).
6. **Sampling** — calls `sample_chemical_fields()`, refreshing every living
   cell's read-only `extracellular` snapshot of its local chemical-field
   values (for recording and survival conditions; moves no mass).
7. **Bounds enforcement** — kills any living cell whose center of mass lies
   outside `environment.size`'s physical extent or on a `wall_map`
   out-of-bounds (`-1`) cell (see
   [Walls and out-of-bounds](#walls-and-out-of-bounds-wall_map) above).
8. **Survival conditions** — kills any living cell whose concentrations
   violate a `survival_conditions` entry (no-op if none were given), then
   delivers any lysis payload configured via `lysis_targets` for the cells
   just killed (see
   [Lysis-based delivery](#lysis-based-delivery) above).
9. **Brownian motion** — if `brownian_motion=True` (the default), applies an
   overdamped-Langevin random kick to every surviving living cell (see
   [Brownian motion](#brownian-motion) below); no-op if `False`.
10. **Contact forces** — applies pairwise Hookean repulsion and torques between
    all overlapping living cells, and between cells and wall geometry (see
    [Contact forces](#contact-forces) and [Wall forces](#wall-forces) below).
11. **Division** — replaces any cell with `cell.ready_to_divide() == True` with
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

env = Environment("uninduced", wall_map=np.zeros((10, 10)))
cell = Cell(id=0, position=[50.0, 50.0], orientation=[1.0, 0.0])
colony = Colony([cell], env)

sim = Simulation(colony, dt=0.1, t_max=5.0)
sim.run()  # simulate before induction

# At t=5.0, add an inducer (e.g. IPTG) uniformly across the medium.
iptg = Field("IPTG", np.full((10, 10), 1.0), is_chemical=True)
induced_env = Environment("+ IPTG", wall_map=np.zeros((10, 10)), fields=[iptg])
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

For a spherical cell (`shape="sphere"`), slender-body theory doesn't apply.
Its drag is isotropic Stokes drag, so `γ_∥ = γ_⊥ = 6π η_local r` and
`γ_rot = 8π η_local r³`, and the same displacement formulas below apply. A
sphere's rotational diffusion matters only because its orientation sets the
axis it next divides along.

For a rod, translational and rotational drag coefficients are computed from
`η_local` and the cell's geometry using slender-body theory for a thin rod of total length
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

A spherical cell is the `length = 0` case: its axis segment collapses to a
point (its center), so the same segment-distance math handles sphere–sphere
and sphere–rod contacts unchanged. Because a contact force on a sphere always
passes through its center, its torque is identically zero. Spheres use
`ζ_t = 3 · drag · radius` and are never rotated by contacts. A rod touching
a sphere still gets the usual torque.

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

#### Wall forces

Cells feel a Hookean repulsion from the environment's `wall_map` geometry
(`env.wall_faces`, `env.wall_corners` — see
[Walls and out-of-bounds](#walls-and-out-of-bounds-wall_map) above), folded
into the same per-cell force and torque sums as cell-cell contact before the
overdamped update runs, so both contact types are resolved together every
step. There are two contact primitives:

- **Flat faces.** Rather than a single closest point on the cell axis (which
  is non-unique for a rod lying parallel to the face — the same degeneracy as
  parallel cell-cell contact), each axis endpoint `e` of the cell is sampled
  independently against every face whose finite extent it falls within (the
  tangential projection of `e` must land between the face's two endpoints —
  this is what lets a cell pass freely through an open end of a wall run
  instead of being pushed off an unbounded plane), and only the single
  *nearest* such face (smallest `|h|`) actually contributes:

  ```
  h = (e − x_w) · n_w      # signed perpendicular distance from e to the face
  δ = R_i − h               # penetration depth
  ```

  where `x_w` is a point on the face and `n_w` its inward unit normal. When
  `δ > 0`, the force `F = k_wall · δ · n_w` (and matching torque) is added.
  Summing over both endpoints handles every orientation with no special
  cases: a rod lying parallel to the face gets an equal push from both ends
  (straight, zero net torque); a tilted rod gets an off-center push from
  whichever end penetrates more, pivoting it to lie flat against the wall.
  A sphere's two "endpoints" coincide at its center, so it samples that one
  point only (counting it twice would double the wall force).
- **Corner points.** Near the end of a finite face, or an inside corner, a
  cell rounds the tip rather than being blocked by an unbounded plane. This
  is point contact between the cell's axis segment and the corner point —
  the same segment-distance geometry as cell-cell contact, with one segment
  collapsed to a point.

Because a wall face's perpendicular distance `h` is measured against its
*infinite* supporting line, an endpoint that has mildly penetrated the near
face of a wall thicker than one grid cell could otherwise also read as
deeply "behind" a far parallel face of the same block — whose line it never
actually approached. Picking only the nearest face by `|h|` (among faces
whose span the endpoint falls within) resolves this correctly regardless of
wall thickness: the near face always wins for a thick block, and for a
thin (single-pixel) wall it's the only candidate regardless of how deep an
endpoint has penetrated, so it keeps pushing back rather than silently
letting the endpoint tunnel through once some fixed depth is exceeded.

### `Simulation`

`Simulation` steps a `Colony` from `t=0` to `t=t_max` in increments of `dt`,
recording every cell's position, orientation, size, shape, alive status, and
chemical concentrations at every timestep. It also owns the simulation
method used to advance every cell's `ReactionNetwork` each step. Pass
`simulation_method="ODE"` (the default), `"SSA"`, or `"CLE"` when
constructing it, plus `ode_method` (default `"RK45"`) to choose the
`solve_ivp` integrator for `"ODE"`, e.g. `"LSODA"` or `"BDF"` for a stiff
network, or `ssa_method` (default `"direct"`; also `"next_reaction"` or
`"tau_leaping"`, with `tau_epsilon`) to choose the stochastic algorithm for
`"SSA"` (see [Choosing an SSA algorithm](#choosing-an-ssa-algorithm-ssa_method)):

```python
import numpy as np
from multicellular import Cell, Colony, Environment, Simulation

env = Environment("LB medium", wall_map=np.zeros((10, 10)))
cell = Cell(id=0, position=[50.0, 50.0], orientation=[1.0, 0.0])
colony = Colony([cell], env)

sim = Simulation(colony, dt=0.1, t_max=10.0, simulation_method="ODE",  # or "SSA"/"CLE"
                 ode_method="RK45")  # or "LSODA", "BDF", "Radau", ...
df = sim.run()

# Stochastic, with adaptive tau-leaping for high-copy-number chemistry:
sim = Simulation(colony, dt=0.1, t_max=10.0, simulation_method="SSA",
                 ssa_method="tau_leaping", tau_epsilon=0.03)  # pandas DataFrame, one row per cell per recorded timestep
```

`sim.run()` records the initial state, then repeatedly calls
`colony.step(dt, sim.simulation_method, sim.ode_method)` and records the new state of every
cell remaining in the colony — including any new daughter cells produced by
division — until `t_max` is reached. The
returned DataFrame has columns `time`, `cell_id`, `parent_id`, `generation`,
`root_id` (see [Lineage tracking](#lineage-tracking)), `alive`, `anchored`, `position_x`,
`position_y`, `orientation_x`, `orientation_y`, `length`, `radius`, `shape`
(`"rod"` or `"sphere"`), plus one
column per chemical species seen in any cell's `concentrations` or
`extracellular` snapshot. A chemical `Field`'s column holds the field's
value at that cell's location. Missing values are `NaN` for cells/species
that don't have that entry. The same data
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
runs (`sim.run(t_max=...)`). Pass `ode_method=` (or `ssa_method=` /
`tau_epsilon=`) to switch integrators or stochastic algorithms for the rest
of the run. Both are useful e.g. to simulate further after calling
`colony.switch_environment(...)` — see
[Switching environments](#switching-environments) above. Calling `run()` a
second time with the same `t_max` it already reached is a no-op.

#### Lineage tracking

Every cell a `Colony` ever holds is recorded in `colony.lineage` (also
`sim.lineage`), a `Lineage`: a forest of family trees, one per founder cell.
Initial cells are founders, born at `t=0`. Each division and death is
stamped with the simulation time at the end of the step it happened in.
Cell ids must be unique within a colony. Pass `id=None` and the colony
assigns one. Each `Cell` also carries `parent_id`, `generation` (divisions
since its founder), and `root_id` (its founder's id). These three also
appear as columns of the history DataFrame, so
`df.groupby("root_id")` follows a clone.

```python
sim = Simulation(Colony([Cell(id=None, position=[50.0, 50.0], orientation=[1.0, 0.0])], env),
                 dt=0.02, t_max=6.0)
sim.run()

lin = sim.lineage_dataframe()  # one row per cell ever created
# id, parent_id, root_id, generation, birth_time, end_time, fate,
# daughter_ids, size_at_birth, size_at_division, shape, species

L = sim.lineage
L.children(0), L.sisters(1), L.ancestors(42), L.descendants(0)
L.leaves(), L.roots(), L.subtree(3)   # subtree(...) is a new Lineage
L.generation_times()                  # interdivision times of non-founders
L.to_newick()                         # branch length = cell lifetime

fig, ax = sim.plot_lineage(color_by="GFP")  # tree vs. time, colored by GFP
```

`fate` is `"alive"`, `"divided"`, `"out_of_bounds"` (killed by
`enforce_bounds`), `"survival_condition"`, or `"killed"` (a manual
`cell.kill()`). `end_time` is `NaN` while a cell is alive. In
`plot_lineage`, each cell is a horizontal branch from birth to
division/death. Deaths are marked at the branch tip, and `color_by` colors
each branch over time by that cell's recorded value. `root_id=` plots one
subtree. It returns `(fig, ax)` without calling `plt.show()`.

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
    env = Environment("LB medium", wall_map=np.zeros((10, 10)))
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
- `simulation_method` (`"ODE"` by default, or `"SSA"`/`"CLE"`),
  `ode_method` (default `"RK45"`), `ssa_method` (default `"direct"`), and
  `tau_epsilon` (default `0.03`) are forwarded to every replicate's
  `Simulation`.
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
    show_progress=True, stride=1, show=True,
)
```

shows a 2D animation of an already-`run()` `Simulation` in a pop-up matplotlib
window:

```python
sim.visualize_colony(red="A", green="B", interval=200)
```

- Each cell is drawn as its rod shape (cylinder + hemispherical caps) using
  its recorded `position`, `orientation`, `length`, and `radius`. A
  spherical cell (`length` 0) is drawn as a circle.
- `red`/`green`/`blue` optionally name chemical species; a cell's color in
  that channel is its concentration of that species, normalized by the
  species' maximum value over the whole simulation. Channels left as `None`
  are `0`, so a cell colored only by `red` is pure red (black when the
  species is absent), and `red` + `green` mix to yellow. With no channel
  set at all, every cell is drawn a uniform mid-gray (`0.3` in each
  channel).
- Dead cells stop appearing in the frame after they die.
- `environment.wall_map` is drawn as the background within `environment.size`:
  media (`0`) is white, walls (`1`) are solid gray, and out-of-bounds (`-1`)
  cells are tinted red — the same red used for the margin outside `size`'s
  physical extent.
- Each frame shows the active environment's `name` on the top-left and the
  current time (`t = ...`) on the top-right. When `colony.switch_environment`
  is called mid-simulation, the name updates automatically at the frame where
  the switch occurred.
- `interval` is the delay between frames in milliseconds.
- `stride` renders only every `stride`-th recorded timestep (default `1`,
  every step), giving a shorter animation without re-running the
  simulation.

`field` optionally names a `Field` (e.g. `"temperature"`) to draw as a light,
semi-transparent heatmap behind the cells, with a colorbar labeled with the
field's name — handy for showing spatial structure (like the extent of a
chemical gradient) that individual cell coloring alone doesn't make clear,
especially where cells are sparse:

```python
sim.visualize_colony(red="temperature", field="temperature", field_cmap="RdYlBu_r")
```

- The heatmap's extent exactly matches `environment.size`, so it's never
  drawn over the red out-of-bounds tint — it sits on top of the `wall_map`
  background instead, staying behind the cells.
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
doesn't already exist:

```python
sim.visualize_colony(red="A", green="B", save_path="./out", filename="colony.gif")
```

#### Choosing GIF or MP4

The extension of `filename` picks the format. Both work out of the box — no
separate `ffmpeg` install is needed.

| | GIF (`.gif`) | MP4 (`.mp4`) |
|---|---|---|
| File size | large | roughly 5–15x smaller |
| Time to write | slower | several times faster |
| Where it plays | embeds directly in notebooks, READMEs, chat and slides | any video player or browser; needs an upload or attachment to share |

**Use a GIF when you want to drop the animation straight into a notebook or a
document.** GIFs are limited to 256 colors, so smooth field heatmaps can look
slightly banded.

**Use an MP4 for anything long or large** — a few hundred frames of a big
colony. It writes faster, takes far less disk space, and keeps full color:

```python
sim.visualize_colony(red="A", green="B", save_path="./out", filename="colony.mp4")
```

`filename` defaults to `"simulation.gif"`, so the format is unchanged unless
you ask for `.mp4`.

#### Saving without watching

By default the animation is also shown interactively after saving. For long
runs — or on a headless machine — pass `show=False` to write the file and skip
the pop-up:

```python
sim.visualize_colony(save_path="./out", filename="colony.mp4", show=False)
```

This also keeps memory use flat: frames are written to the file as they are
drawn, instead of all being held at once (which is hundreds of megabytes for a
long animation). `show=False` returns the path of the file written rather than
an animation object, and requires a `save_path` — without one there would be
nothing to show and nothing to save.

### `Simulation.visualize_field`

```python
sim.visualize_field(
    field_name,
    cmap="viridis",
    vmin=0.0, vmax=None,
    interval=200,
    save_path=None, filename="simulation.gif",
    show_progress=True, stride=1, show=True,
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
- `interval`, `save_path`, `filename`, `show_progress`, `stride`, and `show`
  all behave identically to `visualize_colony` (see above) — including the
  choice between a `.gif` and an `.mp4`.

```python
sim.visualize_field("dye", save_path="./out", filename="dye.gif")
sim.visualize_field("dye", save_path="./out", filename="dye.mp4", show=False)
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

## Examples

The `examples/` directory has runnable notebooks. Their rendered animations
are saved in `examples/outputs/`.

| Notebook | Shows |
|---|---|
| `feature_tour.ipynb` | A short, self-contained demo of every major feature, including spherical cells (section 14) anchored host cells (section 18), and lineage tracking (section 19) |
| `repressilator_demo.ipynb` | Elowitz–Leibler three-gene oscillator |
| `toggle_switch_demo.ipynb` | Gardner–Cantor–Collins bistable toggle: switching and memory |
| `dupOri_cutter.ipynb` | Two incompatible plasmids with mutual nucleases: bistability via low-copy partitioning |
| `quorum_sensing_demo.ipynb` | AHL secretion, diffusion, and density-dependent activation |
| `danino_oscillator_demo.ipynb` | Danino et al. synchronized quorum-sensing clock; voxel volume sized to cell density |
| `synchronized_lysis_demo.ipynb` | Din et al. synchronized lysis in a microfluidic trap, with lysis-based payload delivery |
| `mother_machine_demo.ipynb` | Trap walls, single-file growth, and flush-out in a mother machine |
| `ratkowsky_temperature_demo.ipynb` | Temperature-dependent growth through `growth_rate_law` |

`examples/bench_visualization.py` times each stage of the simulation and
rendering pipeline: `python examples/bench_visualization.py --t-max 9`.

## Unimplemented / stubs

These pieces exist as placeholders (empty classes/functions or
`NotImplementedError`) and are not yet usable:

- **`ReactionNetwork.from_sbml`** (raises `NotImplementedError`) and
  **`multicellular.utils.sbml_parser.parse_sbml`** (an empty stub). Intended
  to construct a `ReactionNetwork` from an SBML model file.

## Running tests

Run the full test suite from the project root:

```bash
pytest
```

Run a specific test file:

```bash
pytest tests/test_cell.py        # Cell geometry (rods and spheres), growth, adder division, low-copy partitioning, lysis targets
pytest tests/test_reactions.py   # Reaction rate laws, stoichiometry, ODE (all solve_ivp methods)/SSA/CLE stepping, compartments, export mass conservation
pytest tests/test_environment.py # Environment/Field construction, wall_map validation and geometry, diffusion, and grid cell volume
pytest tests/test_reaction_diffusion.py # Environment.react: validation, ODE/SSA/CLE correctness, wall exclusion, Colony.step coupling, Turing pattern smoke test
pytest tests/test_colony.py      # Colony bounds, wall/cell-cell contact forces (rods, spheres, mixed), Brownian motion, division, dead cells, field uptake/sensing/export, survival and lysis
pytest tests/test_simulation.py  # Simulation loop and DataFrame export
pytest tests/test_visualization.py # visualize_colony/visualize_field/plot_field: animation output, GIF/MP4 export, stride, static plots
pytest tests/test_parallel.py    # run_replicates: parallel execution, independence, correctness
```
