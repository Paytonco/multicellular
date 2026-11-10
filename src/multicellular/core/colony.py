# core/colony.py

import math
import operator as _operator
from collections import Counter

import numpy as np

from .environment import WATER_DIFFUSIVITY_37C
from .lineage import Lineage
from .reactions import DEFAULT_TAU_EPSILON

_kBT = 1.380649e-23 * 310.15  # J  (k_B × 37°C)

_PARALLEL_TOL = 1e-10  # denom threshold for segment-segment parallel detection

# Comparison operators usable in a Colony survival condition, e.g. ("A", ">", 0).
_COMPARISONS = {
    ">": _operator.gt,
    ">=": _operator.ge,
    "<": _operator.lt,
    "<=": _operator.le,
    "==": _operator.eq,
    "!=": _operator.ne,
}


def _segment_segment_closest(a1x, a1y, b1x, b1y, a2x, a2y, b2x, b2y):
    """
    Closest points between segment [a1, b1] and segment [a2, b2], given as
    raw x/y floats rather than vectors.

    Returns (p1x, p1y, p2x, p2y, d): p1 on seg1, p2 on seg2, d = |p1 - p2|.
    Operates on plain floats (instead of building/dot-producting small numpy
    arrays) since this runs once per candidate contact pair, every step.

    When segments are nearly parallel, uses the midpoint of their overlapping
    axial projection rather than a degenerate endpoint, per the physics spec.
    """
    d1x, d1y = b1x - a1x, b1y - a1y
    d2x, d2y = b2x - a2x, b2y - a2y
    wx, wy = a1x - a2x, a1y - a2y

    a = d1x * d1x + d1y * d1y
    b = d1x * d2x + d1y * d2y
    c = d2x * d2x + d2y * d2y
    e = d1x * wx + d1y * wy
    f = d2x * wx + d2y * wy

    denom = a * c - b * b  # zero iff segments are parallel

    if denom < _PARALLEL_TOL:
        # Parallel or degenerate: project seg2 endpoints onto seg1 and use the
        # midpoint of the overlapping range, or the nearest endpoint pair if
        # there is no axial overlap.
        if a < _PARALLEL_TOL:
            s = 0.0
        else:
            t0 = ((a2x - a1x) * d1x + (a2y - a1y) * d1y) / a
            t1 = ((b2x - a1x) * d1x + (b2y - a1y) * d1y) / a
            lo = max(0.0, min(t0, t1))
            hi = min(1.0, max(t0, t1))
            s = (lo + hi) / 2.0 if lo <= hi else min(max(-e / a, 0.0), 1.0)

        p1x, p1y = a1x + s * d1x, a1y + s * d1y
        if c > _PARALLEL_TOL:
            t = min(max(((p1x - a2x) * d2x + (p1y - a2y) * d2y) / c, 0.0), 1.0)
        else:
            t = 0.0
        p2x, p2y = a2x + t * d2x, a2y + t * d2y
    else:
        # General (non-parallel) case: closed-form minimum with clamped parameters.
        # Clamp s, recompute t, clamp t, then recompute s once more.
        s = min(max((b * f - c * e) / denom, 0.0), 1.0)
        t = min(max((b * s + f) / c, 0.0), 1.0) if c > _PARALLEL_TOL else 0.0
        s = min(max((-e + b * t) / a, 0.0), 1.0) if a > _PARALLEL_TOL else 0.0
        p1x, p1y = a1x + s * d1x, a1y + s * d1y
        p2x, p2y = a2x + t * d2x, a2y + t * d2y

    d = math.hypot(p1x - p2x, p1y - p2y)
    return p1x, p1y, p2x, p2y, d


def _segment_point_closest(a1x, a1y, b1x, b1y, px, py):
    """
    Closest point on segment [a1, b1] to point (px, py).

    Returns (qx, qy, d): q the closest axis point, d = |q - (px, py)|.
    """
    dx, dy = b1x - a1x, b1y - a1y
    len_sq = dx * dx + dy * dy
    if len_sq < _PARALLEL_TOL:
        t = 0.0
    else:
        t = min(max(((px - a1x) * dx + (py - a1y) * dy) / len_sq, 0.0), 1.0)
    qx, qy = a1x + t * dx, a1y + t * dy
    return qx, qy, math.hypot(qx - px, qy - py)


def _pair_contact(ci, cj, ep_i, ep_j, k):
    """
    Hookean contact between cells ci and cj, given their `_axis_endpoints`
    tuples. Returns None if they don't overlap, else (Fx, Fy, pcx, pcy):
    the force on ci (cj feels its negative) and the contact point.
    """
    a1x, a1y, b1x, b1y, cix, ciy = ep_i
    a2x, a2y, b2x, b2y, cjx, cjy = ep_j
    p1x, p1y, p2x, p2y, d = _segment_segment_closest(
        a1x, a1y, b1x, b1y, a2x, a2y, b2x, b2y
    )
    delta = ci.radius + cj.radius - d
    if delta <= 0.0:
        return None

    # Contact normal: unit vector from j toward i. When axes coincide
    # (d≈0), fall back to center-to-center direction, then to ci's
    # perpendicular if centers also coincide.
    if d > 1e-12:
        Nx, Ny = (p1x - p2x) / d, (p1y - p2y) / d
    else:
        sepx, sepy = cix - cjx, ciy - cjy
        sep_norm = math.hypot(sepx, sepy)
        if sep_norm > 1e-12:
            Nx, Ny = sepx / sep_norm, sepy / sep_norm
        else:
            Nx, Ny = -float(ci.orientation[1]), float(ci.orientation[0])

    return k * delta * Nx, k * delta * Ny, (p1x + p2x) / 2.0, (p1y + p2y) / 2.0


class Colony:
    """
    A collection of Cells living within an Environment.

    Anchored cells (`Cell(anchored=True)`) are fixed obstacles: they take
    part in chemistry (reactions, Field exchange, survival conditions,
    lysis) like any other cell, but are never moved by Brownian motion,
    contact forces, or walls. A mobile cell overlapping an anchored one
    receives the full Hookean contact force; overlapping anchored cells
    ignore each other, so tissue cells may be packed freely. Like every
    cell, an anchored cell exchanges mass only with the Field voxel
    containing its center, even if its body spans several voxels.

    Every cell the colony ever holds is recorded in `self.lineage` (a
    `Lineage`): initial cells as founders at construction, then each
    division and death (with its cause) stamped with the simulation time at
    the end of the step it happened in. Cell ids must be unique; cells
    passed with `id=None` are assigned fresh ids.
    """

    def __init__(
        self,
        cells,
        environment,
        k=10.0,
        k_wall=None,
        drag=1.0,
        survival_conditions=None,
        brownian_motion=True,
    ):
        """
        Args:
            cells: initial list of Cell objects.
            environment: the shared Environment.
            k: Hookean contact stiffness (force / length).
            k_wall: Hookean stiffness for cell-wall contacts (wallSpec.txt).
                Defaults to `10 * k` (walls much stiffer than cells, so a
                wall behaves close to a rigid boundary rather than yielding
                like another cell would). Must be tuned alongside `dt`
                like `k` (see `_apply_wall_forces`/README): explicit-Euler
                contact dynamics are only stable while
                `dt < 2 * drag * length / k_wall`.
            drag: isotropic drag constant for contact dynamics.
                  Rods:    ζ_t = drag * length,  ζ_r = (drag / 12) * length³.
                  Spheres: ζ_t = 3 * drag * radius (no contact rotation:
                  every contact force on a sphere passes through its center).
            survival_conditions: optional list of (species, operator, threshold)
                tuples, e.g. [("A", ">", 0)]. Every step, each living cell's
                concentration of `species` is compared against `threshold`
                using `operator` (one of ">", ">=", "<", "<=", "==", "!=");
                a cell dies as soon as any condition is violated. `species`
                may be intracellular or the name of a chemical Field (read
                from the cell's local `extracellular` snapshot); a species
                missing from both is treated as 0.0. A
                dying cell with `lysis_targets` set (see `Cell`) also has its
                lysis payload delivered to the environment; see
                `_deliver_lysis_payloads`.
            brownian_motion: whether `step` applies overdamped-Langevin
                Brownian kicks to living cells each step (default True).
                Set False to disable thermal motion entirely, e.g. for a
                tightly-confined geometry where only contact forces should
                move cells.
        """
        self.cells = list(cells)
        self.environment = environment
        self.k = k
        self.k_wall = 10.0 * k if k_wall is None else k_wall
        self.drag = drag
        self.survival_conditions = self._validate_survival_conditions(
            survival_conditions
        )
        self.brownian_motion = brownian_motion
        existing_ids = [cell.id for cell in self.cells if cell.id is not None]
        duplicates = sorted(i for i, n in Counter(existing_ids).items() if n > 1)
        if duplicates:
            raise ValueError(
                f"Cell ids must be unique within a Colony; duplicated: {duplicates}. "
                "Pass id=None to have the Colony assign one."
            )
        self._next_id = max(existing_ids, default=-1) + 1
        # Simulation time of the colony's latest state; every lineage event
        # in a step is stamped with the time at the end of that step.
        self.time = 0.0
        self.lineage = Lineage()
        self._register_new_cells(self.time)

    def _register_new_cells(self, time):
        """
        Give any id-less cell a fresh id and add every cell not yet in the
        lineage as a founder (its own root unless it already has one).
        """
        for cell in self.cells:
            if cell.id is None:
                cell.id = self._next_id
                self._next_id += 1
            elif cell.id >= self._next_id:
                self._next_id = cell.id + 1
            if cell.id in self.lineage:
                continue
            if cell.root_id is None:
                cell.root_id = cell.id
            self.lineage.register_founder(cell, time)

    def _sync_lineage(self, time):
        """
        Reconcile the lineage with `self.cells` at the start of a step, at
        the current `time`: register cells appended to `cells` directly, and
        record deaths of cells killed outside the colony's own checks (e.g.
        `cell.kill()` between steps).
        """
        self._register_new_cells(time)
        records = self.lineage.records
        for cell in self.cells:
            if not cell.alive and records[cell.id].end_time is None:
                self.lineage.register_death(cell, time, cell.death_cause or "killed")
        self.lineage.time = time

    @staticmethod
    def _validate_survival_conditions(survival_conditions):
        conditions = list(survival_conditions) if survival_conditions else []
        for species, op, threshold in conditions:
            if op not in _COMPARISONS:
                raise ValueError(
                    f"Unknown survival condition operator {op!r} for species "
                    f"{species!r}; must be one of {sorted(_COMPARISONS)}."
                )
        return conditions

    @property
    def living_cells(self):
        return [cell for cell in self.cells if cell.alive]

    def switch_environment(self, environment):
        """
        Replace the colony's environment with a new one.

        Useful for simulating discrete induction/perturbation events
        partway through a simulation (e.g. swapping in an environment whose
        chemical fields represent an inducer being added to the medium),
        without needing to recreate the colony or its cells.
        """
        self.environment = environment

    def _update_growth_rates(self):
        """Call each living cell's growth_rate_law with its current state and local field values."""
        living = [
            cell
            for cell in self.cells
            if cell.alive and not cell.anchored and cell.growth_rate_law is not None
        ]
        if not living:
            return
        positions = np.array([cell.position for cell in living])
        i_idx, j_idx = self._field_indices(positions)
        for cell, i, j in zip(living, i_idx, j_idx):
            extracellular = {
                name: float(field.values[i, j])
                for name, field in self.environment.fields.items()
            }
            cell.growth_rate = cell.growth_rate_law(cell.concentrations, extracellular)

    def step(
        self,
        dt,
        method="ODE",
        ode_method="RK45",
        ssa_method="direct",
        tau_epsilon=DEFAULT_TAU_EPSILON,
    ):
        """
        Advance all cells by one timestep, then enforce bounds and divisions.

        Args:
            dt: timestep size.
            method: simulation method forwarded to each cell's
                `Cell.step` and to the environment's `react()` ("ODE",
                "SSA", or "CLE").
            ode_method: `scipy.integrate.solve_ivp` method used when
                `method` is "ODE" (default "RK45").
            ssa_method: stochastic algorithm used when `method` is "SSA":
                "direct" (default), "next_reaction", or "tau_leaping".
            tau_epsilon: tau-leaping error control (default 0.03).
        """
        t_end = self.time + dt
        # Catch cells added or killed by hand since the last step before
        # this step's events are recorded.
        self._sync_lineage(self.time)
        self.environment.diffuse(dt)
        stochastic = {"ssa_method": ssa_method, "tau_epsilon": tau_epsilon}
        self.environment.react(dt, method, ode_method, **stochastic)
        self._update_growth_rates()
        self._step_cells(dt, method, ode_method, **stochastic)
        self.export_chemical_fields()
        self.sample_chemical_fields()
        self.enforce_bounds(t_end)
        self.enforce_survival_conditions(t_end)
        alive = self.living_cells
        if self.brownian_motion:
            mobile = [cell for cell in alive if not cell.anchored]
            noise = self._draw_brownian_noise(mobile)
            self._apply_brownian_motion(dt, mobile, noise)
        self._apply_contact_forces(dt, alive)
        self.handle_divisions(t_end)
        self.time = t_end
        self.lineage.time = t_end

    def _field_indices(self, positions):
        """Vectorized nearest-grid-point (row, col) indices for an (n, 2) array of positions in μm."""
        return self.environment.grid_indices(positions)

    def _step_cells(
        self,
        dt,
        method,
        ode_method,
        ssa_method="direct",
        tau_epsilon=DEFAULT_TAU_EPSILON,
    ):
        """
        Step every cell, exchanging mass with the chemical Fields its
        reaction network references.

        Any network species sharing a chemical (`is_chemical=True`) Field's
        name is that Field's extracellular species: the cell's reactions
        read it at the voxel containing the cell's center and consume or
        produce it there, via `Cell.step`'s `external` (with the voxel's
        `grid_cell_volume` as its compartment volume), and the result is
        written straight back into the Field. Molecules are conserved
        between the cell and the Field.

        Coupled cells are stepped one at a time, in an order shuffled by
        `environment.rng` each step, each reading its voxel after the
        previous cells' writes. Cells sharing a voxel therefore share (and
        can jointly deplete) one pool, without any cell getting systematic
        first access to it. Uncoupled cells step in list order first.
        """
        chemical = {
            name: field
            for name, field in self.environment.fields.items()
            if field.is_chemical
        }
        coupled, coupled_species = [], []
        for cell in self.cells:
            species = (
                [s for s in cell.network.species if s in chemical]
                if cell.alive and cell.network and chemical
                else []
            )
            if species:
                coupled.append(cell)
                coupled_species.append(species)
            else:
                cell.step(
                    dt,
                    method,
                    ode_method,
                    ssa_method=ssa_method,
                    tau_epsilon=tau_epsilon,
                )
        if not coupled:
            return

        positions = np.array([cell.position for cell in coupled])
        i_idx, j_idx = self._field_indices(positions)
        voxel_volume = self.environment.grid_cell_volume
        for k in self.environment.rng.permutation(len(coupled)):
            i, j = i_idx[k], j_idx[k]
            external = {s: float(chemical[s].values[i, j]) for s in coupled_species[k]}
            updated = coupled[k].step(
                dt,
                method,
                ode_method,
                external,
                voxel_volume,
                ssa_method=ssa_method,
                tau_epsilon=tau_epsilon,
            )
            for s, value in updated.items():
                chemical[s].values[i, j] = value

    def sample_chemical_fields(self):
        """
        Refresh every living cell's `extracellular` snapshot with each
        chemical field's value at the cell's position.

        Read-only: this records what the cell's surroundings look like (for
        `Simulation` history and survival conditions) and never moves mass;
        mass exchange happens in `_step_cells`.
        """
        living = [cell for cell in self.cells if cell.alive]
        if not living:
            return
        chemical_fields = [
            field for field in self.environment.fields.values() if field.is_chemical
        ]
        for cell in living:
            cell.extracellular = {}
        if not chemical_fields:
            return
        positions = np.array([cell.position for cell in living])
        i_idx, j_idx = self._field_indices(positions)
        for field in chemical_fields:
            values = field.values[i_idx, j_idx]
            for cell, value in zip(living, values):
                cell.extracellular[field.name] = float(value)

    def export_chemical_fields(self):
        """
        Deposit each living cell's pending reaction-network exports (set by
        Cell.step from ReactionNetwork.last_exported) into the Field sharing
        the exported species' name, converting molecule counts to a
        concentration change via the environment's grid cell volume.

        Validates every exported species against environment.fields before
        writing anything, so a missing field raises ValueError without
        partially mutating any field.
        """
        living = [cell for cell in self.cells if cell.alive and cell.pending_export]
        if not living:
            return

        positions = np.array([cell.position for cell in living])
        i_idx, j_idx = self._field_indices(positions)
        grid_cell_volume = self.environment.grid_cell_volume

        deltas = {}
        for cell, i, j in zip(living, i_idx, j_idx):
            for species, count in cell.pending_export.items():
                if species not in self.environment.fields:
                    raise ValueError(
                        f"Cell {cell.id} exported species '{species}' but no "
                        f"matching Field exists in the environment."
                    )
                delta = deltas.setdefault(species, np.zeros(self.environment.shape))
                delta[i, j] += count / grid_cell_volume

        for cell in living:
            cell.pending_export = {}
        for species, delta in deltas.items():
            field = self.environment.get_field(species)
            field.values = field.values + delta

    def _draw_brownian_noise(self, alive):
        """
        Draw the 3 standard-normal samples (parallel, perpendicular,
        rotational) each living cell needs for its Brownian kick this step.

        Calls are batched per distinct rng instance (most cells share one
        Generator, inherited from parent to daughters at division) instead of
        issuing one Python-level standard_normal(3) call per cell. A Generator
        only advances its own stream when its own method is called, and
        requesting N draws in one batched call consumes that stream
        identically to N separate calls of the same total size, so each
        cell's draw is bit-for-bit the same as before — this only removes
        per-call dispatch overhead.
        """
        n = len(alive)
        noise = [None] * n
        groups = {}
        for idx, cell in enumerate(alive):
            key = id(cell.rng)
            if key not in groups:
                groups[key] = (cell.rng, [])
            groups[key][1].append(idx)

        for rng, indices in groups.values():
            draws = rng.standard_normal((len(indices), 3))
            for row, idx in zip(draws, indices):
                noise[idx] = row
        return noise

    def _apply_brownian_motion(self, dt, alive, noise):
        """
        Apply overdamped-Langevin Brownian displacements to every living
        cell, computing the per-cell drag/diffusion pipeline as batched
        numpy array ops instead of one Python pass per cell.

        Uses slender-body drag for an anisotropic rod, and isotropic Stokes
        drag (γ_t = 6πηr, γ_r = 8πηr³) for a sphere. Local viscosity and
        diffusivity are sampled from the environment grid at each cell's
        position. Spatial units are μm; dt is in seconds. `noise` is the
        (n, 3) array of (parallel, perpendicular, rotational) standard-normal
        triples drawn for `alive` by `_draw_brownian_noise`. Final position
        and orientation writes are still a per-cell loop (each `Cell` owns
        its own small position/orientation arrays), but that loop does no
        further math — `cell.apply_torque` is reused as-is for the rotation
        update.
        """
        if not alive:
            return

        positions = np.array([cell.position for cell in alive])
        lengths = np.array([cell.length for cell in alive])
        radii = np.array([cell.radius for cell in alive])
        orientations = np.array([cell.orientation for cell in alive])
        is_sphere = np.array([cell.shape == "sphere" for cell in alive])
        xi = np.asarray(noise)

        i_idx, j_idx = self._field_indices(positions)
        eta = self.environment.eta[i_idx, j_idx]
        D_field = self.environment.diffusivity[i_idx, j_idx]

        # Scale viscosity by local diffusivity relative to the reference medium
        # (Stokes-Einstein: η ∝ 1/D at fixed temperature and probe size).
        eta_local = eta * (WATER_DIFFUSIVITY_37C / D_field)

        # Cell geometry in SI (m)
        L_eff = (lengths + 2.0 * radii) * 1e-6
        r = radii * 1e-6
        ln_ratio = np.log(L_eff / r)  # ln(total length / radius), always > 0

        # Slender-body drag coefficients (rods)
        gamma_par = 2.0 * np.pi * eta_local * L_eff / ln_ratio  # kg/s
        gamma_perp = 4.0 * np.pi * eta_local * L_eff / ln_ratio  # kg/s
        gamma_rot = np.pi * eta_local * L_eff**3 / (3.0 * ln_ratio)  # kg·m²/s

        # Stokes drag coefficients (spheres)
        if is_sphere.any():
            gamma_stokes = 6.0 * np.pi * eta_local * r
            gamma_par = np.where(is_sphere, gamma_stokes, gamma_par)
            gamma_perp = np.where(is_sphere, gamma_stokes, gamma_perp)
            gamma_rot = np.where(is_sphere, 8.0 * np.pi * eta_local * r**3, gamma_rot)

        # Diffusion coefficients via Einstein relation
        D_par = _kBT / gamma_par  # m²/s
        D_perp = _kBT / gamma_perp  # m²/s
        D_rot = _kBT / gamma_rot  # rad²/s

        ux, uy = orientations[:, 0], orientations[:, 1]
        uperp_x, uperp_y = -uy, ux
        xi_par, xi_perp, xi_rot = xi[:, 0], xi[:, 1], xi[:, 2]

        s_par = np.sqrt(2.0 * D_par * dt)
        s_perp = np.sqrt(2.0 * D_perp * dt)

        # Translational Brownian kick (convert m → μm)
        dr_x = (ux * xi_par * s_par + uperp_x * xi_perp * s_perp) * 1e6
        dr_y = (uy * xi_par * s_par + uperp_y * xi_perp * s_perp) * 1e6

        # Rotational Brownian kick
        omega = xi_rot * np.sqrt(2.0 * D_rot / dt)

        for idx, cell in enumerate(alive):
            cell.position[0] += dr_x[idx]
            cell.position[1] += dr_y[idx]
            cell.apply_torque(omega[idx], dt)

    def _build_spatial_grid(self, alive, return_bin_size=False):
        """
        Bin cells by center position into a uniform grid for neighbor search.

        The grid cell size is set to the largest possible center-to-center
        distance at which any two cells could overlap (the sum of their
        half-extents, maximized over all living cells). With that sizing,
        any pair of cells that could possibly overlap is guaranteed to lie
        in the same grid cell or one of its 8 neighbors (standard
        linked-cell / cell-list algorithm), so checking that 3x3
        neighborhood is sufficient without missing any contacts.
        """
        max_half_extent = max(
            (cell.length / 2.0 + cell.radius for cell in alive), default=0.0
        )
        cell_size = max(2.0 * max_half_extent, 1e-6)

        grid = {}
        for idx, cell in enumerate(alive):
            gx = int(cell.position[0] // cell_size)
            gy = int(cell.position[1] // cell_size)
            grid.setdefault((gx, gy), []).append(idx)
        return (grid, cell_size) if return_bin_size else grid

    def _apply_contact_forces(self, dt, alive=None):
        """
        Apply pairwise cell-cell and cell-wall Hookean contact forces and
        torques to all living cells.

        For each overlapping cell pair (i, j):
          - overlap δ = R_i + R_j - d  (d: axis-segment distance)
          - force on i: F = k * δ * N  (N: contact normal from j toward i)
          - torque on i: τ = (p_c - c_i) × F  (2D scalar cross product)
        Wall contacts (wallSpec.txt) are folded into the same per-cell force
        and torque sums by `_apply_wall_forces` before dynamics are applied,
        so both contact types go through one overdamped update:
          Δc = F / ζ_t * dt,  ζ_t = drag * length
          Δθ = τ / ζ_r * dt,  ζ_r = (drag / 12) * length³
        A sphere is the length-0 case: its axis segment is a single point, so
        the same segment-distance math covers sphere-sphere and sphere-rod
        contacts. Its contact torque is identically zero, so it uses
        ζ_t = 3 * drag * radius and is never rotated here.

        Anchored cells never move: only the mobile cell of a mobile-anchored
        pair is pushed (by the full force), anchored-anchored pairs are
        skipped, and anchored cells feel no wall forces.

        Candidate mobile-mobile pairs are restricted to cells sharing a grid
        bin or an adjacent one (see _build_spatial_grid), which avoids the
        O(n²) blowup of testing every pair directly while still finding every
        contact. Anchored cells are kept out of that grid, since a single
        large one (e.g. a eukaryotic host cell) would otherwise inflate every
        bin; each instead scans the mobile-cell bins overlapping its own
        reach.

        Per-pair math uses plain floats rather than small numpy arrays/dot
        products, since this loop runs for every candidate pair, every step.
        """
        if alive is None:
            alive = self.living_cells
        mobile = [cell for cell in alive if not cell.anchored]
        n = len(mobile)
        if n == 0:
            return
        anchored = [cell for cell in alive if cell.anchored] if n < len(alive) else []

        forces_x = [0.0] * n
        forces_y = [0.0] * n
        torques = [0.0] * n

        endpoints = [self._axis_endpoints(ci) for ci in mobile]

        if n >= 2 or anchored:
            grid, bin_size = self._build_spatial_grid(mobile, return_bin_size=True)
            k = self.k

            for (gx, gy), home_indices in grid.items():
                neighbor_indices = []
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        neighbor_indices.extend(grid.get((gx + dx, gy + dy), ()))

                for i in home_indices:
                    ci = mobile[i]
                    a1x, a1y, b1x, b1y, cix, ciy = endpoints[i]

                    for j in neighbor_indices:
                        if j <= i:
                            continue

                        cj = mobile[j]
                        a2x, a2y, b2x, b2y, cjx, cjy = endpoints[j]

                        # Inlined rather than calling `_pair_contact`: this
                        # is the hot loop, and the extra call costs ~20%.
                        p1x, p1y, p2x, p2y, d = _segment_segment_closest(
                            a1x, a1y, b1x, b1y, a2x, a2y, b2x, b2y
                        )
                        delta = ci.radius + cj.radius - d

                        if delta <= 0.0:
                            continue

                        # Contact normal: unit vector from j toward i.
                        # When axes coincide (d≈0), fall back to center-to-center direction,
                        # then to ci's perpendicular if centers also coincide.
                        if d > 1e-12:
                            Nx, Ny = (p1x - p2x) / d, (p1y - p2y) / d
                        else:
                            sepx, sepy = cix - cjx, ciy - cjy
                            sep_norm = math.hypot(sepx, sepy)
                            if sep_norm > 1e-12:
                                Nx, Ny = sepx / sep_norm, sepy / sep_norm
                            else:
                                Nx, Ny = -float(ci.orientation[1]), float(
                                    ci.orientation[0]
                                )

                        Fx, Fy = k * delta * Nx, k * delta * Ny
                        pcx, pcy = (p1x + p2x) / 2.0, (p1y + p2y) / 2.0

                        # 2D torque: τ = r_x * F_y - r_y * F_x
                        rix, riy = pcx - cix, pcy - ciy
                        rjx, rjy = pcx - cjx, pcy - cjy

                        forces_x[i] += Fx
                        forces_y[i] += Fy
                        forces_x[j] -= Fx
                        forces_y[j] -= Fy
                        torques[i] += rix * Fy - riy * Fx
                        torques[j] -= rjx * Fy - rjy * Fx  # F_ji = -F

            # Each anchored cell against the mobile cells within its reach.
            max_mobile_half_extent = max(c.length / 2.0 + c.radius for c in mobile)
            for ca in anchored:
                ep_a = self._axis_endpoints(ca)
                reach = ca.length / 2.0 + ca.radius + max_mobile_half_extent
                px, py = ep_a[4], ep_a[5]
                gx0 = int((px - reach) // bin_size)
                gx1 = int((px + reach) // bin_size)
                gy0 = int((py - reach) // bin_size)
                gy1 = int((py + reach) // bin_size)
                for gx in range(gx0, gx1 + 1):
                    for gy in range(gy0, gy1 + 1):
                        for j in grid.get((gx, gy), ()):
                            contact = _pair_contact(
                                ca, mobile[j], ep_a, endpoints[j], k
                            )
                            if contact is None:
                                continue
                            Fx, Fy, pcx, pcy = contact
                            forces_x[j] -= Fx
                            forces_y[j] -= Fy
                            torques[j] -= (pcx - endpoints[j][4]) * Fy - (
                                pcy - endpoints[j][5]
                            ) * Fx

        self._apply_wall_forces(mobile, endpoints, forces_x, forces_y, torques)

        for i, cell in enumerate(mobile):
            if cell.shape == "sphere":
                zeta_t = 3.0 * self.drag * cell.radius
                cell.position[0] += forces_x[i] / zeta_t * dt
                cell.position[1] += forces_y[i] / zeta_t * dt
                continue
            zeta_t = self.drag * cell.length
            zeta_r = (self.drag / 12.0) * cell.length**3
            cell.position[0] += forces_x[i] / zeta_t * dt
            cell.position[1] += forces_y[i] / zeta_t * dt
            cell.apply_torque(torques[i] / zeta_r, dt)

    @staticmethod
    def _axis_endpoints(cell):
        """(a_x, a_y, b_x, b_y, c_x, c_y): a cell's axis segment endpoints and center, as floats."""
        px, py = float(cell.position[0]), float(cell.position[1])
        ox, oy = float(cell.orientation[0]), float(cell.orientation[1])
        hl = cell.length / 2.0
        return (px - hl * ox, py - hl * oy, px + hl * ox, py + hl * oy, px, py)

    def _apply_wall_forces(self, alive, endpoints, forces_x, forces_y, torques):
        """
        Accumulate spherocylinder-wall contact forces/torques (wallSpec.txt)
        into the same per-cell sums `_apply_contact_forces` uses for
        cell-cell contacts.

        Flat faces (secs 1-2): each cell axis endpoint e is sampled
        independently against the face's line — h = (e - x_w)·n_w,
        δ = R_i - h — but only within the face's finite extent (the
        tangential projection of e must fall between its two endpoints);
        outside that span the face doesn't reach, and the adjacent corner
        point (below) takes over, which is what makes an open end of a
        wall run (or the grid boundary) pass cells through freely rather
        than being pushed off an unbounded plane.

        h is a signed distance to a face's *infinite* line, so for a wall
        thicker than one pixel, an endpoint sitting just past the near face
        can also read as deeply "behind" a far parallel face of the same
        block (whose line the point never actually approached). Rather than
        cap that reach at a fixed distance -- which, past the cap, would
        silently stop pushing back on a point that has tunneled into (or
        through) a *thin* wall in a single large step, letting it drift out
        the other side unopposed -- each endpoint interacts with only the
        single face whose line it is nearest to (smallest |h|) among those
        whose span it falls within. That's always the physically relevant
        face: the near one for a thick block, and still the only one for a
        thin wall no matter how deep the penetration.

        A sphere's two axis endpoints coincide at its center, so it samples
        that single point against the faces (sampling both would double the
        face force).

        Corner points (sec 3): point contact between the cell's axis
        segment and the corner, reusing `_segment_point_closest` (the
        cell-cell segment-distance routine with one segment collapsed to a
        point).
        """
        faces = self.environment.wall_faces
        corners = self.environment.wall_corners
        if not faces and not corners:
            return
        k_wall = self.k_wall

        for i, ci in enumerate(alive):
            a1x, a1y, b1x, b1y, cix, ciy = endpoints[i]
            R = ci.radius

            sample_points = (
                ((cix, ciy),) if ci.shape == "sphere" else ((a1x, a1y), (b1x, b1y))
            )
            for ex, ey in sample_points:
                best_h = best_nx = best_ny = None
                for x0, y0, x1, y1, nx, ny in faces:
                    tx, ty = x1 - x0, y1 - y0
                    L = math.hypot(tx, ty)
                    if L < _PARALLEL_TOL:
                        continue
                    tx, ty = tx / L, ty / L
                    s = (ex - x0) * tx + (ey - y0) * ty
                    if s < 0.0 or s > L:
                        continue

                    h = (ex - x0) * nx + (ey - y0) * ny
                    if best_h is None or abs(h) < abs(best_h):
                        best_h, best_nx, best_ny = h, nx, ny

                if best_h is None:
                    continue
                delta = R - best_h
                if delta <= 0.0:
                    continue

                Fx, Fy = k_wall * delta * best_nx, k_wall * delta * best_ny
                forces_x[i] += Fx
                forces_y[i] += Fy
                torques[i] += (ex - cix) * Fy - (ey - ciy) * Fx

            for cx, cy in corners:
                qx, qy, d = _segment_point_closest(a1x, a1y, b1x, b1y, cx, cy)
                delta = R - d
                if delta <= 0.0:
                    continue

                if d > 1e-12:
                    Nx, Ny = (qx - cx) / d, (qy - cy) / d
                else:
                    Nx, Ny = -float(ci.orientation[1]), float(ci.orientation[0])

                Fx, Fy = k_wall * delta * Nx, k_wall * delta * Ny
                forces_x[i] += Fx
                forces_y[i] += Fy
                torques[i] += (qx - cix) * Fy - (qy - ciy) * Fx

    def enforce_bounds(self, time=None):
        """
        Kill any cell whose center of mass has left the environment's
        physical extent or entered a wall_map out-of-bounds (-1) cell,
        recording its death in the lineage at `time` (default `self.time`).
        """
        time = self.time if time is None else time
        for cell in self.cells:
            if cell.alive and not self.environment.in_bounds(cell.position):
                cell.kill("out_of_bounds")
                self.lineage.register_death(cell, time, "out_of_bounds")

    def enforce_survival_conditions(self, time=None):
        """
        Kill any living cell whose concentrations violate a survival
        condition, then deliver lysis payloads (see
        `_deliver_lysis_payloads`) for whichever cells were just killed.
        Deaths are recorded in the lineage at `time` (default `self.time`).
        """
        if not self.survival_conditions:
            return
        time = self.time if time is None else time
        lysed = []
        for cell in self.cells:
            if not cell.alive:
                continue
            for species, op, threshold in self.survival_conditions:
                value = cell.concentrations.get(
                    species, cell.extracellular.get(species, 0.0)
                )
                if not _COMPARISONS[op](value, threshold):
                    cell.kill("survival_condition")
                    self.lineage.register_death(cell, time, "survival_condition")
                    lysed.append(cell)
                    break
        self._deliver_lysis_payloads(lysed)

    def _deliver_lysis_payloads(self, cells):
        """
        Deposit each given cell's configured `lysis_targets` molecule counts
        into the matching Field, at the grid cell the cell's center of mass
        occupies, converting to a Δconcentration via
        `environment.grid_cell_volume` (the same convention
        `export_chemical_fields` uses).

        Only called by `enforce_survival_conditions`, for cells it just
        killed — never for cells killed by `enforce_bounds` (leaving the
        environment or entering a death zone represents outflow, not
        lysis).

        Validates every target species against environment.fields before
        writing anything, so a missing field raises ValueError without
        partially mutating any field.
        """
        lysing = [cell for cell in cells if cell.lysis_targets]
        if not lysing:
            return

        positions = np.array([cell.position for cell in lysing])
        i_idx, j_idx = self._field_indices(positions)
        grid_cell_volume = self.environment.grid_cell_volume

        deltas = {}
        for cell, i, j in zip(lysing, i_idx, j_idx):
            for source_species, target_field in cell.lysis_targets.items():
                if target_field not in self.environment.fields:
                    raise ValueError(
                        f"Cell {cell.id} lyses species '{source_species}' "
                        f"into '{target_field}' but no matching Field "
                        "exists in the environment."
                    )
                delta = deltas.setdefault(
                    target_field, np.zeros(self.environment.shape)
                )
                delta[i, j] += cell.copy_number(source_species) / grid_cell_volume

        for target_field, delta in deltas.items():
            field = self.environment.get_field(target_field)
            field.values = field.values + delta

    def handle_divisions(self, time=None):
        """
        Replace any cell ready to divide with its two daughter cells,
        recording the division in the lineage at `time` (default
        `self.time`).
        """
        time = self.time if time is None else time
        new_cells = []
        for cell in self.cells:
            if cell.alive:
                size_at_division = cell._division_target
                daughters = cell.divide()
            else:
                daughters = None
            if daughters is None:
                new_cells.append(cell)
                continue
            for daughter in daughters:
                daughter.id = self._next_id
                self._next_id += 1
                new_cells.append(daughter)
            self.lineage.register_division(
                cell, daughters, time, size_at_division=size_at_division
            )
        self.cells = new_cells
