# core/colony.py

import math
import operator as _operator

import numpy as np

from .environment import WATER_DIFFUSIVITY_37C

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


class Colony:
    """
    A collection of Cells living within an Environment.
    """

    def __init__(self, cells, environment, k=10.0, drag=1.0, survival_conditions=None):
        """
        Args:
            cells: initial list of Cell objects.
            environment: the shared Environment.
            k: Hookean contact stiffness (force / length).
            drag: isotropic drag constant for contact dynamics.
                  Translational drag: ζ_t = drag * length.
                  Rotational drag:    ζ_r = (drag / 12) * length³.
            survival_conditions: optional list of (species, operator, threshold)
                tuples, e.g. [("A", ">", 0)]. Every step, each living cell's
                concentration of `species` is compared against `threshold`
                using `operator` (one of ">", ">=", "<", "<=", "==", "!=");
                a cell dies as soon as any condition is violated. A species
                missing from a cell's concentrations is treated as 0.0.
        """
        self.cells = list(cells)
        self.environment = environment
        self.k = k
        self.drag = drag
        self.survival_conditions = self._validate_survival_conditions(
            survival_conditions
        )
        existing_ids = [cell.id for cell in self.cells if cell.id is not None]
        self._next_id = max(existing_ids, default=-1) + 1

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

    def step(self, dt):
        """Advance all cells by one timestep, then enforce bounds and divisions."""
        self.apply_chemical_fields()
        for cell in self.cells:
            cell.step(dt)
        self.enforce_bounds()
        self.enforce_survival_conditions()
        alive = self.living_cells
        noise = self._draw_brownian_noise(alive)
        for cell, xi in zip(alive, noise):
            self._apply_brownian_motion(cell, dt, xi)
        self._apply_contact_forces(dt, alive)
        self.handle_divisions()

    def _sample_field(self, field_array, position):
        """Nearest-grid-point lookup of a field array at a position in μm."""
        shape = self.environment.shape
        width, height = self.environment.bounds
        j = int(np.clip(position[0] / width * shape[1], 0, shape[1] - 1))
        i = int(np.clip(position[1] / height * shape[0], 0, shape[0] - 1))
        return field_array[i, j]

    def apply_chemical_fields(self):
        """
        Copy each chemical field's local value into every living cell as the
        concentration of the chemical species sharing the field's name.
        """
        chemical_fields = [
            field for field in self.environment.fields.values() if field.is_chemical
        ]
        if not chemical_fields:
            return
        for cell in self.cells:
            if not cell.alive:
                continue
            for field in chemical_fields:
                cell.concentrations[field.name] = self._sample_field(
                    field.values, cell.position
                )

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

    def _apply_brownian_motion(self, cell, dt, xi):
        """
        Apply overdamped-Langevin Brownian displacements to a living cell.

        Uses slender-body drag for an anisotropic rod. Local viscosity and
        diffusivity are sampled from the environment grid at the cell's
        position. Spatial units are μm; dt is in seconds. `xi` is the
        (parallel, perpendicular, rotational) standard-normal triple drawn
        for this cell by `_draw_brownian_noise`.
        """
        eta = self._sample_field(self.environment.eta, cell.position)
        D_field = self._sample_field(self.environment.diffusivity, cell.position)

        # Scale viscosity by local diffusivity relative to the reference medium
        # (Stokes-Einstein: η ∝ 1/D at fixed temperature and probe size).
        eta_local = eta * (WATER_DIFFUSIVITY_37C / D_field)

        # Cell geometry in SI (m)
        L_eff = (cell.length + 2.0 * cell.radius) * 1e-6
        r = cell.radius * 1e-6
        ln_ratio = math.log(L_eff / r)  # ln(total length / radius), always > 0

        # Slender-body drag coefficients
        gamma_par = 2.0 * math.pi * eta_local * L_eff / ln_ratio  # kg/s
        gamma_perp = 4.0 * math.pi * eta_local * L_eff / ln_ratio  # kg/s
        gamma_rot = math.pi * eta_local * L_eff**3 / (3.0 * ln_ratio)  # kg·m²/s

        # Diffusion coefficients via Einstein relation
        D_par = _kBT / gamma_par  # m²/s
        D_perp = _kBT / gamma_perp  # m²/s
        D_rot = _kBT / gamma_rot  # rad²/s

        ux, uy = float(cell.orientation[0]), float(cell.orientation[1])
        uperp_x, uperp_y = -uy, ux
        xi_par, xi_perp, xi_rot = xi

        s_par = math.sqrt(2.0 * D_par * dt)
        s_perp = math.sqrt(2.0 * D_perp * dt)

        # Translational Brownian kick (convert m → μm)
        dr_x = (ux * xi_par * s_par + uperp_x * xi_perp * s_perp) * 1e6
        dr_y = (uy * xi_par * s_par + uperp_y * xi_perp * s_perp) * 1e6
        cell.position[0] += dr_x
        cell.position[1] += dr_y

        # Rotational Brownian kick
        cell.apply_torque(xi_rot * math.sqrt(2.0 * D_rot / dt), dt)

    def _build_spatial_grid(self, alive):
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
        return grid

    def _apply_contact_forces(self, dt, alive=None):
        """
        Apply pairwise Hookean contact forces and torques to all living cells.

        For each overlapping cell pair (i, j):
          - overlap δ = R_i + R_j - d  (d: axis-segment distance)
          - force on i: F = k * δ * N  (N: contact normal from j toward i)
          - torque on i: τ = (p_c - c_i) × F  (2D scalar cross product)
        Dynamics are overdamped:
          Δc = F / ζ_t * dt,  ζ_t = drag * length
          Δθ = τ / ζ_r * dt,  ζ_r = (drag / 12) * length³

        Candidate pairs are restricted to cells sharing a grid bin or an
        adjacent one (see _build_spatial_grid), which avoids the O(n²) blowup
        of testing every pair directly while still finding every contact.

        Per-pair math uses plain floats rather than small numpy arrays/dot
        products, since this loop runs for every candidate pair, every step.
        """
        if alive is None:
            alive = self.living_cells
        n = len(alive)
        if n < 2:
            return

        forces_x = [0.0] * n
        forces_y = [0.0] * n
        torques = [0.0] * n

        endpoints = []
        for ci in alive:
            px, py = float(ci.position[0]), float(ci.position[1])
            ox, oy = float(ci.orientation[0]), float(ci.orientation[1])
            hl = ci.length / 2.0
            endpoints.append(
                (px - hl * ox, py - hl * oy, px + hl * ox, py + hl * oy, px, py)
            )

        grid = self._build_spatial_grid(alive)

        for (gx, gy), home_indices in grid.items():
            neighbor_indices = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbor_indices.extend(grid.get((gx + dx, gy + dy), ()))

            for i in home_indices:
                ci = alive[i]
                a1x, a1y, b1x, b1y, cix, ciy = endpoints[i]

                for j in neighbor_indices:
                    if j <= i:
                        continue

                    cj = alive[j]
                    a2x, a2y, b2x, b2y, cjx, cjy = endpoints[j]

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
                            Nx, Ny = -float(ci.orientation[1]), float(ci.orientation[0])

                    Fx, Fy = self.k * delta * Nx, self.k * delta * Ny
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

        for i, cell in enumerate(alive):
            zeta_t = self.drag * cell.length
            zeta_r = (self.drag / 12.0) * cell.length**3
            cell.position[0] += forces_x[i] / zeta_t * dt
            cell.position[1] += forces_y[i] / zeta_t * dt
            cell.apply_torque(torques[i] / zeta_r, dt)

    def enforce_bounds(self):
        """Kill any cell whose center of mass has left the environment bounds."""
        for cell in self.cells:
            if cell.alive and not self.environment.in_bounds(cell.position):
                cell.kill()

    def enforce_survival_conditions(self):
        """Kill any living cell whose concentrations violate a survival condition."""
        if not self.survival_conditions:
            return
        for cell in self.cells:
            if not cell.alive:
                continue
            for species, op, threshold in self.survival_conditions:
                value = cell.concentrations.get(species, 0.0)
                if not _COMPARISONS[op](value, threshold):
                    cell.kill()
                    break

    def handle_divisions(self):
        """Replace any cell ready to divide with its two daughter cells."""
        new_cells = []
        for cell in self.cells:
            daughters = cell.divide() if cell.alive else None
            if daughters is None:
                new_cells.append(cell)
                continue
            for daughter in daughters:
                daughter.id = self._next_id
                self._next_id += 1
                new_cells.append(daughter)
        self.cells = new_cells
