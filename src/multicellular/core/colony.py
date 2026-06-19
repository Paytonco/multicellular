# core/colony.py

import numpy as np

_kBT = 1.380649e-23 * 310.15  # J  (k_B × 37°C)

_PARALLEL_TOL = 1e-10  # denom threshold for segment-segment parallel detection


def _segment_segment_closest(a1, b1, a2, b2):
    """
    Closest points between segment [a1, b1] and segment [a2, b2].

    Returns (p1, p2, d): p1 on seg1, p2 on seg2, d = |p1 - p2|.

    When segments are nearly parallel, uses the midpoint of their overlapping
    axial projection rather than a degenerate endpoint, per the physics spec.
    """
    d1 = b1 - a1
    d2 = b2 - a2
    w = a1 - a2

    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    e = np.dot(d1, w)
    f = np.dot(d2, w)

    denom = a * c - b * b  # zero iff segments are parallel

    if denom < _PARALLEL_TOL:
        # Parallel or degenerate: project seg2 endpoints onto seg1 and use the
        # midpoint of the overlapping range, or the nearest endpoint pair if
        # there is no axial overlap.
        if a < _PARALLEL_TOL:
            s = 0.0
        else:
            t0 = np.dot(a2 - a1, d1) / a
            t1 = np.dot(b2 - a1, d1) / a
            lo = max(0.0, min(t0, t1))
            hi = min(1.0, max(t0, t1))
            s = (lo + hi) / 2.0 if lo <= hi else float(np.clip(-e / a, 0.0, 1.0))

        p1 = a1 + s * d1
        t = (
            float(np.clip(np.dot(p1 - a2, d2) / c, 0.0, 1.0))
            if c > _PARALLEL_TOL
            else 0.0
        )
        p2 = a2 + t * d2
    else:
        # General (non-parallel) case: closed-form minimum with clamped parameters.
        # Clamp s, recompute t, clamp t, then recompute s once more.
        s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0))
        t = float(np.clip((b * s + f) / c, 0.0, 1.0)) if c > _PARALLEL_TOL else 0.0
        s = float(np.clip((-e + b * t) / a, 0.0, 1.0)) if a > _PARALLEL_TOL else 0.0
        p1 = a1 + s * d1
        p2 = a2 + t * d2

    d = float(np.linalg.norm(p1 - p2))
    return p1, p2, d


class Colony:
    """
    A collection of Cells living within an Environment.
    """

    def __init__(self, cells, environment, k=10.0, drag=1.0):
        """
        Args:
            cells: initial list of Cell objects.
            environment: the shared Environment.
            k: Hookean contact stiffness (force / length).
            drag: isotropic drag constant for contact dynamics.
                  Translational drag: ζ_t = drag * length.
                  Rotational drag:    ζ_r = (drag / 12) * length³.
        """
        self.cells = list(cells)
        self.environment = environment
        self.k = k
        self.drag = drag
        existing_ids = [cell.id for cell in self.cells if cell.id is not None]
        self._next_id = max(existing_ids, default=-1) + 1

    @property
    def living_cells(self):
        return [cell for cell in self.cells if cell.alive]

    def step(self, dt):
        """Advance all cells by one timestep, then enforce bounds and divisions."""
        for cell in self.cells:
            cell.step(dt)
        self.enforce_bounds()
        for cell in self.living_cells:
            self._apply_brownian_motion(cell, dt)
        self._apply_contact_forces(dt)
        self.handle_divisions()

    def _sample_field(self, field_array, position):
        """Nearest-grid-point lookup of a field array at a position in μm."""
        shape = self.environment.shape
        width, height = self.environment.bounds
        j = int(np.clip(position[0] / width * shape[1], 0, shape[1] - 1))
        i = int(np.clip(position[1] / height * shape[0], 0, shape[0] - 1))
        return field_array[i, j]

    def _apply_brownian_motion(self, cell, dt):
        """
        Apply overdamped-Langevin Brownian displacements to a living cell.

        Uses slender-body drag for an anisotropic rod. Local viscosity and
        diffusivity are sampled from the environment grid at the cell's
        position. Spatial units are μm; dt is in seconds.
        """
        eta = self._sample_field(self.environment.eta, cell.position)
        D_field = self._sample_field(self.environment.diffusivity, cell.position)

        # Scale viscosity by local diffusivity relative to the reference medium
        # (Stokes-Einstein: η ∝ 1/D at fixed temperature and probe size).
        from .environment import WATER_DIFFUSIVITY_37C

        eta_local = eta * (WATER_DIFFUSIVITY_37C / D_field)

        # Cell geometry in SI (m)
        L_eff = (cell.length + 2.0 * cell.radius) * 1e-6
        r = cell.radius * 1e-6
        ln_ratio = np.log(L_eff / r)  # ln(total length / radius), always > 0

        # Slender-body drag coefficients
        gamma_par = 2.0 * np.pi * eta_local * L_eff / ln_ratio  # kg/s
        gamma_perp = 4.0 * np.pi * eta_local * L_eff / ln_ratio  # kg/s
        gamma_rot = np.pi * eta_local * L_eff**3 / (3.0 * ln_ratio)  # kg·m²/s

        # Diffusion coefficients via Einstein relation
        D_par = _kBT / gamma_par  # m²/s
        D_perp = _kBT / gamma_perp  # m²/s
        D_rot = _kBT / gamma_rot  # rad²/s

        u = cell.orientation
        u_perp = np.array([-u[1], u[0]])
        xi_par, xi_perp, xi_rot = cell.rng.standard_normal(3)

        # Translational Brownian kick (convert m → μm)
        dr = u * xi_par * np.sqrt(2.0 * D_par * dt) + u_perp * xi_perp * np.sqrt(
            2.0 * D_perp * dt
        )
        cell.position += dr * 1e6

        # Rotational Brownian kick
        cell.apply_torque(xi_rot * np.sqrt(2.0 * D_rot / dt), dt)

    def _apply_contact_forces(self, dt):
        """
        Apply pairwise Hookean contact forces and torques to all living cells.

        For each overlapping cell pair (i, j):
          - overlap δ = R_i + R_j - d  (d: axis-segment distance)
          - force on i: F = k * δ * N  (N: contact normal from j toward i)
          - torque on i: τ = (p_c - c_i) × F  (2D scalar cross product)
        Dynamics are overdamped:
          Δc = F / ζ_t * dt,  ζ_t = drag * length
          Δθ = τ / ζ_r * dt,  ζ_r = (drag / 12) * length³
        """
        alive = self.living_cells
        n = len(alive)
        if n < 2:
            return

        forces = [np.zeros(2) for _ in range(n)]
        torques = [0.0] * n

        for i in range(n):
            ci = alive[i]
            ai = ci.position - (ci.length / 2.0) * ci.orientation
            bi = ci.position + (ci.length / 2.0) * ci.orientation

            for j in range(i + 1, n):
                cj = alive[j]
                aj = cj.position - (cj.length / 2.0) * cj.orientation
                bj = cj.position + (cj.length / 2.0) * cj.orientation

                pi, pj, d = _segment_segment_closest(ai, bi, aj, bj)
                delta = ci.radius + cj.radius - d

                if delta <= 0.0:
                    continue

                # Contact normal: unit vector from j toward i.
                # When axes coincide (d≈0), fall back to center-to-center direction,
                # then to ci's perpendicular if centers also coincide.
                if d > 1e-12:
                    N = (pi - pj) / d
                else:
                    sep = ci.position - cj.position
                    sep_norm = np.linalg.norm(sep)
                    if sep_norm > 1e-12:
                        N = sep / sep_norm
                    else:
                        N = np.array([-ci.orientation[1], ci.orientation[0]])

                F = self.k * delta * N
                pc = (pi + pj) / 2.0

                # 2D torque: τ = r_x * F_y - r_y * F_x
                ri = pc - ci.position
                rj = pc - cj.position

                forces[i] += F
                forces[j] -= F
                torques[i] += ri[0] * F[1] - ri[1] * F[0]
                torques[j] -= rj[0] * F[1] - rj[1] * F[0]  # F_ji = -F

        for i, cell in enumerate(alive):
            zeta_t = self.drag * cell.length
            zeta_r = (self.drag / 12.0) * cell.length**3
            cell.position += forces[i] / zeta_t * dt
            cell.apply_torque(torques[i] / zeta_r, dt)

    def enforce_bounds(self):
        """Kill any cell whose center of mass has left the environment bounds."""
        for cell in self.cells:
            if cell.alive and not self.environment.in_bounds(cell.position):
                cell.kill()

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
