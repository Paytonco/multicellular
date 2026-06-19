# core/colony.py

import numpy as np

_kBT = 1.380649e-23 * 310.15  # J  (k_B × 37°C)


class Colony:
    """
    A collection of Cells living within an Environment.
    """

    def __init__(self, cells, environment):
        self.cells = list(cells)
        self.environment = environment
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
        self.handle_divisions()

    def _sample_field(self, field_array, position):
        """Nearest-grid-point lookup of a field array at a position in μm."""
        shape = self.environment.shape
        width, height = self.environment.BOUNDS
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
