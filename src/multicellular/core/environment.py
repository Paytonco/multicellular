# core/environment.py

import numpy as np

# Diffusivity of water at 37°C in m²/s (physiological temperature).
WATER_DIFFUSIVITY_37C = 3.0e-9  # m²/s
WATER_VISCOSITY_37C = 6.9e-4  # Pa·s

# Default medium depth (μm): 2x the default Cell radius (0.5 μm), i.e. the
# height of a microfluidics monolayer chamber sized to a single cell.
MONOLAYER_DEPTH_DEFAULT = 1.0  # μm


class Field:
    """
    A scalar field defined over a 2D grid.

    Represents something that varies across the extracellular space, e.g.
    a diffusible chemical's concentration, temperature, or surface
    roughness. The field's extent may be larger than the simulation
    bounds (e.g. to provide a margin for future diffusion dynamics).
    """

    def __init__(
        self, name, values, is_chemical=False, diffuses=False, diffusivity=None
    ):
        self.name = name
        self.values = np.asarray(values, dtype=float)
        self.is_chemical = is_chemical
        self.diffuses = diffuses
        if diffuses:
            if diffusivity is None:
                raise ValueError(
                    f"Field '{name}' has diffuses=True but no diffusivity was given."
                )
            diffusivity = float(diffusivity)
            if diffusivity <= 0:
                raise ValueError(
                    f"Field '{name}' diffusivity must be positive, got {diffusivity}."
                )
        self.diffusivity = diffusivity

    @property
    def shape(self):
        return self.values.shape


class Environment:
    """
    Represents the shared extracellular space that cells inhabit.

    Holds a collection of named Fields (e.g. chemical concentrations,
    temperature, surface roughness), each represented as a matrix of
    values over a shared grid. Fields marked `diffuses=True` are advanced
    each timestep by `diffuse()`; reaction-diffusion coupling within the
    field grid itself (as opposed to inside cells) is not implemented.
    """

    def __init__(
        self,
        name,
        shape,
        bounds=(100.0, 100.0),
        diffusivity=None,
        eta=None,
        fields=None,
        depth=None,
    ):
        self.name = name
        self.shape = tuple(shape)
        self.bounds = tuple(bounds)
        self.diffusivity = (
            np.full(self.shape, WATER_DIFFUSIVITY_37C)
            if diffusivity is None
            else np.asarray(diffusivity, dtype=float)
        )
        self.eta = (
            np.full(self.shape, WATER_VISCOSITY_37C)
            if eta is None
            else np.asarray(eta, dtype=float)
        )
        self.depth = MONOLAYER_DEPTH_DEFAULT if depth is None else float(depth)
        if self.depth <= 0:
            raise ValueError(f"Environment depth must be positive, got {self.depth}.")
        self.fields = {}
        for field in fields or []:
            self.add_field(field)

    @property
    def grid_cell_volume(self):
        """
        Physical volume (μm³) of a single grid cell: dx * dy * depth.

        No meter conversion here (unlike `diffuse`, which works in meters
        because `diffusivity` is in m²/s) — this stays in μm³ to match
        `Cell.compute_volume()` directly, since Field concentrations are on
        the same volumetric basis as Cell concentrations (see
        `Colony.apply_chemical_fields`'s direct-copy convention).
        """
        dx = self.bounds[0] / self.shape[1]
        dy = self.bounds[1] / self.shape[0]
        return dx * dy * self.depth

    def add_field(self, field):
        """Add a Field to the environment, validating its grid shape."""
        if field.shape != self.shape:
            raise ValueError(
                f"Field '{field.name}' has shape {field.shape}, but "
                f"environment requires shape {self.shape}."
            )
        self.fields[field.name] = field

    def get_field(self, name):
        return self.fields[name]

    def in_bounds(self, position):
        """Check whether a 2D position lies within the environment bounds."""
        x, y = position[0], position[1]
        width, height = self.bounds
        return 0 <= x <= width and 0 <= y <= height

    def diffuse(self, dt):
        """
        Advance every diffusive field by dt under the 2D diffusion equation
        ∂C/∂t = D ∇²C, using an explicit centered-difference (FTCS) scheme
        with no-flux (Neumann) boundaries, so each field's total amount is
        conserved rather than leaking out at the grid edges.

        FTCS is only stable for dt <= dx²dy² / (2D(dx²+dy²)); rather than
        require the caller to pick a small enough dt, each field is advanced
        in however many equal sub-steps are needed to satisfy that bound.
        """
        diffusive_fields = [field for field in self.fields.values() if field.diffuses]
        if not diffusive_fields:
            return

        # Grid spacing in meters (bounds/positions are in μm; diffusivity is m²/s).
        dx = self.bounds[0] / self.shape[1] * 1e-6
        dy = self.bounds[1] / self.shape[0] * 1e-6
        inv_dx2 = 1.0 / dx**2
        inv_dy2 = 1.0 / dy**2

        for field in diffusive_fields:
            D = field.diffusivity
            stable_dt = 0.5 / (D * (inv_dx2 + inv_dy2))
            n_steps = max(1, int(np.ceil(dt / stable_dt)))
            sub_dt = dt / n_steps

            C = field.values
            for _ in range(n_steps):
                padded = np.pad(C, 1, mode="edge")
                laplacian = (
                    padded[2:, 1:-1] - 2.0 * padded[1:-1, 1:-1] + padded[:-2, 1:-1]
                ) * inv_dy2 + (
                    padded[1:-1, 2:] - 2.0 * padded[1:-1, 1:-1] + padded[1:-1, :-2]
                ) * inv_dx2
                C = C + D * sub_dt * laplacian
            field.values = C
