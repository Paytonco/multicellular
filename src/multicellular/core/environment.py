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

    The environment's terrain is given by `wall_map`, a matrix (scaled to
    `size`) whose entries are one of:
      -1: out of bounds — a cell whose center strays here is killed.
       0: media — cells move and react freely.
       1: wall — cells feel a Hookean repulsion (see `Colony`, wallSpec.txt).
    `wall_map.shape` becomes the environment's grid `shape`, the same grid
    every Field must match.
    """

    def __init__(
        self,
        name,
        wall_map,
        size=(100.0, 100.0),
        diffusivity=None,
        eta=None,
        fields=None,
        depth=None,
    ):
        self.name = name
        wall_map = np.asarray(wall_map)
        if wall_map.ndim != 2:
            raise ValueError(
                f"wall_map must be a 2D matrix, got shape {wall_map.shape}."
            )
        invalid = ~np.isin(wall_map, (-1, 0, 1))
        if invalid.any():
            raise ValueError(
                "wall_map entries must be -1 (out of bounds), 0 (media), or "
                f"1 (wall); got other values {np.unique(wall_map[invalid]).tolist()}."
            )
        self.wall_map = wall_map.astype(int)
        self.shape = self.wall_map.shape
        self.size = tuple(size)
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
        self.wall_faces, self.wall_corners = _build_wall_geometry(
            self.wall_map, self.size
        )

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
        dx = self.size[0] / self.shape[1]
        dy = self.size[1] / self.shape[0]
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

    def grid_indices(self, positions):
        """
        Vectorized nearest grid-cell (row, col) indices for an (n, 2) array
        of positions in μm, on this environment's grid (shared by every
        Field and by `wall_map`).
        """
        positions = np.asarray(positions, dtype=float)
        width, height = self.size
        j = np.clip(
            (positions[:, 0] / width * self.shape[1]).astype(int), 0, self.shape[1] - 1
        )
        i = np.clip(
            (positions[:, 1] / height * self.shape[0]).astype(int), 0, self.shape[0] - 1
        )
        return i, j

    def in_bounds(self, position):
        """
        Check whether a 2D position lies within the environment's physical
        extent and is not sitting in a `wall_map` out-of-bounds (-1) cell.
        Walls (1) are still in bounds — cells are pushed off them by
        contact forces rather than killed.
        """
        x, y = position[0], position[1]
        width, height = self.size
        if not (0 <= x <= width and 0 <= y <= height):
            return False
        i, j = self.grid_indices(np.array([[x, y]]))
        return bool(self.wall_map[i[0], j[0]] != -1)

    def diffuse(self, dt):
        """
        Advance every diffusive field by dt under the 2D diffusion equation
        ∂C/∂t = D ∇²C, using an explicit centered-difference (FTCS) scheme
        with no-flux (Neumann) boundaries at both the grid edges and any
        `wall_map` wall (1) cell, so each field's total amount is conserved
        rather than leaking out at the grid edges or into solid walls.

        A wall neighbor blocks flux across that face exactly like the domain
        edge already does: the neighbor's (physically meaningless) stored
        value is replaced with the cell's own value, zeroing that face's
        contribution to the Laplacian, rather than letting mass flow into or
        out of a cell that has no medium in it. Wall cells themselves are
        excluded from the update entirely (they never accumulate mass),
        which is what keeps the discrete flux exactly antisymmetric between
        every pair of interacting cells and so conserves total mass. Only
        walls (1) block diffusion this way — out-of-bounds (-1) cells are
        still ordinary diffusive medium.

        FTCS is only stable for dt <= dx²dy² / (2D(dx²+dy²)); rather than
        require the caller to pick a small enough dt, each field is advanced
        in however many equal sub-steps are needed to satisfy that bound.
        """
        diffusive_fields = [field for field in self.fields.values() if field.diffuses]
        if not diffusive_fields:
            return

        # Grid spacing in meters (size/positions are in μm; diffusivity is m²/s).
        dx = self.size[0] / self.shape[1] * 1e-6
        dy = self.size[1] / self.shape[0] * 1e-6
        inv_dx2 = 1.0 / dx**2
        inv_dy2 = 1.0 / dy**2

        is_wall = self.wall_map == 1
        wall_padded = np.pad(is_wall, 1, mode="edge")
        up_wall = wall_padded[2:, 1:-1]
        down_wall = wall_padded[:-2, 1:-1]
        right_wall = wall_padded[1:-1, 2:]
        left_wall = wall_padded[1:-1, :-2]

        for field in diffusive_fields:
            D = field.diffusivity
            stable_dt = 0.5 / (D * (inv_dx2 + inv_dy2))
            n_steps = max(1, int(np.ceil(dt / stable_dt)))
            sub_dt = dt / n_steps

            C = field.values
            for _ in range(n_steps):
                padded = np.pad(C, 1, mode="edge")
                up = np.where(up_wall, C, padded[2:, 1:-1])
                down = np.where(down_wall, C, padded[:-2, 1:-1])
                right = np.where(right_wall, C, padded[1:-1, 2:])
                left = np.where(left_wall, C, padded[1:-1, :-2])

                laplacian = (up - 2.0 * C + down) * inv_dy2 + (
                    right - 2.0 * C + left
                ) * inv_dx2
                C = np.where(is_wall, C, C + D * sub_dt * laplacian)
            field.values = C


# Coordinate tolerance (μm) for treating two wall-face endpoints, or two
# contiguous exposed pixel edges, as touching.
_WALL_GEOMETRY_TOL = 1e-9


def _build_wall_geometry(wall_map, size):
    """
    Decompose a rasterized `wall_map` into the flat faces and corner points
    wallSpec.txt's contact model needs, per wall cell (matrix entry == 1):
    every side bordering a media cell is an exposed edge. Collinear exposed
    edges along the same wall-map row/column boundary are merged into a
    single maximal flat face — merging matters because a straight run of
    wall pixels must read as one long face, not one tiny face per pixel
    with a spurious corner at every pixel seam.

    Each merged face's own endpoints are then a genuine geometric corner
    (a turn, a rounded tip, or the free end of a wall run) — collected,
    deduplicated by coincident location, as corner points.

    A wall cell's edge is *not* exposed against another wall cell (fully
    interior wall pixels contribute no geometry, since cells can never
    reach them), against an out-of-bounds (-1) cell, or against the grid's
    outer boundary. The latter two both terminate a face with no primitive
    at all, same as an open end of a wall run: a -1 cell already kills any
    cell whose center reaches it (see `in_bounds`), and there is no medium
    beyond the grid's edge to push a cell back into, so generating a
    repelling face there would be both moot and, for a wall pixel that
    borders out-of-bounds on a different side than its media-facing side
    (e.g. the trap wall of a mother-machine-style open channel), actively
    wrong — it would push cells along that unrelated face instead of
    leaving that direction untouched.

    Returns:
        faces: list of (x0, y0, x1, y1, nx, ny) tuples — a finite segment
            from (x0, y0) to (x1, y1) with inward unit normal (nx, ny).
        corners: list of (x, y) points.
    """
    n_rows, n_cols = wall_map.shape
    width, height = size
    dx = width / n_cols
    dy = height / n_rows

    def is_media(i, j):
        return 0 <= i < n_rows and 0 <= j < n_cols and wall_map[i, j] == 0

    # Raw exposed pixel edges, keyed by which side of the wall cell they're
    # on. Horizontal edges (south/north) vary along x at a fixed y; vertical
    # edges (west/east) vary along y at a fixed x.
    raw = {"S": [], "N": [], "W": [], "E": []}
    for i in range(n_rows):
        for j in range(n_cols):
            if wall_map[i, j] != 1:
                continue
            x0, x1 = j * dx, (j + 1) * dx
            y0, y1 = i * dy, (i + 1) * dy
            if is_media(i - 1, j):
                raw["S"].append((y0, x0, x1))
            if is_media(i + 1, j):
                raw["N"].append((y1, x0, x1))
            if is_media(i, j - 1):
                raw["W"].append((x0, y0, y1))
            if is_media(i, j + 1):
                raw["E"].append((x1, y0, y1))

    faces = []
    corners = set()

    def merge(entries, normal, horizontal):
        groups = {}
        for fixed, lo, hi in entries:
            groups.setdefault(fixed, []).append((lo, hi))
        for fixed, spans in groups.items():
            spans.sort()
            cur_lo, cur_hi = spans[0]
            merged = []
            for lo, hi in spans[1:]:
                if lo <= cur_hi + _WALL_GEOMETRY_TOL:
                    cur_hi = max(cur_hi, hi)
                else:
                    merged.append((cur_lo, cur_hi))
                    cur_lo, cur_hi = lo, hi
            merged.append((cur_lo, cur_hi))

            for lo, hi in merged:
                if horizontal:
                    x0, x1, y = lo, hi, fixed
                    faces.append((x0, y, x1, y, normal[0], normal[1]))
                    corners.add((round(x0, 9), round(y, 9)))
                    corners.add((round(x1, 9), round(y, 9)))
                else:
                    y0, y1, x = lo, hi, fixed
                    faces.append((x, y0, x, y1, normal[0], normal[1]))
                    corners.add((round(x, 9), round(y0, 9)))
                    corners.add((round(x, 9), round(y1, 9)))

    merge(raw["S"], (0.0, -1.0), horizontal=True)
    merge(raw["N"], (0.0, 1.0), horizontal=True)
    merge(raw["W"], (-1.0, 0.0), horizontal=False)
    merge(raw["E"], (1.0, 0.0), horizontal=False)

    return faces, [np.array(c) for c in corners]
