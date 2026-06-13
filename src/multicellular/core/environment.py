# core/environment.py

import numpy as np


class Field:
    """
    A scalar field defined over a 2D grid.

    Represents something that varies across the extracellular space, e.g.
    a diffusible chemical's concentration, temperature, or surface
    roughness. The field's extent may be larger than the simulation
    bounds (e.g. to provide a margin for future diffusion dynamics).
    """

    def __init__(self, name, values):
        self.name = name
        self.values = np.asarray(values, dtype=float)

    @property
    def shape(self):
        return self.values.shape


class Environment:
    """
    Represents the shared extracellular space that cells inhabit.

    Holds a collection of named Fields (e.g. chemical concentrations,
    temperature, surface roughness), each represented as a matrix of
    values over a shared grid. Reaction-diffusion dynamics are not yet
    implemented; for now the environment is just a container for fields.
    """

    # Hardcoded simulation bounds: a 100um x 100um square (width, height).
    # Fields may have values outside of these bounds.
    BOUNDS = (100.0, 100.0)

    def __init__(self, shape, fields=None):
        self.shape = tuple(shape)
        self.fields = {}
        for field in fields or []:
            self.add_field(field)

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
        width, height = self.BOUNDS
        return 0 <= x <= width and 0 <= y <= height
