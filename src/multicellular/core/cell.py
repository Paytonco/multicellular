# core/cell.py

import numpy as np


class Cell:
    """
    Represents a rod-shaped bacterial cell with
    cylindrical body and spherical end caps.
    """

    def __init__(
        self,
        id,
        position,
        orientation,
        length=2.0,
        radius=0.5,
        species="default",
        network=None,
        rng=None,
    ):
        self.id = id
        self.position = np.array(position, dtype=float)  # Center of mass
        self.orientation = self._normalize(
            np.array(orientation, dtype=float)
        )  # Unit vector
        self.length = length  # Length of cylindrical portion
        self.radius = radius  # Radius of cylinder and caps (constant)
        self.species = species
        self.network = network
        self.age = 0.0
        self.alive = True
        self.rng = rng or np.random.default_rng()

        # Default initial concentrations: all zero if network exists
        self.concentrations = {s: 0.0 for s in network.species} if network else {}

    def compute_volume(self):
        """Compute volume of cylindrical rod with hemispherical caps."""
        cylinder_volume = np.pi * self.radius**2 * self.length
        cap_volume = (4 / 3) * np.pi * self.radius**3
        return cylinder_volume + cap_volume

    def grow(self, dt, growth_rate=0.5):
        """Increase length linearly and age the cell."""
        self.length += growth_rate * dt
        self.age += dt

    def step(self, dt):
        """Advance cell internal state (chemical + growth)."""
        if self.network:
            self.concentrations = self.network.simulate_step(
                self.concentrations, dt, self.compute_volume()
            )
        self.grow(dt)

    def ready_to_divide(self, threshold_length=4.0):
        """Check if cell should divide based on length threshold."""
        return self.length >= threshold_length

    def divide(self):
        """Split into two daughter cells along the longitudinal axis."""
        if not self.ready_to_divide():
            return None

        daughter_length = self.length / 2.0
        offset = (daughter_length / 2.0 + self.radius) * self.orientation

        pos1 = self.position - offset
        pos2 = self.position + offset

        daughter1 = Cell(
            id=None,
            position=pos1,
            orientation=self.orientation,
            length=daughter_length,
            radius=self.radius,
            species=self.species,
            network=self.network.clone() if self.network else None,
            rng=self.rng,
        )
        daughter2 = Cell(
            id=None,
            position=pos2,
            orientation=self.orientation,
            length=daughter_length,
            radius=self.radius,
            species=self.species,
            network=self.network.clone() if self.network else None,
            rng=self.rng,
        )

        daughter1.concentrations = self.concentrations.copy()
        daughter2.concentrations = self.concentrations.copy()

        return daughter1, daughter2

    def apply_force(self, force_vector, dt):
        """Move the cell based on external forces."""
        self.position += force_vector * dt

    def interact_with_environment(self, environment):
        """Placeholder for environment interaction logic."""
        pass

    def kill(self):
        self.alive = False

    def _normalize(self, v):
        """Ensure orientation is a unit vector."""
        norm = np.linalg.norm(v)
        return v / norm if norm != 0 else np.array([1.0, 0.0])

    def to_dict(self):
        return {
            "id": self.id,
            "position": self.position.tolist(),
            "orientation": self.orientation.tolist(),
            "length": self.length,
            "radius": self.radius,
            "volume": self.compute_volume(),
            "species": self.species,
            "age": self.age,
            "alive": self.alive,
            "concentrations": self.concentrations.copy(),
        }
