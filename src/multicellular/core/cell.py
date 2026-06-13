# core/cell.py

import numpy as np


class Cell:
    """
    Represents a rod-shaped bacterial cell with
    cylindrical body and spherical end caps.
    """

    # Above this copy number, partition by sampling from the Gaussian
    # approximation to Binomial(n, 1/2) (CLT) instead of the binomial itself.
    LOW_COPY_GAUSSIAN_THRESHOLD = 35

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

        # Chemical species (by name, from self.concentrations) whose copy
        # number should be partitioned stochastically at division rather
        # than split deterministically by concentration. Use this for
        # low-copy molecules (e.g. plasmids) where stochastic partitioning
        # matters. Designate a species as low-copy via set_concentration().
        self.low_copy_species = set()

        # Default initial concentrations: all zero if network exists
        self.concentrations = {s: 0.0 for s in network.species} if network else {}

    def compute_volume(self):
        """Compute volume of cylindrical rod with hemispherical caps."""
        cylinder_volume = np.pi * self.radius**2 * self.length
        cap_volume = (4 / 3) * np.pi * self.radius**3
        return cylinder_volume + cap_volume

    def copy_number(self, chemical_species):
        """Number of molecules of a chemical species currently in the cell."""
        return self.concentrations.get(chemical_species, 0.0) * self.compute_volume()

    def set_concentration(self, chemical_species, value, low_copy=False):
        """
        Initialize/set the concentration of a chemical species.

        If low_copy is True, the species is designated to have its copy
        number partitioned stochastically (binomial, or its Gaussian
        approximation for large copy numbers) between daughter cells at
        division, rather than having its concentration conserved.
        """
        self.concentrations[chemical_species] = value
        if low_copy:
            self.low_copy_species.add(chemical_species)
        else:
            self.low_copy_species.discard(chemical_species)

    def grow(self, dt, growth_rate=0.5):
        """Increase length linearly and age the cell."""
        self.length += growth_rate * dt
        self.age += dt

    def step(self, dt):
        """Advance cell internal state (chemical + growth). No-op if dead."""
        if not self.alive:
            return
        if self.network:
            self.concentrations = self.network.simulate_step(
                self.concentrations, dt, self.compute_volume()
            )
        self.grow(dt)

    def ready_to_divide(self, threshold_length=4.0):
        """Check if cell should divide based on length threshold."""
        return self.length >= threshold_length

    def _partition_copies(self, n):
        """
        Draw the number of copies (of n total) inherited by one daughter
        cell, with the other daughter receiving the remaining n - x.

        For small n, sample directly from Binomial(n, 1/2). For large n,
        sample from its Gaussian approximation (CLT) instead, which is
        much cheaper and statistically indistinguishable. Either way, the
        result is rounded to a whole number and clamped to [0, n] so that
        copy number is conserved exactly.
        """
        if n <= self.LOW_COPY_GAUSSIAN_THRESHOLD:
            return int(self.rng.binomial(n, 0.5))

        x = self.rng.normal(loc=n / 2.0, scale=np.sqrt(n) / 2.0)
        return int(np.clip(round(x), 0, n))

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

        daughter1.low_copy_species = set(self.low_copy_species)
        daughter2.low_copy_species = set(self.low_copy_species)

        daughter_volume = daughter1.compute_volume()
        conc1 = {}
        conc2 = {}
        for chemical_species, conc in self.concentrations.items():
            if chemical_species in self.low_copy_species:
                n = int(round(self.copy_number(chemical_species)))
                x = self._partition_copies(n)
                conc1[chemical_species] = x / daughter_volume
                conc2[chemical_species] = (n - x) / daughter_volume
            else:
                conc1[chemical_species] = conc
                conc2[chemical_species] = conc

        daughter1.concentrations = conc1
        daughter2.concentrations = conc2

        return daughter1, daughter2

    def apply_force(self, force_vector, dt):
        """Move the cell based on external forces. No-op if dead."""
        if not self.alive:
            return
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
