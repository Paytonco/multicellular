# core/cell.py

import math

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
        growth_rate=np.log(2),
        growth_rate_law=None,
        delta_bar=1.0,
        cv_delta=0.10,
        a=1.0,
        cv_f=0.0,
    ):
        """
        Args:
            growth_rate: Exponential elongation rate λ. Length evolves as
                L(t) = L_b * exp(λ * t). Default ln(2) gives a mean doubling
                time of 1 time unit.
            delta_bar: Mean added length per generation (ΔL̄). Sets the size
                scale: at steady state the mean birth length is delta_bar and
                the mean division length is 2 * delta_bar.
            cv_delta: Coefficient of variation of the added increment. This is
                the primary stochasticity parameter; must be > 0 for the adder
                model. Literature range for E. coli: 0.10–0.15.
            a: Size-control strategy knob. a=1 (default) is a pure adder;
                a=0 is a sizer; a=2 approximates a timer.
            cv_f: CV of the division fraction f ~ Normal(0.5, cv_f * 0.5),
                clipped to (0, 1). Default 0.0 gives a deterministic symmetric
                split. Set ~0.03 to enable partition noise.
        """
        self.id = id
        self.position = np.array(position, dtype=float)
        self.orientation = self._normalize(np.array(orientation, dtype=float))
        self.length = length
        self.radius = radius
        self.species = species
        self.network = network
        self.age = 0.0
        self.alive = True
        self.rng = rng or np.random.default_rng()

        self.growth_rate = growth_rate
        self.growth_rate_law = growth_rate_law
        self.delta_bar = delta_bar
        self.cv_delta = cv_delta
        self.a = a
        self.cv_f = cv_f

        self.length_at_birth = length
        self._division_target = self._sample_division_target()

        self.low_copy_species = set()
        self.concentrations = {s: 0.0 for s in network.species} if network else {}
        # Molecule counts exported (e.g. secreted) by the most recent step,
        # awaiting deposit into the matching Field by Colony.export_chemical_fields.
        self.pending_export = {}

    def _sample_division_target(self):
        """
        Sample the division target for this generation.

        L_d = a * L_b + Delta,  Delta ~ Normal(delta_bar, cv_delta * delta_bar).
        Resamples until Delta > 0 to keep the target above the birth size.
        """
        while True:
            delta = self.rng.normal(self.delta_bar, self.cv_delta * self.delta_bar)
            if delta > 0:
                break
        return self.a * self.length_at_birth + delta

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

    def grow(self, dt):
        """
        Grow exponentially by one timestep, age the cell, and dilute
        concentrations accordingly.

        Growth changes volume but not molecule count, so each species'
        copy number (concentration * volume) is held fixed across the
        volume change: copy numbers are computed at the pre-growth volume,
        then divided by the post-growth volume to get diluted
        concentrations. Reactions (run separately, in `step`) are the only
        thing that changes copy number.
        """
        copy_numbers = {s: self.copy_number(s) for s in self.concentrations}
        self.length *= np.exp(self.growth_rate * dt)
        self.age += dt
        volume = self.compute_volume()
        self.concentrations = {s: n / volume for s, n in copy_numbers.items()}

    def step(self, dt):
        """Advance cell internal state (chemical + growth). No-op if dead."""
        if not self.alive:
            return
        if self.network:
            self.concentrations = self.network.simulate_step(
                self.concentrations, dt, self.compute_volume(), rng=self.rng
            )
            self.pending_export = self.network.last_exported
        self.grow(dt)

    def ready_to_divide(self):
        """Return True if the cell has reached its sampled division target."""
        return self.length >= self._division_target

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
        """
        Split into two daughter cells along the longitudinal axis.

        Daughter lengths are f * L_d and (1 - f) * L_d, where L_d is this
        cell's sampled division target and f is the division fraction (0.5 by
        default; optionally noisy when cv_f > 0). Each daughter samples its
        own division target at construction, continuing the adder lineage.
        """
        if not self.ready_to_divide():
            return None

        if self.cv_f > 0:
            f = float(np.clip(self.rng.normal(0.5, self.cv_f * 0.5), 1e-6, 1.0 - 1e-6))
        else:
            f = 0.5

        L_d = self._division_target
        length1 = f * L_d
        length2 = (1.0 - f) * L_d

        # Place daughters so their surfaces just touch at the division plane.
        pos1 = self.position - ((1.0 - f) * L_d / 2.0 + self.radius) * self.orientation
        pos2 = self.position + (f * L_d / 2.0 + self.radius) * self.orientation

        daughter_kw = dict(
            id=None,
            orientation=self.orientation,
            radius=self.radius,
            species=self.species,
            rng=self.rng,
            growth_rate=self.growth_rate,
            growth_rate_law=self.growth_rate_law,
            delta_bar=self.delta_bar,
            cv_delta=self.cv_delta,
            a=self.a,
            cv_f=self.cv_f,
        )
        daughter1 = Cell(
            position=pos1,
            length=length1,
            network=self.network.clone() if self.network else None,
            **daughter_kw,
        )
        daughter2 = Cell(
            position=pos2,
            length=length2,
            network=self.network.clone() if self.network else None,
            **daughter_kw,
        )

        daughter1.low_copy_species = set(self.low_copy_species)
        daughter2.low_copy_species = set(self.low_copy_species)

        vol1 = daughter1.compute_volume()
        vol2 = daughter2.compute_volume()
        conc1 = {}
        conc2 = {}
        for chemical_species, conc in self.concentrations.items():
            if chemical_species in self.low_copy_species:
                n = int(round(self.copy_number(chemical_species)))
                x = self._partition_copies(n)
                conc1[chemical_species] = x / vol1
                conc2[chemical_species] = (n - x) / vol2
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

    def apply_torque(self, omega, dt):
        """Rotate the cell by omega * dt radians. No-op if dead."""
        if not self.alive:
            return
        angle = omega * dt
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        x, y = float(self.orientation[0]), float(self.orientation[1])
        nx = cos_a * x - sin_a * y
        ny = sin_a * x + cos_a * y
        norm = math.hypot(nx, ny)
        if norm != 0.0:
            nx, ny = nx / norm, ny / norm
        else:
            nx, ny = 1.0, 0.0
        # Mutate in place (each cell owns its own orientation array; see
        # __init__, where np.array(...) always copies) to avoid allocating a
        # new array on every torque application, which happens every step.
        self.orientation[0] = nx
        self.orientation[1] = ny

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
