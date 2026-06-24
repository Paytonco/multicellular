# core/reactions.py

from typing import Callable, Dict, List, Union

import numpy as np


class Reaction:
    """
    Represents a chemical reaction with arbitrary stoichiometry, optional catalysts, and customizable rate laws.

    Example:
        Reaction(
            reactants={"A": 1, "B": 2},
            products={"C": 1},
            catalysts=["E"],
            rate_law_type="mass_action",
            rate_params={"k": 0.1}
        )
    """

    def __init__(
        self,
        reactants: Dict[str, int],
        products: Dict[str, int],
        catalysts: List[str] = None,
        rate_law_type: str = "mass_action",
        rate_params: Dict[str, Union[float, int]] = None,
        custom_rate_law: Callable[
            [
                Dict[str, float],
                Dict[str, float],
                Dict[str, float],
                Dict[str, Union[float, int]],
            ],
            float,
        ] = None,
    ):
        self.reactants = reactants  # e.g., {"A": 1, "B": 2}
        self.products = products  # e.g., {"C": 1}
        self.catalysts = catalysts if catalysts else []  # e.g., ["E"]
        self.rate_law_type = rate_law_type
        self.rate_params = rate_params or {}
        self.custom_rate_law = custom_rate_law  # Optional user-defined function

        self._validate()

    def _validate(self):
        if self.rate_law_type == "custom" and self.custom_rate_law is None:
            raise ValueError(
                "Custom rate law type specified but no custom_rate_law function provided."
            )

    def rate(self, concentrations: Dict[str, float]) -> float:
        """
        Evaluate reaction rate based on current concentrations.

        Args:
            concentrations: Dict of species concentrations (species name → concentration)

        Returns:
            Reaction rate (float)
        """
        reactant_conc = {s: concentrations.get(s, 0.0) for s in self.reactants}
        product_conc = {s: concentrations.get(s, 0.0) for s in self.products}
        catalyst_conc = {s: concentrations.get(s, 0.0) for s in self.catalysts}

        if self.rate_law_type == "mass_action":
            return self._mass_action_rate(reactant_conc)

        elif self.rate_law_type == "michaelis_menten":
            return self._michaelis_menten_rate(reactant_conc, catalyst_conc)

        elif self.rate_law_type == "hill_langmuir":
            return self._hill_langmuir_rate(reactant_conc, catalyst_conc)

        elif self.rate_law_type == "custom" and self.custom_rate_law:
            return self.custom_rate_law(
                reactant_conc, product_conc, catalyst_conc, self.rate_params
            )

        else:
            raise ValueError(f"Unknown rate law type: {self.rate_law_type}")

    def _mass_action_rate(self, reactant_conc: Dict[str, float]) -> float:
        k = self.rate_params.get("k", 1.0)
        rate = k
        for species, stoich in self.reactants.items():
            rate *= reactant_conc.get(species, 0.0) ** stoich
        return rate

    def _michaelis_menten_rate(
        self, reactant_conc: Dict[str, float], catalyst_conc: Dict[str, float]
    ) -> float:
        S = list(reactant_conc.values())[0] if reactant_conc else 0.0
        E = (
            list(catalyst_conc.values())[0] if catalyst_conc else 1.0
        )  # Default catalyst concentration = 1.0 if missing

        Vmax = self.rate_params.get("Vmax", 1.0)
        Km = self.rate_params.get("Km", 1.0)

        return Vmax * E * S / (Km + S)

    def _hill_langmuir_rate(
        self, reactant_conc: Dict[str, float], catalyst_conc: Dict[str, float]
    ) -> float:
        E = catalyst_conc.get(self.catalysts[0], 0.0)
        z = catalyst_conc.get(self.catalysts[1], 0.0)

        alpha = self.rate_params["alpha"]
        beta = self.rate_params["beta"]
        C = self.rate_params["C"]
        n = self.rate_params["n"]  # Hill coefficient

        Cz_n = C * z**n
        return beta * E * (1 + alpha * Cz_n) / (1 + Cz_n)

    def get_stoichiometry_vector(self, species_list: List[str]) -> np.ndarray:
        """
        Return net stoichiometric change vector for this reaction for a given species ordering.

        Args:
            species_list: List of all species in the system.

        Returns:
            NumPy array of stoichiometric changes (products - reactants)
        """
        delta = np.zeros(len(species_list))
        for i, species in enumerate(species_list):
            delta[i] += self.products.get(species, 0) - self.reactants.get(species, 0)
        return delta

    def clone(self):
        """Return a deep copy of the reaction (for cell division use cases)."""
        return Reaction(
            reactants=self.reactants.copy(),
            products=self.products.copy(),
            catalysts=self.catalysts.copy(),
            rate_law_type=self.rate_law_type,
            rate_params=self.rate_params.copy(),
            custom_rate_law=self.custom_rate_law,
        )


class ReactionNetwork:
    """
    Represents a chemical reaction network with associated species and reactions.
    """

    def __init__(
        self, name: str, reactions: Dict[str, Reaction], simulation_method: str = "ODE"
    ):
        self.name = name
        self.reactions = reactions  # dict of name → Reaction
        self.simulation_method = simulation_method.upper()  # ODE, SSA, CLE
        self.species = self._extract_species()

        # Species/reaction ordering and the stoichiometry matrix are fixed for
        # the lifetime of this network (reactions/species don't change after
        # construction), so compute them once instead of on every ODE step.
        self._reaction_names = list(self.reactions.keys())
        self._stoichiometry_matrix = self.get_stoichiometry_matrix(
            self.species, self._reaction_names
        )

    def _extract_species(self) -> List[str]:
        species_set = set()
        for reaction in self.reactions.values():
            species_set.update(reaction.reactants)
            species_set.update(reaction.products)
            species_set.update(reaction.catalysts)
        return sorted(species_set)

    def get_stoichiometry_matrix(
        self, species_list: List[str], reaction_list: List[str]
    ) -> np.ndarray:
        """
        Return net stoichiometric change matrix for this reaction network for a given species and reaction ordering.

        Args:
            species_list: List of all species in the system.
            reaction_list: List of all reactions in the system.

        Returns:
            NumPy array of stoichiometric changes (products - reactants).
            Columns = reactions; Rows = species
        """
        S = np.zeros((len(species_list), len(reaction_list)))
        for j, rxn_name in enumerate(reaction_list):
            rxn = self.reactions[rxn_name]
            S[:, j] = rxn.get_stoichiometry_vector(species_list)
        return S

    def simulate_step(
        self,
        state: Dict[str, float],
        dt: float,
        volume: float,
        rng: np.random.Generator = None,
    ) -> Dict[str, float]:
        """
        Advance the chemical state by one time step using the specified simulation method.

        Args:
            state: dict of species → concentration
            dt: timestep size
            volume: cell volume (used for propensities in SSA/CLE)
            rng: random generator used by SSA/CLE (ignored by ODE). Defaults
                to a fresh `np.random.default_rng()` if not given.

        Returns:
            Updated concentration dictionary
        """
        method = self.simulation_method
        if method == "ODE":
            return self._simulate_ode_step(state, dt)
        elif method == "SSA":
            return self._simulate_ssa_step(
                state, dt, volume, rng or np.random.default_rng()
            )
        elif method == "CLE":
            return self._simulate_cle_step(
                state, dt, volume, rng or np.random.default_rng()
            )
        else:
            raise ValueError(f"Unknown simulation method: {method}")

    def _rate_vector(self, state: Dict[str, float]) -> np.ndarray:
        """Evaluate every reaction's rate law against `state` (a concentration dict)."""
        return np.array(
            [self.reactions[rxn_name].rate(state) for rxn_name in self._reaction_names]
        )

    def _simulate_ode_step(
        self, state: Dict[str, float], dt: float
    ) -> Dict[str, float]:
        """
        Simple forward Euler ODE step.
        """
        species_list = self.species
        S = self._stoichiometry_matrix

        v = self._rate_vector(state)  # Reaction rate vector (concentration/time)

        x = np.array([state.get(s, 0.0) for s in species_list])
        dxdt = S @ v  # Matrix multiplication
        x_new = x + dt * dxdt

        return {
            s: max(x_new[i], 0.0) for i, s in enumerate(species_list)
        }  # Prevent negatives

    def _simulate_ssa_step(
        self,
        state: Dict[str, float],
        dt: float,
        volume: float,
        rng: np.random.Generator,
    ) -> Dict[str, float]:
        """
        Gillespie direct-method SSA step, advancing exactly `dt`.

        Reaction rate laws are defined in concentration space (same as ODE),
        so propensities (molecules/time) are obtained by evaluating each
        rate law against counts/volume and scaling by `volume`. Internally
        this works in integer molecule counts; the incoming/outgoing state
        is still a concentration dict, converted at the boundary via
        `volume`, so callers (Cell.step) need no special handling.
        """
        species_list = self.species
        S = self._stoichiometry_matrix
        n_reactions = len(self._reaction_names)

        counts = np.array(
            [max(0, round(state.get(s, 0.0) * volume)) for s in species_list]
        )

        t = 0.0
        while t < dt:
            conc = {s: counts[i] / volume for i, s in enumerate(species_list)}
            propensities = np.maximum(self._rate_vector(conc), 0.0) * volume
            a0 = propensities.sum()

            if a0 <= 0.0:
                break

            tau = rng.exponential(1.0 / a0)
            if t + tau > dt:
                break

            j = rng.choice(n_reactions, p=propensities / a0)
            counts = np.maximum(counts + S[:, j], 0.0)
            t += tau

        return {s: counts[i] / volume for i, s in enumerate(species_list)}

    def _simulate_cle_step(
        self,
        state: Dict[str, float],
        dt: float,
        volume: float,
        rng: np.random.Generator,
    ) -> Dict[str, float]:
        """
        Chemical Langevin equation step (Euler-Maruyama), in concentration
        space throughout. Derived from the standard count-space CLE
        (dN = S.a(N) dt + S.sqrt(a(N)).dW) via the same propensity bridge
        a(N) = volume * v(C) used by SSA, then dividing through by volume:

            dC = S @ v(C) dt + (1 / sqrt(volume)) * S @ (sqrt(v(C)) * xi) * sqrt(dt)
        """
        species_list = self.species
        S = self._stoichiometry_matrix
        n_reactions = len(self._reaction_names)

        x = np.array([state.get(s, 0.0) for s in species_list])
        v = np.maximum(self._rate_vector(state), 0.0)

        xi = rng.normal(size=n_reactions)
        deterministic = S @ v
        noise = S @ (np.sqrt(v) * xi)

        x_new = x + dt * deterministic + np.sqrt(dt / volume) * noise

        return {s: max(x_new[i], 0.0) for i, s in enumerate(species_list)}

    @classmethod
    def from_sbml(cls, file_path: str) -> "ReactionNetwork":
        """
        Construct a ReactionNetwork from an SBML file.
        Placeholder for future implementation.

        Returns:
            ReactionNetwork instance
        """
        raise NotImplementedError("SBML parsing not yet implemented.")

    def clone(self) -> "ReactionNetwork":
        """
        Return a deep copy of the reaction network.
        Useful for passing independent copies to daughter cells.
        """
        return ReactionNetwork(
            name=self.name,
            reactions={k: v.clone() for k, v in self.reactions.items()},
            simulation_method=self.simulation_method,
        )
