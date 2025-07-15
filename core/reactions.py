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
        S = list(reactant_conc.values())[0] if reactant_conc else 0.0
        E = list(catalyst_conc.values())[0] if catalyst_conc else 1.0

        Vmax = self.rate_params.get("Vmax", 1.0)
        Kd = self.rate_params.get("Kd", 1.0)
        n = self.rate_params.get("n", 1.0)  # Hill coefficient

        return Vmax * E * (S**n) / (Kd**n + S**n)

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
        self, state: Dict[str, float], dt: float, volume: float
    ) -> Dict[str, float]:
        """
        Advance the chemical state by one time step using the specified simulation method.

        Args:
            state: dict of species → concentration
            dt: timestep size
            volume: cell volume (used for propensities in SSA/CLE)

        Returns:
            Updated concentration dictionary
        """
        method = self.simulation_method
        if method == "ODE":
            return self._simulate_ode_step(state, dt)
        elif method == "SSA":
            raise NotImplementedError("SSA method not implemented yet.")
        elif method == "CLE":
            raise NotImplementedError("CLE method not implemented yet.")
        else:
            raise ValueError(f"Unknown simulation method: {method}")

    def _simulate_ode_step(
        self, state: Dict[str, float], dt: float
    ) -> Dict[str, float]:
        """
        Simple forward Euler ODE step.
        """
        species_list = self.species
        reaction_list = list(self.reactions.keys())
        S = self.get_stoichiometry_matrix(species_list, reaction_list)

        v = np.array(
            [self.reactions[rxn_name].rate(state) for rxn_name in reaction_list]
        )  # Reaction rate vector

        x = np.array([state.get(s, 0.0) for s in species_list])
        dxdt = S @ v  # Matrix multiplication
        x_new = x + dt * dxdt

        return {
            s: max(x_new[i], 0.0) for i, s in enumerate(species_list)
        }  # Prevent negatives

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
