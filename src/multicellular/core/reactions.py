# core/reactions.py

from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.integrate import solve_ivp
from tqdm import tqdm

# Every numerical method accepted by scipy.integrate.solve_ivp ("ODE" mode).
ODE_METHODS = ("RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA")


class RateLawNotVectorizedError(Exception):
    """
    Raised by `ReactionNetwork.simulate_ode_batch` when a rate law can't be
    evaluated on arrays of concentrations (typically a `custom` rate law
    written for scalars, e.g. one using `math.exp` or `max`). Callers fall
    back to stepping each compartment separately.
    """


def _resolve_ode_method(ode_method):
    """Case-insensitively match `ode_method` against ODE_METHODS."""
    method = next(
        (m for m in ODE_METHODS if m.upper() == str(ode_method).upper()), None
    )
    if method is None:
        raise ValueError(
            f"Unknown ODE method: {ode_method!r}; expected one of {ODE_METHODS}"
        )
    return method


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
        exports: Dict[str, int] = None,
    ):
        self.reactants = reactants  # e.g., {"A": 1, "B": 2}
        self.products = products  # e.g., {"C": 1}
        self.catalysts = catalysts if catalysts else []  # e.g., ["E"]
        self.rate_law_type = rate_law_type
        self.rate_params = rate_params or {}
        self.custom_rate_law = custom_rate_law  # Optional user-defined function
        # Stoichiometry that leaves the cell entirely (e.g. secretion) rather
        # than being added back into intracellular state, e.g. {"X": 1}.
        # Recommended pattern: reactants={"X": 1}, products={}, exports={"X": 1}
        # (export a species that is also this reaction's sole reactant).
        self.exports = exports or {}

        self._validate()

    def _validate(self):
        if self.rate_law_type == "custom" and self.custom_rate_law is None:
            raise ValueError(
                "Custom rate law type specified but no custom_rate_law function provided."
            )
        if set(self.exports) & set(self.products):
            raise ValueError(
                "A species cannot be both a product and an export of the "
                "same reaction (it would both stay in the cell and leave "
                "it from a single reactant consumption)."
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

    def get_net_changes(self) -> Dict[str, int]:
        """Net stoichiometric change (products - reactants) per species touched."""
        return {
            s: self.products.get(s, 0) - self.reactants.get(s, 0)
            for s in set(self.reactants) | set(self.products)
        }

    def get_export_vector(self, exported_species_list: List[str]) -> np.ndarray:
        """
        Return this reaction's export stoichiometry for a given species ordering.

        Unlike `get_stoichiometry_vector`, this is pure outflow (no reactant
        subtraction): it's the amount of each exported species that leaves
        the cell per unit reaction extent.
        """
        return np.array([self.exports.get(s, 0) for s in exported_species_list])

    def clone(self):
        """Return a deep copy of the reaction (for cell division use cases)."""
        return Reaction(
            reactants=self.reactants.copy(),
            products=self.products.copy(),
            catalysts=self.catalysts.copy(),
            rate_law_type=self.rate_law_type,
            rate_params=self.rate_params.copy(),
            custom_rate_law=self.custom_rate_law,
            exports=self.exports.copy(),
        )


class ReactionNetwork:
    """
    Represents a chemical reaction network with associated species and reactions.

    Reactions may also export species to the cell's extracellular
    environment (e.g. secretion): see `Reaction.exports`. The recommended
    pattern for an export reaction is reactants={species: 1}, products={},
    exports={species: 1} — exporting a species that is also that reaction's
    sole reactant, so it's mass-conserving (the cell's own pool of that
    species is depleted by exactly the amount that leaves).
    """

    def __init__(self, name: str, reactions: Dict[str, Reaction]):
        self.name = name
        self.reactions = reactions  # dict of name → Reaction
        self.species = self._extract_species()
        self.exported_species = sorted(
            {s for r in self.reactions.values() for s in r.exports}
        )

        # Species/reaction ordering and the stoichiometry matrix are fixed for
        # the lifetime of this network (reactions/species don't change after
        # construction), so compute them once instead of on every ODE step.
        self._reaction_names = list(self.reactions.keys())
        self._stoichiometry_matrix = self.get_stoichiometry_matrix(
            self.species, self._reaction_names
        )
        self._export_stoichiometry_matrix = self.get_export_stoichiometry_matrix(
            self.exported_species, self._reaction_names
        )

        # Molecule counts (not concentrations) exported by the most recent
        # `simulate_step` call, keyed by exported species name. Set as a
        # side effect of `simulate_step`; `Cell.step` reads it immediately
        # after each call.
        self.last_exported = {s: 0.0 for s in self.exported_species}

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

    def get_export_stoichiometry_matrix(
        self, exported_species_list: List[str], reaction_list: List[str]
    ) -> np.ndarray:
        """
        Return the export stoichiometry matrix (rows = exported species,
        columns = reactions), the export-side counterpart of
        `get_stoichiometry_matrix`.
        """
        S_exp = np.zeros((len(exported_species_list), len(reaction_list)))
        for j, rxn_name in enumerate(reaction_list):
            rxn = self.reactions[rxn_name]
            S_exp[:, j] = rxn.get_export_vector(exported_species_list)
        return S_exp

    def simulate_step(
        self,
        state: Dict[str, float],
        dt: float,
        volume: float,
        method: str = "ODE",
        rng: np.random.Generator = None,
        ode_method: str = "RK45",
        compartment_volumes: Dict[str, float] = None,
    ) -> Dict[str, float]:
        """
        Advance the chemical state by one time step using the given simulation method.

        Args:
            state: dict of species → concentration
            dt: timestep size
            volume: cell volume (used for propensities in SSA/CLE)
            method: one of "ODE" (numerical integration via
                `scipy.integrate.solve_ivp`), "SSA" (Gillespie), or
                "CLE" (chemical Langevin equation). Case-insensitive.
            rng: random generator used by SSA/CLE (ignored by ODE). Defaults
                to a fresh `np.random.default_rng()` if not given.
            ode_method: `solve_ivp` integrator used when method is "ODE" (ignored
                by SSA/CLE): one of "RK45" (default), "RK23", "DOP853",
                "Radau", "BDF", or "LSODA". Case-insensitive.
            compartment_volumes: optional dict of species → volume of the
                compartment that species lives in, for species that live
                outside `volume` (e.g. an extracellular species living in an
                environment voxel). Every rate law is still evaluated
                against concentrations and scaled by `volume` — a reaction
                fires `v * volume` molecules per unit time — but each
                species' concentration changes by those molecules divided
                by its *own* compartment's volume, so molecule counts are
                conserved across compartments. Species not listed live in
                `volume`. Default None: every species lives in `volume`.

        Returns:
            Updated concentration dictionary
        """
        compartment_volumes = compartment_volumes or {}
        weights = np.array(
            [volume / compartment_volumes.get(s, volume) for s in self.species]
        )
        external = np.array([s in compartment_volumes for s in self.species])
        method = method.upper()
        if method == "ODE":
            return self._simulate_ode_step(state, dt, volume, ode_method, weights)
        elif method == "SSA":
            return self._simulate_ssa_step(
                state, dt, volume, rng or np.random.default_rng(), weights, external
            )
        elif method == "CLE":
            return self._simulate_cle_step(
                state, dt, volume, rng or np.random.default_rng(), weights, external
            )
        else:
            raise ValueError(f"Unknown simulation method: {method}")

    def simulate(
        self,
        initial_state: Dict[str, float],
        t_max: float,
        dt: float,
        volume: float = 1.0,
        method: str = "ODE",
        ode_method: str = "RK45",
        rng: np.random.Generator = None,
        seed: int = None,
        compartment_volumes: Dict[str, float] = None,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """
        Simulate this network on its own (no Cell/Colony) from t=0 to `t_max`.

        Repeatedly calls `simulate_step`, recording the state at t=0 and
        after every step.

        Args:
            initial_state: dict of species → initial concentration. Species
                not listed start at 0; unknown species raise ValueError.
            t_max: end time; the run takes `round(t_max / dt)` steps.
            dt: timestep size.
            volume: compartment volume, as in `simulate_step`. Only matters
                for SSA/CLE, where it converts concentrations to molecule
                counts.
            method: "ODE", "SSA", or "CLE", as in `simulate_step`.
            ode_method: `solve_ivp` method for "ODE", as in `simulate_step`.
            rng: random generator for SSA/CLE. If not given, one is created
                from `seed`, so stochastic runs are reproducible.
            seed: seed for the generator created when `rng` is None.
            compartment_volumes: as in `simulate_step`.
            show_progress: show a tqdm progress bar.

        Returns:
            DataFrame with a `time` column, one column per species (in
            `self.species` order), and, if the network exports species, a
            cumulative `exported_<species>` column (molecule counts) for each.
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if t_max < 0:
            raise ValueError(f"t_max must be non-negative, got {t_max}")
        unknown = set(initial_state) - set(self.species)
        if unknown:
            raise ValueError(
                f"Unknown species in initial_state: {sorted(unknown)}; "
                f"network species are {self.species}"
            )

        state = {s: float(initial_state.get(s, 0.0)) for s in self.species}
        rng = rng or np.random.default_rng(seed)
        exported = {s: 0.0 for s in self.exported_species}

        def record(time):
            row = {"time": time, **state}
            row.update({f"exported_{s}": v for s, v in exported.items()})
            return row

        records = [record(0.0)]
        n_steps = round(t_max / dt)
        for k in tqdm(range(1, n_steps + 1), disable=not show_progress):
            state = self.simulate_step(
                state,
                dt,
                volume,
                method=method,
                rng=rng,
                ode_method=ode_method,
                compartment_volumes=compartment_volumes,
            )
            for s in self.exported_species:
                exported[s] += self.last_exported[s]
            records.append(record(k * dt))

        return pd.DataFrame(records)

    def simulate_ode_batch(
        self,
        state: Dict[str, np.ndarray],
        dt: float,
        ode_method: str = "RK45",
    ) -> Dict[str, np.ndarray]:
        """
        Integrate the reaction ODEs over `dt` for many independent
        compartments at once, e.g. every voxel of an `Environment`.

        Equivalent to calling `simulate_step(..., method="ODE")` on each
        compartment separately, but as a single `solve_ivp` call over the
        stacked state, with every rate law evaluated on whole arrays. That
        removes the per-compartment Python and `solve_ivp` overhead, which
        otherwise dominates reaction-diffusion runs.

        The compartments don't interact, so the Jacobian is block-diagonal
        (one species-by-species block per compartment). The stiff solvers
        are told so, `jac_sparsity` for BDF/Radau and band widths for LSODA,
        so their finite-difference Jacobians stay cheap.

        Error control is shared: `solve_ivp` takes one step size for the
        whole batch and judges each step by the RMS of the scaled error over
        every component, so one compartment's local error can exceed the
        per-component tolerance a separate solve would enforce. In practice
        the difference is far below the scheme's other errors (see
        tests/test_reaction_diffusion.py, which compares the two).

        Export-tagged reactions are clamped as in `simulate_step`, but the
        exported amounts are not tracked (`last_exported` is left alone),
        since a batch has no single owner to deliver them to.

        Args:
            state: species → 1D array of concentrations, one entry per
                compartment (all the same length). Must include every
                species in the network.
            dt: time step.
            ode_method: `solve_ivp` method, as in `simulate_step`.

        Returns:
            species → 1D array of updated concentrations (clamped to >= 0).

        Raises:
            RateLawNotVectorizedError: if a rate law can't be evaluated on
                arrays. Checked before integrating, so callers can fall back
                to per-compartment `simulate_step` calls.
        """
        method = _resolve_ode_method(ode_method)
        species_list = self.species
        n = len(species_list)
        x0 = np.stack(
            [np.asarray(state[s], dtype=float) for s in species_list], axis=1
        )  # (m compartments, n species); flattened compartment-major below
        m = x0.shape[0]
        if m == 0:
            return {s: np.zeros(0) for s in species_list}

        S = self._stoichiometry_matrix
        names = self._reaction_names
        reactions = [self.reactions[name] for name in names]
        export_flags = [bool(rxn.exports) for rxn in reactions]

        def rates(X):
            conc = {s: X[:, i] for i, s in enumerate(species_list)}
            v = np.empty((len(reactions), m))
            for j, rxn in enumerate(reactions):
                v[j] = rxn.rate(conc)  # a scalar (e.g. zero-order) broadcasts
                if export_flags[j]:
                    depleted = np.zeros(m, dtype=bool)
                    for s in rxn.reactants:
                        depleted |= conc[s] <= 0.0
                    v[j] = np.where(depleted, 0.0, np.maximum(v[j], 0.0))
            return v

        X0 = np.maximum(x0, 0.0)
        try:
            with np.errstate(all="ignore"):
                probe = rates(X0)
        except (TypeError, ValueError) as err:
            raise RateLawNotVectorizedError(str(err)) from err
        if probe.shape != (len(reactions), m):
            raise RateLawNotVectorizedError("rate law returned the wrong shape")

        def rhs(_t, y):
            X = np.maximum(y.reshape(m, n), 0.0)
            return (S @ rates(X)).T.ravel()

        options = {}
        if method in ("BDF", "Radau"):
            options["jac_sparsity"] = sparse.block_diag([np.ones((n, n))] * m)
        elif method == "LSODA":
            options["lband"] = options["uband"] = n - 1

        sol = solve_ivp(
            rhs,
            (0.0, dt),
            x0.ravel(),
            method=method,
            rtol=1e-6,
            atol=1e-9,
            **options,
        )
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")
        X_new = np.maximum(sol.y[:, -1].reshape(m, n), 0.0)
        return {s: X_new[:, i] for i, s in enumerate(species_list)}

    def _rate_vector(self, state: Dict[str, float]) -> np.ndarray:
        """Evaluate every reaction's rate law against `state` (a concentration dict)."""
        return np.array(
            [self.reactions[rxn_name].rate(state) for rxn_name in self._reaction_names]
        )

    def _clamp_extents(
        self,
        state: Dict[str, float],
        extent: np.ndarray,
        weights: np.ndarray,
        external: np.ndarray,
    ) -> np.ndarray:
        """
        Clamp the per-reaction firing extent (concentration-equivalent
        advancement, before stoichiometry is applied) of every export-tagged
        reaction to [0, available reactant], and of every reaction touching
        a species in another compartment (`external`) so that it alone
        can't drive any species it touches negative.

        `weights` is each species' `volume / compartment volume` ratio (see
        `simulate_step`'s `compartment_volumes`).

        Export reactions are one-directional (a cell can't un-secrete), and
        can't remove more of a reactant than the cell currently has. Doing
        this clamp once, before deriving *both* the intracellular delta
        (via `_stoichiometry_matrix`) and the exported delta (via
        `_export_stoichiometry_matrix`) from the same `extent`, guarantees
        the two stay exactly consistent — independently clipping the two
        results afterward can otherwise let mass be created or destroyed
        (e.g. CLE noise driving the aggregate exported amount negative,
        clipped to 0, while the aggregate internal state isn't clipped).

        The same reasoning applies to reactions that move mass across
        compartments (e.g. uptake of an extracellular species, whose amount
        is owned by an environment Field): clipping either side after the
        fact would create mass. A positive extent is capped by how much of
        each of the reaction's net reactants is available, and a negative
        extent (CLE noise running the reaction backward) by how much of each
        net product is available.
        """
        weight = dict(zip(self.species, weights))
        is_external = dict(zip(self.species, external))
        for j, name in enumerate(self._reaction_names):
            rxn = self.reactions[name]
            if rxn.exports:
                cap = min(
                    (
                        state.get(s, 0.0) / (stoich * weight[s])
                        for s, stoich in rxn.reactants.items()
                    ),
                    default=np.inf,
                )
                extent[j] = min(max(extent[j], 0.0), cap)
            changes = rxn.get_net_changes()
            if not any(is_external[s] for s in changes):
                continue
            for s, stoich in changes.items():
                if stoich == 0:
                    continue
                available = state.get(s, 0.0) / (abs(stoich) * weight[s])
                if stoich < 0:
                    extent[j] = min(extent[j], available)
                else:
                    extent[j] = max(extent[j], -available)
        return extent

    def _simulate_ode_step(
        self,
        state: Dict[str, float],
        dt: float,
        volume: float,
        ode_method: str = "RK45",
        weights: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        Integrate the reaction ODEs over `dt` with `scipy.integrate.solve_ivp`.

        The integrated state is the species concentrations followed by the
        cumulative concentration-equivalent exported through each export
        species. Export-tagged reactions are one-directional and can't fire
        without reactant, so their rate is clamped to >= 0 and to 0 whenever
        a reactant is depleted; the exported total is integrated alongside
        the intracellular state, so the two stay consistent.

        `weights` (each species' `volume / compartment volume` ratio; see
        `simulate_step`'s `compartment_volumes`) scales each species' rate
        of change into its own compartment's concentration basis.
        """
        method = _resolve_ode_method(ode_method)

        species_list = self.species
        n = len(species_list)
        S = self._stoichiometry_matrix
        S_exp = self._export_stoichiometry_matrix
        export_flags = [self.reactions[r].exports for r in self._reaction_names]
        if weights is None:
            weights = np.ones(n)

        def rhs(_t, y):
            conc = {s: max(y[i], 0.0) for i, s in enumerate(species_list)}
            v = self._rate_vector(conc)
            for j, name in enumerate(self._reaction_names):
                if not export_flags[j]:
                    continue
                rxn = self.reactions[name]
                depleted = any(conc.get(s, 0.0) <= 0.0 for s in rxn.reactants)
                v[j] = 0.0 if depleted else max(v[j], 0.0)
            return np.concatenate([weights * (S @ v), S_exp @ v])

        x0 = np.array([state.get(s, 0.0) for s in species_list])
        y0 = np.concatenate([x0, np.zeros(len(self.exported_species))])
        sol = solve_ivp(rhs, (0.0, dt), y0, method=method, rtol=1e-6, atol=1e-9)
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")
        y_new = sol.y[:, -1]

        exported_delta = np.maximum(y_new[n:], 0.0)
        self.last_exported = {
            s: exported_delta[i] * volume for i, s in enumerate(self.exported_species)
        }

        return {
            s: max(y_new[i], 0.0) for i, s in enumerate(species_list)
        }  # Prevent negatives

    def _simulate_ssa_step(
        self,
        state: Dict[str, float],
        dt: float,
        volume: float,
        rng: np.random.Generator,
        weights: np.ndarray = None,
        external: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        Gillespie direct-method SSA step, advancing exactly `dt`.

        Reaction rate laws are defined in concentration space (same as ODE),
        so propensities (molecules/time) are obtained by evaluating each
        rate law against counts/volume and scaling by `volume`. Internally
        this works in integer molecule counts; the incoming/outgoing state
        is still a concentration dict, converted at the boundary via
        `volume`, so callers (Cell.step) need no special handling.

        Exported molecule counts are tracked alongside `counts` during the
        same firing loop, so they're exact by construction (each firing
        moves an integer number of molecules from one bucket to the other;
        no separate clamping is needed, unlike the ODE/CLE steps).

        Each species is counted in its own compartment's volume
        (`volume / weights`). Species in another compartment (`external`,
        e.g. an extracellular Field's voxel) are *not* rounded to whole
        molecules, so the owning pool's fractional amount isn't created or
        destroyed by rounding; firings still move whole molecules. A firing
        that would drive any count negative is rejected (time advances, the
        state doesn't change), so mass is never created by clamping.
        """
        species_list = self.species
        S = self._stoichiometry_matrix
        S_exp = self._export_stoichiometry_matrix
        n_reactions = len(self._reaction_names)
        if weights is None:
            weights = np.ones(len(species_list))
            external = np.zeros(len(species_list), dtype=bool)
        volumes = volume / weights

        counts = np.array(
            [state.get(s, 0.0) * volumes[i] for i, s in enumerate(species_list)]
        )
        counts = np.maximum(np.where(external, counts, np.round(counts)), 0.0)
        export_counts = np.zeros(len(self.exported_species))

        t = 0.0
        while t < dt:
            conc = {s: counts[i] / volumes[i] for i, s in enumerate(species_list)}
            propensities = np.maximum(self._rate_vector(conc), 0.0) * volume
            a0 = propensities.sum()

            if a0 <= 0.0:
                break

            tau = rng.exponential(1.0 / a0)
            if t + tau > dt:
                break

            j = rng.choice(n_reactions, p=propensities / a0)
            t += tau
            new_counts = counts + S[:, j]
            if (new_counts < 0.0).any():
                continue
            counts = new_counts
            export_counts += S_exp[:, j]

        self.last_exported = {
            s: export_counts[i] for i, s in enumerate(self.exported_species)
        }
        return {s: counts[i] / volumes[i] for i, s in enumerate(species_list)}

    def _simulate_cle_step(
        self,
        state: Dict[str, float],
        dt: float,
        volume: float,
        rng: np.random.Generator,
        weights: np.ndarray = None,
        external: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        Chemical Langevin equation step (Euler-Maruyama), in concentration
        space throughout. Derived from the standard count-space CLE
        (dN = S.a(N) dt + S.sqrt(a(N)).dW) via the same propensity bridge
        a(N) = volume * v(C) used by SSA, then dividing through by volume:

            dC = S @ v(C) dt + (1 / sqrt(volume)) * S @ (sqrt(v(C)) * xi) * sqrt(dt)

        with each species' row then scaled by `weights` into its own
        compartment's concentration basis.
        """
        species_list = self.species
        S = self._stoichiometry_matrix
        n_reactions = len(self._reaction_names)
        if weights is None:
            weights = np.ones(len(species_list))
            external = np.zeros(len(species_list), dtype=bool)

        x = np.array([state.get(s, 0.0) for s in species_list])
        v = np.maximum(self._rate_vector(state), 0.0)

        xi = rng.normal(size=n_reactions)
        extent = self._clamp_extents(
            state,
            dt * v + np.sqrt(dt / volume) * (np.sqrt(v) * xi),
            weights,
            external,
        )

        x_new = x + weights * (S @ extent)

        exported_delta = self._export_stoichiometry_matrix @ extent
        self.last_exported = {
            s: exported_delta[i] * volume for i, s in enumerate(self.exported_species)
        }

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
        )
