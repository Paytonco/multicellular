# tests/test_reactions.py

import numpy as np
import pytest

from multicellular.core.cell import Cell
from multicellular.core.reactions import Reaction, ReactionNetwork


def test_mass_action_rate():
    rxn = Reaction(
        reactants={"A": 1, "B": 1},
        products={"C": 1},
        rate_law_type="mass_action",
        rate_params={"k": 2.0},
    )
    conc = {"A": 1.0, "B": 3.0}
    rate = rxn.rate(conc)
    assert pytest.approx(rate, rel=1e-6) == 6.0  # 2.0 * 1.0 * 3.0


def test_michaelis_menten_rate():
    rxn = Reaction(
        reactants={"S": 1},
        products={"P": 1},
        catalysts=["E"],
        rate_law_type="michaelis_menten",
        rate_params={"Vmax": 4.0, "Km": 2.0},
    )
    conc = {"S": 2.0, "E": 1.0}
    rate = rxn.rate(conc)
    assert pytest.approx(rate, rel=1e-6) == 2.0  # Vmax * S / (Km + S)


def test_hill_langmuir_rate():
    rxn = Reaction(
        reactants={"S": 1},
        products={"P": 1},
        catalysts=["E", "Z"],
        rate_law_type="hill_langmuir",
        rate_params={"alpha": 2.0, "beta": 3.0, "C": 1.0, "n": 2},
    )
    conc = {"E": 1.0, "Z": 1.0}
    rate = rxn.rate(conc)
    # beta * E * (1 + alpha * C * z^n) / (1 + C * z^n) = 3 * 1 * 3 / 2
    assert pytest.approx(rate, rel=1e-6) == 4.5


def test_hill_langmuir_rate_uses_first_two_catalysts_as_e_and_z():
    rxn = Reaction(
        reactants={"S": 1},
        products={"P": 1},
        catalysts=["E", "Z"],
        rate_law_type="hill_langmuir",
        rate_params={"alpha": 0.0, "beta": 2.0, "C": 1.0, "n": 1},
    )
    # alpha=0 collapses the rate to beta * E / (1 + C * z), independent of S.
    conc = {"E": 5.0, "Z": 3.0}
    rate = rxn.rate(conc)
    assert pytest.approx(rate, rel=1e-6) == 2.0 * 5.0 / (1 + 1.0 * 3.0)


@pytest.mark.parametrize("missing", ["alpha", "beta", "C", "n"])
def test_hill_langmuir_missing_required_param_raises(missing):
    rate_params = {"alpha": 1.0, "beta": 1.0, "C": 1.0, "n": 1.0}
    del rate_params[missing]
    rxn = Reaction(
        reactants={"S": 1},
        products={"P": 1},
        catalysts=["E", "Z"],
        rate_law_type="hill_langmuir",
        rate_params=rate_params,
    )
    with pytest.raises(KeyError):
        rxn.rate({"E": 1.0, "Z": 1.0})


def test_stoichiometry_matrix():
    rxn1 = Reaction(
        {"A": 1}, {"B": 1}, rate_law_type="mass_action", rate_params={"k": 1}
    )
    rxn2 = Reaction(
        {"B": 1}, {"C": 1}, rate_law_type="mass_action", rate_params={"k": 1}
    )
    net = ReactionNetwork("simple", {"R1": rxn1, "R2": rxn2})
    S = net.get_stoichiometry_matrix(["A", "B", "C"], ["R1", "R2"])
    expected = np.array(
        [
            [-1, 0],  # A
            [1, -1],  # B
            [0, 1],  # C
        ]
    )
    assert np.allclose(S, expected)


def test_network_simulation_step():
    rxn = Reaction(
        reactants={"A": 1},
        products={"B": 1},
        rate_law_type="mass_action",
        rate_params={"k": 1.0},
    )
    net = ReactionNetwork("linear", {"R": rxn})
    state = {"A": 1.0, "B": 0.0}
    dt = 0.1
    volume = 1.0

    new_state = net.simulate_step(state, dt, volume)

    # A should decrease, B should increase
    assert new_state["A"] < 1.0
    assert new_state["B"] > 0.0
    assert pytest.approx(new_state["A"] + new_state["B"], rel=1e-6) == 1.0


def test_cell_with_reaction_network():
    # Build a simple network A → B
    rxn = Reaction(
        {"A": 1}, {"B": 1}, rate_law_type="mass_action", rate_params={"k": 1.0}
    )
    net = ReactionNetwork("linear", {"R": rxn})

    # Create a cell with the network. growth_rate=0.0 keeps this test
    # focused on reaction-driven conservation, undisturbed by growth-driven
    # dilution (see test_cell.py for dedicated dilution tests).
    cell = Cell(
        id=0,
        position=[0.0, 0.0],
        orientation=[1.0, 0.0],
        length=2.0,
        radius=0.5,
        network=net,
        growth_rate=0.0,
    )
    cell.concentrations = {"A": 1.0, "B": 0.0}

    dt = 0.1
    cell.step(dt)

    assert cell.concentrations["A"] < 1.0
    assert cell.concentrations["B"] > 0.0
    assert (
        pytest.approx(cell.concentrations["A"] + cell.concentrations["B"], rel=1e-6)
        == 1.0
    )


def _linear_ab_network(k=1.0):
    rxn = Reaction(
        {"A": 1}, {"B": 1}, rate_law_type="mass_action", rate_params={"k": k}
    )
    return ReactionNetwork("linear", {"R": rxn})


def test_ssa_step_conserves_integer_counts():
    net = _linear_ab_network(k=1.0)
    volume = 10.0
    state = {"A": 10.0, "B": 0.0}  # 100 copies of A, 0 of B
    rng = np.random.default_rng(0)

    new_state = net.simulate_step(state, dt=0.05, volume=volume, method="SSA", rng=rng)

    count_a = new_state["A"] * volume
    count_b = new_state["B"] * volume
    assert count_a == pytest.approx(round(count_a), abs=1e-6)
    assert count_b == pytest.approx(round(count_b), abs=1e-6)
    assert count_a + count_b == pytest.approx(100.0, abs=1e-6)


def test_ssa_step_zero_propensity_is_noop():
    net = _linear_ab_network(k=1.0)
    rng = np.random.default_rng(0)

    new_state = net.simulate_step(
        {"A": 0.0, "B": 0.0}, dt=1.0, volume=1.0, method="SSA", rng=rng
    )

    assert new_state == {"A": 0.0, "B": 0.0}


def test_ssa_step_matches_ode_mean():
    k = 1.0
    volume = 10.0
    dt = 0.05
    a0 = 100.0  # copies
    net = _linear_ab_network(k=k)

    n_reps = 3000
    b_counts = np.empty(n_reps)
    for i in range(n_reps):
        rng = np.random.default_rng(i)
        new_state = net.simulate_step(
            {"A": a0 / volume, "B": 0.0}, dt=dt, volume=volume, method="SSA", rng=rng
        )
        b_counts[i] = new_state["B"] * volume

    expected_b = a0 * (1.0 - np.exp(-k * dt))
    assert np.mean(b_counts) == pytest.approx(expected_b, abs=1.0)


def test_cle_step_conserves_mass_without_clipping():
    net = _linear_ab_network(k=2.0)
    # Large volume + small dt keeps the noise term well clear of the
    # non-negativity clip, so conservation should hold (near-)exactly.
    state = {"A": 4.0, "B": 1.0}
    volume = 100.0
    dt = 0.02

    for seed in range(20):
        rng = np.random.default_rng(seed)
        new_state = net.simulate_step(
            state, dt=dt, volume=volume, method="CLE", rng=rng
        )
        assert new_state["A"] + new_state["B"] == pytest.approx(5.0, abs=1e-9)


def test_cle_step_matches_ode_mean_and_scales_noise_with_volume():
    k = 2.0
    state = {"A": 4.0, "B": 1.0}
    dt = 0.02
    net = _linear_ab_network(k=k)
    n_reps = 3000

    def run(volume):
        b_vals = np.empty(n_reps)
        for i in range(n_reps):
            rng = np.random.default_rng(i)
            new_state = net.simulate_step(
                state, dt=dt, volume=volume, method="CLE", rng=rng
            )
            b_vals[i] = new_state["B"]
        return b_vals

    b_v1 = run(volume=100.0)
    b_v2 = run(volume=400.0)  # 4x volume -> noise variance should drop ~4x

    expected_b = state["B"] + dt * k * state["A"]
    assert np.mean(b_v1) == pytest.approx(expected_b, abs=0.01)
    assert np.mean(b_v2) == pytest.approx(expected_b, abs=0.01)

    var_ratio = np.var(b_v2) / np.var(b_v1)
    assert var_ratio == pytest.approx(0.25, rel=0.3)


def test_cell_with_ssa_reaction_network():
    net = _linear_ab_network(k=1.0)
    cell = Cell(
        id=0,
        position=[0.0, 0.0],
        orientation=[1.0, 0.0],
        length=2.0,
        radius=0.5,
        network=net,
        growth_rate=0.0,
        rng=np.random.default_rng(0),
    )
    cell.concentrations = {"A": 1.0, "B": 0.0}

    volume = cell.compute_volume()
    expected_copies = round(1.0 * volume)  # initial concentration was 1.0
    cell.step(dt=0.1, method="SSA")

    assert cell.concentrations["A"] >= 0.0
    assert cell.concentrations["B"] >= 0.0
    total_copies = (cell.concentrations["A"] + cell.concentrations["B"]) * volume
    assert total_copies == pytest.approx(expected_copies, abs=1e-6)


def test_reaction_exports_defaults_to_empty():
    rxn = Reaction(
        {"A": 1}, {"B": 1}, rate_law_type="mass_action", rate_params={"k": 1}
    )
    assert rxn.exports == {}


def test_reaction_exports_conflicting_with_products_raises():
    with pytest.raises(ValueError):
        Reaction(
            reactants={"A": 1},
            products={"B": 1},
            exports={"B": 1},
            rate_law_type="mass_action",
            rate_params={"k": 1.0},
        )


def test_reaction_clone_carries_exports():
    rxn = Reaction(
        reactants={"X": 1},
        products={},
        exports={"X": 1},
        rate_law_type="mass_action",
        rate_params={"k": 1.0},
    )
    clone = rxn.clone()
    assert clone.exports == {"X": 1}
    clone.exports["X"] = 99  # mutating the clone must not affect the original
    assert rxn.exports == {"X": 1}


def test_get_export_vector():
    rxn = Reaction(
        reactants={"X": 1},
        products={},
        exports={"X": 1},
        rate_law_type="mass_action",
        rate_params={"k": 1.0},
    )
    assert list(rxn.get_export_vector(["W", "X", "Y"])) == [0, 1, 0]


def _efflux_network(k=1.0):
    rxn = Reaction(
        reactants={"X": 1},
        products={},
        exports={"X": 1},
        rate_law_type="mass_action",
        rate_params={"k": k},
    )
    return ReactionNetwork("efflux", {"R": rxn})


def test_reaction_network_computes_exported_species_and_matrix():
    net = _efflux_network()
    assert net.exported_species == ["X"]
    assert list(net._export_stoichiometry_matrix[:, 0]) == [1]
    assert net.last_exported == {"X": 0.0}


def test_reaction_network_with_no_exports_has_empty_exported_species():
    rxn = Reaction(
        {"A": 1}, {"B": 1}, rate_law_type="mass_action", rate_params={"k": 1}
    )
    net = ReactionNetwork("simple", {"R": rxn})
    assert net.exported_species == []
    assert net.last_exported == {}


def test_ode_efflux_conserves_mass_via_last_exported():
    net = _efflux_network(k=1.0)
    state = {"X": 1.0}
    volume = 2.0
    dt = 0.1

    new_state = net.simulate_step(state, dt, volume)

    lost = (state["X"] - new_state["X"]) * volume
    assert net.last_exported["X"] == pytest.approx(lost, abs=1e-9)
    assert net.last_exported["X"] > 0.0


def test_ode_efflux_caps_export_at_available_pool_for_large_dt():
    # k*dt = 10: the pool is nearly exhausted within one step. The integrated
    # export must follow exponential decay and never exceed what's available.
    net = _efflux_network(k=100.0)
    state = {"X": 0.05}
    volume = 1.0
    dt = 0.1

    new_state = net.simulate_step(state, dt, volume)

    remaining = 0.05 * np.exp(-10.0)
    assert new_state["X"] == pytest.approx(remaining, abs=1e-7)
    assert net.last_exported["X"] == pytest.approx(
        (0.05 - remaining) * volume, abs=1e-7
    )
    assert net.last_exported["X"] <= 0.05 * volume


def test_ssa_efflux_conserves_mass_exactly():
    net = _efflux_network(k=1.0)
    state = {"X": 10.0}
    volume = 10.0  # 100 copies
    rng = np.random.default_rng(0)

    new_state = net.simulate_step(state, dt=0.5, volume=volume, method="SSA", rng=rng)

    remaining = new_state["X"] * volume
    exported = net.last_exported["X"]
    assert exported == pytest.approx(round(exported), abs=1e-9)
    assert remaining + exported == pytest.approx(100.0, abs=1e-9)


def test_ssa_efflux_stops_exporting_once_pool_is_empty():
    net = _efflux_network(k=1.0)
    rng = np.random.default_rng(0)

    new_state = net.simulate_step({"X": 0.0}, dt=1.0, volume=1.0, method="SSA", rng=rng)

    assert new_state["X"] == 0.0
    assert net.last_exported["X"] == 0.0


class _FixedNormalRNG:
    """Stub rng whose .normal() always returns a fixed array, for
    deterministically reproducing a specific CLE noise draw in tests."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def normal(self, size=None):
        return self._values


def test_cle_efflux_conserves_mass_with_large_destabilizing_noise():
    # Regression test for a clipping-asymmetry bug: a large negative noise
    # draw must not be able to increase the internal pool while exporting
    # nothing (or vice versa) -- the shared extent clamp should make the
    # reaction simply not fire instead.
    net = _efflux_network(k=1.0)
    state = {"X": 0.05}
    volume = 1.0
    dt = 0.02
    rng = _FixedNormalRNG([-5.0])

    new_state = net.simulate_step(state, dt, volume, method="CLE", rng=rng)

    assert new_state["X"] == pytest.approx(0.05, abs=1e-9)
    assert net.last_exported["X"] == pytest.approx(0.0, abs=1e-9)


def test_cle_efflux_conserves_mass_across_many_noise_draws():
    net = _efflux_network(k=2.0)
    state = {"X": 4.0}
    volume = 100.0
    dt = 0.02

    for seed in range(20):
        rng = np.random.default_rng(seed)
        new_state = net.simulate_step(
            state, dt=dt, volume=volume, method="CLE", rng=rng
        )
        remaining = new_state["X"] * volume
        exported = net.last_exported["X"]
        assert remaining + exported == pytest.approx(state["X"] * volume, abs=1e-9)


def test_cell_pending_export_populated_after_step():
    net = _efflux_network(k=1.0)
    cell = Cell(
        id=0,
        position=[0.0, 0.0],
        orientation=[1.0, 0.0],
        network=net,
        growth_rate=0.0,
    )
    cell.concentrations = {"X": 1.0}

    cell.step(dt=0.1)

    assert cell.pending_export["X"] > 0.0


def test_cell_with_cle_reaction_network():
    net = _linear_ab_network(k=1.0)
    cell = Cell(
        id=0,
        position=[0.0, 0.0],
        orientation=[1.0, 0.0],
        length=2.0,
        radius=0.5,
        network=net,
        growth_rate=0.0,
        rng=np.random.default_rng(0),
    )
    cell.concentrations = {"A": 1.0, "B": 0.0}

    cell.step(dt=0.1, method="CLE")

    assert cell.concentrations["A"] >= 0.0
    assert cell.concentrations["B"] >= 0.0
    assert (
        pytest.approx(cell.concentrations["A"] + cell.concentrations["B"], abs=1e-6)
        == 1.0
    )


def _decay_network(k=0.5):
    rxn = Reaction(
        reactants={"A": 1},
        products={},
        rate_law_type="mass_action",
        rate_params={"k": k},
    )
    return ReactionNetwork("decay", {"R": rxn})


@pytest.mark.parametrize(
    "ode_method", ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA", "rk45"]
)
def test_ode_methods_match_analytic_decay(ode_method):
    net = _decay_network(k=0.5)
    new_state = net.simulate_step({"A": 2.0}, dt=1.0, volume=1.0, ode_method=ode_method)
    assert new_state["A"] == pytest.approx(2.0 * np.exp(-0.5), rel=1e-3)


@pytest.mark.parametrize("ode_method", ["RK45", "Radau", "LSODA"])
def test_ode_efflux_mass_conserved_for_each_method(ode_method):
    net = _efflux_network(k=1.0)
    new_state = net.simulate_step({"X": 1.0}, dt=0.1, volume=2.0, ode_method=ode_method)
    assert net.last_exported["X"] == pytest.approx(
        (1.0 - new_state["X"]) * 2.0, abs=1e-6
    )


def test_ode_default_is_rk45():
    net = _decay_network()
    a = net.simulate_step({"A": 2.0}, dt=1.0, volume=1.0)
    b = net.simulate_step({"A": 2.0}, dt=1.0, volume=1.0, ode_method="RK45")
    assert a == b


def test_unknown_ode_method_raises():
    with pytest.raises(ValueError):
        _decay_network().simulate_step({"A": 1.0}, 1.0, 1.0, ode_method="Euler")


# --- compartment_volumes (extracellular species) ---


def _uptake_network(k=1.0):
    rxn = Reaction(
        {"A_ext": 1}, {"A_int": 1}, rate_law_type="mass_action", rate_params={"k": k}
    )
    return ReactionNetwork("uptake", {"R": rxn})


@pytest.mark.parametrize("method", ["ODE", "SSA", "CLE"])
def test_compartment_volumes_none_matches_default(method):
    net = _linear_ab_network(k=1.0)
    state = {"A": 3.0, "B": 0.5}
    default = net.simulate_step(state, 0.1, 10.0, method, rng=np.random.default_rng(0))
    explicit = net.simulate_step(
        state,
        0.1,
        10.0,
        method,
        rng=np.random.default_rng(0),
        compartment_volumes={},
    )
    assert explicit == default


def test_ode_compartment_volumes_scale_external_change_and_conserve_mass():
    net = _uptake_network(k=2.0)
    volume, v_ext = 2.0, 20.0
    state = {"A_ext": 5.0, "A_int": 0.0}

    new = net.simulate_step(
        state, 0.5, volume, "ODE", compartment_volumes={"A_ext": v_ext}
    )

    d_int = new["A_int"] - state["A_int"]
    d_ext = new["A_ext"] - state["A_ext"]
    assert d_int > 0.0
    assert d_ext == pytest.approx(-d_int * volume / v_ext, rel=1e-5)
    # Analytic: d[A_ext]/dt = -(volume/v_ext) * k * [A_ext]
    assert new["A_ext"] == pytest.approx(5.0 * np.exp(-0.1 * 2.0 * 0.5), rel=1e-5)


def test_ssa_compartment_volumes_conserve_counts_and_keep_external_fraction():
    net = _uptake_network(k=1.0)
    volume, v_ext = 2.0, 13.7
    state = {"A_ext": 3.3, "A_int": 1.0}  # 45.21 extracellular molecules
    initial = state["A_ext"] * v_ext + state["A_int"] * volume

    for seed in range(20):
        new = net.simulate_step(
            state,
            0.5,
            volume,
            "SSA",
            rng=np.random.default_rng(seed),
            compartment_volumes={"A_ext": v_ext},
        )
        n_int = new["A_int"] * volume
        n_ext = new["A_ext"] * v_ext
        assert n_int == pytest.approx(round(n_int), abs=1e-9)
        # The external pool keeps its fractional part: whole molecules move.
        assert n_ext % 1.0 == pytest.approx(0.21, abs=1e-9)
        assert n_int + n_ext == pytest.approx(initial, abs=1e-9)


def test_ssa_rejects_firing_that_would_overdraw_external_pool():
    net = _uptake_network(k=100.0)
    volume, v_ext = 2.0, 1.0
    state = {"A_ext": 0.5, "A_int": 0.0}  # half a molecule outside

    new = net.simulate_step(
        state,
        1.0,
        volume,
        "SSA",
        rng=np.random.default_rng(0),
        compartment_volumes={"A_ext": v_ext},
    )
    assert new == {"A_ext": 0.5, "A_int": 0.0}


def test_cle_compartment_volumes_conserve_mass_and_stay_nonnegative():
    net = _uptake_network(k=5.0)
    volume, v_ext = 2.0, 1.0
    state = {"A_ext": 0.3, "A_int": 0.05}
    initial = state["A_ext"] * v_ext + state["A_int"] * volume

    for seed in range(200):
        new = net.simulate_step(
            state,
            0.5,
            volume,
            "CLE",
            rng=np.random.default_rng(seed),
            compartment_volumes={"A_ext": v_ext},
        )
        assert new["A_ext"] >= 0.0 and new["A_int"] >= 0.0
        total = new["A_ext"] * v_ext + new["A_int"] * volume
        assert total == pytest.approx(initial, abs=1e-12)


def test_external_only_reaction_changes_only_external_species():
    conversion = Reaction(
        {"S": 1},
        {"P": 1},
        catalysts=["E"],
        rate_law_type="mass_action",
        rate_params={"k": 1.0},
    )
    net = ReactionNetwork("ext_enzyme", {"R": conversion})
    volume, v_ext = 1.0, 10.0
    state = {"S": 2.0, "P": 0.0, "E": 1.0}

    new = net.simulate_step(
        state, 0.1, volume, "ODE", compartment_volumes={"S": v_ext, "P": v_ext}
    )

    assert new["E"] == pytest.approx(1.0)
    assert new["S"] + new["P"] == pytest.approx(2.0)
    assert new["S"] == pytest.approx(2.0 * np.exp(-0.1 * 0.1), rel=1e-5)


# --- ReactionNetwork.simulate ---


def _decay(k=0.5):
    return ReactionNetwork(
        "decay",
        {"decay": Reaction(reactants={"A": 1}, products={}, rate_params={"k": k})},
    )


def _conversion(k=0.5):
    return ReactionNetwork(
        "conversion",
        {"conv": Reaction(reactants={"A": 1}, products={"B": 1}, rate_params={"k": k})},
    )


def test_simulate_ode_matches_analytic_decay():
    k, A0 = 0.5, 10.0
    df = _decay(k).simulate({"A": A0}, t_max=4.0, dt=0.1)
    assert list(df.columns) == ["time", "A"]
    assert len(df) == 41
    np.testing.assert_allclose(df["time"], np.arange(41) * 0.1)
    np.testing.assert_allclose(df["A"], A0 * np.exp(-k * df["time"]), rtol=1e-4)


def test_simulate_missing_species_start_at_zero_and_mass_conserved():
    df = _conversion().simulate({"A": 5.0}, t_max=2.0, dt=0.1)
    assert df["B"].iloc[0] == 0.0
    np.testing.assert_allclose(df["A"] + df["B"], 5.0, rtol=1e-6)


def test_simulate_unknown_species_raises():
    with pytest.raises(ValueError, match="Unknown species"):
        _decay().simulate({"A": 1.0, "Z": 1.0}, t_max=1.0, dt=0.1)


def test_simulate_rejects_bad_dt_and_t_max():
    with pytest.raises(ValueError):
        _decay().simulate({"A": 1.0}, t_max=1.0, dt=0.0)
    with pytest.raises(ValueError):
        _decay().simulate({"A": 1.0}, t_max=-1.0, dt=0.1)


def test_simulate_ssa_reproducible_with_seed():
    net = _conversion()
    kwargs = dict(t_max=5.0, dt=0.5, volume=100.0, method="ssa")
    df1 = net.simulate({"A": 1.0}, seed=1, **kwargs)
    df2 = net.simulate({"A": 1.0}, seed=1, **kwargs)
    df3 = net.simulate({"A": 1.0}, seed=2, **kwargs)
    assert df1.equals(df2)
    assert not df1.equals(df3)


def test_simulate_tracks_cumulative_exports():
    net = ReactionNetwork(
        "secretion",
        {
            "secrete": Reaction(
                reactants={"S": 1},
                products={},
                exports={"S": 1},
                rate_params={"k": 0.3},
            )
        },
    )
    volume = 2.0
    df = net.simulate({"S": 4.0}, t_max=5.0, dt=0.25, volume=volume)
    assert "exported_S" in df.columns
    assert df["exported_S"].iloc[0] == 0.0
    np.testing.assert_allclose(df["S"] * volume + df["exported_S"], 8.0, rtol=1e-5)


def test_simulate_invalid_method_raises():
    with pytest.raises(ValueError, match="Unknown simulation method"):
        _decay().simulate({"A": 1.0}, t_max=1.0, dt=0.1, method="euler")
