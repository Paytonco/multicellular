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


def _linear_ab_network(k=1.0, method="SSA"):
    rxn = Reaction(
        {"A": 1}, {"B": 1}, rate_law_type="mass_action", rate_params={"k": k}
    )
    return ReactionNetwork("linear", {"R": rxn}, simulation_method=method)


def test_ssa_step_conserves_integer_counts():
    net = _linear_ab_network(k=1.0, method="SSA")
    volume = 10.0
    state = {"A": 10.0, "B": 0.0}  # 100 copies of A, 0 of B
    rng = np.random.default_rng(0)

    new_state = net.simulate_step(state, dt=0.05, volume=volume, rng=rng)

    count_a = new_state["A"] * volume
    count_b = new_state["B"] * volume
    assert count_a == pytest.approx(round(count_a), abs=1e-6)
    assert count_b == pytest.approx(round(count_b), abs=1e-6)
    assert count_a + count_b == pytest.approx(100.0, abs=1e-6)


def test_ssa_step_zero_propensity_is_noop():
    net = _linear_ab_network(k=1.0, method="SSA")
    rng = np.random.default_rng(0)

    new_state = net.simulate_step({"A": 0.0, "B": 0.0}, dt=1.0, volume=1.0, rng=rng)

    assert new_state == {"A": 0.0, "B": 0.0}


def test_ssa_step_matches_ode_mean():
    k = 1.0
    volume = 10.0
    dt = 0.05
    a0 = 100.0  # copies
    net = _linear_ab_network(k=k, method="SSA")

    n_reps = 3000
    b_counts = np.empty(n_reps)
    for i in range(n_reps):
        rng = np.random.default_rng(i)
        new_state = net.simulate_step(
            {"A": a0 / volume, "B": 0.0}, dt=dt, volume=volume, rng=rng
        )
        b_counts[i] = new_state["B"] * volume

    expected_b = a0 * (1.0 - np.exp(-k * dt))
    assert np.mean(b_counts) == pytest.approx(expected_b, abs=1.0)


def test_cle_step_conserves_mass_without_clipping():
    net = _linear_ab_network(k=2.0, method="CLE")
    # Large volume + small dt keeps the noise term well clear of the
    # non-negativity clip, so conservation should hold (near-)exactly.
    state = {"A": 4.0, "B": 1.0}
    volume = 100.0
    dt = 0.02

    for seed in range(20):
        rng = np.random.default_rng(seed)
        new_state = net.simulate_step(state, dt=dt, volume=volume, rng=rng)
        assert new_state["A"] + new_state["B"] == pytest.approx(5.0, abs=1e-9)


def test_cle_step_matches_ode_mean_and_scales_noise_with_volume():
    k = 2.0
    state = {"A": 4.0, "B": 1.0}
    dt = 0.02
    net = _linear_ab_network(k=k, method="CLE")
    n_reps = 3000

    def run(volume):
        b_vals = np.empty(n_reps)
        for i in range(n_reps):
            rng = np.random.default_rng(i)
            new_state = net.simulate_step(state, dt=dt, volume=volume, rng=rng)
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
    net = _linear_ab_network(k=1.0, method="SSA")
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
    cell.step(dt=0.1)

    assert cell.concentrations["A"] >= 0.0
    assert cell.concentrations["B"] >= 0.0
    total_copies = (cell.concentrations["A"] + cell.concentrations["B"]) * volume
    assert total_copies == pytest.approx(expected_copies, abs=1e-6)


def test_cell_with_cle_reaction_network():
    net = _linear_ab_network(k=1.0, method="CLE")
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

    cell.step(dt=0.1)

    assert cell.concentrations["A"] >= 0.0
    assert cell.concentrations["B"] >= 0.0
    assert (
        pytest.approx(cell.concentrations["A"] + cell.concentrations["B"], abs=1e-6)
        == 1.0
    )
