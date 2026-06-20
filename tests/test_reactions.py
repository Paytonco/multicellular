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

    # Create a cell with the network
    cell = Cell(
        id=0,
        position=[0.0, 0.0],
        orientation=[1.0, 0.0],
        length=2.0,
        radius=0.5,
        network=net,
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
