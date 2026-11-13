# tests/test_batch_ode.py
"""
The "numba" backend's batched cell stepping (core/_batch.py) against the
one-cell-at-a-time "python" backend.
"""

import numpy as np
import pytest

from multicellular.core.cell import Cell
from multicellular.core.colony import Colony
from multicellular.core.environment import Environment, Field
from multicellular.core.reactions import Reaction, ReactionNetwork

SIZE = 100.0
GRID = 10


def _circuit():
    """Every built-in rate law, plus an export reaction."""
    return ReactionNetwork(
        "circuit",
        {
            "make_A": Reaction({}, {"A": 1}, rate_params={"k": 2.0}),
            "deg_A": Reaction({"A": 1}, {}, rate_params={"k": 0.3}),
            "convert": Reaction(
                {"A": 1},
                {"B": 1},
                catalysts=["E"],
                rate_law_type="michaelis_menten",
                rate_params={"Vmax": 1.5, "Km": 2.0},
            ),
            "express": Reaction(
                {},
                {"G": 1},
                catalysts=["E", "B"],
                rate_law_type="hill_langmuir",
                rate_params={"alpha": 50, "beta": 0.2, "C": 0.5, "n": 2},
            ),
            "deg_G": Reaction({"G": 1}, {}, rate_params={"k": 0.1}),
            "dimerize": Reaction({"B": 2}, {"D": 1}, rate_params={"k": 0.05}),
            "secrete_D": Reaction(
                {"D": 1}, {}, rate_params={"k": 0.4}, exports={"X": 1}
            ),
        },
    )


def _exchange(k_out=0.5, k_in=0.2):
    """A is secreted into, and taken back up from, the AHL field."""
    return ReactionNetwork(
        "exchange",
        {
            "make_A": Reaction({}, {"A": 1}, rate_params={"k": 1.0}),
            "secrete": Reaction({"A": 1}, {"AHL": 1}, rate_params={"k": k_out}),
            "uptake": Reaction({"AHL": 1}, {"A": 1}, rate_params={"k": k_in}),
        },
    )


def _custom_network():
    def law(reactants, products, catalysts, params):
        return params["k"] / (1.0 + max(reactants["A"], 0.0))

    return ReactionNetwork(
        "custom",
        {
            "make": Reaction(
                {"A": 1},
                {"A": 2},
                rate_law_type="custom",
                rate_params={"k": 1.0},
                custom_rate_law=law,
            ),
            "deg_A": Reaction({"A": 1}, {}, rate_params={"k": 0.2}),
        },
    )


def _env(fields=None, diffusivity=1e-11):
    if fields is not None:
        fields = [
            Field(
                name,
                np.full((GRID, GRID), value),
                is_chemical=True,
                diffuses=True,
                diffusivity=diffusivity,
            )
            for name, value in fields.items()
        ]
    return Environment(
        "env",
        wall_map=np.zeros((GRID, GRID)),
        size=(SIZE, SIZE),
        fields=fields,
        rng=np.random.default_rng(0),
    )


def _colony(
    backend,
    network_factory,
    n=12,
    fields=None,
    cluster=False,
    seed=1,
    concentrations=None,
):
    rng = np.random.default_rng(seed)
    cells = []
    for i in range(n):
        if cluster:  # every cell in the same voxel
            position = [55.0 + 0.1 * i, 55.0]
        else:
            position = [10.0 + 7.0 * (i % 10), 15.0 + 25.0 * (i // 10)]
        cell = Cell(
            id=i,
            position=position,
            orientation=[1.0, 0.0],
            length=1.0 + 0.1 * i,
            rng=rng,
            network=network_factory(),
            growth_rate=0.3,
        )
        for s, c in (concentrations or {}).items():
            cell.concentrations[s] = c
        cells.append(cell)
    return Colony(
        cells, _env(fields), brownian_motion=False, k=0.0, k_wall=0.0, backend=backend
    )


def _states(colony):
    return [dict(cell.concentrations) for cell in colony.cells]


def _assert_states_close(a, b, rel, abs_=1e-9):
    assert len(a) == len(b)
    for sa, sb in zip(a, b):
        assert set(sa) == set(sb)
        for s in sa:
            assert sa[s] == pytest.approx(sb[s], rel=rel, abs=abs_), s


@pytest.mark.parametrize("ode_method", ["RK45", "BDF", "LSODA"])
def test_uncoupled_cells_match_python_backend(ode_method):
    init = {"A": 1.0, "E": 1.0}
    fast = _colony("numba", _circuit, concentrations=init, fields={"X": 0.0})
    slow = _colony("python", _circuit, concentrations=init, fields={"X": 0.0})
    for _ in range(5):
        fast.step(0.2, "ODE", ode_method)
        slow.step(0.2, "ODE", ode_method)
    _assert_states_close(_states(fast), _states(slow), rel=1e-5)
    # Exported molecules reach the field identically.
    np.testing.assert_allclose(
        fast.environment.fields["X"].values,
        slow.environment.fields["X"].values,
        rtol=1e-5,
        atol=1e-12,
    )
    assert fast.environment.fields["X"].values.sum() > 0.0


def _total_molecules(colony):
    env = colony.environment
    field = env.fields["AHL"].values.sum() * env.grid_cell_volume
    cells = sum(cell.copy_number("A") for cell in colony.cells)
    return field + cells


@pytest.mark.parametrize("method", ["ODE", "SSA", "CLE"])
def test_coupled_exchange_conserves_molecules(method):
    colony = _colony(
        "numba",
        _exchange,
        n=20,
        fields={"AHL": 0.01},
        cluster=True,
        concentrations={"A": 5.0},
    )
    # make_A creates molecules; switch it off so the total is conserved.
    # No growth: division hands each daughter the parent's concentration,
    # which doesn't conserve copies (on either backend).
    for cell in colony.cells:
        cell.network.reactions["make_A"].rate_params["k"] = 0.0
        cell.growth_rate = 0.0
    # Measure from after the first step: SSA rounds each cell's initial
    # (fractional) copy number to whole molecules once.
    colony.step(0.1, method)
    before = _total_molecules(colony)
    for _ in range(10):
        colony.step(0.1, method)
    after = _total_molecules(colony)
    assert after == pytest.approx(before, rel=1e-9)
    assert colony.environment.fields["AHL"].values.max() > 0.01  # exchange happened


@pytest.mark.parametrize("ode_method", ["RK45", "BDF", "LSODA"])
def test_joint_coupled_solve_agrees_with_sequential_python_at_small_dt(ode_method):
    # Cells sharing a voxel: the joint solve draws from the pool
    # simultaneously, the Python path in sequence. The two agree as dt -> 0.
    kwargs = dict(n=15, fields={"AHL": 0.2}, cluster=True, concentrations={"A": 1.0})
    fast = _colony("numba", _exchange, **kwargs)
    slow = _colony("python", _exchange, **kwargs)
    for _ in range(20):
        fast.step(0.01, "ODE", ode_method)
        slow.step(0.01, "ODE", ode_method)
    _assert_states_close(_states(fast), _states(slow), rel=1e-3)
    np.testing.assert_allclose(
        fast.environment.fields["AHL"].values,
        slow.environment.fields["AHL"].values,
        rtol=1e-3,
    )


def test_custom_rate_law_falls_back_to_python_path():
    fast = _colony("numba", _custom_network, concentrations={"A": 1.0})
    slow = _colony("python", _custom_network, concentrations={"A": 1.0})
    for _ in range(3):
        fast.step(0.1, "ODE")
        slow.step(0.1, "ODE")
    _assert_states_close(_states(fast), _states(slow), rel=0, abs_=0)


def test_mixed_compiled_and_custom_cells_step_together():
    colony = _colony(
        "numba", _circuit, concentrations={"A": 1.0, "E": 1.0}, fields={"X": 0.0}
    )
    custom = Cell(
        id=None,
        position=[50.0, 80.0],
        orientation=[1.0, 0.0],
        network=_custom_network(),
        rng=np.random.default_rng(3),
    )
    custom.concentrations["A"] = 1.0
    colony.cells.append(custom)
    colony.step(0.1, "ODE")
    assert custom.concentrations["A"] != 1.0
    assert all(cell.concentrations["A"] != 1.0 for cell in colony.cells)


def test_rate_parameter_edits_take_effect_between_steps():
    colony = _colony(
        "numba", _circuit, n=3, concentrations={"A": 0.0, "E": 1.0}, fields={"X": 0.0}
    )
    for cell in colony.cells:
        cell.network.reactions["make_A"].rate_params["k"] = 0.0
    colony.step(0.1, "ODE")
    assert all(cell.concentrations["A"] == 0.0 for cell in colony.cells)
    for cell in colony.cells:
        cell.network.reactions["make_A"].rate_params["k"] = 5.0
    colony.step(0.1, "ODE")
    assert all(cell.concentrations["A"] > 0.0 for cell in colony.cells)


def test_cells_without_network_still_grow():
    colony = _colony("numba", _circuit, n=2, fields={"X": 0.0})
    bare = Cell(
        id=None,
        position=[50.0, 80.0],
        orientation=[1.0, 0.0],
        growth_rate=0.5,
        rng=np.random.default_rng(0),
    )
    colony.cells.append(bare)
    length = bare.length
    colony.step(0.1, "ODE")
    assert bare.length == pytest.approx(length * np.exp(0.05))


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        Colony([], _env(), backend="gpu")
