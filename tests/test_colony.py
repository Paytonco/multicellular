# tests/test_colony.py

import copy
import math

import numpy as np
import pytest

from multicellular.core.cell import Cell
from multicellular.core.colony import Colony
from multicellular.core.environment import WATER_DIFFUSIVITY_37C, Environment, Field

_KBT = 1.380649e-23 * 310.15  # must match colony.py's _kBT


def _reference_brownian_step(environment, cell, dt, xi):
    """
    Mirrors the original (pre-vectorization) per-cell scalar implementation
    of Colony._apply_brownian_motion, to regression-test the vectorized
    batch version against it.
    """
    shape = environment.shape
    width, height = environment.bounds
    j = int(np.clip(cell.position[0] / width * shape[1], 0, shape[1] - 1))
    i = int(np.clip(cell.position[1] / height * shape[0], 0, shape[0] - 1))
    eta = environment.eta[i, j]
    D_field = environment.diffusivity[i, j]

    eta_local = eta * (WATER_DIFFUSIVITY_37C / D_field)

    L_eff = (cell.length + 2.0 * cell.radius) * 1e-6
    r = cell.radius * 1e-6
    ln_ratio = math.log(L_eff / r)

    gamma_par = 2.0 * math.pi * eta_local * L_eff / ln_ratio
    gamma_perp = 4.0 * math.pi * eta_local * L_eff / ln_ratio
    gamma_rot = math.pi * eta_local * L_eff**3 / (3.0 * ln_ratio)

    D_par = _KBT / gamma_par
    D_perp = _KBT / gamma_perp
    D_rot = _KBT / gamma_rot

    ux, uy = float(cell.orientation[0]), float(cell.orientation[1])
    uperp_x, uperp_y = -uy, ux
    xi_par, xi_perp, xi_rot = xi

    s_par = math.sqrt(2.0 * D_par * dt)
    s_perp = math.sqrt(2.0 * D_perp * dt)

    dr_x = (ux * xi_par * s_par + uperp_x * xi_perp * s_perp) * 1e6
    dr_y = (uy * xi_par * s_par + uperp_y * xi_perp * s_perp) * 1e6
    cell.position[0] += dr_x
    cell.position[1] += dr_y

    cell.apply_torque(xi_rot * math.sqrt(2.0 * D_rot / dt), dt)


def _make_cell(position, growth_rate=0.0):
    # growth_rate=0.0 by default so mechanics/survival/field tests aren't
    # entangled with growth-driven concentration dilution (see test_cell.py
    # for dedicated dilution tests).
    return Cell(
        id=1,
        position=position,
        orientation=[1.0, 0.0],
        length=2.0,
        radius=0.5,
        growth_rate=growth_rate,
    )


def test_colony_holds_cells_and_environment():
    env = Environment(shape=(10, 10))
    cells = [_make_cell([10.0, 10.0]), _make_cell([20.0, 20.0])]

    colony = Colony(cells, env)

    assert colony.cells == cells
    assert colony.environment is env
    assert colony.living_cells == cells


def test_step_kills_cells_outside_bounds():
    env = Environment(shape=(10, 10))
    inside = _make_cell([50.0, 50.0])
    outside = _make_cell([150.0, 50.0])

    colony = Colony([inside, outside], env)
    colony.step(dt=0.1)

    assert inside.alive
    assert not outside.alive
    assert colony.living_cells == [inside]


def test_step_kills_cells_with_negative_position():
    env = Environment(shape=(10, 10))
    outside = _make_cell([-1.0, 50.0])

    colony = Colony([outside], env)
    colony.step(dt=0.1)

    assert not outside.alive


def test_dead_cells_do_not_grow_or_simulate_or_feel_force():
    cell = _make_cell([50.0, 50.0])
    cell.kill()

    length_before = cell.length
    age_before = cell.age
    position_before = cell.position.copy()

    cell.step(dt=1.0)
    cell.apply_force(np.array([1.0, 0.0]), dt=1.0)

    assert cell.length == length_before
    assert cell.age == age_before
    assert np.array_equal(cell.position, position_before)


def test_dead_cell_stays_dead_after_colony_step():
    env = Environment(shape=(10, 10))
    cell = _make_cell([50.0, 50.0])
    cell.kill()

    colony = Colony([cell], env)
    colony.step(dt=0.1)

    assert not cell.alive
    assert cell.length == 2.0
    assert cell.age == 0.0


def test_survival_condition_kills_cell_when_violated():
    env = Environment(shape=(10, 10))
    cell = _make_cell([50.0, 50.0])
    cell.set_concentration("A", 0.0)

    colony = Colony([cell], env, survival_conditions=[("A", ">", 0)])
    colony.step(dt=0.1)

    assert not cell.alive


def test_survival_condition_keeps_cell_alive_when_satisfied():
    env = Environment(shape=(10, 10))
    cell = _make_cell([50.0, 50.0])
    cell.set_concentration("A", 1.0)

    colony = Colony([cell], env, survival_conditions=[("A", ">", 0)])
    colony.step(dt=0.1)

    assert cell.alive


def test_survival_conditions_any_violation_kills():
    env = Environment(shape=(10, 10))
    ok_cell = _make_cell([50.0, 50.0])
    ok_cell.set_concentration("A", 1.0)
    ok_cell.set_concentration("B", 1.0)

    bad_cell = _make_cell([60.0, 60.0])
    bad_cell.set_concentration("A", 1.0)
    bad_cell.set_concentration("B", 20.0)  # violates the "B" < 10 condition

    colony = Colony(
        [ok_cell, bad_cell], env, survival_conditions=[("A", ">", 0), ("B", "<", 10)]
    )
    colony.step(dt=0.1)

    assert ok_cell.alive
    assert not bad_cell.alive


def test_survival_condition_missing_species_defaults_to_zero():
    env = Environment(shape=(10, 10))
    cell = _make_cell([50.0, 50.0])  # never sets concentration "A"

    colony = Colony([cell], env, survival_conditions=[("A", ">", 0)])
    colony.step(dt=0.1)

    assert not cell.alive


def test_no_survival_conditions_is_a_no_op():
    env = Environment(shape=(10, 10))
    cell = _make_cell([50.0, 50.0])

    colony = Colony([cell], env)
    colony.step(dt=0.1)

    assert cell.alive
    assert colony.survival_conditions == []


def test_invalid_survival_condition_operator_raises():
    env = Environment(shape=(10, 10))
    cell = _make_cell([50.0, 50.0])

    with pytest.raises(ValueError):
        Colony([cell], env, survival_conditions=[("A", "?!", 0)])


def test_step_copies_chemical_field_value_into_cell_concentration():
    values = np.zeros((10, 10))
    values[5, 5] = 7.5  # cell at (50, 50) maps to grid index (5, 5)
    field = Field("glucose", values, is_chemical=True)
    env = Environment(shape=(10, 10), fields=[field])
    cell = _make_cell([50.0, 50.0])

    colony = Colony([cell], env)
    colony.step(dt=0.1)

    assert cell.concentrations["glucose"] == pytest.approx(7.5)


def test_step_chemical_field_concentration_is_not_diluted_by_growth():
    # growth_rate > 0 so the cell actually grows (and would dilute
    # concentrations) within this single step.
    field = Field("glucose", np.full((10, 10), 7.5), is_chemical=True)
    env = Environment(shape=(10, 10), fields=[field])
    cell = _make_cell([50.0, 50.0], growth_rate=np.log(2))

    colony = Colony([cell], env)
    volume_before = cell.compute_volume()
    colony.step(dt=0.1)
    volume_after = cell.compute_volume()

    assert volume_after > volume_before  # sanity check: cell actually grew
    assert cell.concentrations["glucose"] == pytest.approx(7.5)


def test_step_does_not_copy_non_chemical_field_into_concentrations():
    field = Field("temperature", np.full((10, 10), 37.0), is_chemical=False)
    env = Environment(shape=(10, 10), fields=[field])
    cell = _make_cell([50.0, 50.0])

    colony = Colony([cell], env)
    colony.step(dt=0.1)

    assert "temperature" not in cell.concentrations


def test_step_chemical_field_overwrites_existing_concentration():
    field = Field("A", np.full((10, 10), 3.0), is_chemical=True)
    env = Environment(shape=(10, 10), fields=[field])
    cell = _make_cell([50.0, 50.0])
    cell.set_concentration("A", 0.0)

    colony = Colony([cell], env)
    colony.step(dt=0.1)

    assert cell.concentrations["A"] == pytest.approx(3.0)


def test_apply_chemical_fields_skips_dead_cells():
    field = Field("A", np.full((10, 10), 3.0), is_chemical=True)
    env = Environment(shape=(10, 10), fields=[field])
    cell = _make_cell([50.0, 50.0])
    cell.kill()

    colony = Colony([cell], env)
    colony.apply_chemical_fields()

    assert "A" not in cell.concentrations


def test_apply_brownian_motion_matches_reference_per_cell_formula():
    # Non-uniform fields so each cell samples a different (eta, diffusivity)
    # pair, exercising that the vectorized gather maps each cell to its own
    # grid index rather than mixing values up across cells.
    eta = np.array(
        [
            [1e-3, 2e-3, 3e-3, 4e-3],
            [5e-3, 6e-3, 7e-3, 8e-3],
            [9e-3, 1.0e-2, 1.1e-2, 1.2e-2],
            [1.3e-2, 1.4e-2, 1.5e-2, 1.6e-2],
        ]
    )
    diffusivity = 1e-9 * (1.0 + np.arange(16).reshape(4, 4))
    env = Environment(
        shape=(4, 4), bounds=(40.0, 40.0), eta=eta, diffusivity=diffusivity
    )

    cells = [
        Cell(
            id=0,
            position=[10.0, 10.0],
            orientation=[1.0, 0.0],
            length=3.0,
            radius=0.4,
            growth_rate=0.0,
        ),
        Cell(
            id=1,
            position=[30.0, 5.0],
            orientation=[0.0, 1.0],
            length=5.0,
            radius=0.6,
            growth_rate=0.0,
        ),
        Cell(
            id=2,
            position=[15.0, 25.0],
            orientation=[0.6, 0.8],
            length=2.5,
            radius=0.3,
            growth_rate=0.0,
        ),
    ]
    colony = Colony(cells, env)
    noise = [
        np.array([0.3, -0.2, 0.1]),
        np.array([-0.4, 0.5, -0.3]),
        np.array([0.7, 0.1, -0.6]),
    ]
    dt = 0.05

    expected_cells = [copy.deepcopy(c) for c in cells]
    for cell, xi in zip(expected_cells, noise):
        _reference_brownian_step(env, cell, dt, xi)

    colony._apply_brownian_motion(dt, colony.cells, noise)

    for cell, expected in zip(colony.cells, expected_cells):
        assert np.allclose(cell.position, expected.position)
        assert np.allclose(cell.orientation, expected.orientation)


def test_apply_brownian_motion_no_op_on_empty_alive_list():
    env = Environment(shape=(4, 4))
    colony = Colony([], env)

    # Should not raise even though there are no cells to vectorize over.
    colony._apply_brownian_motion(dt=0.1, alive=[], noise=[])


def test_switch_environment_replaces_environment():
    env = Environment(shape=(10, 10))
    other_env = Environment(shape=(10, 10))
    cell = _make_cell([50.0, 50.0])
    colony = Colony([cell], env)

    colony.switch_environment(other_env)

    assert colony.environment is other_env


def test_switch_environment_changes_chemical_field_sampled_on_next_step():
    before = Field("A", np.full((10, 10), 1.0), is_chemical=True)
    env_before = Environment(shape=(10, 10), fields=[before])
    cell = _make_cell([50.0, 50.0])
    colony = Colony([cell], env_before)
    colony.step(dt=0.1)
    assert cell.concentrations["A"] == pytest.approx(1.0)

    after = Field("A", np.full((10, 10), 9.0), is_chemical=True)
    env_after = Environment(shape=(10, 10), fields=[after])
    colony.switch_environment(env_after)
    colony.step(dt=0.1)

    assert cell.concentrations["A"] == pytest.approx(9.0)


def test_step_replaces_dividing_cell_with_daughters():
    env = Environment(shape=(20, 20))
    rng = np.random.default_rng(0)
    cell = Cell(
        id=0, position=[10.0, 10.0], orientation=[1.0, 0.0], length=2.0, rng=rng
    )
    # Advance to the division target so colony.step triggers division.
    cell.length = cell._division_target
    expected_daughter_length = cell._division_target / 2.0

    colony = Colony([cell], env)
    colony.step(dt=0.1)

    assert cell not in colony.cells
    assert len(colony.cells) == 2

    ids = {daughter.id for daughter in colony.cells}
    assert ids == {1, 2}
    for daughter in colony.cells:
        assert daughter.alive
        assert daughter.length == pytest.approx(expected_daughter_length)
