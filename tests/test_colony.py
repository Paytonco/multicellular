# tests/test_colony.py

import numpy as np
import pytest

from multicellular.core.cell import Cell
from multicellular.core.colony import Colony
from multicellular.core.environment import Environment, Field


def _make_cell(position):
    return Cell(
        id=1,
        position=position,
        orientation=[1.0, 0.0],
        length=2.0,
        radius=0.5,
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
