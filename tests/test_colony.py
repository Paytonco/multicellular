# tests/test_colony.py

import numpy as np
import pytest

from multicellular.core.cell import Cell
from multicellular.core.colony import Colony
from multicellular.core.environment import Environment


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
