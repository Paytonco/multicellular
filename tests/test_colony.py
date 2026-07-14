# tests/test_colony.py

import copy
import math

import numpy as np
import pytest

from multicellular.core.cell import Cell
from multicellular.core.colony import Colony
from multicellular.core.environment import WATER_DIFFUSIVITY_37C, Environment, Field
from multicellular.core.reactions import Reaction, ReactionNetwork

_KBT = 1.380649e-23 * 310.15  # must match colony.py's _kBT


def _reference_brownian_step(environment, cell, dt, xi):
    """
    Mirrors the original (pre-vectorization) per-cell scalar implementation
    of Colony._apply_brownian_motion, to regression-test the vectorized
    batch version against it.
    """
    shape = environment.shape
    width, height = environment.size
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
    env = Environment("env", wall_map=np.zeros((10, 10)))
    cells = [_make_cell([10.0, 10.0]), _make_cell([20.0, 20.0])]

    colony = Colony(cells, env)

    assert colony.cells == cells
    assert colony.environment is env
    assert colony.living_cells == cells


def test_step_kills_cells_outside_bounds():
    env = Environment("env", wall_map=np.zeros((10, 10)))
    inside = _make_cell([50.0, 50.0])
    outside = _make_cell([150.0, 50.0])

    colony = Colony([inside, outside], env)
    colony.step(dt=0.1)

    assert inside.alive
    assert not outside.alive
    assert colony.living_cells == [inside]


def test_step_kills_cells_with_negative_position():
    env = Environment("env", wall_map=np.zeros((10, 10)))
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
    env = Environment("env", wall_map=np.zeros((10, 10)))
    cell = _make_cell([50.0, 50.0])
    cell.kill()

    colony = Colony([cell], env)
    colony.step(dt=0.1)

    assert not cell.alive
    assert cell.length == 2.0
    assert cell.age == 0.0


def test_survival_condition_kills_cell_when_violated():
    env = Environment("env", wall_map=np.zeros((10, 10)))
    cell = _make_cell([50.0, 50.0])
    cell.set_concentration("A", 0.0)

    colony = Colony([cell], env, survival_conditions=[("A", ">", 0)])
    colony.step(dt=0.1)

    assert not cell.alive


def test_survival_condition_keeps_cell_alive_when_satisfied():
    env = Environment("env", wall_map=np.zeros((10, 10)))
    cell = _make_cell([50.0, 50.0])
    cell.set_concentration("A", 1.0)

    colony = Colony([cell], env, survival_conditions=[("A", ">", 0)])
    colony.step(dt=0.1)

    assert cell.alive


def test_survival_conditions_any_violation_kills():
    env = Environment("env", wall_map=np.zeros((10, 10)))
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
    env = Environment("env", wall_map=np.zeros((10, 10)))
    cell = _make_cell([50.0, 50.0])  # never sets concentration "A"

    colony = Colony([cell], env, survival_conditions=[("A", ">", 0)])
    colony.step(dt=0.1)

    assert not cell.alive


def test_no_survival_conditions_is_a_no_op():
    env = Environment("env", wall_map=np.zeros((10, 10)))
    cell = _make_cell([50.0, 50.0])

    colony = Colony([cell], env)
    colony.step(dt=0.1)

    assert cell.alive
    assert colony.survival_conditions == []


def test_invalid_survival_condition_operator_raises():
    env = Environment("env", wall_map=np.zeros((10, 10)))
    cell = _make_cell([50.0, 50.0])

    with pytest.raises(ValueError):
        Colony([cell], env, survival_conditions=[("A", "?!", 0)])


def test_step_copies_chemical_field_value_into_cell_concentration():
    values = np.zeros((10, 10))
    values[5, 5] = 7.5  # cell at (50, 50) maps to grid index (5, 5)
    field = Field("glucose", values, is_chemical=True)
    env = Environment("env", wall_map=np.zeros((10, 10)), fields=[field])
    cell = _make_cell([50.0, 50.0])

    colony = Colony([cell], env)
    colony.step(dt=0.1)

    assert cell.concentrations["glucose"] == pytest.approx(7.5)


def test_step_chemical_field_concentration_is_not_diluted_by_growth():
    # growth_rate > 0 so the cell actually grows (and would dilute
    # concentrations) within this single step.
    field = Field("glucose", np.full((10, 10), 7.5), is_chemical=True)
    env = Environment("env", wall_map=np.zeros((10, 10)), fields=[field])
    cell = _make_cell([50.0, 50.0], growth_rate=np.log(2))

    colony = Colony([cell], env)
    volume_before = cell.compute_volume()
    colony.step(dt=0.1)
    volume_after = cell.compute_volume()

    assert volume_after > volume_before  # sanity check: cell actually grew
    assert cell.concentrations["glucose"] == pytest.approx(7.5)


def test_step_diffuses_diffusive_fields():
    values = np.zeros((10, 10))
    values[5, 5] = 100.0
    field = Field("glucose", values, is_chemical=True, diffuses=True, diffusivity=1e-9)
    env = Environment(
        "env", wall_map=np.zeros((10, 10)), size=(50.0, 50.0), fields=[field]
    )
    cell = _make_cell([50.0, 50.0])

    colony = Colony([cell], env)
    colony.step(dt=0.5)

    assert field.values[5, 5] < 100.0
    assert field.values[4, 5] > 0.0


def test_step_does_not_copy_non_chemical_field_into_concentrations():
    field = Field("temperature", np.full((10, 10), 37.0), is_chemical=False)
    env = Environment("env", wall_map=np.zeros((10, 10)), fields=[field])
    cell = _make_cell([50.0, 50.0])

    colony = Colony([cell], env)
    colony.step(dt=0.1)

    assert "temperature" not in cell.concentrations


def test_step_chemical_field_overwrites_existing_concentration():
    field = Field("A", np.full((10, 10), 3.0), is_chemical=True)
    env = Environment("env", wall_map=np.zeros((10, 10)), fields=[field])
    cell = _make_cell([50.0, 50.0])
    cell.set_concentration("A", 0.0)

    colony = Colony([cell], env)
    colony.step(dt=0.1)

    assert cell.concentrations["A"] == pytest.approx(3.0)


def _efflux_network(k=1.0, species="X"):
    rxn = Reaction(
        reactants={species: 1},
        products={},
        exports={species: 1},
        rate_law_type="mass_action",
        rate_params={"k": k},
    )
    return ReactionNetwork("efflux", {"R": rxn})


def _make_exporting_cell(position, species="X", concentration=1.0, k=1.0):
    cell = Cell(
        id=1,
        position=position,
        orientation=[1.0, 0.0],
        length=2.0,
        radius=0.5,
        network=_efflux_network(k=k, species=species),
        growth_rate=0.0,
    )
    cell.set_concentration(species, concentration)
    return cell


def test_export_chemical_fields_deposits_into_matching_field():
    field = Field("X", np.zeros((10, 10)))
    env = Environment(
        "env", wall_map=np.zeros((10, 10)), size=(50.0, 50.0), depth=2.0, fields=[field]
    )
    cell = _make_cell([25.0, 25.0])
    cell.pending_export = {"X": 10.0}  # molecules

    colony = Colony([cell], env)
    colony.export_chemical_fields()

    # grid index for (25,25): dx=dy=5 -> (i,j)=(5,5); grid_cell_volume = 5*5*2=50
    assert field.values[5, 5] == pytest.approx(10.0 / 50.0)
    assert cell.pending_export == {}


def test_export_chemical_fields_sums_contributions_at_same_grid_point():
    field = Field("X", np.zeros((10, 10)))
    env = Environment(
        "env", wall_map=np.zeros((10, 10)), size=(50.0, 50.0), depth=2.0, fields=[field]
    )
    cell1 = _make_cell([25.0, 25.0])
    cell2 = _make_cell([26.0, 26.0])  # same grid cell as cell1
    cell1.pending_export = {"X": 10.0}
    cell2.pending_export = {"X": 5.0}

    colony = Colony([cell1, cell2], env)
    colony.export_chemical_fields()

    assert field.values[5, 5] == pytest.approx(15.0 / 50.0)


def test_export_chemical_fields_is_noop_when_nothing_pending():
    field = Field("X", np.zeros((10, 10)))
    env = Environment("env", wall_map=np.zeros((10, 10)), fields=[field])
    cell = _make_cell([25.0, 25.0])  # pending_export defaults to {}

    colony = Colony([cell], env)
    colony.export_chemical_fields()  # must not raise on the empty-living-list path

    assert np.array_equal(field.values, np.zeros((10, 10)))


def test_export_chemical_fields_raises_for_unmapped_species_without_mutating_fields():
    field = Field("Y", np.zeros((10, 10)))
    env = Environment("env", wall_map=np.zeros((10, 10)), fields=[field])
    cell = _make_cell([25.0, 25.0])
    cell.pending_export = {"X": 10.0}  # no Field named "X" exists

    colony = Colony([cell], env)
    with pytest.raises(ValueError):
        colony.export_chemical_fields()

    assert np.array_equal(field.values, np.zeros((10, 10)))


def test_step_exports_efflux_into_field_and_depletes_cell():
    field = Field("X", np.zeros((10, 10)))
    env = Environment(
        "env", wall_map=np.zeros((10, 10)), size=(50.0, 50.0), fields=[field]
    )
    cell = _make_exporting_cell([25.0, 25.0], concentration=1.0, k=1.0)

    colony = Colony([cell], env)
    colony.step(dt=0.1)

    assert cell.concentrations["X"] < 1.0
    assert field.values[5, 5] > 0.0


def test_step_conserves_mass_across_cell_and_field():
    field = Field("X", np.zeros((10, 10)))
    env = Environment(
        "env", wall_map=np.zeros((10, 10)), size=(50.0, 50.0), depth=2.0, fields=[field]
    )
    cell = _make_exporting_cell([25.0, 25.0], concentration=1.0, k=1.0)
    volume = cell.compute_volume()
    initial_copies = 1.0 * volume

    colony = Colony([cell], env)
    colony.step(dt=0.1)

    remaining_copies = cell.concentrations["X"] * cell.compute_volume()
    deposited_copies = field.values[5, 5] * env.grid_cell_volume
    assert remaining_copies + deposited_copies == pytest.approx(
        initial_copies, abs=1e-9
    )


def test_apply_chemical_fields_skips_dead_cells():
    field = Field("A", np.full((10, 10), 3.0), is_chemical=True)
    env = Environment("env", wall_map=np.zeros((10, 10)), fields=[field])
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
        "env",
        wall_map=np.zeros((4, 4)),
        size=(40.0, 40.0),
        eta=eta,
        diffusivity=diffusivity,
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
    env = Environment("env", wall_map=np.zeros((4, 4)))
    colony = Colony([], env)

    # Should not raise even though there are no cells to vectorize over.
    colony._apply_brownian_motion(dt=0.1, alive=[], noise=[])


def test_switch_environment_replaces_environment():
    env = Environment("env", wall_map=np.zeros((10, 10)))
    other_env = Environment("other", wall_map=np.zeros((10, 10)))
    cell = _make_cell([50.0, 50.0])
    colony = Colony([cell], env)

    colony.switch_environment(other_env)

    assert colony.environment is other_env


def test_switch_environment_changes_chemical_field_sampled_on_next_step():
    before = Field("A", np.full((10, 10), 1.0), is_chemical=True)
    env_before = Environment("before", wall_map=np.zeros((10, 10)), fields=[before])
    cell = _make_cell([50.0, 50.0])
    colony = Colony([cell], env_before)
    colony.step(dt=0.1)
    assert cell.concentrations["A"] == pytest.approx(1.0)

    after = Field("A", np.full((10, 10), 9.0), is_chemical=True)
    env_after = Environment("after", wall_map=np.zeros((10, 10)), fields=[after])
    colony.switch_environment(env_after)
    colony.step(dt=0.1)

    assert cell.concentrations["A"] == pytest.approx(9.0)


def test_step_replaces_dividing_cell_with_daughters():
    env = Environment("env", wall_map=np.zeros((20, 20)))
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


def _wall_block_env(size=(100.0, 100.0), shape=(10, 10)):
    # A 20x20um solid wall block at the center of a 100x100um environment:
    # x in [40, 60], y in [40, 60].
    wall_map = np.zeros(shape)
    wall_map[4:6, 4:6] = 1
    return Environment("wall test", wall_map=wall_map, size=size)


def test_wall_force_pushes_cell_off_a_flat_face():
    env = _wall_block_env()
    # Endpoint just barely penetrating the wall's left face (x=40); the cell
    # itself sits fully to the left of the block, aligned with the face.
    cell = _make_cell([40.7, 50.0])
    colony = Colony([cell], env, k=10.0)

    for _ in range(200):
        colony._apply_contact_forces(dt=0.01, alive=[cell])

    # Both endpoints are nearest to (and so interact with) the west face, so
    # equilibrium isn't reached until the *whole* rod has cleared the block,
    # not just its nearer end: the far endpoint (center + length/2) settles
    # with its cylindrical surface (radius 0.5) just touching x=40, i.e. at
    # x=39.5, putting the center (endpoints are +/- length/2 = +/- 1.0) at
    # x=38.5.
    assert cell.position[0] == pytest.approx(38.5, abs=1e-3)
    assert cell.position[1] == pytest.approx(50.0, abs=1e-9)


def test_wall_force_is_zero_when_cell_is_clear_of_every_face():
    env = _wall_block_env()
    cell = _make_cell([10.0, 10.0])
    colony = Colony([cell], env, k=10.0)

    colony._apply_contact_forces(dt=0.01, alive=[cell])

    assert cell.position == pytest.approx([10.0, 10.0])


def test_wall_force_does_not_trigger_from_the_far_side_of_a_thick_wall():
    # A cell endpoint mildly penetrating the near (left) face of a thick
    # block must not also register a huge spurious contact against the far
    # (right) face's infinite supporting line.
    env = _wall_block_env()
    cell = _make_cell([40.7, 50.0])
    colony = Colony([cell], env, k=10.0)

    colony._apply_contact_forces(dt=0.01, alive=[cell])

    # A single step of mild penetration against the (now much stiffer
    # default) k_wall should nudge the cell left, but not overshoot past
    # the true equilibrium (center at x=38.5, a distance of 2.2 from the
    # start) -- which a spurious contact against the far face's infinite
    # line would do.
    assert -2.2 < cell.position[0] - 40.7 < 0.0


def test_wall_force_face_does_not_reach_beyond_its_finite_extent():
    # A cell sitting just past the end of a wall run, close to where the
    # face's *infinite* supporting line would be, must not be blocked --
    # only the face's own finite span (and the corner point at its end)
    # can produce a contact. This is what makes an open end of a wall run
    # pass cells through freely instead of being pushed off an unbounded
    # plane (wallSpec.txt sec 2).
    wall_map = np.zeros((10, 10))
    wall_map[5, 0:4] = 1  # wall run: x in [0, 40], y in [50, 60]
    env = Environment("open end", wall_map=wall_map, size=(100.0, 100.0))

    cell = Cell(
        id=1,
        position=[41.0, 59.9],
        orientation=[1.0, 0.0],
        length=0.2,
        radius=0.05,
        growth_rate=0.0,
    )
    colony = Colony([cell], env, k=10.0)

    colony._apply_contact_forces(dt=0.01, alive=[cell])

    assert cell.position == pytest.approx([41.0, 59.9])


def test_wall_corner_stops_a_cell_rounding_a_peninsula_tip():
    # A single wall pixel surrounded by media on all sides is a "peninsula
    # tip": a cell axis passing near its corner should be stopped by the
    # corner-point contact even though it isn't aligned with any face.
    wall_map = np.zeros((10, 10))
    wall_map[5, 5] = 1  # x in [50, 60], y in [50, 60]
    env = Environment("tip", wall_map=wall_map, size=(100.0, 100.0))

    # Cell centered directly below the block's corner at (50, 50), close
    # enough for its radius to overlap the corner point. Symmetric about
    # the corner along x, so the contact is a pure -y push with no torque.
    cell = Cell(
        id=1,
        position=[50.0, 49.7],
        orientation=[1.0, 0.0],
        length=2.0,
        radius=0.4,
        growth_rate=0.0,
    )
    colony = Colony([cell], env, k=10.0)

    colony._apply_contact_forces(dt=0.01, alive=[cell])

    # Pushed away from the corner, straight down.
    assert cell.position[1] < 49.7
    assert cell.position[0] == pytest.approx(50.0, abs=1e-9)


def test_k_wall_defaults_to_ten_times_k():
    env = _wall_block_env()
    colony = Colony([], env, k=7.5)
    assert colony.k_wall == 75.0


def test_k_wall_can_be_overridden():
    env = _wall_block_env()
    colony = Colony([], env, k=10.0, k_wall=100.0)
    assert colony.k_wall == 100.0


def test_brownian_motion_defaults_to_true():
    env = Environment("env", wall_map=np.zeros((10, 10)))
    colony = Colony([], env)
    assert colony.brownian_motion is True


def test_brownian_motion_can_be_disabled():
    env = Environment("env", wall_map=np.zeros((10, 10)))
    colony = Colony([], env, brownian_motion=False)
    assert colony.brownian_motion is False


def test_step_moves_cell_via_brownian_motion_when_enabled():
    env = Environment("env", wall_map=np.zeros((10, 10)))
    cell = _make_cell([50.0, 50.0])
    colony = Colony([cell], env, brownian_motion=True)

    colony.step(dt=0.1)

    assert not np.array_equal(cell.position, [50.0, 50.0])


def test_step_does_not_move_isolated_cell_when_brownian_motion_disabled():
    # No neighbors, no walls, and no other force source in this environment,
    # so with Brownian motion off the cell must not move at all.
    env = Environment("env", wall_map=np.zeros((10, 10)))
    cell = _make_cell([50.0, 50.0])
    colony = Colony([cell], env, brownian_motion=False)

    colony.step(dt=0.1)

    assert np.array_equal(cell.position, [50.0, 50.0])


def test_wall_forces_apply_even_with_a_single_living_cell():
    # Regression test: wall contact must not be skipped by the same "n < 2"
    # shortcut that (correctly) skips the cell-cell double loop.
    env = _wall_block_env()
    cell = _make_cell([40.7, 50.0])
    colony = Colony([cell], env, k=10.0)

    colony._apply_contact_forces(dt=0.01, alive=[cell])

    assert cell.position[0] != pytest.approx(40.7)
