# tests/test_environment.py

import numpy as np
import pytest

from multicellular.core.environment import Environment, Field


def test_environment_default_size_is_100_by_100():
    env = Environment("env", wall_map=np.zeros((10, 10)))
    assert env.size == (100.0, 100.0)


def test_environment_size_can_be_specified():
    env = Environment("env", wall_map=np.zeros((10, 10)), size=(20.0, 20.0))
    assert env.size == (20.0, 20.0)
    assert env.in_bounds([10.0, 10.0])
    assert not env.in_bounds([50.0, 50.0])


def test_environment_shape_comes_from_wall_map():
    env = Environment("env", wall_map=np.zeros((7, 3)))
    assert env.shape == (7, 3)


def test_wall_map_rejects_invalid_entries():
    wall_map = np.zeros((5, 5))
    wall_map[2, 2] = 2
    with pytest.raises(ValueError):
        Environment("env", wall_map=wall_map)


def test_wall_map_rejects_non_2d_matrix():
    with pytest.raises(ValueError):
        Environment("env", wall_map=np.zeros((5, 5, 2)))


def test_wall_map_accepts_only_minus_one_zero_one():
    wall_map = np.array([[-1, 0, 1], [1, 0, -1]])
    env = Environment("env", wall_map=wall_map)
    assert np.array_equal(env.wall_map, wall_map)


def test_in_bounds_true_for_media_cell():
    wall_map = np.zeros((10, 10))
    env = Environment("env", wall_map=wall_map, size=(100.0, 100.0))
    assert env.in_bounds([50.0, 50.0])


def test_in_bounds_false_for_out_of_bounds_wall_map_entry():
    wall_map = np.zeros((10, 10))
    wall_map[0, :] = -1  # bottom row (low y) is a death zone
    env = Environment("env", wall_map=wall_map, size=(100.0, 100.0))
    assert not env.in_bounds([50.0, 5.0])
    assert env.in_bounds([50.0, 50.0])


def test_in_bounds_true_for_wall_cell():
    # Walls are still "in bounds" -- cells are pushed off them by contact
    # forces rather than being killed, unlike -1 out-of-bounds cells.
    wall_map = np.zeros((10, 10))
    wall_map[5, 5] = 1
    env = Environment("env", wall_map=wall_map, size=(100.0, 100.0))
    assert env.in_bounds([55.0, 55.0])


def test_in_bounds_false_outside_physical_extent():
    env = Environment("env", wall_map=np.zeros((10, 10)), size=(100.0, 100.0))
    assert not env.in_bounds([150.0, 50.0])
    assert not env.in_bounds([-1.0, 50.0])


def test_add_and_get_field():
    env = Environment("env", wall_map=np.zeros((10, 10)))
    field = Field("glucose", np.ones((10, 10)))

    env.add_field(field)

    assert env.get_field("glucose") is field
    assert np.array_equal(env.get_field("glucose").values, np.ones((10, 10)))


def test_environment_constructor_accepts_fields():
    field = Field("temperature", np.zeros((5, 5)))
    env = Environment("env", wall_map=np.zeros((5, 5)), fields=[field])

    assert env.get_field("temperature") is field


def test_field_shape_must_match_wall_map_shape():
    field = Field("roughness", np.zeros((200, 200)))
    env = Environment("env", wall_map=np.zeros((200, 200)), fields=[field])

    assert env.get_field("roughness").shape == (200, 200)
    assert env.size == (100.0, 100.0)


def test_environment_name_is_stored():
    env = Environment("LB medium", wall_map=np.zeros((10, 10)))
    assert env.name == "LB medium"


def test_field_is_chemical_defaults_to_false():
    field = Field("glucose", np.ones((10, 10)))
    assert field.is_chemical is False


def test_field_is_chemical_can_be_set_true():
    field = Field("glucose", np.ones((10, 10)), is_chemical=True)
    assert field.is_chemical is True


def test_add_field_with_mismatched_shape_raises():
    env = Environment("env", wall_map=np.zeros((10, 10)))
    bad_field = Field("glucose", np.ones((5, 5)))

    with pytest.raises(ValueError):
        env.add_field(bad_field)


def test_constructor_with_mismatched_field_shape_raises():
    bad_field = Field("glucose", np.ones((5, 5)))

    with pytest.raises(ValueError):
        Environment("env", wall_map=np.zeros((10, 10)), fields=[bad_field])


def test_field_diffuses_defaults_to_false():
    field = Field("glucose", np.ones((10, 10)))
    assert field.diffuses is False
    assert field.diffusivity is None


def test_field_diffuses_true_requires_diffusivity():
    with pytest.raises(ValueError):
        Field("glucose", np.ones((10, 10)), diffuses=True)


def test_field_diffuses_true_rejects_nonpositive_diffusivity():
    with pytest.raises(ValueError):
        Field("glucose", np.ones((10, 10)), diffuses=True, diffusivity=0.0)


def test_field_diffuses_true_stores_diffusivity():
    field = Field("glucose", np.ones((10, 10)), diffuses=True, diffusivity=5e-10)
    assert field.diffuses is True
    assert field.diffusivity == 5e-10


def test_diffuse_is_no_op_when_no_diffusive_fields():
    field = Field("temperature", np.full((5, 5), 37.0))
    env = Environment("env", wall_map=np.zeros((5, 5)), fields=[field])

    env.diffuse(1.0)

    assert np.array_equal(field.values, np.full((5, 5), 37.0))


def test_diffuse_leaves_uniform_field_unchanged():
    # A spatially-uniform field has zero Laplacian everywhere, so diffusion
    # should not change it (no-flux boundaries; nothing to spread).
    field = Field("glucose", np.full((10, 10), 3.0), diffuses=True, diffusivity=1e-9)
    env = Environment(
        "env", wall_map=np.zeros((10, 10)), size=(50.0, 50.0), fields=[field]
    )

    env.diffuse(1.0)

    assert np.allclose(field.values, 3.0)


def test_diffuse_conserves_total_mass():
    # No-flux (Neumann) boundaries mean nothing leaves the grid, so the sum
    # of concentration over the grid should be conserved.
    rng = np.random.default_rng(0)
    values = rng.uniform(0.0, 10.0, size=(10, 10))
    field = Field("glucose", values, diffuses=True, diffusivity=1e-9)
    env = Environment(
        "env", wall_map=np.zeros((10, 10)), size=(50.0, 50.0), fields=[field]
    )

    total_before = field.values.sum()
    env.diffuse(2.0)
    total_after = field.values.sum()

    assert total_after == pytest.approx(total_before, rel=1e-9)


def test_diffuse_spreads_a_point_source_toward_its_neighbors():
    # dt chosen small enough that only a single FTCS sub-step elapses, so a
    # 5-point-stencil Laplacian reaches the axis neighbors but not the
    # diagonal ones in this one step.
    values = np.zeros((11, 11))
    values[5, 5] = 100.0
    field = Field("glucose", values, diffuses=True, diffusivity=1e-9)
    env = Environment(
        "env", wall_map=np.zeros((11, 11)), size=(55.0, 55.0), fields=[field]
    )

    env.diffuse(0.005)

    assert field.values[5, 5] < 100.0
    assert field.values[4, 5] > 0.0
    assert field.values[6, 5] > 0.0
    assert field.values[5, 4] > 0.0
    assert field.values[5, 6] > 0.0
    assert field.values[4, 4] == 0.0


def test_diffuse_only_advances_fields_with_diffuses_true():
    static_field = Field("temperature", np.full((10, 10), 37.0))
    diffusing_values = np.zeros((10, 10))
    diffusing_values[5, 5] = 100.0
    diffusing_field = Field(
        "glucose", diffusing_values, diffuses=True, diffusivity=1e-9
    )
    env = Environment(
        "env",
        wall_map=np.zeros((10, 10)),
        size=(50.0, 50.0),
        fields=[static_field, diffusing_field],
    )

    env.diffuse(0.5)

    assert np.array_equal(static_field.values, np.full((10, 10), 37.0))
    assert diffusing_field.values[5, 5] < 100.0


def test_environment_depth_defaults_to_1um():
    env = Environment("env", wall_map=np.zeros((10, 10)))
    assert env.depth == 1.0


def test_environment_depth_can_be_overridden():
    env = Environment("env", wall_map=np.zeros((10, 10)), depth=2.5)
    assert env.depth == 2.5


def test_environment_depth_must_be_positive():
    with pytest.raises(ValueError):
        Environment("env", wall_map=np.zeros((10, 10)), depth=0.0)
    with pytest.raises(ValueError):
        Environment("env", wall_map=np.zeros((10, 10)), depth=-1.0)


def test_grid_cell_volume_computes_dx_dy_depth():
    env = Environment("env", wall_map=np.zeros((10, 5)), size=(100.0, 50.0), depth=2.0)
    # dx = 100/10 = 10, dy = 50/5 = 10, depth = 2 -> volume = 10*10*2 = 200
    assert env.grid_cell_volume == pytest.approx(200.0)


def test_diffuse_is_stable_for_a_large_dt_via_substepping():
    # A dt far larger than the FTCS stability bound for this grid/D should
    # still produce a bounded, non-oscillating result (no internal
    # sub-stepping would blow up to NaN/inf or negative concentrations).
    values = np.zeros((10, 10))
    values[5, 5] = 100.0
    field = Field("glucose", values, diffuses=True, diffusivity=1e-9)
    env = Environment(
        "env", wall_map=np.zeros((10, 10)), size=(20.0, 20.0), fields=[field]
    )

    env.diffuse(100.0)

    assert np.all(np.isfinite(field.values))
    assert np.all(field.values >= 0.0)
    assert field.values.sum() == pytest.approx(100.0, rel=1e-6)


def test_grid_indices_maps_position_to_expected_cell():
    env = Environment("env", wall_map=np.zeros((10, 10)), size=(100.0, 100.0))
    i, j = env.grid_indices(np.array([[25.0, 75.0]]))
    assert (i[0], j[0]) == (7, 2)


def test_no_walls_means_no_faces_or_corners():
    env = Environment("env", wall_map=np.zeros((10, 10)))
    assert env.wall_faces == []
    assert env.wall_corners == []


def test_single_wall_block_produces_four_faces_and_four_corners():
    wall_map = np.zeros((10, 10))
    wall_map[4:6, 4:6] = 1  # 20x20um block: x in [40,60], y in [40,60]
    env = Environment("env", wall_map=wall_map, size=(100.0, 100.0))

    assert len(env.wall_faces) == 4
    normals = {(round(nx), round(ny)) for _, _, _, _, nx, ny in env.wall_faces}
    assert normals == {(0, -1), (0, 1), (-1, 0), (1, 0)}

    corners = {tuple(c) for c in env.wall_corners}
    assert corners == {(40.0, 40.0), (40.0, 60.0), (60.0, 40.0), (60.0, 60.0)}


def test_adjacent_wall_pixels_merge_into_one_face():
    # A 1x3 horizontal run of wall pixels should merge into a single face on
    # each long side, not one tiny face per pixel.
    wall_map = np.zeros((10, 10))
    wall_map[5, 3:6] = 1
    env = Environment("env", wall_map=wall_map, size=(100.0, 100.0))

    long_faces = [f for f in env.wall_faces if abs(f[2] - f[0]) == pytest.approx(30.0)]
    assert len(long_faces) == 2  # top and bottom of the run


def test_wall_wall_boundary_is_not_exposed():
    # A solid 2x2 wall block's internal pixel-pixel boundary shouldn't
    # produce any geometry -- cells can never reach an interior wall face.
    wall_map = np.zeros((4, 4))
    wall_map[1:3, 1:3] = 1
    env = Environment("env", wall_map=wall_map, size=(40.0, 40.0))

    # Still just the 4 outer faces of the 2x2 block, not 8 (one per pixel side).
    assert len(env.wall_faces) == 4


def test_diffuse_does_not_deposit_mass_into_walls():
    wall_map = np.zeros((10, 10))
    wall_map[4:6, 4:6] = 1  # 2x2 wall block in the middle
    values = np.zeros((10, 10))
    values[5, 3] = 100.0  # point source right next to the wall block
    field = Field("dye", values, diffuses=True, diffusivity=1e-9)
    env = Environment("env", wall_map=wall_map, size=(100.0, 100.0), fields=[field])

    env.diffuse(1.0)

    assert np.array_equal(field.values[4:6, 4:6], np.zeros((2, 2)))


def test_diffuse_conserves_mass_with_walls_present():
    wall_map = np.zeros((10, 10))
    wall_map[4:6, 4:6] = 1
    values = np.zeros((10, 10))
    values[5, 3] = 100.0
    field = Field("dye", values, diffuses=True, diffusivity=1e-9)
    env = Environment("env", wall_map=wall_map, size=(100.0, 100.0), fields=[field])

    total_before = field.values.sum()
    for _ in range(50):
        env.diffuse(0.5)
    total_after = field.values.sum()

    assert total_after == pytest.approx(total_before, rel=1e-9)


def test_diffuse_piles_mass_up_against_a_wall_instead_of_passing_through():
    # A full-width wall row completely seals off rows 6-9 from rows 0-4 (no
    # path around it, unlike a small block). A source below the wall should
    # never deposit any mass on the far side.
    wall_map = np.zeros((10, 10))
    wall_map[5, :] = 1  # y in [50, 60] is a solid barrier across the whole width
    values = np.zeros((10, 10))
    values[3, 5] = 100.0  # source below the barrier
    field = Field("dye", values, diffuses=True, diffusivity=1e-9)
    env = Environment("env", wall_map=wall_map, size=(100.0, 100.0), fields=[field])

    total_before = field.values.sum()
    for _ in range(50):
        env.diffuse(0.5)

    assert field.values[4, 5] > 0.0  # near the wall's near face, below it
    assert np.array_equal(field.values[6:, :], np.zeros((4, 10)))  # sealed off above
    assert field.values.sum() == pytest.approx(total_before, rel=1e-9)


def test_diffuse_unaffected_by_walls_when_none_present():
    # Regression check: with an all-media wall_map, wall-aware diffusion must
    # reduce to the original unobstructed behavior.
    values = np.zeros((11, 11))
    values[5, 5] = 100.0
    field = Field("glucose", values, diffuses=True, diffusivity=1e-9)
    env = Environment(
        "env", wall_map=np.zeros((11, 11)), size=(55.0, 55.0), fields=[field]
    )

    env.diffuse(0.005)

    assert field.values[5, 5] < 100.0
    assert field.values[4, 5] > 0.0
    assert field.values[6, 5] > 0.0
    assert field.values[5, 4] > 0.0
    assert field.values[5, 6] > 0.0
    assert field.values[4, 4] == 0.0
