# tests/test_environment.py

import numpy as np
import pytest

from multicellular.core.environment import Environment, Field


def test_environment_default_bounds_are_100_by_100():
    env = Environment(shape=(10, 10))
    assert env.bounds == (100.0, 100.0)


def test_environment_bounds_can_be_specified():
    env = Environment(shape=(10, 10), bounds=(20.0, 20.0))
    assert env.bounds == (20.0, 20.0)
    assert env.in_bounds([10.0, 10.0])
    assert not env.in_bounds([50.0, 50.0])


def test_add_and_get_field():
    env = Environment(shape=(10, 10))
    field = Field("glucose", np.ones((10, 10)))

    env.add_field(field)

    assert env.get_field("glucose") is field
    assert np.array_equal(env.get_field("glucose").values, np.ones((10, 10)))


def test_environment_constructor_accepts_fields():
    field = Field("temperature", np.zeros((5, 5)))
    env = Environment(shape=(5, 5), fields=[field])

    assert env.get_field("temperature") is field


def test_field_can_extend_beyond_bounds():
    # Field grid shape is independent of the (100, 100) simulation bounds.
    field = Field("roughness", np.zeros((200, 200)))
    env = Environment(shape=(200, 200), fields=[field])

    assert env.get_field("roughness").shape == (200, 200)
    assert env.bounds == (100.0, 100.0)


def test_field_is_chemical_defaults_to_false():
    field = Field("glucose", np.ones((10, 10)))
    assert field.is_chemical is False


def test_field_is_chemical_can_be_set_true():
    field = Field("glucose", np.ones((10, 10)), is_chemical=True)
    assert field.is_chemical is True


def test_add_field_with_mismatched_shape_raises():
    env = Environment(shape=(10, 10))
    bad_field = Field("glucose", np.ones((5, 5)))

    with pytest.raises(ValueError):
        env.add_field(bad_field)


def test_constructor_with_mismatched_field_shape_raises():
    bad_field = Field("glucose", np.ones((5, 5)))

    with pytest.raises(ValueError):
        Environment(shape=(10, 10), fields=[bad_field])


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
    env = Environment(shape=(5, 5), fields=[field])

    env.diffuse(1.0)

    assert np.array_equal(field.values, np.full((5, 5), 37.0))


def test_diffuse_leaves_uniform_field_unchanged():
    # A spatially-uniform field has zero Laplacian everywhere, so diffusion
    # should not change it (no-flux boundaries; nothing to spread).
    field = Field("glucose", np.full((10, 10), 3.0), diffuses=True, diffusivity=1e-9)
    env = Environment(shape=(10, 10), bounds=(50.0, 50.0), fields=[field])

    env.diffuse(1.0)

    assert np.allclose(field.values, 3.0)


def test_diffuse_conserves_total_mass():
    # No-flux (Neumann) boundaries mean nothing leaves the grid, so the sum
    # of concentration over the grid should be conserved.
    rng = np.random.default_rng(0)
    values = rng.uniform(0.0, 10.0, size=(10, 10))
    field = Field("glucose", values, diffuses=True, diffusivity=1e-9)
    env = Environment(shape=(10, 10), bounds=(50.0, 50.0), fields=[field])

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
    env = Environment(shape=(11, 11), bounds=(55.0, 55.0), fields=[field])

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
        shape=(10, 10), bounds=(50.0, 50.0), fields=[static_field, diffusing_field]
    )

    env.diffuse(0.5)

    assert np.array_equal(static_field.values, np.full((10, 10), 37.0))
    assert diffusing_field.values[5, 5] < 100.0


def test_environment_depth_defaults_to_1um():
    env = Environment(shape=(10, 10))
    assert env.depth == 1.0


def test_environment_depth_can_be_overridden():
    env = Environment(shape=(10, 10), depth=2.5)
    assert env.depth == 2.5


def test_environment_depth_must_be_positive():
    with pytest.raises(ValueError):
        Environment(shape=(10, 10), depth=0.0)
    with pytest.raises(ValueError):
        Environment(shape=(10, 10), depth=-1.0)


def test_grid_cell_volume_computes_dx_dy_depth():
    env = Environment(shape=(10, 5), bounds=(100.0, 50.0), depth=2.0)
    # dx = 100/10 = 10, dy = 50/5 = 10, depth = 2 -> volume = 10*10*2 = 200
    assert env.grid_cell_volume == pytest.approx(200.0)


def test_diffuse_is_stable_for_a_large_dt_via_substepping():
    # A dt far larger than the FTCS stability bound for this grid/D should
    # still produce a bounded, non-oscillating result (no internal
    # sub-stepping would blow up to NaN/inf or negative concentrations).
    values = np.zeros((10, 10))
    values[5, 5] = 100.0
    field = Field("glucose", values, diffuses=True, diffusivity=1e-9)
    env = Environment(shape=(10, 10), bounds=(20.0, 20.0), fields=[field])

    env.diffuse(100.0)

    assert np.all(np.isfinite(field.values))
    assert np.all(field.values >= 0.0)
    assert field.values.sum() == pytest.approx(100.0, rel=1e-6)
