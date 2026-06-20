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
