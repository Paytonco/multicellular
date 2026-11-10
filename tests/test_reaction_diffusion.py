# tests/test_reaction_diffusion.py

import numpy as np
import pytest

from multicellular.core.colony import Colony
from multicellular.core.environment import Environment, Field
from multicellular.core.reactions import (
    RateLawNotVectorizedError,
    Reaction,
    ReactionNetwork,
)


def _decay_network(k=0.2):
    return ReactionNetwork(
        "decay",
        {"decay": Reaction(reactants={"A": 1}, products={}, rate_params={"k": k})},
    )


def _conversion_network(k=0.2):
    return ReactionNetwork(
        "conversion",
        {
            "convert": Reaction(
                reactants={"A": 1}, products={"B": 1}, rate_params={"k": k}
            )
        },
    )


def _isomerization_network(k_fwd=0.5, k_rev=0.3):
    return ReactionNetwork(
        "isomerization",
        {
            "fwd": Reaction(
                reactants={"A": 1}, products={"B": 1}, rate_params={"k": k_fwd}
            ),
            "rev": Reaction(
                reactants={"B": 1}, products={"A": 1}, rate_params={"k": k_rev}
            ),
        },
    )


def _schnakenberg_network(a=0.1, b=0.9):
    """
    Schnakenberg activator-substrate system as elementary mass-action steps:
        (feed) -> u, u -> (decay)   du/dt += a - u
        (feed) -> v                 dv/dt += b
        2u + v -> 3u                du/dt += u^2 v, dv/dt -= u^2 v
    """
    return ReactionNetwork(
        "schnakenberg",
        {
            "feed_u": Reaction(reactants={}, products={"u": 1}, rate_params={"k": a}),
            "decay_u": Reaction(
                reactants={"u": 1}, products={}, rate_params={"k": 1.0}
            ),
            "feed_v": Reaction(reactants={}, products={"v": 1}, rate_params={"k": b}),
            "autocatalysis": Reaction(
                reactants={"u": 2, "v": 1}, products={"u": 3}, rate_params={"k": 1.0}
            ),
        },
    )


def test_environment_reactions_defaults_to_none():
    env = Environment("env", wall_map=np.zeros((3, 3)))
    assert env.reactions is None


def test_environment_rejects_reactions_with_exports():
    network = ReactionNetwork(
        "export_rxn",
        {
            "secrete": Reaction(
                reactants={"A": 1},
                products={},
                exports={"A": 1},
                rate_params={"k": 1.0},
            )
        },
    )
    field = Field("A", np.zeros((3, 3)))
    with pytest.raises(ValueError):
        Environment("env", wall_map=np.zeros((3, 3)), fields=[field], reactions=network)


def test_react_is_no_op_when_no_reactions():
    field = Field("A", np.full((3, 3), 5.0))
    env = Environment("env", wall_map=np.zeros((3, 3)), fields=[field])

    env.react(1.0)

    assert np.array_equal(field.values, np.full((3, 3), 5.0))


def test_react_raises_for_species_missing_field_without_mutating_anything():
    network = _conversion_network()
    field_a = Field("A", np.full((3, 3), 5.0))
    env = Environment(
        "env", wall_map=np.zeros((3, 3)), fields=[field_a], reactions=network
    )

    with pytest.raises(ValueError):
        env.react(1.0)

    assert np.array_equal(field_a.values, np.full((3, 3), 5.0))


def test_react_ode_matches_analytic_solution():
    network = _conversion_network(k=0.2)
    values_a = np.array([[3.0, 1.0], [3.0, 1.0]])
    expected_a = values_a * np.exp(-0.2)
    expected_b = values_a * (1 - np.exp(-0.2))
    field_a = Field("A", values_a.copy())
    field_b = Field("B", np.zeros((2, 2)))
    env = Environment(
        "env", wall_map=np.zeros((2, 2)), fields=[field_a, field_b], reactions=network
    )

    env.react(1.0, method="ODE")
    assert np.allclose(field_a.values, expected_a, rtol=1e-4)
    assert np.allclose(field_b.values, expected_b, rtol=1e-4)


def test_react_excludes_wall_cells():
    wall_map = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    network = _decay_network(k=1.0)
    field = Field("A", np.full((3, 3), 2.0))
    env = Environment(
        "env", wall_map=wall_map, size=(3.0, 3.0), fields=[field], reactions=network
    )

    env.react(1.0)

    assert field.values[1, 1] == 2.0
    assert np.allclose(field.values[wall_map == 0], 2.0 * np.exp(-1.0), rtol=1e-4)


@pytest.mark.parametrize("ssa_method", ["direct", "next_reaction", "tau_leaping"])
def test_react_ssa_conserves_total_copy_number_for_isomerization(ssa_method):
    network = _isomerization_network()
    field_a = Field("A", np.full((2, 2), 5.0))
    field_b = Field("B", np.full((2, 2), 1.0))
    total_before = field_a.values + field_b.values

    for seed in range(10):
        rng = np.random.default_rng(seed)
        env = Environment(
            "env",
            wall_map=np.zeros((2, 2)),
            size=(2.0, 2.0),
            depth=1.0,
            fields=[
                Field("A", field_a.values.copy()),
                Field("B", field_b.values.copy()),
            ],
            reactions=network,
            rng=rng,
        )
        env.react(1.0, method="SSA", ssa_method=ssa_method)
        total_after = env.fields["A"].values + env.fields["B"].values
        assert np.allclose(total_after, total_before)


def test_react_cle_conserves_mass_without_clipping():
    network = _isomerization_network(k_fwd=2.0, k_rev=1.0)
    # Large volume + small dt keeps the noise term well clear of the
    # non-negativity clip, so conservation should hold (near-)exactly,
    # mirroring test_reactions.py's equivalent per-cell CLE test.
    field_a = Field("A", np.full((2, 2), 4.0))
    field_b = Field("B", np.full((2, 2), 1.0))
    total_before = field_a.values + field_b.values

    for seed in range(10):
        rng = np.random.default_rng(seed)
        env = Environment(
            "env",
            wall_map=np.zeros((2, 2)),
            size=(20.0, 20.0),
            depth=1.0,
            fields=[
                Field("A", field_a.values.copy()),
                Field("B", field_b.values.copy()),
            ],
            reactions=network,
            rng=rng,
        )
        env.react(0.02, method="CLE")
        total_after = env.fields["A"].values + env.fields["B"].values
        assert np.allclose(total_after, total_before, atol=1e-9)


def test_colony_step_runs_field_reactions_with_zero_cells():
    # Uniform field: diffusion is a no-op (zero Laplacian everywhere), so any
    # change after colony.step must come from the reaction, confirming
    # Colony.step actually invokes environment.react even with no cells.
    network = _decay_network(k=0.5)
    field = Field("A", np.full((4, 4), 10.0), diffuses=True, diffusivity=1e-9)
    env = Environment(
        "env",
        wall_map=np.zeros((4, 4)),
        size=(20.0, 20.0),
        fields=[field],
        reactions=network,
    )
    colony = Colony([], env)

    colony.step(1.0)

    assert np.allclose(field.values, 10.0 * np.exp(-0.5), rtol=1e-4)


def _schnakenberg_coupled_reference(u, v, T, a=0.1, b=0.9, Du=1.0, Dv=40.0):
    """Fully coupled forward-Euler solve (small dt), independent of the library."""

    def lap(C):
        P = np.pad(C, 1, mode="edge")
        return P[2:, 1:-1] + P[:-2, 1:-1] + P[1:-1, 2:] + P[1:-1, :-2] - 4 * C

    h = 0.005
    for _ in range(int(round(T / h))):
        uuv = u * u * v
        u, v = u + h * (Du * lap(u) + a - u + uuv), v + h * (Dv * lap(v) + b - uuv)
    return u


def test_schnakenberg_turing_pattern_matches_coupled_reference():
    # The notebook's Turing demo: from 1% noise around the uniform steady
    # state, a diffusion-driven instability grows a spot pattern. With a
    # small enough dt the operator-split library run must match a fully
    # coupled solve, not just "become non-uniform" (a coarse dt also does
    # that, but forms a different, lower-amplitude pattern).
    n, a, b = 16, 0.1, 0.9
    rng = np.random.default_rng(0)
    u0 = (a + b) + rng.normal(0, 0.01, size=(n, n))
    v0 = b / (a + b) ** 2 + rng.normal(0, 0.01, size=(n, n))

    dx_m = 1e-6
    field_u = Field("u", u0, diffuses=True, diffusivity=1.0 * dx_m**2)
    field_v = Field("v", v0, diffuses=True, diffusivity=40.0 * dx_m**2)
    env = Environment(
        "env",
        wall_map=np.zeros((n, n)),
        size=(float(n), float(n)),
        fields=[field_u, field_v],
        reactions=_schnakenberg_network(a, b),
    )
    T, dt = 40.0, 0.02
    for _ in range(int(round(T / dt))):
        env.diffuse(dt)
        env.react(dt, method="ODE")

    u = env.fields["u"].values
    reference = _schnakenberg_coupled_reference(u0, v0, T)
    assert u.max() - u.min() > 3.0  # a strongly patterned field
    assert np.corrcoef(u.ravel(), reference.ravel())[0, 1] > 0.99
    assert u.max() == pytest.approx(reference.max(), rel=0.05)


def _random_grid_env(network, species, shape=(6, 7), wall_map=None, seed=0):
    rng = np.random.default_rng(seed)
    fields = [Field(s, rng.uniform(0.1, 2.0, size=shape)) for s in species]
    return Environment(
        "env",
        wall_map=np.zeros(shape) if wall_map is None else wall_map,
        size=(float(shape[1]), float(shape[0])),
        fields=fields,
        reactions=network,
    )


def _react_voxel_by_voxel(env, dt, ode_method="RK45"):
    """Reference: the per-voxel simulate_step loop react() used before batching."""
    network = env.reactions
    expected = {s: env.fields[s].values.copy() for s in network.species}
    for i in range(env.shape[0]):
        for j in range(env.shape[1]):
            if env.wall_map[i, j] == 1:
                continue
            state = {s: float(expected[s][i, j]) for s in network.species}
            new = network.simulate_step(
                state, dt, env.grid_cell_volume, "ODE", ode_method=ode_method
            )
            for s, value in new.items():
                expected[s][i, j] = value
    return expected


@pytest.mark.parametrize("ode_method", ["RK45", "BDF", "Radau", "LSODA"])
def test_batched_ode_react_matches_per_voxel_steps(ode_method):
    wall_map = np.zeros((6, 7))
    wall_map[2, 3] = 1
    env = _random_grid_env(
        _schnakenberg_network(), ["u", "v"], wall_map=wall_map, seed=1
    )
    wall_before = {s: env.fields[s].values[2, 3] for s in ("u", "v")}
    expected = _react_voxel_by_voxel(env, 0.5, ode_method)

    env.react(0.5, method="ODE", ode_method=ode_method)

    for s in ("u", "v"):
        np.testing.assert_allclose(env.fields[s].values, expected[s], rtol=1e-4)
        assert env.fields[s].values[2, 3] == wall_before[s]  # walls untouched


def test_batched_ode_react_handles_michaelis_menten_and_hill_rate_laws():
    network = ReactionNetwork(
        "mixed",
        {
            "mm": Reaction(
                reactants={"S": 1},
                products={"P": 1},
                catalysts=["E"],
                rate_law_type="michaelis_menten",
                rate_params={"Vmax": 1.0, "Km": 0.5},
            ),
            "hill": Reaction(
                reactants={},
                products={"P": 1},
                catalysts=["E", "S"],
                rate_law_type="hill_langmuir",
                rate_params={"alpha": 50.0, "beta": 0.1, "C": 1.0, "n": 2},
            ),
        },
    )
    env = _random_grid_env(network, ["E", "P", "S"])
    expected = _react_voxel_by_voxel(env, 1.0)
    env.react(1.0, method="ODE")
    for s in ("E", "P", "S"):
        np.testing.assert_allclose(env.fields[s].values, expected[s], rtol=1e-4)


def test_scalar_only_custom_rate_law_falls_back_to_per_voxel_steps():
    import math

    def scalar_only(reactants, products, catalysts, params):
        # math.exp and max reject arrays, so this can't be batched.
        return params["k"] * max(reactants["A"], 0.0) * math.exp(0.0)

    network = ReactionNetwork(
        "custom",
        {
            "decay": Reaction(
                reactants={"A": 1},
                products={},
                rate_law_type="custom",
                custom_rate_law=scalar_only,
                rate_params={"k": 0.5},
            )
        },
    )
    env = _random_grid_env(network, ["A"])
    with pytest.raises(RateLawNotVectorizedError):
        network.simulate_ode_batch({"A": env.fields["A"].values.ravel()}, 1.0)

    before = env.fields["A"].values.copy()
    env.react(1.0, method="ODE")
    np.testing.assert_allclose(env.fields["A"].values, before * np.exp(-0.5), rtol=1e-4)


def test_batched_ode_react_is_a_no_op_on_an_all_wall_grid():
    env = _random_grid_env(
        _schnakenberg_network(), ["u", "v"], wall_map=np.ones((6, 7))
    )
    before = {s: env.fields[s].values.copy() for s in ("u", "v")}
    env.react(1.0, method="ODE")
    for s in ("u", "v"):
        assert np.array_equal(env.fields[s].values, before[s])
