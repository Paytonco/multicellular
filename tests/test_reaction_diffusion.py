# tests/test_reaction_diffusion.py

import numpy as np
import pytest

from multicellular.core.colony import Colony
from multicellular.core.environment import Environment, Field
from multicellular.core.reactions import Reaction, ReactionNetwork


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


def _gray_scott_network(F=0.035, k=0.065):
    """
    Classic Gray-Scott reaction-diffusion system, decomposed into
    elementary mass-action reactions:
        U + 2V -> 3V        (autocatalysis: du/dt -= u*v^2, dv/dt += u*v^2)
        (feed)  -> U         (du/dt += F)
        U ->  (removal)      (du/dt -= F*u), combining with feed to F*(1-u)
        V ->  (kill)         (dv/dt -= (F+k)*v)
    """
    return ReactionNetwork(
        "gray_scott",
        {
            "autocatalysis": Reaction(
                reactants={"U": 1, "V": 2},
                products={"V": 3},
                rate_params={"k": 1.0},
            ),
            "feed_U": Reaction(reactants={}, products={"U": 1}, rate_params={"k": F}),
            "remove_U": Reaction(reactants={"U": 1}, products={}, rate_params={"k": F}),
            "kill_V": Reaction(
                reactants={"V": 1}, products={}, rate_params={"k": F + k}
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


def test_react_ode_matches_hand_computed_forward_euler():
    network = _conversion_network(k=0.2)
    values_a = np.array([[3.0, 1.0], [3.0, 1.0]])
    expected_a = values_a * 0.8
    expected_b = values_a * 0.2
    field_a = Field("A", values_a.copy())
    field_b = Field("B", np.zeros((2, 2)))
    env = Environment(
        "env", wall_map=np.zeros((2, 2)), fields=[field_a, field_b], reactions=network
    )

    env.react(1.0, method="ODE")
    assert np.allclose(field_a.values, expected_a)
    assert np.allclose(field_b.values, expected_b)


def test_react_excludes_wall_cells():
    wall_map = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    network = _decay_network(k=1.0)
    field = Field("A", np.full((3, 3), 2.0))
    env = Environment(
        "env", wall_map=wall_map, size=(3.0, 3.0), fields=[field], reactions=network
    )

    env.react(1.0)

    assert field.values[1, 1] == 2.0
    assert np.all(field.values[wall_map == 0] == 0.0)


def test_react_ssa_conserves_total_copy_number_for_isomerization():
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
        env.react(1.0, method="SSA")
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

    assert np.allclose(field.values, 5.0)


def test_gray_scott_turing_pattern_grows_spatial_variance():
    # Smoke test for the notebook's Turing-pattern demo: starting from a
    # small perturbed patch, the classic Gray-Scott instability should
    # amplify spatial non-uniformity rather than settle back to a flat
    # field. Deterministic (ODE) and kept small so it runs in ~1 second.
    n = 16
    network = _gray_scott_network()

    U = np.ones((n, n))
    V = np.zeros((n, n))
    mid = n // 2
    U[mid - 2 : mid + 2, mid - 2 : mid + 2] = 0.50
    V[mid - 2 : mid + 2, mid - 2 : mid + 2] = 0.25
    rng = np.random.default_rng(0)
    U += rng.normal(0, 0.02, size=(n, n))
    V += rng.normal(0, 0.02, size=(n, n))

    dt = 2.0
    dx_m = (float(n) / n) * 1e-6
    field_u = Field("U", U, diffuses=True, diffusivity=0.16 * dx_m**2)
    field_v = Field("V", V, diffuses=True, diffusivity=0.08 * dx_m**2)
    env = Environment(
        "env",
        wall_map=np.zeros((n, n)),
        size=(float(n), float(n)),
        fields=[field_u, field_v],
        reactions=network,
    )

    std_before = env.fields["V"].values.std()
    for _ in range(300):
        env.diffuse(dt)
        env.react(dt, method="ODE")

    assert np.all(np.isfinite(env.fields["U"].values))
    assert np.all(np.isfinite(env.fields["V"].values))
    assert np.all(env.fields["U"].values >= 0.0)
    assert np.all(env.fields["V"].values >= 0.0)
    assert env.fields["V"].values.std() > std_before
