# tests/test_kernels.py
"""Compiled kernels (core/_kernels.py) against the Python reference implementations."""

import numba
import numpy as np
import pytest
from scipy import stats

from multicellular.core import _kernels as K
from multicellular.core import reactions as R
from multicellular.core.cell import Cell
from multicellular.core.colony import Colony
from multicellular.core.environment import Environment, Field
from multicellular.core.reactions import Reaction, ReactionNetwork


@numba.njit
def _draw(seed, kind, n, lam):
    state = K.rng_init(seed)
    out = np.empty(n)
    for i in range(n):
        if kind == 0:
            out[i] = K.rng_uniform(state)
        elif kind == 1:
            out[i] = K.rng_exponential(state)
        elif kind == 2:
            out[i] = K.rng_normal(state)
        else:
            out[i] = K.rng_poisson(state, lam)
    return out


def test_tau_leaping_constants_mirror_reactions_module():
    assert K._N_CRITICAL == R._N_CRITICAL
    assert K._SSA_FALLBACK_THRESHOLD == R._SSA_FALLBACK_THRESHOLD
    assert K._SSA_FALLBACK_FIRINGS == R._SSA_FALLBACK_FIRINGS


@pytest.mark.parametrize(
    "kind, dist",
    [(0, stats.uniform()), (1, stats.expon()), (2, stats.norm())],
)
def test_rng_continuous_distributions(kind, dist):
    x = _draw(np.uint64(12345), kind, 50_000, 0.0)
    assert stats.kstest(x, dist.cdf).pvalue > 1e-3
    if kind == 0:
        assert x.min() > 0.0 and x.max() <= 1.0


@pytest.mark.parametrize("lam", [0.3, 4.0, 9.99, 10.0, 35.0, 800.0])
def test_rng_poisson_moments(lam):
    x = _draw(np.uint64(7), 3, 100_000, lam)
    assert np.all(x == np.floor(x)) and x.min() >= 0
    se = np.sqrt(lam / len(x))
    assert abs(x.mean() - lam) < 5 * se
    assert x.var() == pytest.approx(lam, rel=0.03)


def test_rng_streams_are_seed_deterministic_and_distinct():
    a = _draw(np.uint64(1), 0, 10, 0.0)
    assert np.array_equal(a, _draw(np.uint64(1), 0, 10, 0.0))
    assert not np.array_equal(a, _draw(np.uint64(2), 0, 10, 0.0))


def _all_laws_network():
    return ReactionNetwork(
        "laws",
        {
            "make_A": Reaction({}, {"A": 1}, rate_params={"k": 2.0}),
            "deg_A2": Reaction({"A": 2}, {}, rate_params={"k": 0.3}),
            "mm": Reaction(
                {"A": 1},
                {"B": 1},
                catalysts=["E"],
                rate_law_type="michaelis_menten",
                rate_params={"Vmax": 1.5, "Km": 2.0},
            ),
            "mm_bare": Reaction(
                {"B": 1},
                {"C": 1},
                rate_law_type="michaelis_menten",
                rate_params={"Vmax": 0.7},
            ),
            "hl": Reaction(
                {},
                {"G": 1},
                catalysts=["E", "B"],
                rate_law_type="hill_langmuir",
                rate_params={"alpha": 50, "beta": 0.2, "C": 0.5, "n": 2.5},
            ),
            "sec": Reaction({"C": 1}, {}, rate_params={"k": 0.4}, exports={"X": 1}),
        },
    )


@pytest.mark.parametrize("C_value", [0.0, 0.8])
def test_ode_derivatives_match_rate_vector(C_value):
    net = _all_laws_network()
    st = K.structure_of(net)
    T = K.build_table([st])
    conc = {"A": 1.3, "B": 0.4, "C": C_value, "E": 2.0, "G": 0.1}
    row = np.array([[conc[s] for s in st.species]])
    params = K.network_params(net, T.S.shape[2])[None]
    dC = np.empty_like(row)
    dE = np.empty((1, T.S_exp.shape[1]))
    K.ode_derivatives(T, np.zeros(1, dtype=np.int64), params, row, dC, dE)

    v = net._rate_vector(conc)
    sec = net._reaction_names.index("sec")
    v[sec] = 0.0 if C_value <= 0.0 else max(v[sec], 0.0)  # export clamp
    np.testing.assert_allclose(dC[0], net._stoichiometry_matrix @ v, rtol=1e-14)
    np.testing.assert_allclose(dE[0], net._export_stoichiometry_matrix @ v, rtol=1e-14)


def test_custom_and_malformed_networks_are_not_compiled():
    custom = ReactionNetwork(
        "c",
        {
            "r": Reaction(
                {"A": 1},
                {},
                rate_law_type="custom",
                custom_rate_law=lambda r, p, c, k: 1.0,
            )
        },
    )
    one_catalyst_hill = ReactionNetwork(
        "h",
        {
            "r": Reaction(
                {},
                {"G": 1},
                catalysts=["E"],
                rate_law_type="hill_langmuir",
                rate_params={"alpha": 1, "beta": 1, "C": 1, "n": 1},
            )
        },
    )
    assert K.structure_of(custom) is None
    assert K.structure_of(one_catalyst_hill) is None


def test_clones_share_one_structure():
    net = _all_laws_network()
    assert K.structure_of(net) is K.structure_of(net.clone())


def _birth_death(k=200.0, gamma=1.0):
    return ReactionNetwork(
        "bd",
        {
            "birth": Reaction({}, {"A": 1}, rate_params={"k": k}),
            "death": Reaction({"A": 1}, {}, rate_params={"k": gamma}),
        },
    )


def _run_compartments(net, method, n, dt, volume=1.0, epsilon=0.03, seed=0):
    """n independent compartments of `net`, all starting at 0, stepped once."""
    st = K.structure_of(net)
    T = K.build_table([st])
    NR = T.S.shape[2]
    conc = np.zeros((n, T.S.shape[1]))
    export = np.zeros((n, T.S_exp.shape[1]))
    seeds = np.random.default_rng(seed).integers(0, 2**64, size=n, dtype=np.uint64)
    K.step_stochastic(
        T,
        method,
        dt,
        epsilon,
        np.zeros(n, dtype=np.int64),
        np.repeat(K.network_params(net, NR)[None], n, axis=0),
        conc,
        np.full(n, volume),
        np.full(conc.shape, -1, dtype=np.int64),
        np.full(n, -1, dtype=np.int64),
        np.zeros((1, 1)),
        1.0,
        seeds,
        np.arange(n + 1, dtype=np.int64),
        np.arange(n, dtype=np.int64),
        export,
    )
    return conc, export


@pytest.mark.parametrize(
    "method",
    [K.SSA_DIRECT, K.SSA_NEXT_REACTION, K.SSA_TAU_LEAPING],
)
def test_birth_death_matches_poisson_statistics(method):
    # From 0, a birth-death process is Poisson with mean k/g (1 - e^{-g t}).
    k, gamma, dt = 200.0, 1.0, 1.5
    conc, _ = _run_compartments(_birth_death(k, gamma), method, 4000, dt)
    x = conc[:, 0]
    mean = k / gamma * (1.0 - np.exp(-gamma * dt))
    assert x.mean() == pytest.approx(mean, rel=0.02)
    assert x.var() == pytest.approx(mean, rel=0.1)


def test_cle_matches_python_cle_step():
    # One Euler-Maruyama step per dt on both paths, so compare the two
    # directly rather than against the exact birth-death distribution.
    net = _birth_death(200.0, 1.0)
    for start in (0.0, 150.0):
        st = K.structure_of(net)
        T = K.build_table([st])
        n = 20000
        conc = np.full((n, 1), start)
        seeds = np.arange(n, dtype=np.uint64)
        K.step_stochastic(
            T,
            K.CLE,
            0.2,
            0.03,
            np.zeros(n, dtype=np.int64),
            np.repeat(K.network_params(net, 2)[None], n, axis=0),
            conc,
            np.full(n, 2.0),
            np.full((n, 1), -1, dtype=np.int64),
            np.full(n, -1, dtype=np.int64),
            np.zeros((1, 1)),
            1.0,
            seeds,
            np.arange(n + 1, dtype=np.int64),
            np.arange(n, dtype=np.int64),
            np.zeros((n, 1)),
        )
        rng = np.random.default_rng(3)
        py = np.array(
            [
                net.simulate_step({"A": start}, 0.2, 2.0, "CLE", rng=rng)["A"]
                for _ in range(20000)
            ]
        )
        assert conc[:, 0].mean() == pytest.approx(py.mean(), rel=0.01)
        assert conc[:, 0].var() == pytest.approx(py.var(), rel=0.05)


def test_python_and_kernel_ssa_agree_on_export_counts():
    net = ReactionNetwork(
        "sec",
        {
            "make": Reaction({}, {"A": 1}, rate_params={"k": 20.0}),
            "sec": Reaction({"A": 1}, {}, rate_params={"k": 0.5}, exports={"A": 1}),
        },
    )
    _, export = _run_compartments(net, K.SSA_DIRECT, 3000, 1.0)
    rng = np.random.default_rng(5)
    py = []
    for _ in range(1500):
        net.simulate_step({}, 1.0, 1.0, "SSA", rng=rng)
        py.append(net.last_exported["A"])
    assert export[:, 0].mean() == pytest.approx(np.mean(py), rel=0.05)
    assert np.all(export == np.floor(export))


def _ssa_colony(seed=0):
    rng = np.random.default_rng(seed)
    env = Environment(
        "env",
        wall_map=np.zeros((5, 5)),
        size=(50.0, 50.0),
        fields=[
            Field(
                "AHL",
                np.full((5, 5), 0.5),
                is_chemical=True,
                diffuses=True,
                diffusivity=1e-11,
            )
        ],
        rng=np.random.default_rng(seed),
    )
    net = ReactionNetwork(
        "x",
        {
            "make": Reaction({}, {"A": 1}, rate_params={"k": 30.0}),
            "sec": Reaction({"A": 1}, {"AHL": 1}, rate_params={"k": 0.5}),
            "up": Reaction({"AHL": 1}, {"A": 1}, rate_params={"k": 0.3}),
        },
    )
    cells = [
        Cell(
            id=None,
            position=rng.uniform(5, 45, size=2),
            orientation=[1.0, 0.0],
            rng=rng,
            network=net.clone(),
        )
        for _ in range(60)
    ]
    return Colony(cells, env)


@pytest.mark.parametrize("ssa_method", ["direct", "tau_leaping"])
def test_results_do_not_depend_on_thread_count(ssa_method):
    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("needs more than one numba thread")
    previous = numba.get_num_threads()
    results = []
    try:
        for n_threads in (1, min(4, numba.config.NUMBA_NUM_THREADS)):
            numba.set_num_threads(n_threads)
            colony = _ssa_colony()
            for _ in range(8):
                colony.step(0.1, "SSA", ssa_method=ssa_method)
            results.append(
                (
                    [dict(c.concentrations) for c in colony.cells],
                    [c.position.copy() for c in colony.cells],
                    colony.environment.fields["AHL"].values.copy(),
                )
            )
    finally:
        numba.set_num_threads(previous)
    (c1, p1, f1), (c2, p2, f2) = results
    assert c1 == c2
    assert all(np.array_equal(a, b) for a, b in zip(p1, p2))
    assert np.array_equal(f1, f2)


def _dense_mechanics_colony(backend, seed):
    """Crowded rods and spheres, two anchored hosts, inside a walled box."""
    rng = np.random.default_rng(seed)
    wall_map = np.zeros((20, 20))
    wall_map[2, 2:18] = wall_map[17, 2:18] = 1
    wall_map[2:18, 2] = wall_map[2:18, 17] = 1
    wall_map[9:11, 9:11] = 1  # a block in the middle
    env = Environment(
        "walls", wall_map=wall_map, size=(40.0, 40.0), rng=np.random.default_rng(seed)
    )
    cells = []
    for i in range(400):
        sphere = i % 5 == 0
        angle = rng.uniform(0, 2 * np.pi)
        cells.append(
            Cell(
                id=None,
                position=rng.uniform(4.0, 36.0, size=2),
                orientation=[np.cos(angle), np.sin(angle)],
                length=0.0 if sphere else rng.uniform(0.5, 3.0),
                radius=rng.uniform(0.3, 0.7),
                shape="sphere" if sphere else "rod",
                growth_rate=0.0,
                rng=np.random.default_rng(i),
            )
        )
    # Exactly coincident pair: exercises the degenerate-normal fallback.
    cells[1].position = cells[2].position.copy()
    cells[1].orientation = cells[2].orientation.copy()
    for x in (12.0, 28.0):
        cells.append(
            Cell(
                id=None,
                position=[x, 20.0],
                orientation=[0.0, 1.0],
                length=4.0,
                radius=2.0,
                anchored=True,
                growth_rate=0.0,
                rng=np.random.default_rng(0),
            )
        )
    return Colony(cells, env, k=5.0, k_wall=20.0, drag=1.0, backend=backend)


@pytest.mark.parametrize("seed", [0, 1])
def test_contact_kernel_matches_python_backend(seed):
    fast = _dense_mechanics_colony("numba", seed)
    slow = _dense_mechanics_colony("python", seed)
    for _ in range(3):
        fast._apply_contact_forces(0.01)
        slow._apply_contact_forces(0.01)
    for a, b in zip(fast.cells, slow.cells):
        np.testing.assert_allclose(a.position, b.position, rtol=0, atol=1e-11)
        np.testing.assert_allclose(a.orientation, b.orientation, rtol=0, atol=1e-11)
    moved = sum(
        not np.array_equal(a.position, b.position)
        for a, b in zip(fast.cells, _dense_mechanics_colony("python", seed).cells)
    )
    assert moved > 100  # the configuration really is crowded


def test_numba_brownian_matches_python_given_same_noise(monkeypatch):
    from multicellular.core import colony as colony_module

    def build(backend):
        colony = _dense_mechanics_colony(backend, 3)
        colony.k = colony.k_wall = 0.0  # Brownian motion only
        return colony

    fast, slow = build("numba"), build("python")
    mobile = [c for c in slow.living_cells if not c.anchored]
    xi = np.random.default_rng(9).standard_normal((len(mobile), 3))
    monkeypatch.setattr(
        colony_module._kernels, "normals_from_seeds", lambda seeds, k: xi
    )
    fast._move_cells_numba(0.05, fast.living_cells)
    slow._apply_brownian_motion(0.05, mobile, xi)
    for a, b in zip(fast.cells, slow.cells):
        np.testing.assert_allclose(a.position, b.position, rtol=0, atol=1e-12)
        np.testing.assert_allclose(a.orientation, b.orientation, rtol=0, atol=1e-12)


def test_normals_from_seeds_are_standard_normal_and_seed_deterministic():
    seeds = np.arange(20000, dtype=np.uint64)
    x = K.normals_from_seeds(seeds, 3)
    assert stats.kstest(x.ravel(), stats.norm.cdf).pvalue > 1e-3
    assert np.array_equal(x, K.normals_from_seeds(seeds, 3))


def test_in_bounds_many_matches_in_bounds():
    wall_map = np.zeros((8, 8))
    wall_map[:2, :] = -1
    wall_map[4, 4] = 1
    env = Environment("e", wall_map=wall_map, size=(40.0, 20.0))
    rng = np.random.default_rng(0)
    pts = rng.uniform(-5.0, 45.0, size=(2000, 2))
    pts[:4] = [[0.0, 0.0], [40.0, 20.0], [40.0, 0.0], [20.0, 10.0]]  # edges
    expected = [env.in_bounds(p) for p in pts]
    assert env.in_bounds_many(pts).tolist() == expected


def test_same_chemistry_under_different_reaction_names_shares_structure():
    def make(names, k):
        return ReactionNetwork(
            "n",
            {
                names[0]: Reaction({}, {"A": 1}, rate_params={"k": k}),
                names[1]: Reaction({"A": 1}, {}, rate_params={"k": 1.0}),
            },
        )

    a, b = make(("birth", "death"), 2.0), make(("make", "decay"), 7.0)
    st = K.structure_of(a)
    assert K.structure_of(b) is st
    assert K.read_params(a, st) == [2.0, 1.0]
    assert K.read_params(b, st) == [7.0, 1.0]


def test_diffusion_kernel_is_bit_identical_to_numpy():
    wall_map = np.zeros((30, 40))
    wall_map[10, 5:35] = 1
    wall_map[20:25, 30] = 1
    wall_map[:3, :3] = -1  # out-of-bounds cells still diffuse
    values = np.random.default_rng(0).random((30, 40))
    results = {}
    for backend in ("python", "numba"):
        env = Environment("e", wall_map=wall_map, size=(80.0, 60.0))
        env.add_field(
            Field(
                "A", values.copy(), is_chemical=True, diffuses=True, diffusivity=6e-10
            )
        )
        env.diffuse(1.0, backend=backend)
        env.diffuse(0.37, backend=backend)
        results[backend] = env.fields["A"].values
    assert np.array_equal(results["python"], results["numba"])


def _voxel_birth_death_env(backend_seed):
    wall_map = np.zeros((40, 40))
    wall_map[20, :] = 1
    env = Environment(
        "e",
        wall_map=wall_map,
        size=(40.0, 40.0),
        fields=[Field("A", np.zeros((40, 40)), is_chemical=True)],
        reactions=_birth_death(k=30.0, gamma=1.0),
        rng=np.random.default_rng(backend_seed),
    )
    env.fields["A"].values[20, :] = 99.0  # wall voxels must not react
    return env


def test_environment_voxel_ssa_kernel_matches_python_statistics():
    stats_by_backend = {}
    for backend in ("python", "numba"):
        env = _voxel_birth_death_env(4)
        env.react(2.0, "SSA", backend=backend)
        A = env.fields["A"].values
        assert np.all(A[20, :] == 99.0)
        medium = np.delete(A, 20, axis=0)
        counts = medium * env.grid_cell_volume
        assert np.allclose(counts, np.round(counts))
        stats_by_backend[backend] = (counts.mean(), counts.var())
    (m_py, v_py), (m_nb, v_nb) = stats_by_backend["python"], stats_by_backend["numba"]
    assert m_nb == pytest.approx(m_py, rel=0.05)
    assert v_nb == pytest.approx(v_py, rel=0.15)


def test_simulation_n_threads_is_applied_and_restored():
    from multicellular.core.simulation import Simulation

    previous = numba.get_num_threads()
    seen = []
    colony = _ssa_colony()
    original_step = colony.step

    def spy(*args, **kwargs):
        seen.append(numba.get_num_threads())
        return original_step(*args, **kwargs)

    colony.step = spy
    Simulation(colony, 0.1, 0.3, simulation_method="SSA", n_threads=1).run(
        show_progress=False
    )
    assert seen and set(seen) == {1}
    assert numba.get_num_threads() == previous


def test_clone_shares_structure_but_not_parameters():
    net = _all_laws_network()
    copy = net.clone()
    assert copy._stoichiometry_matrix is net._stoichiometry_matrix
    assert copy.reactions["make_A"] is not net.reactions["make_A"]
    copy.reactions["make_A"].rate_params["k"] = 99.0
    assert net.reactions["make_A"].rate_params["k"] == 2.0
    copy.last_exported["X"] = 5.0
    assert net.last_exported["X"] == 0.0


def test_founder_streams_are_keyed_by_id_and_unshared():
    def colony(extra):
        shared = np.random.default_rng(11)
        cells = [
            Cell(id=i, position=[10.0 + i, 10.0], orientation=[1.0, 0.0], rng=shared)
            for i in ([-3, 5] + extra)
        ]
        return Colony(cells, Environment("e", wall_map=np.zeros((4, 4))))

    a, b = colony([]), colony([7, 8])
    assert len({id(c.rng) for c in b.cells}) == len(b.cells)
    for ca, cb in zip(a.cells, b.cells):  # ids -3 and 5 in both colonies
        assert ca.rng.random() == cb.rng.random()
