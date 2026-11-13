# core/_kernels.py
"""
Compiled (numba) kernels for stepping many reaction networks at once.

A `ReactionNetwork` is a dict of Python `Reaction` objects, which is
flexible but makes every rate-law evaluation a chain of dict lookups. Here
the *structure* of each network (species ordering, stoichiometry, which
rate law each reaction uses and which species it reads) is packed once into
fixed-shape arrays (`NetworkStructure`), and every network type present in
a colony is stacked into one padded `NetworkTable`. Kernels then step whole
populations of cells (or environment voxels) in a single call, in parallel
over independent work units.

Rate *parameters* are deliberately not part of the cached structure: they
are read from each network's `rate_params` dicts on every call (see
`network_params`), so editing a reaction's parameters between steps takes
effect exactly as it does on the Python path.

Every kernel matches the corresponding `ReactionNetwork` method's
semantics (clamping, rejection of firings that would drive a count
negative, compartment handling, export tracking). Stochastic kernels draw
from a small counter-free xoshiro256** generator seeded per cell, so
results don't depend on how work is split across threads; they agree with
the Python implementations in distribution, not bit for bit.

Networks containing a `custom` rate law (an arbitrary Python callable)
can't be compiled: `structure_of` returns None for them, and callers fall
back to the Python implementation.
"""

import math
from collections import namedtuple

import numpy as np
from numba import njit, prange

# Rate-law codes.
MASS_ACTION, MICHAELIS_MENTEN, HILL_LANGMUIR = 0, 1, 2
_LAW_CODES = {
    "mass_action": MASS_ACTION,
    "michaelis_menten": MICHAELIS_MENTEN,
    "hill_langmuir": HILL_LANGMUIR,
}
N_PARAMS = 4  # parameter slots per reaction (Hill-Langmuir needs all four)

# Stochastic method codes.
SSA_DIRECT, SSA_NEXT_REACTION, SSA_TAU_LEAPING, CLE = 0, 1, 2, 3
STOCHASTIC_CODES = {
    "direct": SSA_DIRECT,
    "next_reaction": SSA_NEXT_REACTION,
    "tau_leaping": SSA_TAU_LEAPING,
}

# Mirrors of reactions.py's tau-leaping constants (imported there would be
# circular; tests check they agree).
_N_CRITICAL = 10
_SSA_FALLBACK_THRESHOLD = 10.0
_SSA_FALLBACK_FIRINGS = 100


# ---------------------------------------------------------------------------
# Network structure
# ---------------------------------------------------------------------------


class NetworkStructure:
    """
    The fixed, parameter-free shape of one `ReactionNetwork`, as arrays.

    Two networks with the same `signature` (same species, reactions, rate
    law types and the species each reads, in the same order) share one
    structure, so every clone of a cell's network maps to the same entry of
    a `NetworkTable`.
    """

    def __init__(self, network, signature):
        self.signature = signature
        self.species = list(network.species)
        self.exported_species = list(network.exported_species)
        names = network._reaction_names
        reactions = [network.reactions[name] for name in names]
        self.reaction_names = names
        self.laws = [rxn.rate_law_type for rxn in reactions]
        index = {s: i for i, s in enumerate(self.species)}
        ns, nr = len(self.species), len(reactions)

        self.law = np.array([_LAW_CODES[law] for law in self.laws], dtype=np.int64)
        self.reactants = [
            [(index[s], float(n)) for s, n in rxn.reactants.items()]
            for rxn in reactions
        ]
        self.s_idx = np.full(nr, -1, dtype=np.int64)
        self.e_idx = np.full(nr, -1, dtype=np.int64)
        self.z_idx = np.full(nr, -1, dtype=np.int64)
        for j, rxn in enumerate(reactions):
            if rxn.rate_law_type == "michaelis_menten":
                if rxn.reactants:
                    self.s_idx[j] = index[next(iter(rxn.reactants))]
                if rxn.catalysts:
                    self.e_idx[j] = index[rxn.catalysts[0]]
            elif rxn.rate_law_type == "hill_langmuir":
                self.e_idx[j] = index[rxn.catalysts[0]]
                self.z_idx[j] = index[rxn.catalysts[1]]
        self.is_export = np.array([bool(rxn.exports) for rxn in reactions])
        self.touched = np.zeros((ns, nr), dtype=np.bool_)
        for j, rxn in enumerate(reactions):
            for s in set(rxn.reactants) | set(rxn.products):
                self.touched[index[s], j] = True
        self.S = np.array(network._stoichiometry_matrix, dtype=float).reshape(ns, nr)
        self.S_exp = np.array(network._export_stoichiometry_matrix, dtype=float)
        self.S_exp = self.S_exp.reshape(len(self.exported_species), nr)
        self.dependency_graph = [list(deps) for deps in network._dependency_graph]

        # Where each reaction's parameters go in a flattened (NR, N_PARAMS)
        # parameter row, and how to read them (key, default; None: required).
        self.param_slots = []
        self.param_keys = []
        for j, law in enumerate(self.laws):
            for slot, key, default in _PARAM_LAYOUT[law]:
                self.param_slots.append(j * N_PARAMS + slot)
                self.param_keys.append((j, key, default))
        self.tau_orders = np.array(network._tau_orders, dtype=float)

    @property
    def n_species(self):
        return len(self.species)

    @property
    def n_reactions(self):
        return len(self.reaction_names)


# Interned structures, keyed by signature. Bounded by the number of distinct
# network *structures* a session creates, which is small.
_STRUCTURES = {}


def _signature(network):
    """
    A hashable fingerprint of `network`'s structure, or None if it can't be
    compiled (a `custom` or unknown rate law, or a Hill-Langmuir reaction
    without its two catalysts, which the Python path rejects at rate time).
    """
    parts = []
    for name in network._reaction_names:
        rxn = network.reactions[name]
        if rxn.rate_law_type not in _LAW_CODES:
            return None
        if rxn.rate_law_type == "hill_langmuir" and len(rxn.catalysts) < 2:
            return None
        parts.append(
            (
                rxn.rate_law_type,
                tuple(rxn.reactants.items()),
                tuple(rxn.products.items()),
                tuple(rxn.catalysts),
                tuple(sorted(rxn.exports.items())),
            )
        )
    return tuple(parts)


def structure_of(network):
    """
    The interned `NetworkStructure` for `network`, or None if it can't be
    compiled. Cached on the network object: a network's reactions (unlike
    their parameters) don't change after construction.
    """
    try:
        return network._kernel_structure
    except AttributeError:
        pass
    signature = _signature(network)
    if signature is None:
        structure = None
    else:
        structure = _STRUCTURES.get(signature)
        if structure is None:
            structure = _STRUCTURES[signature] = NetworkStructure(network, signature)
    network._kernel_structure = structure
    return structure


# Per rate law: (slot, rate_params key, default or None if required), as
# read by `Reaction.rate` and stored in the slots `_rate` reads.
_PARAM_LAYOUT = {
    "mass_action": ((0, "k", 1.0),),
    "michaelis_menten": ((0, "Vmax", 1.0), (1, "Km", 1.0)),
    "hill_langmuir": (
        (0, "alpha", None),
        (1, "beta", None),
        (2, "C", None),
        (3, "n", None),
    ),
}


def read_params(network, structure):
    """
    `network`'s current rate parameters, in `structure.param_slots` order.
    Read from the live `rate_params` dicts on every call, so edits made
    between steps take effect.
    """
    reactions = network.reactions
    # The network's own names: networks sharing a structure need only
    # list the same reactions in the same order, not under the same names.
    names = network._reaction_names
    return [
        (
            reactions[names[j]].rate_params.get(key, default)
            if default is not None
            else reactions[names[j]].rate_params[key]
        )
        for j, key, default in structure.param_keys
    ]


def network_params(network, n_reactions_max):
    """
    `network`'s current rate parameters as an (n_reactions_max, N_PARAMS)
    array, in the slots `_rate` reads: mass action [k], Michaelis-Menten
    [Vmax, Km], Hill-Langmuir [alpha, beta, C, n]. Read fresh on every call.
    """
    out = np.zeros((n_reactions_max, N_PARAMS))
    for j, name in enumerate(network._reaction_names):
        rxn = network.reactions[name]
        p = rxn.rate_params
        law = rxn.rate_law_type
        if law == "mass_action":
            out[j, 0] = p.get("k", 1.0)
        elif law == "michaelis_menten":
            out[j, 0] = p.get("Vmax", 1.0)
            out[j, 1] = p.get("Km", 1.0)
        else:  # hill_langmuir: required keys, as on the Python path
            out[j, 0] = p["alpha"]
            out[j, 1] = p["beta"]
            out[j, 2] = p["C"]
            out[j, 3] = p["n"]
    return out


NetworkTable = namedtuple(
    "NetworkTable",
    [
        "n_species",  # (K,)
        "n_reactions",  # (K,)
        "n_exported",  # (K,)
        "law",  # (K, NR)
        "n_reactants",  # (K, NR)
        "r_idx",  # (K, NR, MAXR)
        "r_pow",  # (K, NR, MAXR)
        "s_idx",  # (K, NR)
        "e_idx",  # (K, NR)
        "z_idx",  # (K, NR)
        "is_export",  # (K, NR)
        "touched",  # (K, NS, NR)
        "S",  # (K, NS, NR)
        "S_exp",  # (K, NE, NR)
        "n_deps",  # (K, NR)
        "deps",  # (K, NR, NR)
        "tau_orders",  # (K, NS)
    ],
)


def build_table(structures):
    """Stack `structures` into one zero-padded `NetworkTable` (entry k = structures[k])."""
    K = len(structures)
    NS = max((st.n_species for st in structures), default=0) or 1
    NR = max((st.n_reactions for st in structures), default=0) or 1
    NE = max((len(st.exported_species) for st in structures), default=0) or 1
    MAXR = max((len(r) for st in structures for r in st.reactants), default=0) or 1

    n_species = np.zeros(K, dtype=np.int64)
    n_reactions = np.zeros(K, dtype=np.int64)
    n_exported = np.zeros(K, dtype=np.int64)
    law = np.zeros((K, NR), dtype=np.int64)
    n_reactants = np.zeros((K, NR), dtype=np.int64)
    r_idx = np.zeros((K, NR, MAXR), dtype=np.int64)
    r_pow = np.zeros((K, NR, MAXR))
    s_idx = np.full((K, NR), -1, dtype=np.int64)
    e_idx = np.full((K, NR), -1, dtype=np.int64)
    z_idx = np.full((K, NR), -1, dtype=np.int64)
    is_export = np.zeros((K, NR), dtype=np.bool_)
    touched = np.zeros((K, NS, NR), dtype=np.bool_)
    S = np.zeros((K, NS, NR))
    S_exp = np.zeros((K, NE, NR))
    n_deps = np.zeros((K, NR), dtype=np.int64)
    deps = np.zeros((K, NR, NR), dtype=np.int64)
    tau_orders = np.zeros((K, NS))

    for k, st in enumerate(structures):
        ns, nr, ne = st.n_species, st.n_reactions, len(st.exported_species)
        n_species[k], n_reactions[k], n_exported[k] = ns, nr, ne
        law[k, :nr] = st.law
        for j, reactants in enumerate(st.reactants):
            n_reactants[k, j] = len(reactants)
            for m, (i, power) in enumerate(reactants):
                r_idx[k, j, m] = i
                r_pow[k, j, m] = power
        s_idx[k, :nr] = st.s_idx
        e_idx[k, :nr] = st.e_idx
        z_idx[k, :nr] = st.z_idx
        is_export[k, :nr] = st.is_export
        touched[k, :ns, :nr] = st.touched
        S[k, :ns, :nr] = st.S
        S_exp[k, :ne, :nr] = st.S_exp
        for j, dependents in enumerate(st.dependency_graph):
            n_deps[k, j] = len(dependents)
            deps[k, j, : len(dependents)] = dependents
        tau_orders[k, :ns] = st.tau_orders

    return NetworkTable(
        n_species,
        n_reactions,
        n_exported,
        law,
        n_reactants,
        r_idx,
        r_pow,
        s_idx,
        e_idx,
        z_idx,
        is_export,
        touched,
        S,
        S_exp,
        n_deps,
        deps,
        tau_orders,
    )


# ---------------------------------------------------------------------------
# Rate laws
# ---------------------------------------------------------------------------


@njit(cache=True)
def _rate(T, k, j, conc, params):
    """Reaction j of network k's rate at concentrations `conc` (matches `Reaction.rate`)."""
    law = T.law[k, j]
    if law == MASS_ACTION:
        rate = params[j, 0]
        for m in range(T.n_reactants[k, j]):
            rate *= conc[T.r_idx[k, j, m]] ** T.r_pow[k, j, m]
        return rate
    if law == MICHAELIS_MENTEN:
        s = T.s_idx[k, j]
        e = T.e_idx[k, j]
        S = conc[s] if s >= 0 else 0.0
        E = conc[e] if e >= 0 else 1.0
        return params[j, 0] * E * S / (params[j, 1] + S)
    E = conc[T.e_idx[k, j]]
    z = conc[T.z_idx[k, j]]
    Cz_n = params[j, 2] * z ** params[j, 3]
    return params[j, 1] * E * (1.0 + params[j, 0] * Cz_n) / (1.0 + Cz_n)


@njit(parallel=True, cache=True)
def ode_derivatives(T, net, params, C, dC, dE):
    """
    Reaction-only ODE right-hand side for m independent compartments.

    For each row i of `C` (concentrations, already clamped to >= 0, in
    network `net[i]`'s species order), writes S @ v into dC[i] and the
    export rate S_exp @ v into dE[i]. Export-tagged reactions are clamped
    as in `ReactionNetwork._simulate_ode_step`: rate >= 0, and 0 whenever a
    reactant is depleted.
    """
    m = C.shape[0]
    for i in prange(m):
        k = net[i]
        ns = T.n_species[k]
        nr = T.n_reactions[k]
        ne = T.n_exported[k]
        v = np.empty(nr)
        for j in range(nr):
            r = _rate(T, k, j, C[i], params[i])
            if T.is_export[k, j]:
                depleted = False
                for m_ in range(T.n_reactants[k, j]):
                    if C[i, T.r_idx[k, j, m_]] <= 0.0:
                        depleted = True
                r = 0.0 if depleted else max(r, 0.0)
            v[j] = r
        for s in range(dC.shape[1]):
            acc = 0.0
            if s < ns:
                for j in range(nr):
                    acc += T.S[k, s, j] * v[j]
            dC[i, s] = acc
        for e in range(dE.shape[1]):
            acc = 0.0
            if e < ne:
                for j in range(nr):
                    acc += T.S_exp[k, e, j] * v[j]
            dE[i, e] = acc


# ---------------------------------------------------------------------------
# Random numbers: xoshiro256** seeded through splitmix64
# ---------------------------------------------------------------------------

_U64 = np.uint64


@njit(cache=True)
def _splitmix64(x):
    x = x + _U64(0x9E3779B97F4A7C15)
    z = x
    z = (z ^ (z >> _U64(30))) * _U64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> _U64(27))) * _U64(0x94D049BB133111EB)
    return x, z ^ (z >> _U64(31))


@njit(cache=True)
def rng_init(seed):
    """A fresh xoshiro256** state (uint64[4]) from a 64-bit seed."""
    state = np.empty(4, dtype=np.uint64)
    x = _U64(seed)
    for i in range(4):
        x, state[i] = _splitmix64(x)
    return state


@njit(cache=True)
def _rotl(x, k):
    return (x << _U64(k)) | (x >> _U64(64 - k))


@njit(cache=True)
def _next_u64(state):
    s0, s1, s2, s3 = state[0], state[1], state[2], state[3]
    result = _rotl(s1 * _U64(5), 7) * _U64(9)
    t = s1 << _U64(17)
    s2 ^= s0
    s3 ^= s1
    s1 ^= s2
    s0 ^= s3
    s2 ^= t
    s3 = _rotl(s3, 45)
    state[0], state[1], state[2], state[3] = s0, s1, s2, s3
    return result


@njit(cache=True)
def rng_uniform(state):
    """Uniform on (0, 1] (never 0, so it's safe under log)."""
    return ((_next_u64(state) >> _U64(11)) + _U64(1)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def rng_exponential(state):
    """Standard exponential (mean 1)."""
    return -math.log(rng_uniform(state))


@njit(cache=True)
def rng_normal(state):
    """Standard normal (Marsaglia's polar method)."""
    while True:
        u = 2.0 * rng_uniform(state) - 1.0
        v = 2.0 * rng_uniform(state) - 1.0
        s = u * u + v * v
        if 0.0 < s < 1.0:
            return u * math.sqrt(-2.0 * math.log(s) / s)


@njit(cache=True)
def rng_poisson(state, lam):
    """Poisson(lam): multiplication method below 10, Hörmann's PTRS above (as numpy)."""
    if lam <= 0.0:
        return 0.0
    if lam < 10.0:
        limit = math.exp(-lam)
        k = 0.0
        p = rng_uniform(state)
        while p > limit:
            k += 1.0
            p *= rng_uniform(state)
        return k
    slam = math.sqrt(lam)
    loglam = math.log(lam)
    b = 0.931 + 2.53 * slam
    a = -0.059 + 0.02483 * b
    invalpha = 1.1239 + 1.1328 / (b - 3.4)
    vr = 0.9277 - 3.6224 / (b - 2.0)
    while True:
        U = rng_uniform(state) - 0.5
        V = rng_uniform(state)
        us = 0.5 - abs(U)
        k = math.floor((2.0 * a / us + b) * U + lam + 0.43)
        if us >= 0.07 and V <= vr:
            return k
        if k < 0.0 or (us < 0.013 and V > us):
            continue
        if (math.log(V) + math.log(invalpha) - math.log(a / (us * us) + b)) <= (
            -lam + k * loglam - math.lgamma(k + 1.0)
        ):
            return k


@njit(parallel=True, cache=True)
def normals_from_seeds(seeds, k):
    """An (n, k) array of standard normals, row i drawn from a stream seeded by seeds[i]."""
    n = seeds.shape[0]
    out = np.empty((n, k))
    for i in prange(n):
        state = rng_init(seeds[i])
        for j in range(k):
            out[i, j] = rng_normal(state)
    return out


@njit(cache=True)
def _choose(state, weights, n, total):
    """Index in [0, n) drawn with probability weights[j] / total."""
    target = rng_uniform(state) * total
    acc = 0.0
    last = -1
    for j in range(n):
        if weights[j] > 0.0:
            last = j
            acc += weights[j]
            if target <= acc:
                return j
    return last


@njit(cache=True)
def _choose_masked(state, weights, mask, n, total):
    """Like `_choose`, restricted to indices where mask[j] is True."""
    target = rng_uniform(state) * total
    acc = 0.0
    last = -1
    for j in range(n):
        if mask[j] and weights[j] > 0.0:
            last = j
            acc += weights[j]
            if target <= acc:
                return j
    return last


# ---------------------------------------------------------------------------
# Stochastic steps on one compartment (counts are modified in place)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _propensities(T, k, params, counts, volumes, volume, conc, a):
    ns = T.n_species[k]
    for s in range(ns):
        conc[s] = counts[s] / volumes[s]
    a0 = 0.0
    for j in range(T.n_reactions[k]):
        a[j] = max(_rate(T, k, j, conc, params), 0.0) * volume
        a0 += a[j]
    return a0


@njit(cache=True)
def _try_fire(T, k, counts, export, j, times):
    """Apply `times` firings of reaction j unless any count would go negative."""
    ns = T.n_species[k]
    for s in range(ns):
        if counts[s] + times * T.S[k, s, j] < 0.0:
            return False
    for s in range(ns):
        counts[s] += times * T.S[k, s, j]
    for e in range(T.n_exported[k]):
        export[e] += times * T.S_exp[k, e, j]
    return True


@njit(cache=True)
def _ssa_direct(
    T, k, params, counts, export, volumes, volume, state, t, t_end, max_firings
):
    """Port of `ReactionNetwork._ssa_direct_firings`; returns the time reached."""
    nr = T.n_reactions[k]
    a = np.empty(nr)
    conc = np.empty(T.n_species[k])
    fired = 0
    while t < t_end and (max_firings < 0 or fired < max_firings):
        a0 = _propensities(T, k, params, counts, volumes, volume, conc, a)
        if a0 <= 0.0:
            return t_end
        tau = rng_exponential(state) / a0
        if t + tau > t_end:
            return t_end
        j = _choose(state, a, nr, a0)
        t += tau
        fired += 1
        _try_fire(T, k, counts, export, j, 1.0)
    return t


@njit(cache=True)
def _sift_up(heap, pos, times, i):
    while i > 0:
        parent = (i - 1) // 2
        if times[heap[i]] < times[heap[parent]]:
            a, b = heap[i], heap[parent]
            heap[i], heap[parent] = b, a
            pos[b], pos[a] = i, parent
            i = parent
        else:
            return


@njit(cache=True)
def _sift_down(heap, pos, times, i, n):
    while True:
        left, right, smallest = 2 * i + 1, 2 * i + 2, i
        if left < n and times[heap[left]] < times[heap[smallest]]:
            smallest = left
        if right < n and times[heap[right]] < times[heap[smallest]]:
            smallest = right
        if smallest == i:
            return
        a, b = heap[i], heap[smallest]
        heap[i], heap[smallest] = b, a
        pos[b], pos[a] = i, smallest
        i = smallest


@njit(cache=True)
def _heap_update(heap, pos, times, item, time, n):
    old = times[item]
    times[item] = time
    if time < old:
        _sift_up(heap, pos, times, pos[item])
    else:
        _sift_down(heap, pos, times, pos[item], n)


@njit(cache=True)
def _next_reaction(T, k, params, counts, export, volumes, volume, state, dt):
    """Port of `ReactionNetwork._simulate_next_reaction_step`."""
    nr = T.n_reactions[k]
    ns = T.n_species[k]
    if nr == 0:
        return
    conc = np.empty(ns)
    for s in range(ns):
        conc[s] = counts[s] / volumes[s]
    a = np.empty(nr)
    times = np.empty(nr)
    for j in range(nr):
        a[j] = max(_rate(T, k, j, conc, params), 0.0) * volume
    for j in range(nr):
        draw = rng_exponential(state)
        times[j] = draw / a[j] if a[j] > 0.0 else np.inf
    heap = np.arange(nr)
    pos = np.arange(nr)
    for i in range(nr // 2 - 1, -1, -1):
        _sift_down(heap, pos, times, i, nr)

    while True:
        mu = heap[0]
        t = times[mu]
        if t > dt:
            break
        if not _try_fire(T, k, counts, export, mu, 1.0):
            _heap_update(heap, pos, times, mu, t + rng_exponential(state) / a[mu], nr)
            continue
        for s in range(ns):
            if T.S[k, s, mu] != 0.0:
                conc[s] = counts[s] / volumes[s]
        for d in range(T.n_deps[k, mu]):
            kk = T.deps[k, mu, d]
            if kk == mu:
                continue
            a_old = a[kk]
            a_new = max(_rate(T, k, kk, conc, params), 0.0) * volume
            a[kk] = a_new
            tau_k = times[kk]
            if a_new <= 0.0:
                _heap_update(heap, pos, times, kk, np.inf, nr)
            elif a_old <= 0.0 or not np.isfinite(tau_k):
                _heap_update(
                    heap, pos, times, kk, t + rng_exponential(state) / a_new, nr
                )
            else:
                _heap_update(
                    heap, pos, times, kk, t + (a_old / a_new) * (tau_k - t), nr
                )
        a[mu] = max(_rate(T, k, mu, conc, params), 0.0) * volume
        new_time = t + rng_exponential(state) / a[mu] if a[mu] > 0.0 else np.inf
        _heap_update(heap, pos, times, mu, new_time, nr)


@njit(cache=True)
def _tau_leap_size(T, k, counts, a, noncritical, epsilon):
    """Port of `ReactionNetwork._tau_leap_size`."""
    ns = T.n_species[k]
    nr = T.n_reactions[k]
    best = np.inf
    for s in range(ns):
        g = T.tau_orders[k, s]
        if g <= 0.0:
            continue
        mu = 0.0
        sigma2 = 0.0
        for j in range(nr):
            if noncritical[j]:
                S_sj = T.S[k, s, j]
                mu += abs(S_sj) * a[j]
                sigma2 += S_sj * S_sj * a[j]
        if mu == 0.0 and not sigma2 > 0.0:
            continue
        bound = max(epsilon * counts[s] / g, 1.0)
        if mu != 0.0:
            best = min(best, bound / abs(mu))
        if sigma2 > 0.0:
            best = min(best, bound * bound / sigma2)
    return best


@njit(cache=True)
def _tau_leap(T, k, params, counts, export, volumes, volume, state, dt, epsilon):
    """Port of `ReactionNetwork._simulate_tau_leap_step`."""
    nr = T.n_reactions[k]
    ns = T.n_species[k]
    a = np.empty(nr)
    conc = np.empty(ns)
    critical = np.zeros(nr, dtype=np.bool_)
    noncritical = np.zeros(nr, dtype=np.bool_)
    firings = np.zeros(nr)
    new_counts = np.empty(ns)
    tau_cap = np.inf
    t = 0.0
    while t < dt:
        a0 = _propensities(T, k, params, counts, volumes, volume, conc, a)
        if a0 <= 0.0:
            break

        a0_crit = 0.0
        for j in range(nr):
            limit = np.inf
            for s in range(ns):
                S_sj = T.S[k, s, j]
                if S_sj < 0.0:
                    limit = min(limit, math.floor(counts[s] / -S_sj))
            critical[j] = a[j] > 0.0 and limit < _N_CRITICAL
            noncritical[j] = a[j] > 0.0 and not critical[j]
            if critical[j]:
                a0_crit += a[j]

        tau_p = min(_tau_leap_size(T, k, counts, a, noncritical, epsilon), tau_cap)
        if tau_p < _SSA_FALLBACK_THRESHOLD / a0:
            t = _ssa_direct(
                T,
                k,
                params,
                counts,
                export,
                volumes,
                volume,
                state,
                t,
                dt,
                _SSA_FALLBACK_FIRINGS,
            )
            tau_cap = np.inf
            continue

        tau_pp = rng_exponential(state) / a0_crit if a0_crit > 0.0 else np.inf
        tau = min(tau_p, tau_pp, dt - t)

        for j in range(nr):
            firings[j] = rng_poisson(state, a[j] * tau) if noncritical[j] else 0.0
        negative = False
        for s in range(ns):
            acc = counts[s]
            for j in range(nr):
                acc += T.S[k, s, j] * firings[j]
            new_counts[s] = acc
            if acc < 0.0:
                negative = True
        if negative:
            tau_cap = tau_p / 2.0
            continue

        if tau == tau_pp:
            j = _choose_masked(state, a, critical, nr, a0_crit)
            ok = True
            for s in range(ns):
                if new_counts[s] + T.S[k, s, j] < 0.0:
                    ok = False
            if ok:
                for s in range(ns):
                    new_counts[s] += T.S[k, s, j]
                firings[j] += 1.0

        for s in range(ns):
            counts[s] = new_counts[s]
        for e in range(T.n_exported[k]):
            for j in range(nr):
                export[e] += T.S_exp[k, e, j] * firings[j]
        t = dt if tau == dt - t else t + tau
        tau_cap = np.inf


@njit(cache=True)
def _cle(T, k, params, conc, external, weights, volume, dt, state, out, export):
    """
    Port of `ReactionNetwork._simulate_cle_step`, in concentrations: writes
    the new concentrations into `out` and the exported molecule counts
    (unclamped, as on the Python path) into `export`.
    """
    nr = T.n_reactions[k]
    ns = T.n_species[k]
    extent = np.empty(nr)
    scale = math.sqrt(dt / volume)
    for j in range(nr):
        v = max(_rate(T, k, j, conc, params), 0.0)
        extent[j] = dt * v + scale * (math.sqrt(v) * rng_normal(state))

    # _clamp_extents
    for j in range(nr):
        if T.is_export[k, j]:
            cap = np.inf
            for m in range(T.n_reactants[k, j]):
                s = T.r_idx[k, j, m]
                cap = min(cap, conc[s] / (T.r_pow[k, j, m] * weights[s]))
            extent[j] = min(max(extent[j], 0.0), cap)
        crosses = False
        for s in range(ns):
            if T.touched[k, s, j] and external[s]:
                crosses = True
        if not crosses:
            continue
        for s in range(ns):
            S_sj = T.S[k, s, j]
            if S_sj == 0.0:
                continue
            available = conc[s] / (abs(S_sj) * weights[s])
            if S_sj < 0.0:
                extent[j] = min(extent[j], available)
            else:
                extent[j] = max(extent[j], -available)

    for s in range(ns):
        acc = 0.0
        for j in range(nr):
            acc += T.S[k, s, j] * extent[j]
        out[s] = max(conc[s] + weights[s] * acc, 0.0)
    for e in range(T.n_exported[k]):
        acc = 0.0
        for j in range(nr):
            acc += T.S_exp[k, e, j] * extent[j]
        export[e] = acc * volume


@njit(cache=True)
def _step_one(
    T,
    k,
    params,
    method,
    conc_in,
    external,
    volumes,
    volume,
    dt,
    epsilon,
    state,
    conc_out,
    export,
):
    """
    Advance one compartment by `dt`. `conc_in` holds every species'
    concentration in its own compartment; `volumes` each species'
    compartment volume (`volume` for the network's own, e.g. a voxel's for
    an `external` species). Writes `conc_out` and `export` (molecules).
    """
    ns = T.n_species[k]
    for e in range(export.shape[0]):
        export[e] = 0.0
    if method == CLE:
        weights = np.empty(ns)
        for s in range(ns):
            weights[s] = volume / volumes[s]
        _cle(
            T,
            k,
            params,
            conc_in,
            external,
            weights,
            volume,
            dt,
            state,
            conc_out,
            export,
        )
        return

    counts = np.empty(ns)
    for s in range(ns):
        c = conc_in[s] * volumes[s]
        if not external[s]:
            c = np.round(c)
        counts[s] = max(c, 0.0)
    if method == SSA_DIRECT:
        _ssa_direct(T, k, params, counts, export, volumes, volume, state, 0.0, dt, -1)
    elif method == SSA_NEXT_REACTION:
        _next_reaction(T, k, params, counts, export, volumes, volume, state, dt)
    else:
        _tau_leap(T, k, params, counts, export, volumes, volume, state, dt, epsilon)
    for s in range(ns):
        conc_out[s] = counts[s] / volumes[s]


@njit(parallel=True, cache=True)
def step_stochastic(
    T,
    method,
    dt,
    epsilon,
    net,
    params,
    conc,
    volume,
    ext_field,
    voxel,
    pool,
    voxel_volume,
    seeds,
    unit_ptr,
    unit_items,
    export,
):
    """
    Stochastically step many compartments (cells or voxels) in parallel.

    Compartment i runs network `net[i]` with rate parameters `params[i]`
    and volume `volume[i]`, starting from `conc[i]` (concentrations in that
    network's species order). A species s with `ext_field[i, s] = f >= 0`
    is extracellular: it is read from, and written back to,
    `pool[f, voxel[i]]` (in a compartment of `voxel_volume`) instead of
    `conc[i, s]`, exactly as `Cell.step`'s `external` is.

    Compartments are grouped into work units (`unit_items[unit_ptr[u]:
    unit_ptr[u + 1]]`). Units run in parallel and must not share a pool
    voxel; within a unit, compartments run sequentially in the given order,
    each seeing the pool after the previous one's update, as the Python
    path steps cells sharing a voxel. `seeds[i]` seeds compartment i's
    random stream for this step, so results don't depend on the thread
    count. Writes the results into `conc`/`pool` and the exported molecule
    counts into `export[i]`.
    """
    n_units = unit_ptr.shape[0] - 1
    NS = conc.shape[1]
    for u in prange(n_units):
        conc_in = np.empty(NS)
        conc_out = np.empty(NS)
        volumes = np.empty(NS)
        external = np.zeros(NS, dtype=np.bool_)
        for idx in range(unit_ptr[u], unit_ptr[u + 1]):
            i = unit_items[idx]
            k = net[i]
            ns = T.n_species[k]
            for s in range(ns):
                f = ext_field[i, s]
                if f >= 0:
                    conc_in[s] = pool[f, voxel[i]]
                    volumes[s] = voxel_volume
                    external[s] = True
                else:
                    conc_in[s] = conc[i, s]
                    volumes[s] = volume[i]
                    external[s] = False
            state = rng_init(seeds[i])
            _step_one(
                T,
                k,
                params[i],
                method,
                conc_in,
                external,
                volumes,
                volume[i],
                dt,
                epsilon,
                state,
                conc_out,
                export[i],
            )
            for s in range(ns):
                f = ext_field[i, s]
                if f >= 0:
                    pool[f, voxel[i]] = conc_out[s]
                else:
                    conc[i, s] = conc_out[s]


# ---------------------------------------------------------------------------
# Diffusion
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def ftcs_diffuse(C, is_wall, coef, inv_dy2, inv_dx2, n_steps):
    """
    `n_steps` explicit FTCS sub-steps of 2D diffusion on `C`, returning the
    result (`C` itself is not modified).

    Port of `Environment.diffuse`'s numpy loop, without its per-sub-step
    temporaries: a neighbor outside the grid, or inside a wall, contributes
    the cell's own value (no flux across that face), and wall cells are
    never updated. `coef` is D * sub_dt; the arithmetic matches the numpy
    version term for term.
    """
    n_rows, n_cols = C.shape
    A = C.copy()
    B = np.empty_like(A)
    for _ in range(n_steps):
        for i in prange(n_rows):
            for j in range(n_cols):
                c = A[i, j]
                if is_wall[i, j]:
                    B[i, j] = c
                    continue
                up = A[i + 1, j] if i + 1 < n_rows and not is_wall[i + 1, j] else c
                down = A[i - 1, j] if i > 0 and not is_wall[i - 1, j] else c
                right = A[i, j + 1] if j + 1 < n_cols and not is_wall[i, j + 1] else c
                left = A[i, j - 1] if j > 0 and not is_wall[i, j - 1] else c
                laplacian = (up - 2.0 * c + down) * inv_dy2 + (
                    right - 2.0 * c + left
                ) * inv_dx2
                B[i, j] = c + coef * laplacian
        A, B = B, A
    return A
