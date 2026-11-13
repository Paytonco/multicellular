# core/_batch.py
"""
Batched stepping of a whole colony's reaction networks.

`step_cells` is the "numba" backend of `Colony._step_cells`: rather than
calling `Cell.step` once per cell (one `solve_ivp`, or one Python Gillespie
loop, per cell), it packs every cell whose network can be compiled (see
`_kernels.structure_of`) into arrays and advances them all at once:

- ODE: one `solve_ivp` over the joint system of every cell's intracellular
  state plus one copy of each extracellular (chemical Field) species per
  occupied voxel. Cells sharing a voxel draw from and add to that pool
  simultaneously, and molecules are conserved between cells and Fields.
- SSA / CLE: one parallel kernel call (`_kernels.step_stochastic`). Cells
  not coupled to a Field are independent work units; cells coupled to a
  Field are grouped by voxel, and each voxel's cells run one after another
  in an order drawn from `environment.rng` (as on the Python path), each
  seeing the pool after the previous cell's update.

Cells whose network can't be compiled (a `custom` rate law) are returned to
the caller to step on the Python path.
"""

import contextlib

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from threadpoolctl import ThreadpoolController

from . import _kernels
from .reactions import _resolve_ode_method, _resolve_ssa_method

_THREADPOOLS = None


@contextlib.contextmanager
def single_threaded_blas():
    """
    Limit BLAS to one thread inside the block.

    Between numba's parallel kernels, numpy/scipy make many small BLAS
    calls (e.g. `solve_ivp`'s Runge-Kutta stage sums over a batched state).
    Threaded BLAS gains nothing on those, and its worker threads keep
    spinning afterwards, starving numba's threads: on 8 cores this made a
    3000-cell batched ODE step ~8x slower.
    """
    global _THREADPOOLS
    if _THREADPOOLS is None:
        _THREADPOOLS = ThreadpoolController()
    with _THREADPOOLS.limit(limits=1, user_api="blas"):
        yield


# NetworkTables for each distinct combination of network structures seen,
# keyed by the structures' ids (structures are interned, so ids are stable).
_TABLES = {}


def table_for(structures):
    """The (cached) `NetworkTable` stacking `structures`, in that order."""
    key = tuple(id(st) for st in structures)
    table = _TABLES.get(key)
    if table is None:
        table = _TABLES[key] = _kernels.build_table(structures)
    return table


def _pack(cells, chemical, environment):
    """
    Pack `cells` (all living, all with compilable networks) into kernel arrays.

    Cells are gathered one network structure at a time, so the per-cell work
    is a couple of list comprehensions. Returns a dict of arrays plus what
    `_unpack` needs to write the results back.
    """
    structures, structure_index, net = [], {}, []
    for cell in cells:
        st = _kernels.structure_of(cell.network)
        k = structure_index.get(id(st))
        if k is None:
            k = structure_index[id(st)] = len(structures)
            structures.append(st)
        net.append(k)
    net = np.asarray(net, dtype=np.int64)
    table = table_for(structures)
    m = len(cells)
    NS = table.S.shape[1]
    NR = table.S.shape[2]
    NE = table.S_exp.shape[1]

    field_names = list(chemical)
    field_index = {name: f for f, name in enumerate(field_names)}

    conc = np.zeros((m, NS))
    params = np.zeros((m, NR * _kernels.N_PARAMS))
    ext_field = np.full((m, NS), -1, dtype=np.int64)
    coupled = np.zeros(m, dtype=bool)
    groups = []
    for k, st in enumerate(structures):
        rows = np.flatnonzero(net == k)
        group = [cells[i] for i in rows]
        names = st.species
        conc[rows, : st.n_species] = [
            [cell.concentrations.get(name, 0.0) for name in names] for cell in group
        ]
        params[np.ix_(rows, st.param_slots)] = [
            _kernels.read_params(cell.network, st) for cell in group
        ]
        ext_cols = [s for s, name in enumerate(names) if name in field_index]
        if ext_cols:
            ext_field[np.ix_(rows, ext_cols)] = [
                field_index[names[s]] for s in ext_cols
            ]
            coupled[rows] = True
        internal = [s for s, name in enumerate(names) if name not in field_index]
        groups.append((rows, group, [names[s] for s in internal], internal))

    lengths = np.array([cell.length for cell in cells], dtype=float)
    radii = np.array([cell.radius for cell in cells], dtype=float)
    volume = _volumes(lengths, radii)

    voxel = np.full(m, -1, dtype=np.int64)
    if coupled.any():
        positions = np.array([cells[i].position for i in np.flatnonzero(coupled)])
        rows_, cols_ = environment.grid_indices(positions)
        voxel[coupled] = np.asarray(rows_) * environment.shape[1] + np.asarray(cols_)

    return {
        "table": table,
        "structures": structures,
        "groups": groups,
        "net": net,
        "params": params.reshape(m, NR, _kernels.N_PARAMS),
        "conc": conc,
        "volume": volume,
        "lengths": lengths,
        "radii": radii,
        "ext_field": ext_field,
        "coupled": coupled,
        "voxel": voxel,
        "field_names": field_names,
        "NE": NE,
    }


def _volumes(lengths, radii):
    """`Cell.compute_volume`, vectorized."""
    return np.pi * radii**2 * lengths + (4 / 3) * np.pi * radii**3


def _pool_from_fields(chemical, field_names):
    """Every chemical Field's values as rows of one (n_fields, n_voxels) float array."""
    if not field_names:
        return np.zeros((1, 1))
    return np.stack(
        [np.asarray(chemical[name].values, dtype=float).ravel() for name in field_names]
    )


def _pool_to_fields(pool, chemical, field_names, touched):
    """Write the rows of `pool` for the fields in `touched` back into their Fields, in place."""
    for f in touched:
        field = chemical[field_names[f]]
        field.values[...] = pool[f].reshape(field.values.shape)


def _unpack(cells, packed, conc, export, dt):
    """
    Store each cell's new state and exports, then grow it: the tail of
    `Cell.step`, with `Cell.grow` vectorized over the batch (same
    arithmetic: copy numbers at the old volume, divided by the new one).
    """
    m = len(cells)
    growth_rate = np.array([cell.growth_rate for cell in cells], dtype=float)
    is_sphere = np.array([cell.shape == "sphere" for cell in cells])
    anchored = np.array([cell.anchored for cell in cells])
    lengths, radii = packed["lengths"], packed["radii"]
    old_volume = packed["volume"]

    grows = ~anchored
    new_lengths = np.where(
        grows & ~is_sphere, lengths * np.exp(growth_rate * dt), lengths
    )
    new_radii = np.where(
        grows & is_sphere, radii * np.exp(growth_rate * dt / 3.0), radii
    )
    new_volume = _volumes(new_lengths, new_radii)
    copies = conc * old_volume[:, None]
    diluted = np.where(grows[:, None], copies / new_volume[:, None], conc)

    structures = packed["structures"]
    net = packed["net"]
    for rows, group, internal_names, internal in packed["groups"]:
        st = structures[net[rows[0]]]
        exported_names = st.exported_species
        values = diluted[np.ix_(rows, internal)].tolist()
        exports = export[rows, : len(exported_names)].tolist()
        for cell, row, exp_row in zip(group, values, exports):
            cell.concentrations = dict(zip(internal_names, row))
            exported = dict(zip(exported_names, exp_row))
            cell.network.last_exported = exported
            cell.pending_export = exported

    new_lengths = new_lengths.tolist()
    new_radii = new_radii.tolist()
    for i in range(m):
        cell = cells[i]
        cell.age += dt
        if grows[i]:
            if is_sphere[i]:
                cell.radius = new_radii[i]
            else:
                cell.length = new_lengths[i]


def step_cells(
    cells,
    environment,
    dt,
    method,
    ode_method="RK45",
    ssa_method="direct",
    tau_epsilon=0.03,
):
    """
    Step every living cell in `cells` (reactions, Field exchange, growth).

    Returns the cells it did not step (their networks can't be compiled);
    the caller steps those on the Python path. Dead cells are skipped, and
    cells without a network only grow, as in `Cell.step`.
    """
    chemical = {
        name: field for name, field in environment.fields.items() if field.is_chemical
    }
    compiled, fallback = [], []
    for cell in cells:
        if not cell.alive:
            continue
        if not cell.network:
            cell.grow(dt)
        elif _kernels.structure_of(cell.network) is None:
            fallback.append(cell)
        else:
            compiled.append(cell)
    if not compiled:
        return fallback

    method = str(method).upper()
    if method not in ("ODE", "SSA", "CLE"):
        raise ValueError(f"Unknown simulation method: {method}")
    if method == "ODE":
        ode_method = _resolve_ode_method(ode_method)
        if ode_method == "LSODA" and _any_coupled(compiled, chemical):
            # LSODA only takes a banded Jacobian, which the cell <-> voxel
            # coupling breaks; a dense one would be far too large. Step
            # these cells one at a time instead, as the Python path does.
            return fallback + compiled

    packed = _pack(compiled, chemical, environment)
    pool = _pool_from_fields(chemical, packed["field_names"])
    used_fields = np.unique(packed["ext_field"][packed["ext_field"] >= 0])

    if method == "ODE":
        conc, export = _step_ode(
            packed, pool, environment.grid_cell_volume, dt, ode_method
        )
    else:
        code = (
            _kernels.CLE
            if method == "CLE"
            else _kernels.STOCHASTIC_CODES[_resolve_ssa_method(ssa_method)]
        )
        if code == _kernels.SSA_TAU_LEAPING and not tau_epsilon > 0:
            raise ValueError(f"tau_epsilon must be positive, got {tau_epsilon}")
        conc, export = _step_stochastic(
            packed, compiled, pool, environment, dt, code, tau_epsilon
        )

    _pool_to_fields(pool, chemical, packed["field_names"], used_fields)
    _unpack(compiled, packed, conc, export, dt)
    return fallback


def _any_coupled(cells, chemical):
    return bool(chemical) and any(
        s in chemical for cell in cells for s in cell.network.species
    )


def _work_units(packed, rng):
    """
    Work-unit layout for `step_stochastic`: one unit per uncoupled cell,
    and one per occupied voxel holding that voxel's coupled cells, ordered
    within the voxel by a permutation of all coupled cells drawn from `rng`
    (the same draw the Python path makes).
    """
    coupled_idx = np.flatnonzero(packed["coupled"])
    uncoupled_idx = np.flatnonzero(~packed["coupled"])
    if len(coupled_idx):
        order = coupled_idx[rng.permutation(len(coupled_idx))]
        order = order[np.argsort(packed["voxel"][order], kind="stable")]
        _, starts = np.unique(packed["voxel"][order], return_index=True)
    else:
        order = coupled_idx
        starts = np.zeros(0, dtype=np.int64)
    items = np.concatenate([uncoupled_idx, order]).astype(np.int64)
    ptr = np.concatenate(
        [
            np.arange(len(uncoupled_idx)),
            len(uncoupled_idx) + starts,
            [len(items)],
        ]
    ).astype(np.int64)
    return ptr, items


def _step_stochastic(packed, cells, pool, environment, dt, code, tau_epsilon):
    # One raw 64-bit draw from each cell's own stream seeds its kernel
    # generator for this step.
    seeds = np.fromiter(
        (cell.rng.bit_generator.random_raw() for cell in cells),
        dtype=np.uint64,
        count=len(cells),
    )
    ptr, items = _work_units(packed, environment.rng)
    conc = packed["conc"]
    export = np.zeros((len(cells), packed["NE"]))
    _kernels.step_stochastic(
        packed["table"],
        code,
        float(dt),
        float(tau_epsilon),
        packed["net"],
        packed["params"],
        conc,
        packed["volume"],
        packed["ext_field"],
        packed["voxel"],
        pool,
        float(environment.grid_cell_volume),
        seeds,
        ptr,
        items,
        export,
    )
    return conc, export


def _step_ode(packed, pool, voxel_volume, dt, ode_method):
    """
    One `solve_ivp` over every packed cell plus the occupied voxels' pools.

    State layout: per cell, its NS species then its NE cumulative exports
    (so each cell's block is contiguous), followed by one entry per
    occupied (field, voxel) pool slot.
    """
    T = packed["table"]
    net, params, volume = packed["net"], packed["params"], packed["volume"]
    m, NS = packed["conc"].shape
    NE = packed["NE"]
    B = NS + NE  # per-cell block width

    ext_cell, ext_sp = np.nonzero(packed["ext_field"] >= 0)
    ext_f = packed["ext_field"][ext_cell, ext_sp]
    ext_flat = ext_f * pool.shape[1] + packed["voxel"][ext_cell]
    slot_keys, ext_slot = np.unique(ext_flat, return_inverse=True)
    q = len(slot_keys)
    # Molecules a cell's reactions move per unit concentration change in
    # its own volume, re-expressed as concentration in the voxel.
    ext_scale = volume[ext_cell] / voxel_volume

    y0 = np.zeros(m * B + q)
    blocks = y0[: m * B].reshape(m, B)
    blocks[:, :NS] = packed["conc"]
    y0[m * B :] = pool.ravel()[slot_keys]

    C = np.empty((m, NS))
    dC = np.empty((m, NS))
    dE = np.empty((m, max(NE, 1)))

    def rhs(_t, y):
        Y = y[: m * B].reshape(m, B)
        np.maximum(Y[:, :NS], 0.0, out=C)
        P = np.maximum(y[m * B :], 0.0)
        if q:
            C[ext_cell, ext_sp] = P[ext_slot]
        _kernels.ode_derivatives(T, net, params, C, dC, dE)
        out = np.empty_like(y)
        out_blocks = out[: m * B].reshape(m, B)
        if q:
            flux = dC[ext_cell, ext_sp] * ext_scale
            dC[ext_cell, ext_sp] = 0.0
            out[m * B :] = np.bincount(ext_slot, weights=flux, minlength=q)
        out_blocks[:, :NS] = dC
        out_blocks[:, NS:] = dE[:, :NE]
        return out

    options = {}
    if ode_method in ("BDF", "Radau"):
        options["jac_sparsity"] = _jacobian_sparsity(
            m, NS, NE, q, ext_cell, ext_sp, ext_slot
        )
    elif ode_method == "LSODA":  # only reached with no coupling
        options["lband"] = options["uband"] = B - 1

    sol = solve_ivp(
        rhs, (0.0, dt), y0, method=ode_method, rtol=1e-6, atol=1e-9, **options
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    y = sol.y[:, -1]
    Y = y[: m * B].reshape(m, B)
    conc = np.maximum(Y[:, :NS], 0.0)
    export = np.maximum(Y[:, NS:], 0.0) * volume[:, None]
    if q:
        flat = pool.reshape(-1)
        flat[slot_keys] = np.maximum(y[m * B :], 0.0)
    return conc, export


def _jacobian_sparsity(m, NS, NE, q, ext_cell, ext_sp, ext_slot):
    """
    Sparsity pattern of the joint system's Jacobian: each cell's block
    depends only on itself and on the pool slots it reads, and each pool
    slot depends on the cells that read it.
    """
    B = NS + NE
    blocks = sparse.block_diag([np.ones((B, B))] * m, format="csr")
    if not q:
        return blocks
    cell_rows = (ext_cell[:, None] * B + np.arange(B)[None, :]).ravel()
    slot_cols = np.repeat(m * B + ext_slot, B)
    # Every species row of a coupled cell can depend on its pool slots, and
    # each slot depends on every species of the cells exchanging with it.
    rows = np.concatenate([cell_rows, slot_cols])
    cols = np.concatenate([slot_cols, cell_rows])
    coupling = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(m * B + q, m * B + q)
    )
    pool_diag = sparse.diags(np.r_[np.zeros(m * B), np.ones(q)], format="csr")
    full = sparse.bmat([[blocks, None], [None, sparse.csr_matrix((q, q))]])
    pattern = (full + coupling + pool_diag) != 0
    return pattern.astype(float)


def step_compartments(
    network, conc, volume, method, dt, ssa_method, tau_epsilon, seeds
):
    """
    Stochastically step independent compartments (e.g. environment voxels)
    all running `network`, in parallel: `conc` is (m, n_species) in
    `network.species` order and is updated in place. Returns False (and
    leaves `conc` alone) if `network` can't be compiled.
    """
    st = _kernels.structure_of(network)
    if st is None:
        return False
    method = str(method).upper()
    code = (
        _kernels.CLE
        if method == "CLE"
        else _kernels.STOCHASTIC_CODES[_resolve_ssa_method(ssa_method)]
    )
    if code == _kernels.SSA_TAU_LEAPING and not tau_epsilon > 0:
        raise ValueError(f"tau_epsilon must be positive, got {tau_epsilon}")
    table = table_for([st])
    m = conc.shape[0]
    NS, NR, NE = table.S.shape[1], table.S.shape[2], table.S_exp.shape[1]
    packed = np.zeros((m, NS))
    packed[:, : st.n_species] = conc
    params = np.broadcast_to(
        _kernels.network_params(network, NR), (m, NR, _kernels.N_PARAMS)
    )
    with single_threaded_blas():
        _kernels.step_stochastic(
            table,
            code,
            float(dt),
            float(tau_epsilon),
            np.zeros(m, dtype=np.int64),
            np.ascontiguousarray(params),
            packed,
            np.full(m, float(volume)),
            np.full((m, NS), -1, dtype=np.int64),
            np.full(m, -1, dtype=np.int64),
            np.zeros((1, 1)),
            1.0,
            np.asarray(seeds, dtype=np.uint64),
            np.arange(m + 1, dtype=np.int64),
            np.arange(m, dtype=np.int64),
            np.zeros((m, NE)),
        )
    conc[...] = packed[:, : st.n_species]
    return True
