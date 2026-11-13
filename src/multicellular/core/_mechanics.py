# core/_mechanics.py
"""
Contact mechanics: the geometry helpers shared by both Colony backends, and
the compiled (numba) kernel used by the "numba" backend.

The geometry helpers are plain-float Python functions. The "python" backend
calls them directly; the kernel calls numba-compiled copies of the very same
functions, so the two backends can't drift apart.

`contact_step` computes every mobile cell's cell-cell, cell-anchored and
cell-wall contact force and torque, and applies the overdamped update, in
one parallel pass over cells. Each mobile cell gathers its own forces from
its neighbors (found through a sorted cell list), so every pair is
evaluated twice, once from each side, but no two threads ever write to the
same cell. Each pair is always evaluated with its lower-index cell as the
first argument, exactly as the Python path's i < j loop does, so the two
backends agree to floating-point summation order.
"""

import math

import numpy as np
from numba import njit, prange

_PARALLEL_TOL = 1e-10  # denom threshold for segment-segment parallel detection


def _segment_segment_closest(a1x, a1y, b1x, b1y, a2x, a2y, b2x, b2y):
    """
    Closest points between segment [a1, b1] and segment [a2, b2], given as
    raw x/y floats rather than vectors.

    Returns (p1x, p1y, p2x, p2y, d): p1 on seg1, p2 on seg2, d = |p1 - p2|.
    Operates on plain floats (instead of building/dot-producting small numpy
    arrays) since this runs once per candidate contact pair, every step.

    When segments are nearly parallel, uses the midpoint of their overlapping
    axial projection rather than a degenerate endpoint, per the physics spec.
    """
    d1x, d1y = b1x - a1x, b1y - a1y
    d2x, d2y = b2x - a2x, b2y - a2y
    wx, wy = a1x - a2x, a1y - a2y

    a = d1x * d1x + d1y * d1y
    b = d1x * d2x + d1y * d2y
    c = d2x * d2x + d2y * d2y
    e = d1x * wx + d1y * wy
    f = d2x * wx + d2y * wy

    denom = a * c - b * b  # zero iff segments are parallel

    if denom < _PARALLEL_TOL:
        # Parallel or degenerate: project seg2 endpoints onto seg1 and use the
        # midpoint of the overlapping range, or the nearest endpoint pair if
        # there is no axial overlap.
        if a < _PARALLEL_TOL:
            s = 0.0
        else:
            t0 = ((a2x - a1x) * d1x + (a2y - a1y) * d1y) / a
            t1 = ((b2x - a1x) * d1x + (b2y - a1y) * d1y) / a
            lo = max(0.0, min(t0, t1))
            hi = min(1.0, max(t0, t1))
            s = (lo + hi) / 2.0 if lo <= hi else min(max(-e / a, 0.0), 1.0)

        p1x, p1y = a1x + s * d1x, a1y + s * d1y
        if c > _PARALLEL_TOL:
            t = min(max(((p1x - a2x) * d2x + (p1y - a2y) * d2y) / c, 0.0), 1.0)
        else:
            t = 0.0
        p2x, p2y = a2x + t * d2x, a2y + t * d2y
    else:
        # General (non-parallel) case: closed-form minimum with clamped parameters.
        # Clamp s, recompute t, clamp t, then recompute s once more.
        s = min(max((b * f - c * e) / denom, 0.0), 1.0)
        t = min(max((b * s + f) / c, 0.0), 1.0) if c > _PARALLEL_TOL else 0.0
        s = min(max((-e + b * t) / a, 0.0), 1.0) if a > _PARALLEL_TOL else 0.0
        p1x, p1y = a1x + s * d1x, a1y + s * d1y
        p2x, p2y = a2x + t * d2x, a2y + t * d2y

    d = math.hypot(p1x - p2x, p1y - p2y)
    return p1x, p1y, p2x, p2y, d


def _segment_point_closest(a1x, a1y, b1x, b1y, px, py):
    """
    Closest point on segment [a1, b1] to point (px, py).

    Returns (qx, qy, d): q the closest axis point, d = |q - (px, py)|.
    """
    dx, dy = b1x - a1x, b1y - a1y
    len_sq = dx * dx + dy * dy
    if len_sq < _PARALLEL_TOL:
        t = 0.0
    else:
        t = min(max(((px - a1x) * dx + (py - a1y) * dy) / len_sq, 0.0), 1.0)
    qx, qy = a1x + t * dx, a1y + t * dy
    return qx, qy, math.hypot(qx - px, qy - py)


def _contact_force(ep_i, ep_j, r_i, r_j, ox_i, oy_i, k):
    """
    Hookean contact between cells i and j, given their axis endpoints
    `(a_x, a_y, b_x, b_y, c_x, c_y)`, radii, and i's orientation (the
    normal of last resort). Returns (hit, Fx, Fy, pcx, pcy): the force on i
    (j feels its negative) and the contact point; hit is False if they
    don't overlap.
    """
    a1x, a1y, b1x, b1y, cix, ciy = ep_i
    a2x, a2y, b2x, b2y, cjx, cjy = ep_j
    p1x, p1y, p2x, p2y, d = _segment_segment_closest(
        a1x, a1y, b1x, b1y, a2x, a2y, b2x, b2y
    )
    delta = r_i + r_j - d
    if delta <= 0.0:
        return False, 0.0, 0.0, 0.0, 0.0

    # Contact normal: unit vector from j toward i. When axes coincide
    # (d≈0), fall back to center-to-center direction, then to i's
    # perpendicular if centers also coincide.
    if d > 1e-12:
        Nx, Ny = (p1x - p2x) / d, (p1y - p2y) / d
    else:
        sepx, sepy = cix - cjx, ciy - cjy
        sep_norm = math.hypot(sepx, sepy)
        if sep_norm > 1e-12:
            Nx, Ny = sepx / sep_norm, sepy / sep_norm
        else:
            Nx, Ny = -oy_i, ox_i

    return True, k * delta * Nx, k * delta * Ny, (p1x + p2x) / 2.0, (p1y + p2y) / 2.0


# Compiled copies of the same functions, for use inside the kernels below.
_segment_segment_closest_nb = njit(cache=True)(_segment_segment_closest)
_segment_point_closest_nb = njit(cache=True)(_segment_point_closest)


@njit(cache=True)
def _contact_force_nb(ep_i, ep_j, r_i, r_j, ox_i, oy_i, k):
    a1x, a1y, b1x, b1y, cix, ciy = ep_i
    a2x, a2y, b2x, b2y, cjx, cjy = ep_j
    p1x, p1y, p2x, p2y, d = _segment_segment_closest_nb(
        a1x, a1y, b1x, b1y, a2x, a2y, b2x, b2y
    )
    delta = r_i + r_j - d
    if delta <= 0.0:
        return False, 0.0, 0.0, 0.0, 0.0
    if d > 1e-12:
        Nx, Ny = (p1x - p2x) / d, (p1y - p2y) / d
    else:
        sepx, sepy = cix - cjx, ciy - cjy
        sep_norm = math.hypot(sepx, sepy)
        if sep_norm > 1e-12:
            Nx, Ny = sepx / sep_norm, sepy / sep_norm
        else:
            Nx, Ny = -oy_i, ox_i
    return True, k * delta * Nx, k * delta * Ny, (p1x + p2x) / 2.0, (p1y + p2y) / 2.0


@njit(cache=True)
def _endpoints(px, py, ox, oy, length, i):
    hl = length[i] / 2.0
    return (
        px[i] - hl * ox[i],
        py[i] - hl * oy[i],
        px[i] + hl * ox[i],
        py[i] + hl * oy[i],
        px[i],
        py[i],
    )


@njit(cache=True)
def _wall_force(ep, R, is_sphere, ox, oy, k_wall, faces, corners):
    """
    One cell's total wall force and torque (port of
    `Colony._apply_wall_forces`, see its docstring for the model).
    """
    a1x, a1y, b1x, b1y, cix, ciy = ep
    Fx_tot = 0.0
    Fy_tot = 0.0
    T_tot = 0.0
    n_points = 1 if is_sphere else 2
    for p in range(n_points):
        if is_sphere:
            ex, ey = cix, ciy
        elif p == 0:
            ex, ey = a1x, a1y
        else:
            ex, ey = b1x, b1y
        found = False
        best_h = 0.0
        best_nx = 0.0
        best_ny = 0.0
        for f in range(faces.shape[0]):
            x0, y0, x1, y1, nx, ny = (
                faces[f, 0],
                faces[f, 1],
                faces[f, 2],
                faces[f, 3],
                faces[f, 4],
                faces[f, 5],
            )
            tx, ty = x1 - x0, y1 - y0
            L = math.hypot(tx, ty)
            if L < _PARALLEL_TOL:
                continue
            tx, ty = tx / L, ty / L
            s = (ex - x0) * tx + (ey - y0) * ty
            if s < 0.0 or s > L:
                continue
            h = (ex - x0) * nx + (ey - y0) * ny
            if not found or abs(h) < abs(best_h):
                found = True
                best_h, best_nx, best_ny = h, nx, ny
        if not found:
            continue
        delta = R - best_h
        if delta <= 0.0:
            continue
        Fx, Fy = k_wall * delta * best_nx, k_wall * delta * best_ny
        Fx_tot += Fx
        Fy_tot += Fy
        T_tot += (ex - cix) * Fy - (ey - ciy) * Fx

    for c in range(corners.shape[0]):
        cx, cy = corners[c, 0], corners[c, 1]
        qx, qy, d = _segment_point_closest_nb(a1x, a1y, b1x, b1y, cx, cy)
        delta = R - d
        if delta <= 0.0:
            continue
        if d > 1e-12:
            Nx, Ny = (qx - cx) / d, (qy - cy) / d
        else:
            Nx, Ny = -oy, ox
        Fx, Fy = k_wall * delta * Nx, k_wall * delta * Ny
        Fx_tot += Fx
        Fy_tot += Fy
        T_tot += (qx - cix) * Fy - (qy - ciy) * Fx
    return Fx_tot, Fy_tot, T_tot


@njit(parallel=True, cache=True)
def contact_step(
    px,
    py,
    ox,
    oy,
    length,
    radius,
    is_sphere,
    apx,
    apy,
    aox,
    aoy,
    alength,
    aradius,
    k,
    k_wall,
    drag,
    dt,
    faces,
    corners,
):
    """
    Contact forces and the overdamped update for every mobile cell, in place.

    `px..is_sphere` describe the mobile cells (positions and orientations
    are updated in place); `apx..aradius` the anchored ones (never moved).
    See `Colony._apply_contact_forces` for the model.
    """
    n = px.shape[0]
    na = apx.shape[0]
    if n == 0:
        return

    # Cell list: bins as wide as the largest possible contact distance, so
    # every overlapping pair lies in the same or an adjacent bin.
    max_half = 0.0
    for i in range(n):
        max_half = max(max_half, length[i] / 2.0 + radius[i])
    bin_size = max(2.0 * max_half, 1e-6)
    gx = np.empty(n, dtype=np.int64)
    gy = np.empty(n, dtype=np.int64)
    for i in range(n):
        gx[i] = np.int64(math.floor(px[i] / bin_size))
        gy[i] = np.int64(math.floor(py[i] / bin_size))
    gx0, gx1 = gx.min(), gx.max()
    gy0, gy1 = gy.min(), gy.max()
    ny_bins = gy1 - gy0 + 1
    keys = (gx - gx0) * ny_bins + (gy - gy0)
    order = np.argsort(keys, kind="mergesort")
    sorted_keys = keys[order]

    fx = np.zeros(n)
    fy = np.zeros(n)
    tq = np.zeros(n)
    for i in prange(n):
        ep_i = _endpoints(px, py, ox, oy, length, i)
        Fx_i = 0.0
        Fy_i = 0.0
        T_i = 0.0
        cix, ciy = ep_i[4], ep_i[5]
        for dx in range(-1, 2):
            bx = gx[i] + dx
            if bx < gx0 or bx > gx1:
                continue
            for dy in range(-1, 2):
                by = gy[i] + dy
                if by < gy0 or by > gy1:
                    continue
                key = (bx - gx0) * ny_bins + (by - gy0)
                lo = np.searchsorted(sorted_keys, key, side="left")
                hi = np.searchsorted(sorted_keys, key, side="right")
                for p in range(lo, hi):
                    j = order[p]
                    if j == i:
                        continue
                    ep_j = _endpoints(px, py, ox, oy, length, j)
                    if i < j:
                        hit, Fx, Fy, pcx, pcy = _contact_force_nb(
                            ep_i, ep_j, radius[i], radius[j], ox[i], oy[i], k
                        )
                        if hit:
                            Fx_i += Fx
                            Fy_i += Fy
                            T_i += (pcx - cix) * Fy - (pcy - ciy) * Fx
                    else:
                        hit, Fx, Fy, pcx, pcy = _contact_force_nb(
                            ep_j, ep_i, radius[j], radius[i], ox[j], oy[j], k
                        )
                        if hit:
                            Fx_i -= Fx
                            Fy_i -= Fy
                            T_i -= (pcx - cix) * Fy - (pcy - ciy) * Fx

        # Anchored cells push this one (with the anchored cell as the first
        # argument, as on the Python path) and never move themselves.
        half_i = length[i] / 2.0 + radius[i]
        for a in range(na):
            reach = alength[a] / 2.0 + aradius[a] + half_i
            if abs(apx[a] - cix) > reach or abs(apy[a] - ciy) > reach:
                continue
            ep_a = _endpoints(apx, apy, aox, aoy, alength, a)
            hit, Fx, Fy, pcx, pcy = _contact_force_nb(
                ep_a, ep_i, aradius[a], radius[i], aox[a], aoy[a], k
            )
            if hit:
                Fx_i -= Fx
                Fy_i -= Fy
                T_i -= (pcx - cix) * Fy - (pcy - ciy) * Fx

        if faces.shape[0] or corners.shape[0]:
            wFx, wFy, wT = _wall_force(
                ep_i, radius[i], is_sphere[i], ox[i], oy[i], k_wall, faces, corners
            )
            Fx_i += wFx
            Fy_i += wFy
            T_i += wT

        fx[i] = Fx_i
        fy[i] = Fy_i
        tq[i] = T_i

    # Overdamped update, after every force is known (positions are read
    # above, so they must not move until all forces are summed).
    for i in prange(n):
        if is_sphere[i]:
            zeta_t = 3.0 * drag * radius[i]
            px[i] += fx[i] / zeta_t * dt
            py[i] += fy[i] / zeta_t * dt
            continue
        zeta_t = drag * length[i]
        zeta_r = (drag / 12.0) * length[i] ** 3
        px[i] += fx[i] / zeta_t * dt
        py[i] += fy[i] / zeta_t * dt
        angle = tq[i] / zeta_r * dt
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        x, y = ox[i], oy[i]
        nx = cos_a * x - sin_a * y
        ny = sin_a * x + cos_a * y
        norm = math.hypot(nx, ny)
        if norm != 0.0:
            nx, ny = nx / norm, ny / norm
        else:
            nx, ny = 1.0, 0.0
        ox[i] = nx
        oy[i] = ny
