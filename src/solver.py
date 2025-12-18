"""Eikonal equation solver using Fast Marching Method."""

import numpy as np
from numba import njit


@njit
def _solve_quadratic(T, slowness, y, x, dx, ny, nx):
    slow = slowness[y, x]
    t_horiz, t_vert = np.inf, np.inf
    if x > 0 and T[y, x-1] < t_horiz: t_horiz = T[y, x-1]
    if x < nx-1 and T[y, x+1] < t_horiz: t_horiz = T[y, x+1]
    if y > 0 and T[y-1, x] < t_vert: t_vert = T[y-1, x]
    if y < ny-1 and T[y+1, x] < t_vert: t_vert = T[y+1, x]
    t1, t2 = (t_horiz, t_vert) if t_horiz < t_vert else (t_vert, t_horiz)
    if t1 == np.inf: return np.inf
    if t2 < np.inf:
        a, b, c = 2.0, -2*(t1+t2), t1**2 + t2**2 - (slow*dx)**2
        disc = b*b - 4*a*c
        if disc >= 0:
            t_new = (-b + np.sqrt(disc)) / (2*a)
            if t_new >= t2: return t_new
    return t1 + slow*dx


@njit
def _fmm_core(slowness, sy, sx, dx):
    ny, nx = slowness.shape
    T = np.full((ny, nx), np.inf)
    frozen = np.zeros((ny, nx), dtype=np.bool_)
    T[sy, sx] = 0.0
    heap_t = np.full(ny*nx, np.inf)
    heap_y = np.zeros(ny*nx, dtype=np.int32)
    heap_x = np.zeros(ny*nx, dtype=np.int32)
    heap_t[0], heap_y[0], heap_x[0] = 0.0, sy, sx
    hs = 1
    dy_arr = np.array([-1, 1, 0, 0], dtype=np.int32)
    dx_arr = np.array([0, 0, -1, 1], dtype=np.int32)
    while hs > 0:
        mi, mt = 0, heap_t[0]
        for i in range(1, hs):
            if heap_t[i] < mt: mt, mi = heap_t[i], i
        y, x = heap_y[mi], heap_x[mi]
        hs -= 1
        if mi < hs:
            heap_t[mi], heap_y[mi], heap_x[mi] = heap_t[hs], heap_y[hs], heap_x[hs]
        if frozen[y, x]: continue
        frozen[y, x] = True
        for d in range(4):
            ny2, nx2 = y + dy_arr[d], x + dx_arr[d]
            if 0 <= ny2 < ny and 0 <= nx2 < nx and not frozen[ny2, nx2]:
                tn = _solve_quadratic(T, slowness, ny2, nx2, dx, ny, nx)
                if tn < T[ny2, nx2]:
                    T[ny2, nx2] = tn
                    if hs < ny*nx:
                        heap_t[hs], heap_y[hs], heap_x[hs] = tn, ny2, nx2
                        hs += 1
    return T


def solve_eikonal(n_field, source, dx=1.0):
    slowness = n_field.astype(np.float64)
    return _fmm_core(slowness, int(source[0]), int(source[1]), dx)


def warmup():
    w = np.ones((50, 50), dtype=np.float64)
    _ = solve_eikonal(w, (25, 5))
