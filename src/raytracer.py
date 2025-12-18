"""Ray tracing by gradient descent on arrival time field."""

import numpy as np
from numba import njit


@njit
def _gradient(T, y, x):
    ny, nx = T.shape
    y = max(0.5, min(ny-1.5, y))
    x = max(0.5, min(nx-1.5, x))
    iy, ix = int(y), int(x)
    fy, fx = y - iy, x - ix
    iy0, ix0 = max(0, min(ny-2, iy)), max(0, min(nx-2, ix))
    t00, t01 = T[iy0, ix0], T[iy0, ix0+1]
    t10, t11 = T[iy0+1, ix0], T[iy0+1, ix0+1]
    gx = (1-fy)*(t01-t00) + fy*(t11-t10)
    gy = (1-fx)*(t10-t00) + fx*(t11-t01)
    return gy, gx


@njit
def _trace_core(T, ty, tx, sy, sx, step, maxs):
    ny, nx = T.shape
    py, px = np.zeros(maxs), np.zeros(maxs)
    y, x = float(ty), float(tx)
    py[0], px[0] = y, x
    n = 1
    for _ in range(1, maxs):
        gy, gx = _gradient(T, y, x)
        gn = np.sqrt(gy**2 + gx**2)
        if gn < 1e-10: break
        y -= step * gy / gn
        x -= step * gx / gn
        if y < 0 or y >= ny or x < 0 or x >= nx: break
        py[n], px[n] = y, x
        n += 1
        if np.sqrt((y-sy)**2 + (x-sx)**2) < step*2: break
    return py[:n], px[:n]


def trace_ray(T, target, source, step=0.5):
    py, px = _trace_core(T, float(target[0]), float(target[1]),
                         float(source[0]), float(source[1]), step, 30000)
    return np.column_stack([py, px])


def trace_rays(T, targets, source, step=0.5):
    return [trace_ray(T, t, source, step) for t in targets]


def warmup():
    T = np.ones((50, 50), dtype=np.float64)
    _ = trace_ray(T, (40, 40), (25, 5))
