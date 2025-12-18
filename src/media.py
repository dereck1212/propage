"""Refractive index field generators for various media."""

import numpy as np
from scipy.ndimage import gaussian_filter


def homogeneous(shape, n=1.5):
    return np.full(shape, n, dtype=np.float64)


def vertical_gradient(shape, n_min=1.0, n_max=2.0):
    ny, nx = shape
    t = np.linspace(0, 1, ny)[:, np.newaxis]
    return (n_min + (n_max - n_min) * np.broadcast_to(t, shape)).astype(np.float64)


def planar_diopter(shape, interface_frac=0.5, n1=1.0, n2=1.5):
    ny, nx = shape
    n = np.full(shape, n1, dtype=np.float64)
    n[:, int(nx * interface_frac):] = n2
    return n


def double_diopter(shape, x1_frac=0.33, x2_frac=0.67, n1=1.0, n2=1.5, n3=1.0):
    ny, nx = shape
    n = np.full(shape, n1, dtype=np.float64)
    n[:, int(nx*x1_frac):int(nx*x2_frac)] = n2
    n[:, int(nx*x2_frac):] = n3
    return n


def circular_lens(shape, center_frac=(0.5, 0.5), radius_frac=0.2, n_lens=1.8, n_bg=1.0):
    ny, nx = shape
    cy, cx = int(ny*center_frac[0]), int(nx*center_frac[1])
    r = int(min(ny, nx) * radius_frac)
    y, x = np.ogrid[0:ny, 0:nx]
    dist = np.sqrt((y-cy)**2 + (x-cx)**2)
    n = np.where(dist <= r, n_lens, n_bg)
    return n.astype(np.float64)


def multi_lens(shape, n_lenses=3, n_lens=2.0, n_bg=1.0, radius=0.1):
    ny, nx = shape
    y, x = np.ogrid[0:ny, 0:nx]
    y_n, x_n = y / ny, x / nx
    n = np.full(shape, n_bg, dtype=np.float64)
    centers = [(0.3, 0.5), (0.6, 0.3), (0.6, 0.7)]
    for cy, cx in centers[:n_lenses]:
        dist = np.sqrt((x_n - cx)**2 + (y_n - cy)**2)
        n[dist < radius] = n_lens
    return n


def turbulent(shape, n_min=1.0, n_max=2.5, n_scales=5, seed=42):
    np.random.seed(seed)
    ny, nx = shape
    n = np.zeros(shape)
    for scale in range(n_scales):
        sigma = 2 ** (n_scales - scale - 1) * 5
        amp = 1.0 / (scale + 1)
        n += amp * gaussian_filter(np.random.randn(ny, nx), sigma=sigma)
    n = (n - n.min()) / (n.max() - n.min())
    return (n_min + (n_max - n_min) * n).astype(np.float64)


def layered(shape, n_layers=5, n_values=None):
    ny, nx = shape
    if n_values is None:
        n_values = [1.0, 1.8, 1.2, 2.0, 1.5]
    n = np.zeros(shape, dtype=np.float64)
    h = ny // n_layers
    for i in range(n_layers):
        n[i*h:(i+1)*h if i < n_layers-1 else ny, :] = n_values[i % len(n_values)]
    return n


def complex_structure(shape, seed=123):
    np.random.seed(seed)
    ny, nx = shape
    n = np.full(shape, 1.0, dtype=np.float64)
    n[:, nx//2:] = 1.5
    cy, cx, r = ny//3, nx//4, min(ny, nx)//6
    y, x = np.ogrid[0:ny, 0:nx]
    n[np.sqrt((y-cy)**2 + (x-cx)**2) < r] = 2.2
    n[ny//2-ny//20:ny//2+ny//20, :] = 1.8
    return n


def maze(shape, maze_size=15, n_wall=10000.0, n_path=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    mh, mw = maze_size, maze_size
    grid = np.ones((mh, mw), dtype=np.int8)
    
    def carve(cy, cx):
        grid[cy, cx] = 0
        dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        np.random.shuffle(dirs)
        for dy, dx in dirs:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < mh and 0 <= nx < mw and grid[ny, nx] == 1:
                grid[cy + dy//2, cx + dx//2] = 0
                carve(ny, nx)
    
    start_y, start_x = 1, 1
    carve(start_y, start_x)
    
    end_y, end_x = mh - 2, mw - 2
    if grid[end_y, end_x] == 1:
        grid[end_y, end_x] = 0
        for dy, dx in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            ny, nx = end_y + dy, end_x + dx
            if 0 <= ny < mh and 0 <= nx < mw and grid[ny, nx] == 0:
                break
        else:
            grid[end_y - 1, end_x] = 0
    
    ny, nx = shape
    n = np.full(shape, n_wall, dtype=np.float64)
    
    cell_h, cell_w = ny // mh, nx // mw
    for my in range(mh):
        for mx in range(mw):
            if grid[my, mx] == 0:
                y0, y1 = my * cell_h, (my + 1) * cell_h
                x0, x1 = mx * cell_w, (mx + 1) * cell_w
                n[y0:y1, x0:x1] = n_path
    
    start_pos = (start_y * cell_h + cell_h // 2, start_x * cell_w + cell_w // 2)
    end_pos = (end_y * cell_h + cell_h // 2, end_x * cell_w + cell_w // 2)
    
    return n, start_pos, end_pos
