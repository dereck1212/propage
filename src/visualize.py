"""Visualization functions for eikonal wave propagation."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def get_colormap():
    colors = ['#000428', '#004e92', '#007991', '#00bf8f',
              '#7dd87d', '#fcff9d', '#ffb347', '#ff6b6b',
              '#c23b23', '#7d1424', '#3d0c11']
    return LinearSegmentedColormap.from_list('wave', colors, N=512)


def plot_medium(ax, n_field, source=None, title=""):
    ax.set_facecolor('#050510')
    im = ax.imshow(n_field, cmap='viridis', origin='lower', interpolation='bilinear')
    if source:
        ax.plot(source[1], source[0], 'w*', markersize=20, markeredgecolor='red', markeredgewidth=2)
    ax.set_title(title, fontsize=14, color='white', pad=10)
    ax.tick_params(colors='white')
    return im


def plot_arrival_time(ax, T, source=None, n_contours=20, title=""):
    ax.set_facecolor('#050510')
    T_plot = np.ma.masked_invalid(T)
    T_plot = np.ma.masked_where(T_plot > 1e10, T_plot)
    im = ax.imshow(T_plot, cmap=get_colormap(), origin='lower', interpolation='bilinear')
    ny, nx = T.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    T_valid = T[np.isfinite(T) & (T < 1e10)]
    if len(T_valid) > 0:
        levels = np.linspace(T_valid.min(), T_valid.max() * 0.92, n_contours)
        ax.contour(X, Y, T_plot, levels=levels, colors='white', linewidths=0.4, alpha=0.6)
    if source:
        ax.plot(source[1], source[0], 'w*', markersize=20, markeredgecolor='red', markeredgewidth=2)
    ax.set_title(title, fontsize=14, color='white', pad=10)
    ax.tick_params(colors='white')
    return im


def plot_rays(ax, T, rays, source, targets, n_field=None, title=""):
    ax.set_facecolor('#050510')
    T_plot = np.ma.masked_invalid(T)
    T_plot = np.ma.masked_where(T_plot > 1e10, T_plot)
    if n_field is not None:
        ax.imshow(n_field, cmap='viridis', origin='lower', alpha=0.5)
    else:
        ax.imshow(T_plot, cmap=get_colormap(), origin='lower', alpha=0.7)
    colors = ['#ff6b6b', '#feca57', '#48dbfb', '#ff9ff3', '#54a0ff',
              '#00d2d3', '#ff9f43', '#10ac84', '#ee5a24', '#9b59b6']
    for i, ray in enumerate(rays):
        c = colors[i % len(colors)]
        ax.plot(ray[:, 1], ray[:, 0], color='white', linewidth=4, alpha=0.3)
        ax.plot(ray[:, 1], ray[:, 0], color=c, linewidth=2)
    ax.plot(source[1], source[0], 'w*', markersize=25, markeredgecolor='#ff6b6b', markeredgewidth=2, zorder=10)
    for i, t in enumerate(targets):
        ax.plot(t[1], t[0], 'o', markersize=10, color=colors[i % len(colors)], 
                markeredgecolor='white', markeredgewidth=2, zorder=10)
    ax.set_title(title, fontsize=14, color='white', pad=10)
    ax.tick_params(colors='white')


def create_experiment_figure(n_field, T, rays, source, targets, name, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor='#050510')
    plot_medium(axes[0], n_field, source, "Refractive Index n(x,y)")
    plot_arrival_time(axes[1], T, source, 25, "Arrival Time T(x,y)")
    plot_rays(axes[2], T, rays, source, targets, n_field, "Ray Tracing")
    for ax in axes:
        cbar = plt.colorbar(ax.images[0] if ax.images else None, ax=ax, shrink=0.7, pad=0.02)
        if cbar: 
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(cbar.ax.get_yticklabels(), color='white')
    plt.suptitle(name, fontsize=18, color='white', y=0.98, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#050510')
    return fig
