"""Maze solving experiments using eikonal wave propagation."""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, '/Users/dereckewane/propage')
sys.setrecursionlimit(5000)

from src.solver import solve_eikonal, warmup as solver_warmup
from src.raytracer import trace_ray, warmup as ray_warmup
from src import media


def get_colormap():
    colors = ['#000428', '#004e92', '#007991', '#00bf8f',
              '#7dd87d', '#fcff9d', '#ffb347', '#ff6b6b']
    return LinearSegmentedColormap.from_list('wave', colors, N=256)


def visualize_maze_solution(n_field, T, ray, source, target, maze_size, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor='#050510')
    
    ax1 = axes[0]
    ax1.set_facecolor('#050510')
    ax1.imshow(n_field < 100, cmap='binary_r', origin='lower')
    ax1.plot(source[1], source[0], 'go', markersize=15, label='Source (End)')
    ax1.plot(target[1], target[0], 'ro', markersize=15, label='Target (Start)')
    ax1.set_title(f'Maze {maze_size}x{maze_size}', fontsize=16, color='white')
    ax1.legend(loc='upper right', facecolor='#1a1a2e', labelcolor='white')
    ax1.tick_params(colors='white')
    ax1.axis('off')
    
    ax2 = axes[1]
    ax2.set_facecolor('#050510')
    T_masked = np.ma.masked_where(T > 1e9, T)
    im2 = ax2.imshow(T_masked, cmap=get_colormap(), origin='lower')
    ax2.plot(source[1], source[0], 'w*', markersize=20)
    ax2.set_title('Arrival Time Map', fontsize=16, color='white')
    cbar = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar.set_label('Time', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')
    ax2.tick_params(colors='white')
    ax2.axis('off')
    
    ax3 = axes[2]
    ax3.set_facecolor('#050510')
    ax3.imshow(n_field < 100, cmap='binary_r', origin='lower', alpha=0.8)
    ax3.imshow(T_masked, cmap=get_colormap(), origin='lower', alpha=0.4)
    ax3.plot(ray[:, 1], ray[:, 0], 'w-', linewidth=3, alpha=0.3)
    ax3.plot(ray[:, 1], ray[:, 0], '#00ff88', linewidth=2, label='Optimal Path')
    ax3.plot(source[1], source[0], 'go', markersize=15)
    ax3.plot(target[1], target[0], 'ro', markersize=15)
    ax3.set_title('Solved Path', fontsize=16, color='white')
    ax3.legend(loc='upper right', facecolor='#1a1a2e', labelcolor='white')
    ax3.tick_params(colors='white')
    ax3.axis('off')
    
    plt.suptitle(f'Maze Solving via Eikonal Propagation ({maze_size}x{maze_size})', 
                 fontsize=18, color='white', y=0.98, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#050510')
    
    return fig


def run_maze_experiments():
    print("=" * 70)
    print("  MAZE SOLVING WITH EIKONAL WAVE PROPAGATION")
    print("  Finding optimal paths through labyrinths")
    print("=" * 70)
    
    print("\n[JIT Warmup...]")
    solver_warmup()
    ray_warmup()
    print("Done.\n")
    
    RES = 600
    maze_sizes = [11, 21, 31, 41, 51]
    
    for i, ms in enumerate(maze_sizes):
        print(f"\n[Maze {i+1}/{len(maze_sizes)}] Size: {ms}x{ms}")
        
        n_field, start_pos, end_pos = media.maze((RES, RES), maze_size=ms, seed=42+i)
        
        source = end_pos
        target = start_pos
        
        print(f"  Source (end): {source}")
        print(f"  Target (start): {target}")
        
        t0 = time.perf_counter()
        T = solve_eikonal(n_field, source)
        print(f"  Eikonal solved in: {time.perf_counter() - t0:.3f}s")
        
        step = RES / 2000
        t0 = time.perf_counter()
        ray = trace_ray(T, target, source, step=step)
        print(f"  Path traced in: {time.perf_counter() - t0:.3f}s")
        print(f"  Path length: {len(ray)} points")
        
        path_time = T[int(target[0]), int(target[1])]
        print(f"  Travel time: {path_time:.2f}")
        
        save_path = f"/Users/dereckewane/propage/results/maze_{ms}x{ms}.png"
        fig = visualize_maze_solution(n_field, T, ray, source, target, ms, save_path)
        print(f"  Saved: results/maze_{ms}x{ms}.png")
        plt.close(fig)
    
    print("\n" + "=" * 70)
    print("  ALL MAZES SOLVED!")
    print("=" * 70)


if __name__ == "__main__":
    run_maze_experiments()
