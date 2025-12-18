"""Main experiment runner for eikonal wave propagation simulations."""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/dereckewane/propage')
from src.solver import solve_eikonal, warmup as solver_warmup
from src.raytracer import trace_rays, warmup as ray_warmup
from src import media
from src.visualize import create_experiment_figure


def run_experiments():
    print("=" * 70)
    print("  EIKONAL WAVE PROPAGATION EXPERIMENTS")
    print("  Classical equation: |nabla T| = n(x,y)")
    print("=" * 70)

    print("\n[JIT Warmup...]")
    solver_warmup()
    ray_warmup()
    print("Done.\n")

    RES = 800
    shape = (RES, RES)
    source = (RES // 2, int(RES * 0.05))
    n_targets = 9
    dy_vals = np.linspace(-RES * 0.35, RES * 0.35, n_targets)
    targets = [(int(RES // 2 + dy), int(RES * 0.92)) for dy in dy_vals]

    experiments = [
        ("1_Homogeneous", media.homogeneous(shape, n=1.5)),
        ("2_Vertical_Gradient", media.vertical_gradient(shape, n_min=1.0, n_max=2.5)),
        ("3_Planar_Diopter", media.planar_diopter(shape, interface_frac=0.5, n1=1.0, n2=1.7)),
        ("4_Double_Diopter", media.double_diopter(shape, n1=1.0, n2=2.0, n3=1.0)),
        ("5_Circular_Lens", media.circular_lens(shape, radius_frac=0.25, n_lens=2.2, n_bg=1.0)),
        ("6_Multi_Lens", media.multi_lens(shape, n_lenses=3, n_lens=2.5, n_bg=1.0, radius=0.12)),
        ("7_Turbulent", media.turbulent(shape, n_min=1.0, n_max=2.8, n_scales=5)),
        ("8_Complex_Structure", media.complex_structure(shape)),
    ]

    for name, n_field in experiments:
        print(f"\n[{name}]")
        
        t0 = time.perf_counter()
        T = solve_eikonal(n_field, source)
        eik_time = time.perf_counter() - t0
        print(f"  Eikonal solved in: {eik_time:.3f}s")

        step = RES / 2500
        t0 = time.perf_counter()
        rays = trace_rays(T, targets, source, step=step)
        ray_time = time.perf_counter() - t0
        print(f"  {n_targets} rays traced in: {ray_time:.3f}s")

        save_path = f"/Users/dereckewane/propage/results/{name}.png"
        fig = create_experiment_figure(n_field, T, rays, source, targets, name.replace("_", " "), save_path)
        print(f"  Saved: results/{name}.png")
        plt.close(fig)

    print("\n" + "=" * 70)
    print("  ALL EXPERIMENTS COMPLETED!")
    print(f"  Output directory: results/")
    print("=" * 70)


if __name__ == "__main__":
    run_experiments()
