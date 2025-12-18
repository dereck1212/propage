"""Snell-Descartes law convergence study at multiple resolutions."""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/dereckewane/propage')
from src.solver import solve_eikonal, warmup as solver_warmup
from src.raytracer import trace_ray, warmup as ray_warmup
from src import media


def measure_angles(ray, interface_x, margin=80):
    xc, yc = ray[:, 1], ray[:, 0]
    crossing = None
    for i in range(len(ray) - 1):
        if (xc[i] > interface_x >= xc[i+1]) or (xc[i] < interface_x <= xc[i+1]):
            crossing = i
            break
    if crossing is None:
        return None, None
    pts_m2 = ray[max(0, crossing-margin):crossing]
    pts_m1 = ray[crossing+1:min(len(ray), crossing+1+margin)]
    if len(pts_m2) < 15 or len(pts_m1) < 15:
        return None, None
    try:
        s2 = np.polyfit(pts_m2[:, 1], pts_m2[:, 0], 1)[0]
        s1 = np.polyfit(pts_m1[:, 1], pts_m1[:, 0], 1)[0]
        return np.arctan(abs(s1)), np.arctan(abs(s2))
    except:
        return None, None


def run_convergence_study():
    print("=" * 75)
    print("  SNELL-DESCARTES CONVERGENCE STUDY")
    print("  Classical Eikonal: |nabla T| = n(x,y)")
    print("=" * 75)
    
    print("\n[JIT Warmup...]")
    solver_warmup()
    ray_warmup()
    print("Done.\n")
    
    n1, n2 = 1.0, 1.5
    resolutions = [500, 750, 1000, 1500, 2000, 2500, 3000]
    n_rays = 40
    
    results = []
    
    for res in resolutions:
        print(f"\n[Resolution {res}x{res}]")
        
        shape = (res, res)
        interface_x = res // 2
        source = (res // 2, int(res * 0.03))
        
        dy_vals = np.linspace(-res * 0.45, res * 0.45, n_rays)
        targets = [(int(res // 2 + dy), int(res * 0.97)) for dy in dy_vals]
        
        n_field = media.planar_diopter(shape, interface_frac=0.5, n1=n1, n2=n2)
        
        t0 = time.perf_counter()
        T = solve_eikonal(n_field, source)
        eik_time = time.perf_counter() - t0
        print(f"  Eikonal: {eik_time:.2f}s")
        
        step = res / 5000
        margin = max(60, res // 12)
        
        errors = []
        angles_data = []
        
        for target in targets:
            ray = trace_ray(T, target, source, step=step)
            t1, t2 = measure_angles(ray, interface_x, margin)
            if t1 is not None and t2 is not None and t2 > 0.01:
                ratio = (n1 * np.sin(t1)) / (n2 * np.sin(t2))
                err = abs(ratio - 1.0) * 100
                errors.append(err)
                angles_data.append((np.degrees(t1), np.degrees(t2), ratio, err))
        
        if errors:
            mean_err = np.mean(errors)
            std_err = np.std(errors)
            max_err = np.max(errors)
            min_err = np.min(errors)
            
            print(f"  Valid rays: {len(errors)}/{n_rays}")
            print(f"  Error: {mean_err:.4f}% +/- {std_err:.4f}%")
            print(f"  Range: [{min_err:.4f}%, {max_err:.4f}%]")
            
            results.append({
                'res': res,
                'mean': mean_err,
                'std': std_err,
                'min': min_err,
                'max': max_err,
                'n_valid': len(errors),
                'angles': angles_data
            })
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='#050510')
    
    ax1 = axes[0, 0]
    ax1.set_facecolor('#050510')
    ress = [r['res'] for r in results]
    means = [r['mean'] for r in results]
    stds = [r['std'] for r in results]
    ax1.errorbar(ress, means, yerr=stds, fmt='o-', color='#48dbfb', 
                 linewidth=2, markersize=10, capsize=5, capthick=2)
    ax1.axhline(y=1.0, color='#feca57', linestyle='--', linewidth=1.5, label='1% threshold')
    ax1.axhline(y=0.5, color='#00ff88', linestyle=':', linewidth=1.5, label='0.5% threshold')
    ax1.set_xlabel('Resolution (pixels)', fontsize=12, color='white')
    ax1.set_ylabel('Snell Error (%)', fontsize=12, color='white')
    ax1.set_title('Convergence with Resolution', fontsize=14, color='white')
    ax1.tick_params(colors='white')
    ax1.legend(facecolor='#1a1a2e', labelcolor='white')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.set_facecolor('#050510')
    log_h = np.log(np.array(ress))
    log_e = np.log(np.array(means))
    slope, intercept = np.polyfit(log_h, log_e, 1)
    ax2.loglog(ress, means, 'o-', color='#48dbfb', linewidth=2, markersize=10)
    fit_line = np.exp(intercept) * np.array(ress) ** slope
    ax2.loglog(ress, fit_line, '--', color='#ff6b6b', linewidth=2, 
               label=f'Fit: O(h^{{{-slope:.2f}}})')
    ax2.set_xlabel('Resolution (log)', fontsize=12, color='white')
    ax2.set_ylabel('Error % (log)', fontsize=12, color='white')
    ax2.set_title(f'Convergence Order = {-slope:.2f}', fontsize=14, color='white')
    ax2.tick_params(colors='white')
    ax2.legend(facecolor='#1a1a2e', labelcolor='white')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.set_facecolor('#050510')
    high_res = results[-1]
    angles_1 = [a[0] for a in high_res['angles']]
    ratios = [a[2] for a in high_res['angles']]
    ax3.scatter(angles_1, ratios, c='#48dbfb', s=50, alpha=0.8)
    ax3.axhline(y=1.0, color='#00ff88', linewidth=2, label='Perfect (ratio=1)')
    ax3.set_xlabel('Incident Angle theta_1 (degrees)', fontsize=12, color='white')
    ax3.set_ylabel('n1*sin(theta1) / n2*sin(theta2)', fontsize=12, color='white')
    ax3.set_title(f'Snell Ratio at {high_res["res"]}x{high_res["res"]}', fontsize=14, color='white')
    ax3.tick_params(colors='white')
    ax3.legend(facecolor='#1a1a2e', labelcolor='white')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.9, 1.1)
    
    ax4 = axes[1, 1]
    ax4.set_facecolor('#050510')
    ax4.axis('off')
    ax4.text(0.5, 0.9, 'SUMMARY', fontsize=20, color='white', ha='center',
             fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.5, 0.78, f'n1 = {n1}, n2 = {n2}', fontsize=14, color='#48dbfb',
             ha='center', transform=ax4.transAxes)
    
    y = 0.65
    ax4.text(0.1, y, 'Resolution', fontsize=11, color='#feca57', fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.4, y, 'Mean Err', fontsize=11, color='#feca57', fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.65, y, 'Rays', fontsize=11, color='#feca57', fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.85, y, 'Status', fontsize=11, color='#feca57', fontweight='bold', transform=ax4.transAxes)
    
    y -= 0.06
    for r in results:
        status = 'OK' if r['mean'] < 1.0 else '~' if r['mean'] < 2.0 else 'X'
        color = '#00ff88' if r['mean'] < 1.0 else '#feca57' if r['mean'] < 2.0 else '#ff6b6b'
        ax4.text(0.1, y, f"{r['res']}x{r['res']}", fontsize=10, color='white', transform=ax4.transAxes)
        ax4.text(0.4, y, f"{r['mean']:.4f}%", fontsize=10, color='white', transform=ax4.transAxes)
        ax4.text(0.65, y, f"{r['n_valid']}/{n_rays}", fontsize=10, color='white', transform=ax4.transAxes)
        ax4.text(0.85, y, status, fontsize=14, color=color, fontweight='bold', transform=ax4.transAxes)
        y -= 0.055
    
    final = results[-1]
    verdict = "SNELL'S LAW VERIFIED" if final['mean'] < 1.0 else "Approximately Verified"
    v_color = '#00ff88' if final['mean'] < 1.0 else '#feca57'
    ax4.text(0.5, 0.08, verdict, fontsize=16, color=v_color, ha='center',
             fontweight='bold', transform=ax4.transAxes,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e', edgecolor=v_color))
    
    plt.suptitle("Snell-Descartes Law Verification - Convergence Study",
                 fontsize=18, color='white', y=0.98, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = '/Users/dereckewane/propage/results/snell_convergence.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#050510')
    print(f"\n-> Saved: results/snell_convergence.png")
    
    print("\n" + "=" * 75)
    print("  CONVERGENCE STUDY COMPLETE")
    print(f"  Convergence order: O(h^{-slope:.2f})")
    print(f"  At {results[-1]['res']}x{results[-1]['res']}: error = {results[-1]['mean']:.4f}%")
    print("=" * 75)
    
    plt.show()


if __name__ == "__main__":
    run_convergence_study()
