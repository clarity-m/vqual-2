"""Driver: V2 ceiling-height estimate (chained light motion vs endpoint gate PnP)
plus grid pitch. Spacing ratios re-measured too so the pitch is self-contained."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import skylight_pitch as SP

print('GRID SPACING (units of H), strafe session:', flush=True)
sp = SP.grid_spacing()
for ax, (s, n) in sp.items():
    print(f'  axis {ax}: spacing/H = {None if s is None else round(s, 4)} (n={n})', flush=True)

print('CEILING HEIGHT H, V2 (chained frame-to-frame light motion, endpoint-smoothed PnP):', flush=True)
ratios = SP.height_from_strafe()
if ratios:
    H = np.array([x['H'] for x in ratios])
    b = np.array([x['base_m'] for x in ratios])
    med = float(np.median(H))
    print(f'  n={len(H)} windows: H median {med:.3f} m, MAD {np.median(np.abs(H - med)):.3f}, '
          f'p10 {np.percentile(H, 10):.3f}, p90 {np.percentile(H, 90):.3f}')
    print(f'  baselines: median {np.median(b):.2f} m, p10 {np.percentile(b, 10):.2f}, '
          f'p90 {np.percentile(b, 90):.2f}')
    for ax, (s, n) in sp.items():
        if s:
            print(f'  -> grid pitch axis {ax}: {s * med:.3f} m')
else:
    print('  no usable windows')
