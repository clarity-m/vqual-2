"""Sweep the chain length in the V2 height estimator: longer light-motion chains ->
gate-PnP endpoint noise shrinks relative to the baseline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import skylight_pitch as SP

for min_net, max_steps in [(0.25, 120), (0.5, 300), (1.0, 600), (1.5, 900)]:
    ratios = SP.height_from_strafe(min_net=min_net, max_steps=max_steps)
    if not ratios:
        print(f'min_net {min_net}: no windows', flush=True)
        continue
    H = np.array([x['H'] for x in ratios])
    b = np.array([x['base_m'] for x in ratios])
    med = float(np.median(H))
    print(f'min_net {min_net:4.2f}: n={len(H):3d}  H median {med:.3f}  '
          f'MAD {np.median(np.abs(H - med)):.3f}  p10 {np.percentile(H, 10):.3f}  '
          f'p90 {np.percentile(H, 90):.3f}  base_m median {np.median(b):.2f}', flush=True)
    if min_net >= 1.0:
        for x in ratios:
            print(f'    H {x["H"]:6.3f}  base {x["base_m"]:5.2f} m  steps {x["n_steps"]}',
                  flush=True)
