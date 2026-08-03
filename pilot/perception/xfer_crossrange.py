"""xfer_crossrange.py -- a ground-truth-free proxy for |reported - true| pos_body.

VQ2 blocks pose, so there is no truth to difference against. But the race packet says
WHEN a gate plane was crossed, and at that instant the true range from the drone to that
gate's centre is bounded by the aperture: <= 0.75 m in-plane, plus whatever the race
packet's own latency costs. So extrapolating the reported current-gate range to the
crossing time and asking how far it misses zero is a bounded-truth range check.

Caveats, stated up front:
  * it measures RANGE error along the line of sight only, not full pos_body error;
  * it conflates our error with the drone's genuine off-centre passage (<= 0.75 m) and
    with race-packet timing;
  * it samples only the very-near regime, which is where our detector is BEST.
"""
import os, sys, csv, json
import numpy as np

sys.path.insert(0, 'C:/Users/USER/Projects/vqual-2/pilot')
import producer as P

PR = 'C:/Users/USER/Projects/vqual-2/pilot/perception/producer_runs/'
SESS = 'C:/Users/USER/Projects/vqual-2/pilot/sessions/'

RUNS = [('gentle 121520', 'XFER-gentle121520', '20260801-121520-vq2-lap-0-15', False),
        ('race   153626', 'XFER-fast153626',
         '20260802-153626-vm-fast-lap-0-10-with-collisions', False),
        ('race   161838', 'XFER-fast161838', '20260802-161838', True)]

WIN = 1.0      # seconds of approach used for the linear extrapolation


def crossings(sess, cut):
    frames = [r for r in P.load_csv(SESS + sess + '/frames.csv') if r['file']]
    race = P.load_csv(SESS + sess + '/race.csv')
    if cut:
        r = P.find_last_reset(race, frames)
        if r is not None:
            frames = [f for f in frames if float(f['t_recv_wall_ns']) >= r[0]]
    t0 = float(frames[0]['t_recv_wall_ns'])
    t1 = float(frames[-1]['t_recv_wall_ns'])
    out, prev = [], None
    for r in race:
        t = float(r['t_wall_ns'])
        a = int(r['active_gate_index'])
        if prev is not None and a == prev + 1 and t0 <= t <= t1:
            out.append(((t - t0) / 1e9, prev))
        prev = a
    return out


print('crossing-range residual: reported current-gate range extrapolated to the moment')
print('the race packet advanced, where the true range is <= ~0.75 m\n')
allrows = {}
for name, run, sess, cut in RUNS:
    if not os.path.exists(PR + run + '/diag.csv'):
        print('%s: no run yet' % name)
        continue
    dg = np.loadtxt(PR + run + '/diag.csv', delimiter=',', skiprows=1)
    t, act, val, rng = dg[:, 0], dg[:, 1], dg[:, 2], dg[:, 4]
    st = dg[:, 5]
    rows = []
    for tc, g in crossings(sess, cut):
        m = (t > tc - WIN) & (t <= tc) & (val > 0.5) & (act == g) & np.isfinite(rng) \
            & (st < 0.2)
        if m.sum() < 5:
            rows.append((g, np.nan, np.nan, int(m.sum())))
            continue
        A = np.polyfit(t[m], rng[m], 1)
        rhat = np.polyval(A, tc)
        rows.append((g, float(rhat), float(-A[0]), int(m.sum())))
    ok = np.array([r[1] for r in rows if np.isfinite(r[1])])
    sp = np.array([r[2] for r in rows if np.isfinite(r[1])])
    allrows[name] = rows
    print('== %s ==  %d crossings, %d usable' % (name, len(rows), len(ok)))
    print('   ' + '  '.join('g%d:%+.1f' % (r[0], r[1]) for r in rows if np.isfinite(r[1])))
    if len(ok):
        print('   residual  median %+.2f m   |median| %.2f   MAD %.2f   p90|.| %.2f   '
              'rms %.2f' % (np.median(ok), abs(np.median(ok)),
                            np.median(np.abs(ok - np.median(ok))),
                            np.percentile(np.abs(ok), 90), np.sqrt((ok ** 2).mean())))
        print('   closing speed at crossing, median %.1f m/s (fit slope; sanity only)'
              % np.median(sp))
json.dump({k: [[float(x) for x in r] for r in v] for k, v in allrows.items()},
          open('C:/Users/USER/Projects/vqual-2/pilot/perception/xfer_crossrange.json',
               'w'), indent=1)
