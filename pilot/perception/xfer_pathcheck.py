"""xfer_pathcheck.py -- flown path length per leg from the drag speed model, against the
map chord. path/chord must be >= 1, and the drag speed is the BODY-HORIZONTAL in-plane
component only, so the measured ratio UNDERSTATES the truth.

    |v_xy| = sqrt(|a_xy| / k),  k = 0.0425 /m   (fit vs VQ1 truth velocity, 2026-08-02)
"""
import sys, json
import numpy as np
sys.path.insert(0, 'C:/Users/USER/Projects/vqual-2/pilot')
import producer as P

K = 0.0425
SESS = 'C:/Users/USER/Projects/vqual-2/pilot/sessions/'
C = json.load(open('C:/Users/USER/Projects/vqual-2/pilot/course/course_vq2.json'))
POS = np.array([g['position_m'] for g in C['gates']], float)

RUNS = [('race 153626', '20260802-153626-vm-fast-lap-0-10-with-collisions', False),
        ('race 161838', '20260802-161838', True),
        ('gentle 121520', '20260801-121520-vq2-lap-0-15', False)]

acc = {}
for name, sess, cut in RUNS:
    imu = P.load_csv(SESS + sess + '/imu.csv')
    race = P.load_csv(SESS + sess + '/race.csv')
    frames = [r for r in P.load_csv(SESS + sess + '/frames.csv') if r['file']]
    t_lo = 0.0
    if cut:
        r = P.find_last_reset(race, frames)
        if r is not None:
            t_lo = r[0]
    ti = np.array([float(r['t_wall_ns']) for r in imu])
    ax = np.array([float(r['xacc']) for r in imu])
    ay = np.array([float(r['yacc']) for r in imu])
    ah = np.hypot(ax, ay)
    v = np.sqrt(np.maximum(ah, 0.0) / K)
    # crossing times
    cr, prev = [], None
    for r in race:
        t, a = float(r['t_wall_ns']), int(r['active_gate_index'])
        if prev is not None and a == prev + 1 and t >= t_lo:
            cr.append((t, prev))
        prev = a
    print('== %s ==' % name)
    for (ta, ga), (tb, gb) in zip(cr[:-1], cr[1:]):
        if gb != ga + 1:
            continue
        m = (ti >= ta) & (ti <= tb)
        if m.sum() < 20:
            continue
        path = float(np.trapezoid(v[m], ti[m] / 1e9))
        chord = float(np.linalg.norm(POS[gb] - POS[ga]))
        dur = (tb - ta) / 1e9
        print('   leg %2d-%-2d  dur %5.2f s  chord %6.2f m  drag path %6.2f m  '
              'path/chord %5.2f  mean|v_xy| %5.2f m/s'
              % (ga, gb, dur, chord, path, path / chord, path / dur))
        acc.setdefault((ga, gb), []).append(path / chord)
print()
print('--- pooled path/chord per leg (race laps only) ---')
rat = {k: v for k, v in acc.items()}
for k in sorted(rat):
    print('   leg %2d-%-2d  n=%d  ratios %s' % (k[0], k[1], len(rat[k]),
                                                ' '.join('%.2f' % x for x in rat[k])))
allr = np.array([x for v in rat.values() for x in v])
print('   median over all legs/laps: %.2f  (must be >= 1.0)' % np.median(allr))
