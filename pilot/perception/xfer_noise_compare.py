"""xfer_noise_compare.py -- measured producer statistics vs the surrogate's synthetic
detector table (STATE_SURROGATE_FOR_PERCEPTION.md section 2), split by regime.

Reads producer_runs/RUN/obs.npy (N x 73, interface.Observation.to_vector layout) and
diag.csv. Read-only apart from xfer_* artifacts.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PR = 'C:/Users/USER/Projects/vqual-2/pilot/perception/producer_runs/'
VC = 'C:/Users/USER/Projects/vqual-2/pilot/perception/vercheck/'
os.makedirs(VC, exist_ok=True)

RUNS = [('gentle 20260801-121520 exploratory lap 0-16', 'XFER-gentle121520'),
        ('race   20260802-153626 fast lap 0-11 collisions', 'XFER-fast153626'),
        ('race   20260802-161838 fast lap 0-15 post-reset', 'XFER-fast161838')]

IV, INV, IPV, ICF, IST = 6, 7, 8, 9, 10
RB_VALID, RB_PXF = 45, 46
GYRO = slice(48, 51)
ATT_CONF = 56


def gbase(g):
    return 11 * g


def runs_of(mask):
    out, n = [], 0
    for m in mask:
        if m:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return np.array(out, float)


def analyse(name, run):
    v = np.load(PR + run + '/obs.npy')
    dg = np.loadtxt(PR + run + '/diag.csv', delimiter=',', skiprows=1)
    N = len(v)
    t = dg[:, 0]
    dt = float(np.median(np.diff(t)))
    b0 = gbase(0)
    val = v[:, b0 + IV] > 0.5
    nrm = v[:, b0 + INV] > 0.5
    pos = v[:, b0 + IPV] > 0.5
    st = v[:, b0 + IST]
    rng = np.linalg.norm(v[:, b0:b0 + 3], axis=1)
    gy = np.linalg.norm(v[:, GYRO], axis=1)
    fresh = val & (st <= 1e-6)
    p = v[:, b0:b0 + 3]
    same = np.zeros(N, bool)
    same[1:] = np.all(p[1:] == p[:-1], axis=1) & val[1:] & val[:-1]
    r = dict(name=name, run=run, N=N, dur=float(t[-1] - t[0]), dt=dt,
             valid=float(val.mean()),
             normal_valid=float(nrm.mean()), normal_valid_c=float(nrm[val].mean()),
             pose_valid=float(pos.mean()), pose_valid_c=float(pos[val].mean()),
             stale_mean=float(st[val].mean()), stale_med=float(np.median(st[val])),
             stale_p90=float(np.percentile(st[val], 90)),
             unchanged_bitwise=float(same.mean()),
             unrefreshed=float((val & (st > 1e-6)).mean()),
             ribbon=float((v[:, RB_VALID] > 0.5).mean()),
             med_range=float(np.median(rng[val])),
             gyro_med=float(np.median(gy)), gyro_p90=float(np.percentile(gy, 90)),
             gyro_hi=float((gy > 1.0).mean()),
             att_conf=float(np.median(v[:, ATT_CONF])),
             lookahead1=float((v[:, gbase(1) + IV] > 0.5).mean()),
             lookahead2=float((v[:, gbase(2) + IV] > 0.5).mean()),
             attn=[float(v[:, 69 + k].mean()) for k in range(4)])
    rb = v[:, RB_VALID] > 0.5
    r['ribbon_pxf'] = float(np.median(v[rb, RB_PXF])) if rb.any() else float('nan')
    gaps = runs_of(~fresh) * dt
    r['gap_n'] = int(len(gaps))
    for k, q in [('gap_med', 50), ('gap_p90', 90)]:
        r[k] = float(np.percentile(gaps, q)) if len(gaps) else float('nan')
    r['gap_max'] = float(gaps.max()) if len(gaps) else float('nan')
    r['gap_long'] = float((gaps > 0.2).mean()) if len(gaps) else float('nan')
    x = fresh.astype(float) - fresh.mean()
    den = float((x * x).sum())
    r['acf'] = [float((x[:N - k] * x[k:]).sum() / den) for k in (1, 2, 3, 5, 10, 20, 30)]
    sel = val & (st < 1.0)
    edges = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 10.0])
    curve = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = sel & (gy >= a) & (gy < b)
        curve.append((float(a), float(b), int(m.sum()),
                      float(fresh[m].mean()) if m.sum() else float('nan')))
    r['gyro_curve'] = curve
    r['_a'] = dict(gy=gy, fresh=fresh, gaps=gaps, val=val, st=st)
    return r


res = [analyse(n, rn) for n, rn in RUNS if os.path.exists(PR + rn + '/obs.npy')]


def row(lbl, f, fmt='%.1f%%', mul=100.0):
    return [lbl] + [(fmt % (f(r) * mul)) for r in res]


rows = [
    row('current gate valid', lambda r: r['valid']),
    row('normal_valid  (all frames)', lambda r: r['normal_valid']),
    row('normal_valid  (given valid)', lambda r: r['normal_valid_c']),
    row('pose_valid    (all frames)', lambda r: r['pose_valid']),
    row('pose_valid    (given valid)', lambda r: r['pose_valid_c']),
    row('staleness mean on valid, s', lambda r: r['stale_mean'], '%.3f', 1.0),
    row('staleness median on valid, s', lambda r: r['stale_med'], '%.3f', 1.0),
    row('staleness p90 on valid, s', lambda r: r['stale_p90'], '%.3f', 1.0),
    row('unchanged since last step, bitwise', lambda r: r['unchanged_bitwise']),
    row('no fresh measurement this step', lambda r: r['unrefreshed']),
    row('ribbon valid', lambda r: r['ribbon']),
    row('ribbon cyan pixel frac, median', lambda r: r['ribbon_pxf'], '%.4f', 1.0),
    row('median range of current gate, m', lambda r: r['med_range'], '%.1f', 1.0),
    row('lookahead slot 1 valid', lambda r: r['lookahead1']),
    row('lookahead slot 2 valid', lambda r: r['lookahead2']),
    row('gyro magnitude median, rad/s', lambda r: r['gyro_med'], '%.2f', 1.0),
    row('gyro magnitude p90, rad/s', lambda r: r['gyro_p90'], '%.2f', 1.0),
    row('frames with gyro over 1 rad/s', lambda r: r['gyro_hi']),
    row('attitude_conf median', lambda r: r['att_conf'], '%.2f', 1.0),
]
print('=' * 104)
for r in res:
    print('%-50s N=%5d  %6.1f s  dt=%.4f s' % (r['name'], r['N'], r['dur'], r['dt']))
print('=' * 104)
w = max(len(x[0]) for x in rows) + 2
print(('%-*s' % (w, 'metric')) + ''.join('%-20s' % r['name'].split()[1][-6:] for r in res))
for x in rows:
    print(('%-*s' % (w, x[0])) + ''.join('%-20s' % c for c in x[1:]))

print()
print('--- dropout burst structure: runs of no-fresh-measurement ---')
for r in res:
    print('  %-50s n=%4d  median %.2f s  p90 %.2f s  max %.2f s  %.0f pct over 0.2 s'
          % (r['name'], r['gap_n'], r['gap_med'], r['gap_p90'], r['gap_max'],
             100 * r['gap_long']))
print()
print('--- autocorrelation of the fresh-measurement indicator, lags 1 2 3 5 10 20 30 ---')
for r in res:
    print('  %-50s %s' % (r['name'], ' '.join('%+.2f' % a for a in r['acf'])))
print()
print('--- P(fresh) vs gyro magnitude, gate tracked and seen under 1 s ago ---')
for r in res:
    print('  ' + r['name'])
    for a, b, n, p in r['gyro_curve']:
        print('     gyro %4.2f-%5.2f rad/s  n=%5d  P(fresh)=%s'
              % (a, b, n, ('%.3f' % p) if n else '   --'))
print()
print('--- attention mix SEARCH / GATE_CURRENT / GATE_NEXT / RIBBON ---')
for r in res:
    print('  %-50s %s' % (r['name'], ' '.join('%.1f pct' % (100 * a) for a in r['attn'])))

json.dump([{k: vv for k, vv in r.items() if k != '_a'} for r in res],
          open('C:/Users/USER/Projects/vqual-2/pilot/perception/xfer_noise_compare.json',
               'w'), indent=1)


fig, ax = plt.subplots(2, 2, figsize=(13, 9))
cols = ['#1f77b4', '#d62728', '#ff7f0e']
for r, c in zip(res, cols):
    A = r['_a']
    lab = r['name'].split()[0] + ' ' + r['name'].split()[1][-6:]
    xs = [0.5 * (a + b) for a, b, n, p in r['gyro_curve'] if n > 30]
    ys = [p for a, b, n, p in r['gyro_curve'] if n > 30]
    ax[0, 0].plot(xs, ys, 'o-', color=c, label=lab)
    g = A['gaps']
    if len(g):
        ax[0, 1].hist(g, bins=np.arange(0, 1.55, 0.05), histtype='step',
                      color=c, density=True, label=lab)
    f = A['fresh'].astype(float)
    ax[1, 0].plot(range(1, 31),
                  [float(np.corrcoef(f[:-k], f[k:])[0, 1]) for k in range(1, 31)],
                  color=c, label=lab)
    st = A['st'][A['val']]
    ax[1, 1].hist(st, bins=np.arange(0, 1.02, 0.02), histtype='step', color=c,
                  density=True, label=lab)
ax[0, 0].set_xlabel('body angular rate magnitude (rad/s)')
ax[0, 0].set_ylabel('P(fresh measurement)')
ax[0, 0].set_title('detection rate vs body angular rate\ngate tracked and seen under 1 s ago')
ax[0, 0].set_ylim(0, 1)
ax[0, 1].set_xlabel('dropout gap length (s)')
ax[0, 1].set_ylabel('density')
ax[0, 1].set_title('dropout is bursty: gap-length distribution')
ax[0, 1].set_yscale('log')
ax[1, 0].set_xlabel('lag (frames at 30 Hz)')
ax[1, 0].set_ylabel('autocorrelation of fresh indicator')
ax[1, 0].set_title('validity is strongly autocorrelated')
ax[1, 0].axhline(0, color='k', lw=0.5)
ax[1, 1].set_xlabel('staleness_s on valid frames')
ax[1, 1].set_ylabel('density')
ax[1, 1].set_title('staleness distribution')
ax[1, 1].set_yscale('log')
for a in ax.ravel():
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
fig.suptitle('vqual-2 perception: real producer statistics by regime', fontsize=13)
fig.tight_layout()
fig.savefig(VC + 'xfer_noise_shape.png', dpi=110)
print('wrote ' + VC + 'xfer_noise_shape.png')
