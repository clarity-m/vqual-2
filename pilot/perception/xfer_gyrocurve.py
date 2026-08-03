"""xfer_gyrocurve.py -- does REAL detection probability fall with body angular rate?

The surrogate's detect.py makes p_detect a function of range and obliquity only; motion
blur is not modelled (their section 8). This measures the real curve and controls the
obvious confound: at high rate the gate is also near the frame edge, so a decline could
be framing rather than blur.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PR = 'C:/Users/USER/Projects/vqual-2/pilot/perception/producer_runs/'
VC = 'C:/Users/USER/Projects/vqual-2/pilot/perception/vercheck/'
RUNS = [('gentle 121520', 'XFER-gentle121520'), ('race 153626', 'XFER-fast153626'),
        ('race 161838', 'XFER-fast161838')]

GY, VAL, ST, RNG, BRG, ELV = 9, 2, 5, 4, 6, 16

gy, fresh, rng, brg, elv, tag = [], [], [], [], [], []
for name, run in RUNS:
    d = np.loadtxt(PR + run + '/diag.csv', delimiter=',', skiprows=1)
    v = np.load(PR + run + '/obs.npy')
    m = (d[:, VAL] > 0.5) & (d[:, ST] < 1.0)
    gy.append(d[m, GY]); fresh.append(d[m, ST] <= 1e-6)
    rng.append(d[m, RNG]); brg.append(d[m, BRG]); elv.append(d[m, ELV])
    tag.append(np.full(m.sum(), name))
gy = np.concatenate(gy); fresh = np.concatenate(fresh); rng = np.concatenate(rng)
brg = np.concatenate(brg); elv = np.concatenate(elv); tag = np.concatenate(tag)

edges = np.array([0, .2, .4, .6, .8, 1.0, 1.25, 1.5, 2.0, 3.0])


def curve(mask, lbl):
    xs, ys, ns, lo, hi = [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = mask & (gy >= a) & (gy < b)
        n = int(m.sum())
        if n < 25:
            continue
        p = float(fresh[m].mean())
        se = np.sqrt(p * (1 - p) / n)
        xs.append(0.5 * (a + b)); ys.append(p); ns.append(n)
        lo.append(p - 1.96 * se); hi.append(p + 1.96 * se)
    print('  %s' % lbl)
    for x, y, n, l, h in zip(xs, ys, ns, lo, hi):
        print('     |w| ~ %4.2f  n=%5d  P(fresh) %.3f  [%.3f, %.3f]' % (x, n, y, l, h))
    return np.array(xs), np.array(ys), np.array(ns), np.array(lo), np.array(hi)


print('POOLED over all three regimes (frames where the current gate is tracked and was')
print('measured within the last second):')
base = np.ones(len(gy), bool)
c_all = curve(base, 'all such frames')
print()
print('CONTROL 1 -- gate well inside the frame (|bearing| < 20 deg, elevation within')
print('the +49.4/-9.4 deg span by >= 8 deg margin), so framing cannot explain a decline:')
inner = (np.abs(brg) < 20.0) & (elv > -1.4) & (elv < 41.4)
c_in = curve(inner, 'well-framed only')
print()
print('CONTROL 2 -- matched range band 5-15 m:')
c_r = curve(base & (rng > 5) & (rng < 15), 'range 5-15 m')

# a one-parameter blur model for their detect.py
m = base & (gy < 2.0)
p0 = float(fresh[m & (gy < 0.2)].mean())
x, y, n = c_all[0], c_all[1], c_all[2]
k = np.polyfit(x, np.log(np.clip(y / p0, 1e-3, 1)), 1, w=np.sqrt(n))
print()
print('one-parameter fit  p(|w|) = p0 * exp(-|w| / w0)   with p0 = %.3f (|w| < 0.2)' % p0)
print('   w0 = %.2f rad/s   (weighted log fit over the pooled curve)' % (-1.0 / k[0]))
print('   i.e. detection probability falls ~%.0f%% per rad/s of body rate in 0-1.5 rad/s'
      % (100 * (1 - np.exp(-1.0 / (-1.0 / k[0])))))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for (xs, ys, ns, lo, hi), lbl, c in [(c_all, 'all tracked frames', '#1f77b4'),
                                     (c_in, 'well-framed only', '#2ca02c'),
                                     (c_r, 'range 5-15 m', '#d62728')]:
    ax[0].errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt='o-', color=c, label=lbl,
                   capsize=3)
xx = np.linspace(0, 2, 50)
ax[0].plot(xx, p0 * np.exp(k[0] * xx), 'k--', lw=1,
           label='p0*exp(-|w|/%.2f)' % (-1.0 / k[0]))
ax[0].set_xlabel('|gyro| (rad/s)'); ax[0].set_ylabel('P(fresh measurement)')
ax[0].set_title('real detection rate vs body angular rate\n(pooled, 3 sessions)')
ax[0].set_ylim(0, 1); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
for name, c in zip([r[0] for r in RUNS], ['#1f77b4', '#d62728', '#ff7f0e']):
    m = tag == name
    ax[1].hist(gy[m], bins=np.arange(0, 2.55, 0.1), histtype='step', density=True,
               color=c, label=name)
ax[1].set_xlabel('|gyro| (rad/s)'); ax[1].set_ylabel('density')
ax[1].set_title('how much of each regime is spent at rate'); ax[1].grid(alpha=.3)
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(VC + 'xfer_gyro_detect.png', dpi=110)
print('\nwrote ' + VC + 'xfer_gyro_detect.png')
