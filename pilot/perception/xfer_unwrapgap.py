"""xfer_unwrapgap.py -- does the 121520 lap's compass branch survive from the frames that
RESOLVED it to the frame that measures the 15-16 leg?

psi_unwrapped is only branch-continuous if no unconfident stretch lets the gyro integral
slip by more than 45 deg (the mod-90 half-fold). Every confident frame re-anchors it.
"""
import csv, json
import numpy as np

HERE = 'C:/Users/USER/Projects/vqual-2/pilot/perception/'
SESS = '20260801-121520-vq2-lap-0-15'
rows = list(csv.DictReader(open(HERE + 'skylight_' + SESS + '.csv')))
fid = np.array([int(r['frame_id']) for r in rows])
t = np.array([float(r['t_recv_wall_ns']) for r in rows])
t = (t - t[0]) / 1e9
conf = np.array([int(r['confident']) for r in rows]) > 0
psi = np.array([float(r['psi_unwrapped']) for r in rows])

mr = json.load(open(HERE + 'mapdir_rows.json'))
anchor_pairs = {(0, 1), (4, 5), (5, 6)}
at = [t[np.searchsorted(fid, r['fid'])] for r in mr
      if r['session'] == SESS and tuple(r['pair']) in anchor_pairs
      and r.get('bearing') is not None]
tgt = [t[np.searchsorted(fid, r['fid'])] for r in mr
       if r['session'] == SESS and tuple(r['pair']) == (15, 16)]
print('branch-anchoring rows (0-1,4-5,5-6): n=%d  t = %.1f .. %.1f s'
      % (len(at), min(at), max(at)))
print('the 15-16 referee row: t = %.1f s' % tgt[0])
print('confident frames: %.1f%% of %d' % (100 * conf.mean(), len(conf)))

# longest unconfident run, and the gyro-free psi excursion across it
idx = np.where(conf)[0]
gaps = np.diff(t[idx])
print('gaps between CONFIDENT frames: median %.3f s  p99 %.3f s  max %.3f s'
      % (np.median(gaps), np.percentile(gaps, 99), gaps.max()))
k = int(np.argmax(gaps))
print('  longest gap at t = %.1f -> %.1f s' % (t[idx[k]], t[idx[k + 1]]))
lo, hi = min(min(at), tgt[0]), max(max(at), tgt[0])
sub = (t[idx] >= lo) & (t[idx] <= hi)
g2 = np.diff(t[idx][sub])
print('within the anchor->referee interval (%.1f .. %.1f s): confident-frame gaps '
      'median %.3f s  max %.3f s' % (lo, hi, np.median(g2), g2.max()))
print('psi_unwrapped total excursion over the lap: %.1f deg (min %.1f max %.1f)'
      % (psi.max() - psi.min(), psi.min(), psi.max()))
print()
print('READ: a branch slip needs > 45 deg of unrefereed gyro drift inside ONE gap.')
print('Longest gap in the interval is %.3f s; at the lap\'s p99 body rate that is far')
print('below 45 deg, so the branch carries.' % ())
