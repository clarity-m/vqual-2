"""policy_dryrun.py -- offline shakeout of a Policy against a REAL recorded session.

Walks a session the same way producer.py --replay does (frames + raw IMU + race),
feeds each Observation to the policy, and reports what it would have commanded. No
sim needed, so signs, mode logic and gain sanity get checked before anything flies.

Two outputs:
  * mode/action statistics -- does the policy spend the lap tracking, or blind?
  * directional agreement vs cmd.csv -- on a hand-flown lap the pilot was steering at
    the same gates, so the PHYSICAL roll command should correlate in sign. cmd.csv is
    wire convention; the policy is canonical; both are converted to canonical here.
    Expect positive-but-modest agreement (she flies harder and earlier than a P-loop);
    a NEGATIVE roll agreement means a mirrored axis and must stop the show.

    python3 pilot/policy_dryrun.py <session-dir-or-name> [--limit N] [--stride N]
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from producer import HERE, Producer, load_csv
from policy_servo import ServoPolicy

# Wire -> canonical for cmd.csv (CONVENTIONS.md: all three rates mirrored).
SIGN = -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--no-net', action='store_true')
    args = ap.parse_args()

    session = args.session
    if not os.path.isdir(session):
        session = os.path.join(HERE, 'sessions', session)

    frames = [r for r in load_csv(os.path.join(session, 'frames.csv')) if r['file']]
    imu = load_csv(os.path.join(session, 'imu.csv'))
    race = load_csv(os.path.join(session, 'race.csv'))
    cmd = load_csv(os.path.join(session, 'cmd.csv'))
    frames = frames[::args.stride]
    if args.limit:
        frames = frames[:args.limit]

    imu_t = np.array([float(r['t_wall_ns']) for r in imu])
    race_t = np.array([float(r['t_wall_ns']) for r in race]) if race else np.array([0.0])
    cmd_t = np.array([float(r['t_wall_ns']) for r in cmd]) if cmd else None

    prod = Producer(use_net=not args.no_net)
    pol = ServoPolicy()
    t0 = float(frames[0]['t_recv_wall_ns'])
    imu_i = 0
    modes = {}
    acts, recs = [], []

    for fr in frames:
        t_ns = float(fr['t_recv_wall_ns'])
        j = int(np.searchsorted(imu_t, t_ns))
        rows, imu_i = imu[imu_i:j], j
        ri = max(0, int(np.searchsorted(race_t, t_ns)) - 1)
        active = int(race[ri]['active_gate_index']) if race else 0
        img = cv2.imread(os.path.join(session, 'frames', fr['file']))
        obs = prod.step(img, rows, {'active_gate_index': active, 'armed': True,
                                    'n_gates_total': 17}, (t_ns - t0) / 1e9)
        a = pol(obs)
        modes[pol._mode] = modes.get(pol._mode, 0) + 1
        g = obs.gates[0]
        brg = math.degrees(g.bearing_rad) if g.valid else np.nan
        acts.append((a.roll_rate, a.pitch_rate, a.thrust,
                     brg, 1.0 if pol._mode in ('track', 'aim') else 0.0))
        if cmd_t is not None and len(cmd_t):
            k = min(len(cmd) - 1, max(0, int(np.searchsorted(cmd_t, t_ns)) - 1))
            recs.append((SIGN * float(cmd[k]['roll_rate']),
                         SIGN * float(cmd[k]['pitch_rate']),
                         float(cmd[k]['thrust'])))

    acts = np.array(acts)
    n = len(acts)
    print('\n%d frames  modes: %s' % (
        n, '  '.join('%s %.0f%%' % (k, 100.0 * v / n)
                     for k, v in sorted(modes.items(), key=lambda kv: -kv[1]))))
    print('policy roll  cmd: mean %+.2f  p90|.| %.2f rad/s' % (
        acts[:, 0].mean(), np.percentile(np.abs(acts[:, 0]), 90)))
    print('policy pitch cmd: mean %+.2f  p90|.| %.2f rad/s' % (
        acts[:, 1].mean(), np.percentile(np.abs(acts[:, 1]), 90)))
    print('policy thrust   : mean %.3f  p10 %.3f  p90 %.3f' % (
        acts[:, 2].mean(), *np.percentile(acts[:, 2], [10, 90])))

    if recs:
        recs = np.array(recs)
        for i, name in ((0, 'roll'), (1, 'pitch')):
            m = (np.abs(acts[:, i]) > 0.05) & (np.abs(recs[:, i]) > 0.05)
            if m.sum() > 20:
                agree = float(np.mean(np.sign(acts[m, i]) == np.sign(recs[m, i])))
                r = float(np.corrcoef(acts[m, i], recs[m, i])[0, 1])
                print('%s vs recorded (canonical, %d frames both active): '
                      'sign-agree %.0f%%  corr %+.2f%s'
                      % (name, m.sum(), 100 * agree, r,
                         '   << MIRRORED AXIS?' if agree < 0.45 else ''))
        thr = float(np.corrcoef(acts[:, 2], recs[:, 2])[0, 1])
        print('thrust vs recorded: corr %+.2f' % thr)

        # The cut that actually discriminates a mirror: gate well off to one side and
        # the policy steering at it. There a human flying the same course rolls the
        # same way; chance-level agreement HERE (not overall) is the alarm.
        m = (acts[:, 4] > 0) & (np.abs(acts[:, 3]) > 10.0) & (np.abs(recs[:, 0]) > 0.05)
        if m.sum() > 20:
            agree = float(np.mean(np.sign(acts[m, 0]) == np.sign(recs[m, 0])))
            print('roll agree, |bearing|>10deg & steering (%d frames): %.0f%%%s'
                  % (m.sum(), 100 * agree,
                     '   << MIRRORED AXIS?' if agree < 0.45 else ''))


if __name__ == '__main__':
    main()
