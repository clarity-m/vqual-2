"""Is this recording usable for map building? Run it the moment a lap lands.

An hour of flying is expensive and a bad recording is not obvious from the cockpit. This
answers the four questions that decide whether the session can build a map, before any
time is spent on it:

  1. Does active_gate_index actually advance?   <- the ONLY source of race order
  2. Are frames present, and paced like 30 Hz?
  3. Do gates get detected, and at what rate?
  4. Do the Station columns read?               <- identity when the race packet is silent

Written 2026-07-31 against the fact that every VQ2 session recorded so far tops out at
active_gate_index 1, which was not noticed until a map had already been built on one.

    python3 pilot/perception/checksession.py <session-dir>
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import label as L  # noqa: E402

OK, BAD, WARN = 'PASS', 'FAIL', 'WARN'


def verdict(tag, msg):
    print(f'  [{tag}] {msg}')
    return tag


def check(session, expect_gates, sample):
    import cv2
    import detect as D

    results = []
    print(f'\n{os.path.basename(os.path.normpath(session))}')

    # --- 1. race order -------------------------------------------------------------
    race = list(L.load_csv(os.path.join(session, 'race.csv')))
    idx = sorted({int(r['active_gate_index']) for r in race if r.get('active_gate_index')})
    crossed = max(idx) if idx else -1
    print('\nrace order')
    if crossed >= expect_gates - 1:
        results.append(verdict(OK, f'active_gate_index reached {crossed} — full course'))
    elif crossed >= 1:
        results.append(verdict(WARN, f'active_gate_index reached {crossed} of {expect_gates - 1}'
                                     f' — partial course, map covers only those gates'))
    else:
        results.append(verdict(BAD, f'active_gate_index never advanced past {crossed}. '
                                    'No gate was crossed, so there is NO race order in this '
                                    'recording and no map can be built from it.'))

    # --- 2. frames -----------------------------------------------------------------
    frames = [r for r in L.load_csv(os.path.join(session, 'frames.csv')) if r['file']]
    print('\nframes')
    if not frames:
        results.append(verdict(BAD, 'no frames recorded'))
        return results
    t = np.array([float(r['t_recv_wall_ns']) for r in frames]) / 1e9
    dur = t[-1] - t[0]
    fps = len(frames) / max(dur, 1e-6)
    on_disk = sum(1 for r in frames[::37]
                  if os.path.exists(os.path.join(session, 'frames', r['file'])))
    results.append(verdict(OK if fps > 20 else WARN,
                           f'{len(frames)} frames, {dur:.0f} s, {fps:.1f} fps'))
    if on_disk < len(frames[::37]):
        results.append(verdict(BAD, 'frames.csv references JPEGs that are not on disk'))
    gap = np.diff(t)
    if gap.size and gap.max() > 1.0:
        results.append(verdict(WARN, f'largest frame gap {gap.max():.1f} s — a stall or a '
                                     'reset; tracks will not survive it'))

    # --- 3. detection --------------------------------------------------------------
    print('\ndetection')
    sel = frames[::max(1, len(frames) // sample)][:sample]
    counts = []
    for r in sel:
        img = cv2.imread(os.path.join(session, 'frames', r['file']))
        if img is not None:
            counts.append(len(D.detections(img)))
    counts = np.array(counts) if counts else np.array([0])
    multi = float((counts >= 2).mean())
    results.append(verdict(OK if multi > 0.5 else WARN,
                           f'{counts.mean():.2f} gates/frame, max {counts.max()}, '
                           f'{100 * multi:.0f}% of frames carry 2+ '
                           '(co-visibility is what makes an edge)'))

    # --- 4. stations ---------------------------------------------------------------
    print('\nstations')
    tmpl = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'station_templates.npz')
    if not os.path.exists(tmpl):
        results.append(verdict(WARN, 'no station templates built; skipping'))
    else:
        import stations as S
        d = np.load(tmpl)
        T, keys = d['templates'], [str(x) for x in d['digits']]
        seen, nread = set(), 0
        for r in sel:
            img = cv2.imread(os.path.join(session, 'frames', r['file']))
            if img is None:
                continue
            rd = S.read_frame(img, T, keys)
            if rd:
                nread += 1
            seen.update(x['station'] for x in rd)
        cov = nread / max(1, len(sel))
        results.append(verdict(OK if cov > 0.2 else WARN,
                               f'{100 * cov:.0f}% of frames read a station, '
                               f'{len(seen)} distinct: {sorted(seen)}'))
        if cov < 0.2:
            print('        low coverage usually means the session was flown fast — '
                  'the columns blur.')

    print('\n' + ('FAIL — do not build on this recording' if BAD in results else
                  'usable' + (' (with warnings)' if WARN in results else '')))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--gates', type=int, default=17, help='expected gate count')
    ap.add_argument('--sample', type=int, default=120)
    args = ap.parse_args()
    check(args.session, args.gates, args.sample)


if __name__ == '__main__':
    main()
