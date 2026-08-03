"""build_tilt.py -- per-gate TILT-FROM-VERTICAL, measured, for the course export.

Estimator (pilot/perception/gatetilt.py, edge_tilt): the plane through the camera centre
and the image line of one gate edge has normal N = K^T l; the edge's 3-D direction lies
in it, so a gravity-vertical edge satisfies N.g = 0 and

    phi = asin(N_hat . g_hat)

is that edge's lean out of vertical -- from the IMAGE LINE and GRAVITY only. No PnP
rotation, no IPPE twin, no compass, so it survives exactly the head-on degeneracy that
makes gate-plane YAW unmeasurable at gates 8/12/13.

Validated on an independent frame: gate8.png (a fresh sim screenshot, not one of the five
mapped sessions) shows a gate edge-on as a narrow orange bar. Fitting that bar's axis and
running the same formula (90 deg HFOV, camera pitched 20 deg up) returns 19.7 deg lean --
against 3.9 deg of pure perspective lean for a TRUE vertical at that image position, and
1.0-1.2 deg measured on two hangar columns in the same frame. So the estimator resolves a
real ~15-20 deg tilt when one is present.

KNOWN BLIND SPOT, stated because it decides how to read the result: a gate that leans
BACK or FORWARD (tilt about its own horizontal in-plane axis) projects its side edges as
vertical lines when viewed HEAD-ON. The lean only appears off-axis, and the detector's
aspect-ratio filter throws away the most oblique views. So this measurement is strongest
where the flight passed a gate side-on and weakest on approach-only gates.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PERC = os.path.normpath(os.path.join(HERE, '..', 'perception'))
sys.path.insert(0, PERC)
import gatetilt as GT        # noqa: E402

ROWS = os.path.join(PERC, 'gatetilt_rows.json')
OUT = os.path.join(HERE, 'tilt_measured.json')

# Above this share of rows leaning more than 10 deg, a gate is called SUSPECT rather than
# vertical. Every control gate the map accepted sits at 0.01-0.11; gate 8 sits at 0.27.
SUSPECT_FRAC = 0.18


def main():
    R = {int(k): v for k, v in json.load(open(ROWS)).items()}
    per = {}
    print('%-5s %6s %6s %6s %6s %7s  %s' %
          ('gate', 'n', 'med', 'p75', 'p90', 'frac>10', 'verdict'))
    for g in range(17):
        rows = R.get(g, [])
        v, obl = [], []
        for r in rows:
            e = GT.edge_tilt(r)
            if e is None or r['size_px'] < 12:
                continue
            q = np.asarray(r['quad'], float).reshape(4, 2)
            Ls = [np.linalg.norm(q[(j + 1) % 4] - q[j]) for j in range(4)]
            a02, a13 = 0.5 * (Ls[0] + Ls[2]), 0.5 * (Ls[1] + Ls[3])
            obl.append(math.degrees(math.acos(
                max(0.0, min(1.0, min(a02, a13) / max(a02, a13))))))
            v.append(e['lean_abs'])
        v = np.array(v)
        obl = np.array(obl)
        if len(v) < 20:
            per[g] = {'tilt_from_vertical_deg': 0.0, 'tilt_sigma_deg': 4.0,
                      'tilt_n': int(len(v)),
                      'tilt_status': 'assumed vertical -- too few rows to measure'}
            print('%-5d %6d %6s %6s %6s %7s  assumed vertical (thin)'
                  % (g, len(v), '-', '-', '-', '-'))
            continue
        frac = float((v > 10.0).mean())
        med, p75, p90 = (float(np.median(v)), float(np.percentile(v, 75)),
                         float(np.percentile(v, 90)))
        obl_hi = float(np.median(v[obl > 35])) if (obl > 35).sum() >= 20 else None
        if frac > SUSPECT_FRAC:
            st = ('SUSPECT: heavier lean tail than every other gate -- consistent with a '
                  'real tilt but not resolved; see structure.tilted_gate')
            tilt, sig = None, None
        else:
            st = 'vertical (measured)'
            tilt, sig = 0.0, round(max(2.0, med), 2)
        per[g] = {
            'tilt_from_vertical_deg': tilt,
            'tilt_sigma_deg': sig,
            'tilt_n': int(len(v)),
            'tilt_median_deg': round(med, 2),
            'tilt_p90_deg': round(p90, 2),
            'tilt_frac_over_10deg': round(frac, 3),
            'tilt_median_oblique_views_deg': (round(obl_hi, 2)
                                              if obl_hi is not None else None),
            'tilt_status': st,
        }
        print('%-5d %6d %6.2f %6.2f %6.2f %7.3f  %s'
              % (g, len(v), med, p75, p90, frac, st))

    # The one screenshot-confirmed tilted gate is either 8 or 9 and the evidence does not
    # separate them, so BOTH carry an unresolved tilt rather than a committed value.
    for g, why in ((8, 'data outlier: 27% of rows lean >10 deg, p90 12.9 deg -- the only '
                       'gate that stands out, but its sessions disagree'),
                   (9, 'pilot\'s visual identification of the ~15-20 deg tilted gate in '
                       'gate8.png; our rows read 4.0 deg median and do NOT confirm it')):
        per[g]['tilt_from_vertical_deg'] = None
        per[g]['tilt_sigma_deg'] = None
        per[g]['tilt_prior'] = {'kind': 'uniform', 'low_deg': 0.0, 'high_deg': 20.0}
        per[g]['tilt_status'] = 'UNRESOLVED -- randomize. ' + why

    doc = {
        'method': __doc__.strip(),
        'per_gate': {str(g): per[g] for g in range(17)},
        'structure': {
            'gates_are_vertical': (
                'Measured: every gate reads a median side-edge lean of 1.3-4.9 deg out '
                'of gravity-vertical, which is the noise floor of the estimator, and '
                'nothing rises with viewing obliquity the way a real tilt must. Treat '
                'the course as vertical planes. This is a usable PRIOR for a pose '
                'estimator: it resolves the two-fold IPPE tilt ambiguity, which is '
                'exactly the ambiguity that made gate YAW unmeasurable at 8/12/13.'),
            'tilted_gate': (
                'ONE EXCEPTION EXISTS AND WE COULD NOT NAIL WHICH GATE IT IS. An '
                'independent sim screenshot (gate8.png) shows a gate edge-on leaning '
                '19.7 deg off vertical -- unambiguous, and 5x the 3.9 deg of perspective '
                'lean expected for a true vertical at that image position. The pilot '
                'first read it as gate 8, then as gate 9. Our five mapped sessions do '
                'not confirm either: gate 8 has by far the heaviest lean tail (27% of '
                'rows over 10 deg, p90 12.9 deg, against 1-11% and p90 4-10 deg for '
                'every other gate) but its own sessions disagree (median 10.5 deg on '
                '121520, 3.1 deg on 202110); gate 9 reads 4.0 deg median, 10% over 10 '
                'deg -- unremarkable. Because a lean-BACK tilt is invisible head-on and '
                'our rows are mostly approach views, absence of evidence here is weak '
                'evidence of absence. RECOMMENDED HANDLING: randomize the tilt of gates '
                '8 AND 9 over 0-20 deg (course_vq2.sample does this) rather than '
                'committing to either.'),
            'gate_faces_and_the_grid': (
                'The pilot reports that gate faces are parallel to the hangar columns, '
                'i.e. gate yaw is quantized to the ceiling/floor/column grid (which the '
                'skylight compass reads, and which the station rows share within sketch '
                'accuracy). OUR MEASURED YAWS DO NOT REPRODUCE THAT: the 14 accepted '
                'plane azimuths, taken mod 90, spread over 11-88 deg with no cluster at '
                'the grid bin (deviations -37 to +44 deg). They agree far better with '
                'the RACE-LINE BISECTOR (median |error| 10.9 deg, and under 6 deg for '
                'gates 1, 3, 6, 10, 14, 16). Both facts are exported per gate -- '
                'yaw_deg, yaw_grid_bin_deg + yaw_grid_dev_deg, yaw_race_bisector_deg -- '
                'and the conflict is flagged rather than smoothed. Given how noisy IPPE '
                'azimuth proved to be on this course, the visual claim may well be '
                'right and our azimuths wrong; do not build anything that needs an '
                'exact gate yaw.'),
        },
    }
    json.dump(doc, open(OUT, 'w'), indent=1)
    print('\nwrote %s' % OUT)


if __name__ == '__main__':
    main()
