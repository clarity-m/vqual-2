"""build_course.py -- generate pilot/course/course_vq2.json from the perception map.

Run:  python3 pilot/course/build_course.py

Reads  ../perception/map_vq2.json  (layout_directions_2026_08_02 + measured_pairs)
       ../perception/gatetilt_rows.json  (per-detection gate-plane tilt, optional)
Writes course_vq2.json

Nothing here re-litigates a measurement. It reshapes the map into an interchange file
and attaches an UNCERTAINTY MODEL calibrated against the map's own leave-one-out
errors, so the control side can domain-randomize at the size of our real error.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PERC = os.path.normpath(os.path.join(HERE, '..', 'perception'))
MAP = os.path.join(PERC, 'map_vq2.json')
OUT = os.path.join(HERE, 'course_vq2.json')

N_GATES = 17
INNER_M = 1.5
OUTER_M = 2.7

# ---- uncertainty model ----------------------------------------------------------------
# Calibration (see README): the map's own leave-one-out check, run WITH directions on the
# 8 edges that sit inside a braced loop, gives |d_err| rms 0.60 m and |bearing_err| rms
# 1.68 deg. A LOO error is (measurement error) + (prediction error from the rest of the
# graph), so a single edge's own sigma is ~ err/sqrt(2): 0.42 m and 1.19 deg. The formal
# standard error of each pair median (1.4826*MAD/sqrt(n)) is 0.02-0.44 m -- i.e. SMALLER
# than the observed spread for almost every edge. The floor, not the row count, is what
# the evidence supports, so the floor is the model and n only ever makes it worse.
SIGMA_D_FLOOR = 0.40         # m
SIGMA_B_FLOOR = 1.20         # deg
SIGMA_DZ_FLOOR = 0.15        # m
THIN_N = 15                  # edges below this row count get the thin penalty
THIN_MULT = 1.5
ONE_SESSION_MULT = 1.3
BRIDGE_MULT = 1.5            # no loop can check a bridge edge -- honest inflation
N_MC = 20000


def load_map():
    return json.load(open(MAP))


def edge_table(mapj):
    L = mapj['layout_directions_2026_08_02']
    MP = mapj['measured_pairs']
    BR = L['pair_bearings_grid_deg']
    DZ = L['pair_dz_up_m']
    loo = {tuple(r['pair']): r for r in L['residuals']['loo_with_directions']}

    edges = []
    for key, b in BR.items():
        a, c = (int(v) for v in key.split('-'))
        mp = MP[key]
        dz = DZ[key]
        bridge = bool(loo[(a, c)]['bridge'])
        n, mad = int(mp['n']), float(mp['mad_m'])
        ns = int(b['n_sessions'])

        mult = 1.0
        flags = []
        if n < THIN_N:
            mult *= THIN_MULT
            flags.append('thin(n=%d)' % n)
        if ns == 1:
            mult *= ONE_SESSION_MULT
            flags.append('single-session')
        if bridge:
            mult *= BRIDGE_MULT
            flags.append('bridge(no loop check)')
        if mp.get('gatenet_only'):
            flags.append('single-channel')

        sd = max(SIGMA_D_FLOOR, 1.4826 * mad / math.sqrt(max(n, 1))) * mult
        xs = b['cross_session_spread_deg']
        sb = max(SIGMA_B_FLOOR, float(b['row_mad_deg']),
                 float(xs) if xs is not None else 0.0) * (
                     ONE_SESSION_MULT if ns == 1 else 1.0)
        sz = max(SIGMA_DZ_FLOOR,
                 1.4826 * float(dz['mad']) / math.sqrt(max(int(dz['n']), 1))) * mult

        if key == '1-2':
            # Not Gaussian: a REFUSED 34-row channel reads 13.0 m against the accepted
            # 8.32 m and the map calls this its least-trusted edge. Widened by hand and
            # exported with the rival value so a consumer can sample the alternative.
            sd = 1.20
            flags.append('CONTESTED: rival hypothesis 13.0 m (refused, see map status)')

        e = {
            'pair': [a, c],
            'd_horiz_m': round(float(mp['horiz_m']), 3),
            'd_3d_m': round(float(mp['dist_m']), 3),
            'mad_m': round(mad, 3),
            'n_rows': n,
            'n_sessions': ns,
            'bearing_deg': round(float(b['bearing_sketch_frame']), 3),
            'bearing_row_mad_deg': round(float(b['row_mad_deg']), 3),
            'bearing_cross_session_spread_deg': (round(float(xs), 3)
                                                 if xs is not None else None),
            'dz_up_m': round(float(dz['dz']), 3),
            'dz_mad_m': round(float(dz['mad']), 3),
            'dz_n': int(dz['n']),
            'bridge': bridge,
            'sigma_d_m': round(sd, 3),
            'sigma_bearing_deg': round(sb, 3),
            'sigma_dz_m': round(sz, 3),
            'flags': flags,
        }
        if not bridge:
            e['loo_d_err_m'] = round(float(loo[(a, c)]['d_err']), 3)
            e['loo_bearing_err_deg'] = round(float(loo[(a, c)]['b_err']), 3)
        if key == '1-2':
            e['alt_hypothesis_d_m'] = 13.0
        edges.append(e)
    edges.sort(key=lambda e: (e['pair'][0], e['pair'][1]))
    return edges


# ---- the chain solve (shared with course_vq2.py) ---------------------------------------


def solve_layout(edges, d, bear_deg, dz):
    """Weighted least squares: p_b - p_a = d * u(bearing); gate 0 pinned at the origin.

    Same linear form the map's own direction solve uses. Overdetermined wherever the
    graph has a loop, which is what keeps a sampled course self-consistent.
    """
    idx = {g: g - 1 for g in range(1, N_GATES)}       # gate 0 is the gauge
    m, nun = len(edges), N_GATES - 1
    A = np.zeros((2 * m, 2 * nun))
    bvec = np.zeros(2 * m)
    Az = np.zeros((m, nun))
    bz = np.zeros(m)
    for i, e in enumerate(edges):
        a, c = e['pair']
        th = math.radians(bear_deg[i])
        v = d[i] * np.array([math.cos(th), math.sin(th)])
        w = 1.0 / math.hypot(e['sigma_d_m'],
                             d[i] * math.radians(e['sigma_bearing_deg']))
        for k in (0, 1):
            if a != 0:
                A[2 * i + k, 2 * idx[a] + k] = -w
            if c != 0:
                A[2 * i + k, 2 * idx[c] + k] = w
            bvec[2 * i + k] = w * v[k]
        wz = 1.0 / e['sigma_dz_m']
        if a != 0:
            Az[i, idx[a]] = -wz
        if c != 0:
            Az[i, idx[c]] = wz
        bz[i] = wz * dz[i]
    xy = np.linalg.lstsq(A, bvec, rcond=None)[0].reshape(nun, 2)
    z = np.linalg.lstsq(Az, bz, rcond=None)[0]
    P = np.zeros((N_GATES, 3))
    P[1:, :2] = xy
    P[1:, 2] = z
    return P


def montecarlo(edges, nominal, n=N_MC, seed=0):
    rng = np.random.default_rng(seed)
    d0 = np.array([e['d_horiz_m'] for e in edges])
    b0 = np.array([e['bearing_deg'] for e in edges])
    z0 = np.array([e['dz_up_m'] for e in edges])
    sd = np.array([e['sigma_d_m'] for e in edges])
    sb = np.array([e['sigma_bearing_deg'] for e in edges])
    sz = np.array([e['sigma_dz_m'] for e in edges])
    base = solve_layout(edges, d0, b0, z0)
    acc = np.zeros((n, N_GATES, 3))
    for i in range(n):
        P = solve_layout(edges, d0 + rng.normal(0, sd), b0 + rng.normal(0, sb),
                         z0 + rng.normal(0, sz))
        acc[i] = nominal + (P - base)
    return acc


# ---- gate plane angles -----------------------------------------------------------------


GRID_MOD90 = None            # filled from the map


def plane_block(mapj, nominal):
    L = mapj['layout_directions_2026_08_02']
    GA = L['gate_plane_angles_deg']
    grid = L['frame']['grid_to_sketch_rotation_deg'] % 90.0
    out = {}
    for g in range(N_GATES):
        a = GA.get(str(g))
        # race-line bisector: the direction a gate would face if it faced the racing line
        if g == 0:
            u_in = None
        else:
            u_in = nominal[g, :2] - nominal[g - 1, :2]
            u_in = u_in / np.linalg.norm(u_in)
        if g == N_GATES - 1:
            u_out = None
        else:
            u_out = nominal[g + 1, :2] - nominal[g, :2]
            u_out = u_out / np.linalg.norm(u_out)
        bb = u_in if u_out is None else u_out if u_in is None else (u_in + u_out)
        bb = bb / np.linalg.norm(bb)
        bis = math.degrees(math.atan2(bb[1], bb[0])) % 180.0
        rec = {'yaw_race_bisector_deg': round(bis, 2)}
        if a is None:
            rec.update({'yaw_deg': None, 'yaw_mad_deg': None, 'yaw_n': 0,
                        'yaw_grid_bin_deg': None, 'yaw_grid_dev_deg': None,
                        'yaw_status': 'REFUSED by the map (IPPE head-on valley, '
                                      'MAD > 20 deg) -- use the bisector or randomize'})
        else:
            az = float(a['normal_az_sketch_frame_mod180'])
            k = round((az - grid) / 90.0)
            binv = (grid + 90.0 * k) % 180.0
            dev = (az - binv + 90.0) % 180.0 - 90.0
            rec.update({'yaw_deg': round(az, 2),
                        'yaw_mad_deg': round(float(a['mad_deg']), 2),
                        'yaw_n': int(a['n']),
                        'yaw_grid_bin_deg': round(binv, 2),
                        'yaw_grid_dev_deg': round(dev, 2),
                        'yaw_status': 'measured (PnP normal + skylight compass, mod 180)'})
        out[g] = rec
    return out, grid


def main():
    mapj = load_map()
    L = mapj['layout_directions_2026_08_02']
    P = L['positions_m']
    nominal = np.array([[P[str(g)]['x'], P[str(g)]['y'], P[str(g)]['z_up_m']]
                        for g in range(N_GATES)])

    edges = edge_table(mapj)
    acc = montecarlo(edges, nominal)
    sxy = np.sqrt(0.5 * (acc[:, :, 0].var(axis=0) + acc[:, :, 1].var(axis=0)))
    sx = acc[:, :, 0].std(axis=0)
    sy = acc[:, :, 1].std(axis=0)
    sz = acc[:, :, 2].std(axis=0)
    # LOCAL uncertainty: how well the step from the previous gate is known. This is the
    # number a policy actually feels; sigma_xy_m is measured from gate 0 and therefore
    # accumulates down the chain (gauge choice, not extra ignorance).
    dl = acc[:, 1:, :] - acc[:, :-1, :]
    sxy_loc = np.concatenate([[0.0], np.sqrt(
        0.5 * (dl[:, :, 0].var(axis=0) + dl[:, :, 1].var(axis=0)))])
    sz_loc = np.concatenate([[0.0], dl[:, :, 2].std(axis=0)])

    planes, grid = plane_block(mapj, nominal)

    tilt = json.load(open(os.path.join(HERE, 'tilt_measured.json')))

    gates = []
    for g in range(N_GATES):
        rec = {
            'id': g,
            'position_m': [round(float(nominal[g, 0]), 3),
                           round(float(nominal[g, 1]), 3),
                           round(float(nominal[g, 2]), 3)],
            'sigma_xy_m': round(float(sxy[g]), 3),
            'sigma_x_m': round(float(sx[g]), 3),
            'sigma_y_m': round(float(sy[g]), 3),
            'sigma_z_m': round(float(sz[g]), 3),
            'sigma_xy_local_m': round(float(sxy_loc[g]), 3),
            'sigma_z_local_m': round(float(sz_loc[g]), 3),
        }
        rec.update(planes[g])
        rec.update(tilt['per_gate'][str(g)])
        gates.append(rec)

    doc = {
        'schema': 'course_vq2/1',
        'built': '2026-08-02 (edge 10-11 corrected)',
        'source': ('pilot/perception/map_vq2.json -- layout_directions_2026_08_02 '
                   '(positions, bearings, signed dz, gate plane angles) and '
                   'measured_pairs (distances, MAD, n). Vision + gravity + the '
                   'skylight compass only; no pose stream, no sim ground truth.'),
        'frame': {
            'kind': 'RELATIVE. No absolute origin or heading w.r.t. the sim world.',
            'origin': 'gate 0 centre = (0, 0, 0)',
            'handedness': 'right-handed (x, y, z)',
            'x': ('along-hangar. +x points toward INCREASING station number on the '
                  '12-20 column row, i.e. BACK toward the start/pad. The race runs '
                  '0 -> 16 in the -x direction (gate 16 is at x = -217 m). '
                  'corr(x, sketch along_station) = +0.996 over 17 gates.'),
            'y': ('across-hangar. +y points toward the column row numbered 21-29 '
                  '(the "Station 26/27/28" side); -y toward the row numbered 12-20. '
                  'corr(y, sketch across) = +0.929. With z up and travel along -x, '
                  '+y is on the RIGHT of the direction of travel.'),
            'z': 'metres UP, gravity-referenced (accel/gyro filter). gate 0 = 0.',
            'rotation_note': ('these axes are the map\'s sketch-aligned axes: the '
                              'levelled ceiling-grid frame rotated by '
                              'grid_to_sketch_rotation_deg and y-mirrored. The grid '
                              'itself is only known mod 90 deg, so the WHOLE course '
                              'may be rotated by a multiple of 90 deg (and only that) '
                              'relative to the sim world.'),
            'grid_to_sketch_rotation_deg':
                L['frame']['grid_to_sketch_rotation_deg'],
            'grid_axis_mod90_deg': round(grid, 3),
        },
        'race_order': list(range(N_GATES)),
        'aperture': {
            'inner_m': INNER_M,
            'outer_m': OUTER_M,
            'note': ('spec-exact 1500 mm inner square, 2700 mm outer frame. Apparent '
                     'size is a calibrated rangefinder: range_m = 480 / gate_px '
                     '(fx = 320 px at 640x360).'),
        },
        'gates': gates,
        'edges': edges,
        'uncertainty_model': {
            'how': ('per-edge sigma on distance / bearing / signed dz, floors set by '
                    'the map\'s own leave-one-out residuals rather than by row counts '
                    '(the formal standard error of every pair median is smaller than '
                    'the observed spread, so the floor is what the evidence supports). '
                    'Per-gate sigma is then the Monte-Carlo marginal of re-solving the '
                    'whole chain with perturbed edges -- it is a DERIVED quantity, not '
                    'an independent input.'),
            'calibration': {
                'loo_nonbridge_d_err_rms_m': 0.601,
                'loo_nonbridge_bearing_err_rms_deg': 1.677,
                'single_edge_sigma_from_loo_m': round(0.601 / math.sqrt(2), 3),
                'single_edge_sigma_from_loo_deg': round(1.677 / math.sqrt(2), 3),
            },
            'sigma_d_floor_m': SIGMA_D_FLOOR,
            'sigma_bearing_floor_deg': SIGMA_B_FLOOR,
            'sigma_dz_floor_m': SIGMA_DZ_FLOOR,
            'multipliers': {'thin_n_lt_%d' % THIN_N: THIN_MULT,
                            'single_session': ONE_SESSION_MULT,
                            'bridge_edge': BRIDGE_MULT},
            'n_monte_carlo': N_MC,
            'not_modelled': [
                'the mod-90 branch of the ceiling grid: 2-3, 6-7 and 15-16 take their '
                'quadrant from the sketch, not from a shared measurement. If a branch '
                'is wrong that LEG rotates by 90 deg -- a discrete failure, not a sigma.',
                'per-session compass bias: edges measured in one session share it, so '
                'the independent-edge sampler understates whole-leg rotation risk.',
                'the 1-2 rival value (13.0 m) is exported as alt_hypothesis_d_m, not '
                'folded into its sigma.',
            ],
        },
        'structure': tilt['structure'],
        'limits': [
            'RELATIVE MAP ONLY. There is no absolute origin and no absolute heading. '
            'A policy may not consume world coordinates; it may consume gate-relative '
            'geometry and the layout\'s shape.',
            'Bridge edges (0-1, 1-2, 2-3, 3-4, 4-5, 5-6, 6-7, 9-10, 15-16) sit on a '
            'chain with no second path, so nothing in the data can check them. Their '
            'sigma is inflated, not verified.',
            'EDGE 10-11 WAS CORRECTED ON 2026-08-02: it read 33.84 m and was a real '
            'measurement of the WRONG pair (the far detection was gate 12), taken from '
            'a single vantage so the static-pair viewpoint referee never fired. It is '
            '16.49 m, measured in 20260802-180755-vm-strafe-10-11 from four staged '
            'vantages; 10-12 = 34.97 m entered as its own edge and 10-11-12 now closes '
            'as a triangle (0.46 m over a 71 m perimeter). Course length fell from '
            '268.22 m to 250.93 m. Any policy trained on a pre-correction copy of this '
            'file learned a course 17 m too long with gates 11-16 ~15 m out of place. '
            'See map_vq2.json status.CORRECTION_2026_08_02_edge_10_11.',
            '1-2 is the least-trusted edge in the map: 8.32 m on n=5 rows from one '
            'session, against a refused 34-row channel reading 13.0 m.',
            'Thin edges: 14-15 (n=9), 9-10 (n=12), 12-13 (n=14), 12-14 (n=15).',
            'Gate plane YAW is refused by the map for gates 8, 12 and 13 (viewed '
            'head-on, where IPPE\'s rotation is degenerate). Do not rely on an exact '
            'gate plane at those three.',
            'Heights are gravity-referenced and are the most reliable part of this '
            'file (height-solve residual ~0.00 m, per-pair dz MAD 0.03-0.44 m).',
            'The 4-6 edge in earlier versions of the map was REFUSED: it re-measures '
            '4-5 under a wrong name. It is absent here on purpose.',
        ],
    }
    json.dump(doc, open(OUT, 'w'), indent=1)
    print('wrote %s' % OUT)
    print('per-gate sigma_xy: ' + ' '.join('%d:%.2f' % (g, sxy[g])
                                           for g in range(N_GATES)))
    print('per-gate sigma_z : ' + ' '.join('%d:%.2f' % (g, sz[g])
                                           for g in range(N_GATES)))


if __name__ == '__main__':
    main()
