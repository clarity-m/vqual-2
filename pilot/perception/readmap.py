"""Turn Claire's hand-drawn aerial map into a machine-readable gate map.

`map-aerial/approx_map.png` is a top-down sketch built by hand from aerial captures: two
rows of numbered Station columns, and one red bar per gate giving its position AND its
orientation in the hangar plane. 17 bars, matching the gate count.

This is the piece vision could not supply. mapbuild.py can measure inter-gate DISTANCES
well but cannot say which gate is which, because identity needs either a surviving track
or a race-packet crossing, and VQ2 recordings have neither past gate 1. The sketch carries
identity directly, keyed to the station grid the camera can already read.

Coordinates come out in STATION UNITS, not pixels:
    along  -- station number, interpolated (12.0 is level with Station 12)
    across -- 0 at the left row, 1 at the right row

Station units are chosen because they are what the camera can measure: stations.py reads
the numbers, so a gate at along=24.3 is directly comparable to a live reading. Converting
to metres needs the station pitch, which is NOT in the sketch -- see fit_scale().

    python3 pilot/perception/readmap.py --out pilot/perception/map_approx.json
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

# Left row counts DOWN the image, right row counts UP; a facing pair sums to 41. Confirmed
# on the sketch (12|29 ... 20|21) and independently from the camera in stations.py.
SUM_INVARIANT = 41


def find_stations(img):
    """The grey label boxes -> one centre per station, split into two rows."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = ((g > 80) & (g < 140)).astype(np.uint8) * 255
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if w < 15 or h < 15 or a < 200:
            continue
        if not (0.6 < w / h < 1.7):        # the boxes are square-ish
            continue
        boxes.append((float(cent[i][0]), float(cent[i][1])))
    if len(boxes) < 4:
        raise SystemExit(f'found {len(boxes)} station boxes, expected many more')
    xs = np.array([b[0] for b in boxes])
    split = (xs.min() + xs.max()) / 2
    left = sorted([b for b in boxes if b[0] < split], key=lambda b: b[1])
    right = sorted([b for b in boxes if b[0] >= split], key=lambda b: b[1])
    if len(left) != len(right):
        raise SystemExit(f'rows are uneven: {len(left)} left, {len(right)} right')
    return left, right


def find_gates(img):
    """The red bars -> centre, orientation, length."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m = (((h <= 12) | (h >= 168)) & (s > 90) & (v > 90)).astype(np.uint8) * 255
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        if stats[i][4] < 30:
            continue
        pts = np.column_stack(np.where(lab == i))[:, ::-1].astype(np.float32)
        mean = pts.mean(0)
        u, sv, vt = np.linalg.svd(pts - mean, full_matrices=False)
        d = vt[0]
        out.append({'px': float(mean[0]), 'py': float(mean[1]),
                    # Bar direction is the gate's PLANE, so its normal is perpendicular.
                    # Stored as the plane angle; the caller decides which it wants.
                    'plane_deg': float(np.degrees(np.arctan2(d[1], d[0])) % 180.0),
                    'len_px': float(2 * sv[0] / np.sqrt(len(pts)))})
    return sorted(out, key=lambda g: g['py'])


def build(map_png, first_left):
    img = cv2.imread(map_png)
    if img is None:
        raise SystemExit(f'cannot read {map_png}')
    left, right = find_stations(img)
    nrow = len(left)
    # Left row descends from `first_left`; the right row is fixed by the sum invariant, so
    # it is DERIVED rather than read -- and the derivation is checked against the geometry
    # below, which is the whole point of having an invariant.
    left_ids = [first_left + i for i in range(nrow)]
    right_ids = [SUM_INVARIANT - i for i in left_ids]

    ys = np.array([p[1] for p in left])
    pitch = float(np.median(np.diff(ys)))
    x_left = float(np.median([p[0] for p in left]))
    x_right = float(np.median([p[0] for p in right]))
    # Rows must be level with each other, or the sum invariant is being applied to columns
    # that do not actually face one another.
    skew = float(np.max(np.abs(np.array([p[1] for p in left]) -
                               np.array([p[1] for p in right]))))
    if skew > 0.3 * pitch:
        raise SystemExit(f'rows are not level (max offset {skew:.0f} px vs pitch {pitch:.0f})')

    gates = []
    for g in find_gates(img):
        along = left_ids[0] + (g['py'] - ys[0]) / pitch
        across = (g['px'] - x_left) / (x_right - x_left)
        gates.append({'along_station': round(along, 3), 'across': round(across, 3),
                      'plane_deg': round(g['plane_deg'], 1),
                      'span_px': round(g['len_px'], 1),
                      'px': round(g['px'], 1), 'py': round(g['py'], 1)})
    return {
        'source': os.path.basename(map_png),
        'frame': {'along': 'station number, interpolated', 'across': '0 = left row, 1 = right row',
                  'plane_deg': 'gate plane bearing in the hangar plane, 0..180'},
        'station_pitch_px': round(pitch, 2),
        'row_separation_px': round(x_right - x_left, 2),
        'metres_per_station': None,
        'stations': {'left': dict(zip(map(str, left_ids), [round(p[1], 1) for p in left])),
                     'right': dict(zip(map(str, right_ids), [round(p[1], 1) for p in right]))},
        'n_gates': len(gates),
        # Claire's own caveat, kept with the data rather than in a commit message: the ORDER
        # and the overall curve are certain (cross-referenced against video of a completed
        # lap), the positions and plane angles are rough, and there is NO vertical component
        # at all -- this is a top-down sketch. Treat it as a prior for pre-turning and
        # attention, never as terminal guidance.
        'accuracy': {'order': 'certain', 'topology': 'certain',
                     'positions': 'approximate', 'plane_angles': 'approximate',
                     'vertical': 'ABSENT — the sketch is top-down only'},
        'n_gates': len(gates),
        'gates': gates,
    }


def race_order(map_png, gates):
    """Race order, by arc length along the drawn course line.

    Claire added the flown path to the sketch (`approx_map_path.png`), cross-referenced
    against video of a completed lap. That line is the only statement of race order we have,
    and it is authoritative in ORDER even though she is explicit that positions and angles
    are rough and vertical height is absent entirely.

    Order comes from a GEODESIC distance along the line, not from sorting by y and not from
    chaining nearest neighbours. Both of those break on this course: it doubles back on
    itself around the Station 22 and 26 loops, so two points that are close in the image can
    be far apart along the path, and a y-sort would interleave the two halves of a loop.
    A breadth-first flood from the start end measures distance THROUGH the line, which is
    exactly the quantity that orders gates.
    """
    import heapq

    img = cv2.imread(map_png)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    line = ((h > 95) & (h < 130) & (s > 80) & (v > 80))
    ys, xs = np.where(line)
    if ys.size < 100:
        raise SystemExit('no course line found')
    # Start at the BOTTOM end: the lap enters low (past Station 20/21) and exits top.
    start = (int(ys.max()), int(xs[np.argmax(ys)]))

    INF = np.inf
    dist = np.full(line.shape, INF)
    dist[start] = 0.0
    pq = [(0.0, start[0], start[1])]
    nbr = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
           (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
    H, W = line.shape
    while pq:
        d, y, x = heapq.heappop(pq)
        if d > dist[y, x]:
            continue
        for dy, dx, w in nbr:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and line[ny, nx] and d + w < dist[ny, nx]:
                dist[ny, nx] = d + w
                heapq.heappush(pq, (d + w, ny, nx))
    reached = np.isfinite(dist) & line
    if reached.sum() < 0.9 * line.sum():
        raise SystemExit(f'course line is broken: flood reached {reached.sum()} of '
                         f'{line.sum()} pixels. Order would be wrong -- fix the drawing.')

    pts = np.column_stack(np.where(reached))          # (y, x)
    dvals = dist[reached]
    arcs = []
    for i, g in enumerate(gates):
        d2 = (pts[:, 0] - g['py']) ** 2 + (pts[:, 1] - g['px']) ** 2
        j = int(np.argmin(d2))
        arcs.append((float(dvals[j]), float(np.sqrt(d2[j])), i))
    arcs.sort()
    return arcs, float(dist[reached].max())


def map_pair_distances(m):
    """All inter-gate distances implied by the sketch, in station units."""
    k = m['row_separation_px'] / m['station_pitch_px']    # across-units -> station-units
    P = np.array([[g['along_station'], g['across'] * k] for g in m['gates']])
    iu = np.triu_indices(len(P), 1)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    return D, D[iu]


def fit_scale(m, edges_json, lo=2.0, hi=60.0, n=600):
    """Metres per station, from vision's measured inter-gate distances.

    The sketch has no scale and the recordings have no identity, but the two are
    complementary: mapbuild.py measures inter-gate distances in METRES without knowing
    which gates they join, and the sketch knows every gate pair's separation in STATION
    UNITS without knowing metres. One unknown scalar relates them.

    So: sweep the scale, and for each measured distance take the closest sketch pair
    distance. The residual is scored RELATIVE to the measurement, because an absolute
    residual is minimised trivially by shrinking the scale until every sketch distance
    sits near zero.

    This is an assignment-free fit -- it never has to decide which gate is which, which is
    exactly the thing that cannot yet be decided. Its honesty rests entirely on the sweep
    having a sharp minimum; a flat curve means the sketch distances are dense enough to
    absorb any scale, and the answer would be meaningless. main() prints the curve.
    """
    ed = json.load(open(edges_json))['edges']
    # Well-measured edges only: an edge whose own scatter is large cannot pin a scale.
    d = np.array([e[2] for e in ed if e[4] < 2.5])
    if d.size < 20:
        raise SystemExit(f'only {d.size} tight edges; need more to fit a scale')
    _D, pairs_u = map_pair_distances(m)
    pairs_u = pairs_u[pairs_u > 1e-6]
    scales = np.linspace(lo, hi, n)
    cost = []
    for s in scales:
        pred = s * pairs_u
        resid = np.abs(d[:, None] - pred[None, :]).min(axis=1) / d
        cost.append(float(np.median(resid)))
    cost = np.array(cost)
    best = int(np.argmin(cost))
    # DEGENERACY TEST, and it fires on the real data. 17 gates give 136 pair distances,
    # which form a near-continuum: almost any measured distance lands close to SOME sketch
    # pair at almost any scale, so the minimum is meaningless unless it is sharp. Measure
    # sharpness as the share of the swept range that is nearly as good as the best.
    near = cost < 2.0 * cost[best]
    breadth = float(near.mean())
    return {'scales': scales, 'cost': cost, 'best': float(scales[best]),
            'best_cost': float(cost[best]), 'n_edges': d.size, 'breadth': breadth,
            'degenerate': breadth > 0.2,
            'plateau': [float(scales[near].min()), float(scales[near].max())]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default='map-aerial/approx_map.png')
    ap.add_argument('--first-left', type=int, default=12,
                    help='station number of the TOPMOST left-row box')
    ap.add_argument('--expect', type=int, default=17)
    ap.add_argument('--path', default='',
                    help='sketch WITH the flown course line drawn on it; supplies race order. '
                         'Gates are still read from --map, because the line crosses each bar '
                         'and splits it into two components (30 "gates" instead of 17).')
    ap.add_argument('--fit-scale', default='', help='edges JSON from mapbuild cache')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    m = build(args.map, args.first_left)
    print(f'stations : {len(m["stations"]["left"])} per row, pitch {m["station_pitch_px"]} px, '
          f'separation {m["row_separation_px"]} px')
    print(f'  left  {list(m["stations"]["left"])}')
    print(f'  right {list(m["stations"]["right"])}')
    print(f'gates    : {m["n_gates"]}' +
          ('' if m['n_gates'] == args.expect else f'  <-- EXPECTED {args.expect}'))
    for g in m['gates']:
        print('  along %6.2f  across %5.2f  plane %5.1f deg' %
              (g['along_station'], g['across'], g['plane_deg']))
    if args.path:
        arcs, total = race_order(args.path, m['gates'])
        far = [(r, off) for r, (_a, off, _i) in enumerate(arcs) if off > 25]
        print(f'\nrace order from {os.path.basename(args.path)}  (path {total:.0f} px)')
        print('  race  arc   off    along  across  plane')
        for r, (a, off, i) in enumerate(arcs):
            g = m['gates'][i]
            m['gates'][i]['race_index'] = r
            m['gates'][i]['arc_px'] = round(a, 1)
            print('  %4d %5.0f %5.0f   %6.2f %6.2f  %5.1f'
                  % (r, a, off, g['along_station'], g['across'], g['plane_deg']))
        if far:
            print(f'  NOTE gates sitting >25 px off the line: {far} '
                  '— their place in the order is the least certain')
        m['gates'] = sorted(m['gates'], key=lambda g: g['race_index'])
        m['race_order'] = {
            'source': os.path.basename(args.path),
            'derived': 'geodesic arc length along the drawn course line, flooded from the '
                       'bottom (entry) end. Not a y-sort: the course doubles back around the '
                       'Station 22 and 26 loops, which a y-sort would interleave.',
            'confirmed_landmarks': 'loop around 22, then 16+17, then 26, then 13, then exit',
        }

    if args.fit_scale:
        f = fit_scale(m, args.fit_scale)
        scales, cost = f['scales'], f['cost']
        print(f'\nscale fit against {f["n_edges"]} tight measured edges')
        step = max(1, len(scales) // 28)
        for i in range(0, len(scales), step):
            bar = '#' * int(60 * (1 - (cost[i] - cost.min()) / (cost.max() - cost.min())))
            print(f'  {scales[i]:5.1f} m  {cost[i]:.3f} {bar}')
        if f['degenerate']:
            print(f'\n  DEGENERATE — scales from {f["plateau"][0]:.1f} to {f["plateau"][1]:.1f} '
                  f'm/station are all within 2x of the best cost '
                  f'({100 * f["breadth"]:.0f}% of the swept range).')
            print('  17 gates give 136 pair distances, a near-continuum, so any measured\n'
                  '  distance matches SOME pair at almost any scale. The apparent optimum\n'
                  f'  ({f["best"]:.1f} m) is noise. metres_per_station stays null.')
            m['metres_per_station'] = None
        else:
            print(f'\n  best {f["best"]:.2f} m/station, residual {f["best_cost"]:.3f}')
            m['metres_per_station'] = round(f['best'], 2)
        m['scale_fit'] = {'degenerate': f['degenerate'], 'n_edges': int(f['n_edges']),
                          'apparent_best': round(f['best'], 2),
                          'plateau': [round(x, 1) for x in f['plateau']]}

    if args.out:
        json.dump(m, open(args.out, 'w'), indent=1)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
