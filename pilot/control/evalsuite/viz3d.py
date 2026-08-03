"""Fly a few seeds and render the courses and flown paths as one offline 3-D page.

    python pilot/control/evalsuite/viz3d.py --policy none --seeds 0-3
    python pilot/control/evalsuite/viz3d.py --policy baseline --seeds 0-5 --open
    python pilot/control/evalsuite/viz3d.py --policy rl --ckpt run1_s5046272 --seeds 0-11

`run1` sat at 0% completion for five million steps and the baseline stalls around gate 2
of 18-22. The only inspection tool for that was `_probe_baseline.py`, which prints one
line per seed: a cause and a miss vector. That says a gate frame was struck 0.9 m
off-centre. It cannot say what approach put the aircraft there, whether the gate was ever
in camera frame, or how near the path ran to the corridor wall -- and those are exactly
the live hypotheses. This writes the same rollout out as geometry you can turn over.

Output is a single self-contained HTML file with no external references of any kind: the
repo has no plotting dependency, no dependency manifest to add one to, and
`user-vm-cmds.md` warns that unpinned installs here shadow conda's numpy and break
precisely that class of library. So the 3-D is hand-rolled on a canvas and the geometry
travels as JSON inside the page. It opens offline, from a file:// URL, forever.

UNITS AND FRAME, once: metres, seconds, radians. The world is NED with z DOWN, so
altitude is `-z` and `z_ceil` is negative. The body frame is FRD. Detection positions are
body-frame. The page converts to a z-up display frame exactly once, on load.

Three things the surrogate does that the obvious rendering would get wrong:

  * The last `course.RUNOUT` gates are NOT physical. `env._gate_geometry` gates collisions
    on `exists = idx < n_gates` and `detect.tick` gates visibility on the same expression,
    so those gates cannot be hit and cannot be seen. They shape the corridor polyline and
    the bisector normals, nothing else. Drawn dashed, and the legend says so.
  * `info["collision"]` is one flag covering gate frame, floor AND ceiling. The subtype
    has to be re-derived from the position, against the env's own thresholds
    (`env.py:455-456`) -- not the slightly tighter tolerance `_probe_baseline` uses, which
    leaves near-ceiling strikes falling through to "collision?".
  * `run_policy.REASON_FLAGS` and `_probe_baseline` order the termination flags
    differently and will label the same episode two ways. This follows the probe, and
    ships all four raw booleans alongside so the label is never load-bearing.

And the one that bites everybody: `VecSurrogate` auto-resets inside `step()`, so the true
state after a terminating step belongs to the NEXT episode. Everything here is captured
BEFORE the step, and the terminal point is solved along the last segment.
"""

import argparse
import ast
import datetime
import json
import os
import sys
import webbrowser

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pilot.control.evalsuite.run_policy import (                       # noqa: E402
    _Env, add_common_args, add_policy_args, build_config, config_dict,
    make_policy, parse_pair, parse_seeds, steps_for, write_json)
from pilot.control.surrogate import camera, course, vmath              # noqa: E402

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viz3d_template.html")
PAYLOAD_ID = 'id="payload"'

# NED: -z is up. `_probe_baseline.py:57-59` builds the gate's own axes this way;
# `env.py:370` uses the exact negation, but multiplies it by a zero-mean Gaussian, so the
# sign is unobservable there and there is nothing to reconcile. This file is the one that
# has to be consistent, because the page draws with these axes and never picks its own.
UP0 = np.array([0.0, 0.0, -1.0])

CUES = ("gate", "ribbon", "blind")
CAUSE_ORDER = ("FINISHED", "GATE FRAME", "FLOOR", "CEILING", "CORRIDOR", "TIMEOUT",
               "FRAME?", "MAX STEPS", "COURSE ONLY")


# -- small geometry helpers ------------------------------------------------------------
def _unit(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _gate_axes(nrm):
    """Per-gate in-plane axes `(side, up)` for `[S, 3]` travel normals."""
    n = _unit(nrm)
    side = np.cross(np.broadcast_to(UP0, n.shape), n)
    bad = np.linalg.norm(side, axis=-1) < 1e-9          # gate flown straight up; cannot
    side[bad] = np.array([0.0, 1.0, 0.0])               # happen at elev <= 22 deg
    side = _unit(side)
    return side, np.cross(n, side)                      # right-handed, already unit


def _rot_wb(q):
    """Body->world 3x3 for one (w, x, y, z) quaternion, via the surrogate's own algebra."""
    return vmath.quat_to_rot(np.asarray(q, dtype=np.float64).reshape(1, 4))[0]


def _to_body(p, q, pts):
    """World `[..., 3]` -> body, the way `detect.py:99-104` does it: R^T @ (pt - p)."""
    return (np.asarray(pts, dtype=np.float64) - np.asarray(p, dtype=np.float64)) @ _rot_wb(q)


def _seg_frac(a, b, target):
    """Fraction along a->b where the scalar crosses `target`, clamped to [0, 1]."""
    span = b - a
    if abs(span) < 1e-12:
        return 1.0
    return float(np.clip((target - a) / span, 0.0, 1.0))


def _r(a, nd):
    return [round(float(x), nd) for x in np.asarray(a).ravel()]


# -- capture ---------------------------------------------------------------------------
def _capture_world(env, cfg):
    """Everything about the course that does not change during the episode."""
    ts = env.true_state()
    g_pos = np.asarray(ts["gate_pos"], dtype=np.float64)
    g_nrm = _unit(ts["gate_nrm"])
    side, up = _gate_axes(g_nrm)

    # `start_p` anchors corridor segment 0 and is the only privileged read this tool
    # actually requires that `single.true_state()` does not expose. Degrade rather than
    # crash if a refactor moves it: one mean segment back along gate 0's normal is close
    # enough to draw, and the warning says the corridor's first leg is approximate.
    start = None
    vec = getattr(env, "vec", None)
    if vec is not None:
        sp = getattr(vec, "start_p", None)
        if sp is not None:
            start = np.asarray(sp[0], dtype=np.float64).copy()
    approx_start = start is None
    if approx_start:
        start = g_pos[0] - g_nrm[0] * float(getattr(cfg, "seg_len_easy", (30.0,))[0])

    return dict(p=g_pos, n=g_nrm, s=side, u=up, start=start,
                raced=int(ts["n_gates"]), slots=int(len(g_pos)),
                z_ceil=float(ts["z_ceil"]), corridor_r=float(ts["corridor_r"]),
                sphere_r=float(ts["sphere_r"]), approx_start=approx_start)


def _det_row(g, is_fp):
    """One `interface.GateObs` -> the compact record the page draws."""
    flags = (1 * int(bool(g.valid)) + 2 * int(bool(g.normal_valid))
             + 4 * int(bool(g.pose_valid)) + 8 * int(bool(is_fp)))
    return (list(np.asarray(g.pos_body, dtype=np.float64).ravel()),
            float(g.confidence), float(g.staleness_s), float(g.size_px),
            int(g.index), flags)


def capture_episode(policy, cfg, seed, max_steps=None, want_det=True):
    """Fly one seed, recording privileged truth BEFORE every step. Full rate."""
    env = _Env(cfg, seed)
    obs = env.reset()
    if policy is not None:
        policy.reset()
    world = _capture_world(env, cfg)
    ts0 = env.true_state()
    ep = dict(seed=int(seed), world=world, start_gate=int(ts0["active_gate"]),
              n_gates=world["raced"])

    if policy is None:
        ep.update(cause="COURSE ONLY", flags=dict(finished=0, collision=0, corridor=0,
                                                  timeout=0),
                  steps=0, gates=0, t_end=0.0, ret=0.0, rows=[], term=None)
        return ep

    if max_steps is None:
        max_steps = steps_for(cfg)
    vec = getattr(env, "vec", None)
    rows, term, info = [], None, {}
    for i in range(1, int(max_steps) + 1):
        prev = env.true_state()
        act = policy(obs)

        # The gate the race is currently chasing, in body frame, through the real camera
        # model. Computed here rather than in the page so the page's ported projection has
        # something to be checked against -- see `--check`.
        a_idx = min(int(prev["active_gate"]), world["slots"] - 1)
        fov = bool(camera.in_frustum(
            _to_body(prev["p"], prev["q"], world["p"][a_idx])))

        det = []
        if want_det:
            fp = None
            if vec is not None and getattr(vec, "det", None) is not None:
                fp = getattr(vec.det, "is_fp", None)
            for j, g in enumerate(obs.gates):
                det.append(_det_row(g, False if fp is None else bool(fp[0, j])))

        cue = -1
        last = getattr(policy, "last", None)
        if isinstance(last, dict) and last.get("cue") in CUES:
            cue = CUES.index(last["cue"])

        t_gate = float(prev["t_s"])
        if vec is not None and getattr(vec, "t_gate", None) is not None:
            t_gate = float(vec.t_gate[0])

        rows.append(dict(t=float(prev["t_s"]), p=np.asarray(prev["p"], dtype=np.float64),
                         v=np.asarray(prev["v"], dtype=np.float64),
                         q=np.asarray(prev["q"], dtype=np.float64),
                         a=int(prev["active_gate"]), t_gate=t_gate, cue=cue, fov=fov,
                         u=(float(act.roll_rate), float(act.pitch_rate), float(act.thrust)),
                         det=det, sphere_r=float(prev["sphere_r"])))

        dt = float(obs.dt_s)
        obs, info = env.step(act)
        if info.get("done"):
            term = classify_terminal(info, rows[-1], dt, world, cfg)
            break

    if term is None:
        term = dict(kind="MAX STEPS", p=list(rows[-1]["p"]), gate=-1, lat=None,
                    miss_side=None, miss_up=None, sub=None)
        info = dict(info)

    ep.update(cause=term["kind"],
              flags=dict(finished=int(bool(info.get("finished"))),
                         collision=int(bool(info.get("collision"))),
                         corridor=int(bool(info.get("corridor_exit"))),
                         timeout=int(bool(info.get("timeout")))),
              steps=len(rows), t_end=float(rows[-1]["t"]),
              gates=int(info.get("gates_passed", rows[-1]["a"])) - ep["start_gate"],
              ret=float(info.get("episode_return", 0.0)), rows=rows, term=term)
    return ep


def classify_terminal(info, prev, dt, world, cfg):
    """Why the episode ended, and WHERE -- solved on the segment, not extrapolated to it.

    `_probe_baseline.py:33` marks the end at `p + v*dt`, which at 10 m/s and 55 Hz is up
    to ~0.2 m past the event: enough to place the marker on the wrong side of a 0.6 m
    frame ring. The cause logic is the probe's; the position is solved for the actual
    crossing so the picture and the label agree.
    """
    p0 = np.asarray(prev["p"], dtype=np.float64)
    end = p0 + np.asarray(prev["v"], dtype=np.float64) * dt
    r = float(prev["sphere_r"])
    ceil_h = -float(world["z_ceil"])
    out = dict(kind=None, p=list(end), gate=-1, lat=None, miss_side=None, miss_up=None,
               sub=None)

    if bool(info.get("finished")):
        out["kind"] = "FINISHED"
        return out
    if bool(info.get("corridor_exit")):
        out["kind"] = "CORRIDOR"
        return out
    if bool(info.get("timeout")):
        out["kind"] = "TIMEOUT"
        # env.py:484 tests the clocks AFTER the step advances them, and everything here
        # was captured before it. Compare against the post-step values or a gate stall
        # that fires at 11.99 s reads as a wall-clock timeout.
        gt = float(getattr(cfg, "gate_timeout_s", 12.0))
        out["sub"] = "gate stall" if float(prev["t_gate"]) + dt > gt else "wall clock"
        return out
    # `gate_timeout` is its OWN info flag, not part of `timeout` -- env.py split them so
    # PPO would stop bootstrapping a stuck policy with the value of a course it was never
    # going to fly. This branch was written before that split and never updated, so every
    # gate stall fell through to FRAME? below and the dominant ending of a pass-through
    # run got reported as "cause unknown".
    if bool(info.get("gate_timeout")):
        out["kind"] = "TIMEOUT"
        out["sub"] = "gate stall"
        return out
    if not bool(info.get("collision")):
        out["kind"] = "FRAME?"
        return out

    # Floor and ceiling use the env's OWN thresholds (env.py:455-456), expressed in
    # altitude: floor_hit is alt < sphere_r, ceil_hit is alt > ceil_h - sphere_r.
    alt0, alt1 = -p0[2], -end[2]
    if alt1 <= r:
        out["kind"] = "FLOOR"
        out["p"] = list(p0 + (end - p0) * _seg_frac(alt0, alt1, r))
        return out
    if alt1 >= ceil_h - r:
        out["kind"] = "CEILING"
        out["p"] = list(p0 + (end - p0) * _seg_frac(alt0, alt1, ceil_h - r))
        return out

    # Otherwise a gate frame. Only the three slots the env itself tests can collide, and
    # only where the slot is a real raced gate (`exists = idx < n_gates`, env.py:584).
    inner = 0.5 * course.GATE_INNER_M
    outer = 0.5 * float(getattr(cfg, "gate_outer_m", course.GATE_OUTER_M))
    a = int(prev["a"])
    for k in range(max(a, 0), min(a + 3, world["raced"])):
        g, n = world["p"][k], world["n"][k]
        d0, d1 = float((p0 - g) @ n), float((end - g) @ n)
        if not (d0 < 0.0 <= d1):
            continue
        x = p0 + (end - p0) * (-d0 / max(d1 - d0, 1e-9)) - g
        x = x - (x @ n) * n
        lat = float(np.linalg.norm(x))
        if lat + r > inner and lat - r < outer:          # env.py:603, this episode's r
            out.update(kind="GATE FRAME", p=list(g + x), gate=k, lat=lat,
                       miss_side=float(x @ world["s"][k]), miss_up=float(x @ world["u"][k]))
            return out
    out["kind"] = "FRAME?"
    return out


# -- checks ----------------------------------------------------------------------------
def check_episode(ep, cfg, warn):
    """Cheap invariants. These catch a wrong VISUALIZER, not a wrong surrogate."""
    w = ep["world"]
    n, s, u = w["n"], w["s"], w["u"]
    for arr, name in ((n, "n"), (s, "s"), (u, "u")):
        if np.max(np.abs(np.linalg.norm(arr, axis=1) - 1.0)) > 1e-9:
            warn("seed %d: gate axis %s not unit" % (ep["seed"], name))
    dots = [np.max(np.abs(np.einsum('ij,ij->i', a, b)))
            for a, b in ((n, s), (n, u), (s, u))]
    if max(dots) > 1e-9:
        warn("seed %d: gate axes not orthogonal (max |dot| %.2e)" % (ep["seed"], max(dots)))

    lo = float(getattr(cfg, "floor_clear_m", 1.8)) + 0.5 * course.GATE_INNER_M
    hi = (-w["z_ceil"] - float(getattr(cfg, "ceil_clear_m", 1.0))
          - 0.5 * course.GATE_INNER_M)
    alt = -w["p"][:w["raced"], 2]
    if alt.min() < lo - 1e-6 or alt.max() > hi + 1e-6:
        warn("seed %d: raced gate altitude %.2f..%.2f outside band %.2f..%.2f"
             % (ep["seed"], alt.min(), alt.max(), lo, hi))

    if not ep["rows"]:
        return
    # Corridor: the env measures against the CURRENT pair of segments only
    # (env.py:607-615). If this reproduction disagrees on a non-CORRIDOR episode, the
    # tool's corridor model is wrong -- which is the bug that would otherwise ship as a
    # confidently drawn tube in the wrong place.
    p = np.array([r["p"] for r in ep["rows"]])
    a = np.clip(np.array([r["a"] for r in ep["rows"]]), 0, w["slots"] - 1)
    ga, gb = w["p"][a], w["p"][np.clip(a + 1, 0, w["slots"] - 1)]
    prev = np.where((a <= 0)[:, None], w["start"], w["p"][np.clip(a - 1, 0, w["slots"] - 1)])
    d = np.minimum(course.point_segment_distance(p, prev, ga),
                   course.point_segment_distance(p, ga, gb))
    bad = int(np.count_nonzero(d[:-1] > w["corridor_r"] + 1e-6))
    if bad and ep["cause"] != "CORRIDOR":
        warn("seed %d: %d path samples outside the corridor on a %s episode (max %.2f "
             "of %.2f) -- the tool's corridor model disagrees with env._corridor_exit"
             % (ep["seed"], bad, ep["cause"], d[:-1].max(), w["corridor_r"]))


# -- serialisation ---------------------------------------------------------------------
def _detail_indices(ep, stride, tail):
    """Strided, plus every crossing step and the last `tail` -- the frames that answer it."""
    n = len(ep["rows"])
    keep = set(range(0, n, max(int(stride), 1)))
    keep.update(range(max(n - int(tail), 0), n))
    a = [r["a"] for r in ep["rows"]]
    keep.update(i for i in range(n - 1) if a[i + 1] > a[i])
    return sorted(keep)


def _crossings(ep):
    """Where the path actually pierced a gate plane, decomposed in that gate's own axes."""
    w, rows, out = ep["world"], ep["rows"], []
    for i in range(len(rows) - 1):
        k = rows[i]["a"]
        if rows[i + 1]["a"] <= k or k >= w["raced"]:
            continue
        p0, p1, g, n = rows[i]["p"], rows[i + 1]["p"], w["p"][k], w["n"][k]
        d0, d1 = float((p0 - g) @ n), float((p1 - g) @ n)
        x = p0 + (p1 - p0) * (-d0 / max(d1 - d0, 1e-9)) - g
        x = x - (x @ n) * n
        out.append(dict(k=int(k), i=i, t=round(rows[i]["t"], 3),
                        lat=round(float(np.linalg.norm(x)), 3),
                        side=round(float(x @ w["s"][k]), 3),
                        up=round(float(x @ w["u"][k]), 3)))
    return out


def episode_blob(ep, stride, tail, want_det):
    w, rows = ep["world"], ep["rows"]
    blob = dict(seed=ep["seed"], cause=ep["cause"], flags=ep["flags"], gates=ep["gates"],
                start_gate=ep["start_gate"], n_gates=ep["n_gates"], steps=ep["steps"],
                t_end=round(ep["t_end"], 3), ret=round(ep["ret"], 3),
                world=dict(start=_r(w["start"], 2), z_ceil=round(w["z_ceil"], 2),
                           corridor_r=round(w["corridor_r"], 2),
                           sphere_r=round(w["sphere_r"], 3), raced=w["raced"],
                           approx_start=int(w["approx_start"]),
                           g=dict(cx=_r(w["p"][:, 0], 2), cy=_r(w["p"][:, 1], 2),
                                  cz=_r(w["p"][:, 2], 2),
                                  nx=_r(w["n"][:, 0], 4), ny=_r(w["n"][:, 1], 4),
                                  nz=_r(w["n"][:, 2], 4),
                                  sx=_r(w["s"][:, 0], 4), sy=_r(w["s"][:, 1], 4),
                                  sz=_r(w["s"][:, 2], 4),
                                  ux=_r(w["u"][:, 0], 4), uy=_r(w["u"][:, 1], 4),
                                  uz=_r(w["u"][:, 2], 4))))
    if ep["term"] is not None:
        t = dict(ep["term"])
        t["p"] = _r(t["p"], 2)
        for k in ("lat", "miss_side", "miss_up"):
            if t[k] is not None:
                t[k] = round(float(t[k]), 3)
        blob["term"] = t
    if not rows:
        blob["path"] = dict(t=[], x=[], y=[], z=[], spd=[], a=[], u0=[], u1=[], u2=[])
        blob["detail"] = dict(i=[])
        blob["cross"] = []
        return blob

    p = np.array([r["p"] for r in rows])
    blob["path"] = dict(
        t=_r([r["t"] for r in rows], 3), x=_r(p[:, 0], 2), y=_r(p[:, 1], 2),
        z=_r(p[:, 2], 2),
        spd=_r([float(np.linalg.norm(r["v"])) for r in rows], 2),
        a=[int(r["a"]) for r in rows],
        u0=_r([r["u"][0] for r in rows], 3), u1=_r([r["u"][1] for r in rows], 3),
        u2=_r([r["u"][2] for r in rows], 3))
    blob["cross"] = _crossings(ep)

    idx = _detail_indices(ep, stride, tail)
    sel = [rows[i] for i in idx]
    detail = dict(i=idx,
                  qw=_r([r["q"][0] for r in sel], 4), qx=_r([r["q"][1] for r in sel], 4),
                  qy=_r([r["q"][2] for r in sel], 4), qz=_r([r["q"][3] for r in sel], 4),
                  tg=_r([r["t_gate"] for r in sel], 2),
                  cue=[int(r["cue"]) for r in sel],
                  fov=[int(bool(r["fov"])) for r in sel])
    if want_det and sel and sel[0]["det"]:
        detail["det"] = [
            dict(x=_r([r["det"][j][0][0] for r in sel], 2),
                 y=_r([r["det"][j][0][1] for r in sel], 2),
                 z=_r([r["det"][j][0][2] for r in sel], 2),
                 c=_r([r["det"][j][1] for r in sel], 3),
                 s=_r([r["det"][j][2] for r in sel], 3),
                 px=_r([r["det"][j][3] for r in sel], 1),
                 gi=[int(r["det"][j][4]) for r in sel],
                 f=[int(r["det"][j][5]) for r in sel])
            for j in range(len(sel[0]["det"]))]
    blob["detail"] = detail
    return blob


def build_blob(episodes, args, name, cfg, stride, tail, want_det):
    meta = dict(policy=name, ckpt=args.ckpt, difficulty=args.difficulty,
                speed_cap=args.speed_cap, normal_starts=int(not args.random_starts),
                detail_stride=int(stride), title=args.title or name,
                gate_inner_m=course.GATE_INNER_M,
                gate_outer_m=float(getattr(cfg, "gate_outer_m", course.GATE_OUTER_M)),
                cam=dict(fx=camera.FX, fy=camera.FY, cx=camera.CX, cy=camera.CY,
                         w=camera.W_PX, h=camera.H_PX,
                         # The 20 deg UP tilt puts body-forward BELOW image centre, at
                         # v = 296 of 360. camera.py calls that "the whole trap"; it ships
                         # explicitly so the page cannot quietly re-derive it wrong.
                         fwd_v=round(camera.CY + camera.FY
                                     * np.tan(np.radians(camera.TILT_DEG)), 1),
                         R_CB=[[round(float(v), 12) for v in row] for row in camera.R_CB]))
    if not args.no_timestamp:
        meta["generated"] = datetime.datetime.now().replace(microsecond=0).isoformat()
    if args.full_config:
        meta["config"] = config_dict(cfg)
    return dict(meta=meta,
                episodes=[episode_blob(e, stride, tail, want_det) for e in episodes])


def render_html(blob, out_path, template=TEMPLATE):
    """Substitute the payload into the template. The result references nothing external."""
    if not os.path.isfile(template):
        raise SystemExit("missing template %s -- it ships beside this script" % template)
    with open(template, "r", encoding="utf-8") as fh:
        html = fh.read()
    hits = [ln for ln in html.splitlines() if PAYLOAD_ID in ln]
    if len(hits) != 1:
        raise SystemExit("template %s must contain exactly one %s line (found %d)"
                         % (template, PAYLOAD_ID, len(hits)))
    # `<\/` is a legal JSON string escape and cannot terminate the script element, which
    # is the one way an embedded payload can break out of its own tag.
    text = json.dumps(blob, separators=(",", ":")).replace("</", "<\\/")
    line = '<script %s type="application/json">%s</script>' % (PAYLOAD_ID, text)
    html = html.replace(hits[0], line)
    d = os.path.dirname(os.path.abspath(out_path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_path


# -- reporting -------------------------------------------------------------------------
def print_table(episodes):
    print("%5s %-11s %5s %5s %7s %6s %6s %-28s %s"
          % ("seed", "why", "gate", "need", "t_s", "alt", "ceil", "miss at the plane",
             "cue g/r/b"))
    for ep in episodes:
        w, t = ep["world"], ep["term"]
        alt = -ep["rows"][-1]["p"][2] if ep["rows"] else 0.0
        miss = ""
        if t and t.get("gate", -1) >= 0:
            miss = "g%d side%+.2f up%+.2f |%.2f|" % (t["gate"], t["miss_side"],
                                                     t["miss_up"], t["lat"])
        elif t and t.get("sub"):
            miss = t["sub"]
        cues = [0, 0, 0]
        for r in ep["rows"]:
            if 0 <= r["cue"] < 3:
                cues[r["cue"]] += 1
        print("%5d %-11s %5d %5d %7.2f %6.2f %6.2f %-28s g%d/r%d/b%d"
              % (ep["seed"], ep["cause"], ep["gates"], ep["n_gates"] - ep["start_gate"],
                 ep["t_end"], alt, -w["z_ceil"], miss, cues[0], cues[1], cues[2]))
    done = sum(1 for e in episodes if e["cause"] == "FINISHED")
    tally = ", ".join("%s %d" % (c, sum(1 for e in episodes if e["cause"] == c))
                      for c in CAUSE_ORDER
                      if any(e["cause"] == c for e in episodes))
    print("\nfinished %d/%d | gates mean %.2f | %s"
          % (done, len(episodes), np.mean([e["gates"] for e in episodes]) if episodes
             else 0.0, tally))


# -- CLI -------------------------------------------------------------------------------
def _default_out(args, name):
    stem = "rollout3d-%s" % args.policy
    if args.policy == "rl" and args.ckpt:
        stem += "-%s" % os.path.splitext(os.path.basename(args.ckpt))[0]
    return os.path.join(_ROOT, "pilot", "evidence",
                        "%s-%s.html" % (datetime.date.today().isoformat(), stem))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 conflict_handler="resolve")
    add_policy_args(ap)
    # `add_policy_args` pins choices to baseline|rl; "none" draws courses with no drone.
    ap.add_argument("--policy", default="baseline", choices=("baseline", "rl", "none"))
    add_common_args(ap)
    ap.add_argument("--out", default=None, help="output HTML (default pilot/evidence/)")
    ap.add_argument("--open", dest="open_", action="store_true", help="open when written")
    ap.add_argument("--detail-stride", type=int, default=3)
    ap.add_argument("--tail-steps", type=int, default=40)
    ap.add_argument("--random-starts", action="store_true",
                    help="keep the training start mix; default forces normal starts")
    ap.add_argument("--max-episodes", type=int, default=24)
    ap.add_argument("--no-detections", action="store_true")
    ap.add_argument("--only-cause", default=None, help="keep only this ending, e.g. FLOOR")
    ap.add_argument("--title", default=None)
    ap.add_argument("--no-timestamp", action="store_true")
    ap.add_argument("--full-config", action="store_true")
    ap.add_argument("--no-check", action="store_true", help="skip the geometry invariants")
    # Arbitrary EnvConfig overrides, python literals, same spelling as train.py's.
    # Needed because a checkpoint can be trained under env settings this tool had no flag
    # for -- `gate_contact_terminates=False` and `gate_inner_scale>1` in particular. A
    # pass-through policy rendered with terminal gates ends every episode at its first
    # clip, which is the one thing it was trained not to do, and the picture is a lie.
    ap.add_argument("--env-kwarg", action="append", default=[], metavar="KEY=VALUE",
                    help="EnvConfig override, e.g. gate_inner_scale=1.45 (repeatable)")
    args = ap.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    # vq2_frac MUST match training (1.0 for a VQ2-only checkpoint). Default normal
    # starts: the training mix can spawn mid-course / outside the corridor and dies
    # on step 1, which makes the pictures unreadable.
    cfg = build_config(args.difficulty, args.speed_cap, parse_pair(args.decision_hz),
                       parse_pair(args.n_gates), args.time_penalty,
                       vq2_frac=args.vq2_frac, random_starts=args.random_starts)
    for item in args.env_kwarg:
        k, _, v = str(item).partition("=")
        k = k.strip()
        if not hasattr(cfg, k):
            raise SystemExit(f"viz3d: EnvConfig has no field {k!r}")
        try:
            val = ast.literal_eval(v.strip())
        except (ValueError, SyntaxError):
            val = v.strip()
        setattr(cfg, k, val)
        print(f"[viz3d] env override: {k} = {val!r}")

    policy, name = (None, "none")
    if args.policy != "none":
        policy, name = make_policy(args.policy, args.ckpt, args.gains, args.supervisor,
                                   args.device)
    print("%s | difficulty %.2f speed_cap %.2f | %d seed(s)%s"
          % (name, args.difficulty, args.speed_cap, len(seeds),
             "" if args.random_starts else " | normal starts"))

    warnings = []
    episodes = []
    for s in seeds:
        ep = capture_episode(policy, cfg, s, args.max_steps, not args.no_detections)
        if args.only_cause and ep["cause"] != args.only_cause:
            continue
        if not args.no_check:
            check_episode(ep, cfg, warnings.append)
        episodes.append(ep)
        if len(episodes) >= args.max_episodes:
            left = len(seeds) - seeds.index(s) - 1
            if left:
                print("  --max-episodes %d reached; %d later seed(s) not flown"
                      % (args.max_episodes, left))
            break

    if not episodes:
        raise SystemExit("no episodes to render"
                         + (" (--only-cause %s matched nothing)" % args.only_cause
                            if args.only_cause else ""))
    if not args.quiet and args.policy != "none":
        print_table(episodes)
    for w in warnings:
        print("  geometry warning: %s" % w)

    blob = build_blob(episodes, args, name, cfg, args.detail_stride, args.tail_steps,
                      not args.no_detections)
    out = args.out or _default_out(args, name)
    render_html(blob, out)
    print("\nwrote %s (%.0f kB, %d episode(s))"
          % (out, os.path.getsize(out) / 1024.0, len(episodes)))
    write_json(args.json, blob)
    if args.open_:
        webbrowser.open("file://" + os.path.abspath(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
