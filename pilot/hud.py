"""
Perception HUD -- see what the aircraft believes, overlaid on the camera view.

This is a debugging instrument and a spectator toy: it renders the whole perception
stack on top of the 640x360 camera frame, so a human flying teleop (or watching a
replay) can SEE detection / pose / heading / map-attention as they happen.

It draws nothing the pilot flies on. Every layer here is a READOUT of an existing module
in pilot/perception -- detect.py, gatenet + the confidence head, skylight.py, and
map_vq2.json -- composited with cheap OpenCV drawing. It never sends a command and never
touches the control path.

LAYERS (each toggleable; keys 1..6 in an interactive window):
  1  contour detections (detect.py): inner quads GREEN, outer-fallback AMBER,
     colour-rejected contours dim RED dots.
  2  gatenet quads on detector-seeded crops, coloured by TRUST (confidence head p at the
     deployed operating point 0.082, AND the PnP snap residual gate): green trusted,
     yellow pose-caution, red rejected. Reads out p, predicted corner error, residual.
     Net quads failing the producer's strict interior-colour decoration test (bright /
     white interior) are dim red with reason "interior" -- never trust-coloured.
  3  PnP: range in metres + the approach-normal arrow, for pose-valid gates.
  4  skylight compass: heading dial (mod 90) + confidence, faint light-quad outlines,
     and the gravity horizon line.
  5  map strip: race order 0..16 with active_gate_index highlighted, plus a small
     top-down minimap and the active->next internal cue (map + compass).
  6  status: per-stage latency, frame staleness, body rate, |accel|-g (attitude trust).

BUDGET. The HUD must DEGRADE, not lag. In live mode a background worker measures each
stage; if the full stack runs over ~25 ms it drops gatenet (then the compass) to every
Nth frame and SAYS SO on the overlay. teleop stays smooth because the worker runs off the
control loop entirely -- the control thread only blits the most recent finished overlay.

MODES:
  * replay:   python3 pilot/hud.py --replay <session-dir> [--frames-out DIR | --out x.mp4]
              [--show] [--preview N] [--start i --end j --stride s] [--layers 123456]
              [--no-net]
  * live:     pilot/teleop.py --hud  (hooks this module's LiveHUD behind the frame path)

Conventions come from CONVENTIONS.md via label.py / skylight.py; this file re-derives no
signs. Gyro reaching here is already canonical (mirror applied); accel is canonical.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PERC = os.path.join(HERE, "perception")
sys.path.insert(0, PERC)

import detect as D          # noqa: E402  contour detector (imports label)
import label as L           # noqa: E402  camera geometry + projection
import skylight as SK       # noqa: E402  ceiling-light compass
# The strict interior-colour decoration test, borrowed VERBATIM from the producer so the
# HUD shows what the producer would decide (real aperture interiors are dark; decoration
# interiors are bright/white -- a wordmark quad must not render as a trusted gate).
from producer import (interior_colour, is_decoration,   # noqa: E402
                      n_corners_in_frame, MIN_CORNERS_IN_FRAME)

RUNS = os.path.join(PERC, "gatenet_runs")
TRUNK_CKPT = os.path.join(RUNS, "colab-v3-rot", "best.pt")   # deployed regressor trunk
HEAD_CKPT = os.path.join(RUNS, "colab-conf-rot", "best.pt")  # paired confidence head
MAP_JSON = os.path.join(PERC, "map_vq2.json")

# Deployed confidence operating point for the rot pairing (TRAINING.md 2026-08-02):
# p >= 0.082 rejects 89.7% of decoration FPs at 0.00% clean-gate loss.
P_OP = 0.082
# PnP snap residual gate. A physically-realizable square projection sits ~0 px from a
# clean gatenet quad; a large raw-vs-snapped residual means the 8 regressed numbers are
# not a consistent pose. SIZE-RELATIVE as of residgate.py (2026-08-02): the old flat
# 0.3 px absolute floor rejected excellent poses on big/close gates (residual scales
# with apparent size -- Claire measured r=0.87 px, 0.17% of gate size, rejected on
# final approach) while a fixed-size analysis showed a 1% relative threshold keeps the
# net's catastrophe-catch property. Kept in lockstep with producer.py's
# PNP_RESID_REL/PNP_RESID_ABS_NET -- these two must not drift, so the corner-count half
# of the same rule is IMPORTED from producer rather than restated here.
PNP_RESID_GATE_ABS = 0.3
PNP_RESID_GATE_REL = 0.05
TOP_K_NET = 3               # net runs on at most this many seeds (interface carries 3)
BUDGET_MS = 25.0            # live: over this, start dropping expensive layers

K = np.array([[L.FX, 0.0, L.CX], [0.0, L.FY, L.CY], [0.0, 0.0, 1.0]], np.float64)

# colours (BGR)
C_INNER = (70, 220, 70)
C_OUTER = (0, 180, 255)
C_REJECT = (70, 70, 160)
C_TRUST = (40, 230, 40)
C_CAUTION = (0, 220, 220)
C_NOTRUST = (60, 60, 240)
C_NORMAL = (255, 200, 40)
C_LIGHT = (150, 150, 90)
C_HORIZON = (110, 200, 130)
C_DIAL = (210, 255, 255)
C_MAP_ACT = (0, 215, 255)
C_MAP_NEXT = (0, 255, 180)
C_MAP_OTH = (140, 140, 140)
C_TXT = (240, 240, 240)
C_PANEL = (18, 18, 18)


def _text(img, s, org, scale=0.4, col=C_TXT, thick=1, bg=True):
    x, y = int(org[0]), int(org[1])
    if bg:
        (w, h), b = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(img, (x - 1, y - h - 2), (x + w + 1, y + b), (0, 0, 0), -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


class _Ema:
    """Per-stage millisecond timer with a light exponential average for the readout."""

    def __init__(self, a=0.25):
        self.a = a
        self.v = {}

    def put(self, k, ms):
        self.v[k] = ms if k not in self.v else (1 - self.a) * self.v[k] + self.a * ms

    def get(self, k):
        return self.v.get(k, 0.0)


class _PsiBridge:
    """Levelled gyro yaw integral that BRIDGES between strided compass fixes.

    Mirror of producer.ImuFilter's psi_int / anchor_psi: integrate the levelled body
    yaw rate (g . w, gravity-projected gyro) every frame -- cheap, no budget impact --
    and let a skylight measurement (mod 90) only CORRECT the integral onto its nearest
    mod-90 branch, never set it. This is what keeps the dial continuous when the
    latency degrade strides the compass to every Nth frame: mod-90 headings need
    frame-to-frame continuity to hold the quadrant branch, and the integral supplies
    it between fixes. corr_ema tracks |correction| at anchors as branch health."""

    def __init__(self):
        self.psi = None          # unwrapped deg, positive nose-right; None until anchored
        self._t = None
        self.corr_ema = 0.0      # EMA of |anchor correction| in deg (branch health)

    def integrate(self, t, gravity, gyro):
        """Per-frame: advance the integral by the levelled yaw increment."""
        if self._t is None:
            self._t = t
            return
        dt = min(max(t - self._t, 0.0), 0.15)
        self._t = t
        if gravity is None or gyro is None or self.psi is None:
            return
        w = np.asarray(gyro, float)
        g = np.asarray(gravity, float)
        self.psi += math.degrees(float(g @ w) * dt)

    def anchor(self, psi_mod90):
        """Compass fix: pull psi onto the measurement's nearest mod-90 branch."""
        if self.psi is None:
            self.psi = float(psi_mod90)
            return 0.0
        corr = (float(psi_mod90) - self.psi + 45.0) % 90.0 - 45.0
        self.psi += corr
        self.corr_ema = 0.7 * self.corr_ema + 0.3 * abs(corr)
        return corr


class PerceptionHUD:
    """Stateful renderer. Call render(frame, ctx) per frame; returns an annotated copy.

    ctx keys (all optional except the frame): gravity (unit-down body 3-vec or None),
    accel (m/s^2 body 3-vec, canonical), gyro (rad/s body, canonical), active_gate (int),
    fps (camera fps), t (wall seconds), frame_id, psi_deg (precomputed unwrapped heading).
    """

    def __init__(self, layers="123456", use_net=True, adapt=True, draw_status=True):
        self.on = {i: (str(i) in layers) for i in range(1, 7)}
        self.use_net = use_net
        self.adapt = adapt            # live: degrade under budget. replay: measure full.
        self.draw_status = draw_status
        self.ema = _Ema()
        self.net_stride = 1
        self.comp_stride = 1
        self._fc = 0
        self._trunk = self._head = self._dev = None
        self._net_err = None
        self._net_cache = ([], -1)    # (results, frame_count) for skipped frames
        self._comp_cache = (None, -1)
        self._psi = _PsiBridge()      # gyro yaw integral bridging strided compass fixes
        self._prev_quads = []         # crude carry-forward tracking for net seeds
        self._last_seen_t = None
        self._map = self._load_map()
        self._help = False

    # -- one-time model load (lazy: keeps `import hud` torch-free until needed) --------
    def _ensure_net(self):
        if self._trunk is not None or not self.use_net:
            return
        try:
            import torch
            import gatenet as G
            import gatenet_conf as GC
            self._G, self._GC = G, GC
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            trunk = G.GateNet(1.0).to(dev)
            trunk.load_state_dict(torch.load(TRUNK_CKPT, map_location=dev,
                                             weights_only=False)["model"])
            trunk.eval()
            head = GC.ConfHead().to(dev)
            head.load_state_dict(torch.load(HEAD_CKPT, map_location=dev,
                                            weights_only=False)["head"])
            head.eval()
            for p in trunk.parameters():
                p.requires_grad_(False)
            self._trunk, self._head, self._dev = trunk, head, dev
            # Warm up: the first forward pays CUDA init + kernel load (~seconds). It is
            # async, so without a sync the cost lands on frame 1's measured call and
            # swamps the latency EMA. Warm every batch size the live path uses (1..K),
            # then block until the queue drains so the cost stays here, untimed.
            for b in range(1, TOP_K_NET + 1):
                GC.conf_forward(trunk, head,
                                torch.zeros(b, 3, G.RES, G.RES, device=dev))
            if dev.type == "cuda":
                torch.cuda.synchronize()
        except Exception as e:                        # torch/ckpt missing -> net off
            self.use_net = False
            self._net_err = str(e)[:60]

    def _load_map(self):
        try:
            d = json.load(open(MAP_JSON))
            pos = d["layout_directions_2026_08_02"]["positions_m"]
            order = sorted(int(k) for k in pos)
            xy = {int(k): (pos[k]["x"], pos[k]["y"]) for k in pos}
            rot = d["layout_directions_2026_08_02"]["frame"].get(
                "grid_to_sketch_rotation_deg", 0.0)
            return {"order": order, "xy": xy, "rot": float(rot), "n": len(order)}
        except Exception:
            return None

    # -- key toggles (interactive replay window / routed from teleop) ------------------
    def on_key(self, code):
        try:
            ch = chr(code & 0xFF)
        except ValueError:
            return
        if ch in "123456":
            self.on[int(ch)] = not self.on[int(ch)]
        elif ch in ("n", "N"):
            self.use_net = not self.use_net
        elif ch in ("h", "H", "?"):
            self._help = not self._help

    # ================================ main entry =====================================
    def render(self, frame, ctx):
        self._fc += 1
        view = frame.copy()
        H, W = view.shape[:2]
        t = ctx.get("t", time.time())
        # Top-left y at which the status block starts. Producer mode draws the numeric
        # Observation readout first (at 74, under teleop's own top lines) and moves this
        # down so the two stack cleanly instead of the readout falling off the bottom.
        self._status_y0 = 74
        ms = {}
        # heading bridge: integrate the levelled gyro yaw EVERY frame (independent of
        # layer toggles and compass stride), so strided compass fixes land on a live
        # integral instead of a stale mod-90 sample.
        self._psi.integrate(t, ctx.get("gravity"), ctx.get("gyro"))

        # ---- LAYER 1: contour detections ------------------------------------------
        dets, rejects = [], []
        obs_mode = ctx.get("obs") is not None
        need_dets = (not obs_mode) and (self.on[1] or self.on[2] or self.on[3])
        if need_dets:
            t0 = time.perf_counter()
            try:
                dets = D.detections(frame, rejects)
            except Exception:
                dets = []
            ms["det"] = (time.perf_counter() - t0) * 1e3
        if self.on[1] and not obs_mode:
            for r in rejects:
                cv2.circle(view, tuple(np.int32(r[0])), 3, C_REJECT, -1)
            for d in dets:
                col = C_INNER if d["source"] == "inner" else C_OUTER
                cv2.polylines(view, [np.int32(d["quad"])], True, col, 1, cv2.LINE_AA)
                c = np.int32(d["centre"])
                _text(view, "unassigned %s %.0fpx %.1fm" % (d["source"][0], d["size_px"],
                      d["range_m"]), (c[0] + 4, c[1] - 4), 0.32, col)

        # ---- producer mode: draw the filled Observation instead of layers 1-3 -----
        obs = ctx.get("obs")
        if obs is not None:
            # Producer mode's primary panel: top-left, always shown regardless of layer
            # toggles, and it pushes the status block down (never anchored to the window
            # bottom, which clips).
            self._status_y0 = self._draw_obs_readout(view, obs, 74)
            if self.on[1] or self.on[2] or self.on[3]:
                t0 = time.perf_counter()
                self._draw_observation(view, obs, ctx.get("producer"))
                ms["prod"] = (time.perf_counter() - t0) * 1e3
                if any(g.valid for g in obs.gates):
                    self._last_seen_t = t

        # ---- LAYER 2/3: gatenet + confidence + PnP --------------------------------
        if obs is None and (self.on[2] or self.on[3]):
            run_net = self.use_net and (self._fc % max(1, self.net_stride) == 0)
            if run_net:
                t0 = time.perf_counter()
                results = self._run_net(frame, dets)
                ms["net"] = (time.perf_counter() - t0) * 1e3
                self._net_cache = (results, self._fc)
            else:
                results = self._net_cache[0]
            stale_net = self._net_cache[1] != self._fc
            if results:
                self._last_seen_t = t
            self._draw_net(view, results, stale_net)

        # ---- LAYER 4: skylight compass --------------------------------------------
        if self.on[4]:
            run_comp = ctx.get("gravity") is not None and \
                (self._fc % max(1, self.comp_stride) == 0)
            if run_comp:
                t0 = time.perf_counter()
                comp = self._run_compass(frame, ctx["gravity"])
                ms["comp"] = (time.perf_counter() - t0) * 1e3
                self._comp_cache = (comp, self._fc)
                # fresh fix -> mod-90 branch correction of the integral (never a jump)
                if comp is not None and comp.get("psi") is not None \
                        and comp["n_seg"] >= 4:
                    self._psi.anchor(comp["psi"])
            comp = self._comp_cache[0]
            if comp is not None:
                self._draw_compass(view, comp, ctx)

        # ---- LAYER 5: map strip + minimap -----------------------------------------
        if self.on[5] and self._map is not None:
            t0 = time.perf_counter()
            self._draw_map(view, ctx)
            ms["map"] = (time.perf_counter() - t0) * 1e3

        # ---- LAYER 6: status ------------------------------------------------------
        for k, v in ms.items():
            self.ema.put(k, v)
        total = sum(ms.values())
        self.ema.put("total", total)
        if self.adapt:
            self._degrade()
        if self.on[6]:
            self._draw_status(view, ctx, t)
        if self._help:
            self._draw_help(view)
        return view

    # -- gatenet inference on detector-seeded crops -----------------------------------
    def _run_net(self, frame, dets):
        self._ensure_net()
        if self._trunk is None:
            return []
        import torch
        G, GC = self._G, self._GC
        seeds = [np.asarray(d["quad"], np.float64) for d in dets[:TOP_K_NET]]
        # crude carry-forward: keep last quads that no current detection overlaps, so a
        # gate the contour momentarily drops still gets a net answer (light tracking).
        for q in self._prev_quads:
            if len(seeds) >= TOP_K_NET:
                break
            qc = q.mean(0)
            if all(np.linalg.norm(qc - s.mean(0)) > 25 for s in seeds):
                seeds.append(q)
        if not seeds:
            self._prev_quads = []
            return []
        crops, geos = [], []
        for q in seeds:
            cx, cy, S = G.crop_params(q)
            crop = G.make_crop(frame, cx, cy, S)
            x = np.ascontiguousarray(crop.transpose(2, 0, 1)).astype(np.float32)
            crops.append(x)
            geos.append((cx, cy, S))
        xb = torch.from_numpy(np.stack(crops)).to(self._dev).div_(255.).sub_(0.45).div_(0.25)
        pred8, logit, logerr = GC.conf_forward(self._trunk, self._head, xb)
        pred8 = pred8.detach().cpu().numpy()
        p = 1.0 / (1.0 + np.exp(-logit.detach().cpu().numpy()))
        errn = np.power(10.0, logerr.detach().cpu().numpy())
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        out, prev = [], []
        for i, (cx, cy, S) in enumerate(geos):
            quad = G.from_norm(pred8[i], cx, cy, S).astype(np.float64)
            err_px = float(errn[i] * S / 2.0)
            # interior-colour rejection (producer's strict variant on net quads, with
            # the dark-gap escape hatch: a real aperture against a LIT CEILING still
            # has dark patches a decoration fill cannot produce -- interiorgate.py
            # 2026-08-02). Import kept in lockstep with producer.py per this module's
            # own contract.
            inw, inv, ind = interior_colour(hsv, quad)
            if is_decoration(inw, inv, ind):
                out.append({"quad": quad, "p": float(p[i]), "err_px": err_px,
                            "range_m": None, "normal_cam": None, "resid": float("nan"),
                            "interior": (inw, inv, ind)})
                prev.append(quad)
                continue
            rng, normal_cam, resid = self._pnp(quad)
            out.append({"quad": quad, "p": float(p[i]), "err_px": err_px,
                        "range_m": rng, "normal_cam": normal_cam, "resid": resid,
                        "interior": None})
            prev.append(quad)
        self._prev_quads = prev
        return out

    def _pnp(self, quad):
        """(range_m, normal_cam signed toward camera, snap-residual px) or (None,...)."""
        try:
            ok, rvec, tvec = cv2.solvePnP(D.OBJ, D.order_quad(quad), K, D.DIST,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                return None, None, float("nan")
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3)
            n = R[:, 2]
            if float(n @ t) > 0:          # sign toward the camera (approach side)
                n = -n
            proj, _ = cv2.projectPoints(D.OBJ, rvec, tvec, K, D.DIST)
            resid = float(np.sqrt(np.mean((proj.reshape(4, 2) - D.order_quad(quad)) ** 2)))
            return float(np.linalg.norm(t)), (t, n), resid
        except cv2.error:
            return None, None, float("nan")

    def _draw_net(self, view, results, stale):
        for r in results:
            if r.get("interior") is not None:      # decoration: dim-red rejected class
                inw, inv, ind = r["interior"]
                cv2.polylines(view, [np.int32(r["quad"])], True, C_REJECT,
                              1 if stale else 2, cv2.LINE_AA)
                c = r["quad"].mean(0)
                _text(view, "interior w%.2f v%.0f d%.2f" % (inw, inv, ind),
                      (c[0] - 30, c[1] + 4), 0.34, C_REJECT)
                continue
            qsize = float(max(np.linalg.norm(r["quad"][(k + 1) % 4] - r["quad"][k])
                             for k in range(4)))
            n_in = n_corners_in_frame(r["quad"])
            trusted = r["p"] >= P_OP and np.isfinite(r["resid"]) \
                and n_in >= MIN_CORNERS_IN_FRAME \
                and r["resid"] <= max(PNP_RESID_GATE_ABS, PNP_RESID_GATE_REL * qsize)
            col = C_TRUST if trusted else (C_CAUTION if r["p"] >= P_OP else C_NOTRUST)
            th = 1 if stale else 2
            cv2.polylines(view, [np.int32(r["quad"])], True, col, th, cv2.LINE_AA)
            c = r["quad"].mean(0)
            # c<n> is the in-frame corner count -- the trust variable for close gates.
            # Shown so a rejected big gate reads as "clipped past the limit" rather than
            # as an unexplained refusal (Claire's request, 2026-08-02).
            # "unassigned": these quads come from this module's own recompute and carry
            # NO race identity -- only the producer assigns gate indices. Saying so keeps
            # them from being read as an identified gate (2026-08-02).
            tag = "unassigned p%.2f e%.1f r%.2f c%d" % (r["p"], r["err_px"], r["resid"],
                                                        n_in)
            _text(view, tag + ("~" if stale else ""), (c[0] - 30, c[1] + 4), 0.34, col)
            # LAYER 3: PnP range + approach-normal arrow
            if self.on[3] and r["normal_cam"] is not None:
                t, n = r["normal_cam"]
                d = float(np.clip(r["range_m"] * 0.35, 0.8, 4.0))
                P = t + d * n
                if P[2] > 0.1:
                    uv = (K @ P)[:2] / P[2]
                    p0 = tuple(np.int32(c))
                    p1 = tuple(np.int32(uv))
                    cv2.arrowedLine(view, p0, p1, C_NORMAL, 2, cv2.LINE_AA, tipLength=0.3)
                _text(view, "%.1fm" % r["range_m"], (c[0] - 12, c[1] - 10), 0.4, C_NORMAL)

    # -- producer Observation rendering (--use-producer) ------------------------------
    def _proj(self, p_body):
        """Body-frame 3-vector -> (u, v) pixels, or None if behind the camera."""
        cam = L.body_to_cam() @ np.asarray(p_body, float)
        if cam[2] <= 0.05:
            return None
        return np.array([L.CX + L.FX * cam[0] / cam[2], L.CY + L.FY * cam[1] / cam[2]])

    def _draw_obs_readout(self, view, obs, y0):
        """Numeric dump of the interface.Observation the control policy consumes: per-gate
        body position (x fwd, y right, z down, metres), range, approach normal + validity
        flags, confidence and coast age, plus the attention index. This is the actual
        flight-time perception OUTPUT -- the numbers that go to the controller -- not the
        HUD's independent recompute. Drawn top-left starting at y0 on a dark backing;
        returns the y just below it so the status block can stack underneath (keeping the
        panel off the window bottom, which clips)."""
        act = obs.race.active_gate_index
        gates = [g for g in obs.gates if g.valid]
        gates.sort(key=lambda g: (g.index != act, g.index))
        lines = ["OBS->policy act=g%d ribbon=%s"
                 % (act, "Y" if obs.ribbon.valid else "-")]
        for g in gates:
            slot = "CUR" if g.index == act else "%+d" % (g.index - act)
            p = np.asarray(g.pos_body, float)
            n = np.asarray(g.normal_body, float)
            lines.append(
                "g%-2d %-3s r%5.1f p[%+5.1f%+5.1f%+5.1f] n[%+.2f%+.2f%+.2f]%s%s c%.2f%s"
                % (g.index, slot, g.range_m, p[0], p[1], p[2], n[0], n[1], n[2],
                   "V" if g.normal_valid else "-", "P" if g.pose_valid else "-",
                   g.confidence,
                   " ~%.1fs" % g.staleness_s if g.staleness_s > 0.05 else ""))
        if len(lines) == 1:
            lines.append("(no valid gate)")
        row_h, pad = 15, 4
        h = pad * 2 + row_h * len(lines)
        cv2.rectangle(view, (4, y0), (4 + 398, y0 + h), (0, 0, 0), -1)
        for i, ln in enumerate(lines):
            _text(view, ln, (8, y0 + pad + row_h * i + 11), 0.36,
                  (255, 255, 255) if i == 0 else (170, 255, 170), 1)
        return y0 + h + 4

    def _draw_observation(self, view, obs, prod=None):
        """Render a filled interface.Observation: what the PRODUCER believes this frame
        (gates in the body frame, attention, ribbon). Distinct from my independent
        recompute -- this shows the actual flight-time perception output."""
        H, W = view.shape[:2]
        for g in obs.gates:
            if not g.valid:
                continue
            c = self._proj(g.pos_body)
            if c is None:
                continue
            trusted = g.pose_valid and g.confidence >= P_OP
            col = C_TRUST if trusted else (C_CAUTION if g.confidence >= P_OP else C_NOTRUST)
            if g.staleness_s > 0.15:
                col = tuple(int(x * 0.5) for x in col)     # dim a coasting track
            s = max(6.0, g.size_px)
            p0 = np.int32([c[0] - s / 2, c[1] - s / 2])
            p1 = np.int32([c[0] + s / 2, c[1] + s / 2])
            cv2.rectangle(view, tuple(p0), tuple(p1), col, 2)
            # IDENTITY FIRST (Claire, 2026-08-02). Which gate the producer believes this
            # quad IS is the thing the association bugs got wrong, so the absolute race
            # index leads the label and the CURRENT slot is drawn distinctly: bigger,
            # brighter and tagged CUR, lookahead slots as +1 / +2. A misassigned index is
            # then readable at a glance instead of hiding inside a range readout.
            is_cur = g.index == obs.race.active_gate_index
            slot = "CUR" if is_cur else "+%d" % (g.index - obs.race.active_gate_index)
            tag = "g%d %s %.1fm p%.2f%s" % (g.index, slot, g.range_m, g.confidence,
                                            " P" if g.pose_valid else "")
            _text(view, tag, (c[0] - 28, c[1] - s / 2 - 4),
                  0.44 if is_cur else 0.32,
                  (255, 255, 255) if is_cur else col, 2 if is_cur else 1)
            if is_cur:
                cv2.rectangle(view, tuple(np.int32([c[0] - s / 2 - 3, c[1] - s / 2 - 3])),
                              tuple(np.int32([c[0] + s / 2 + 3, c[1] + s / 2 + 3])),
                              (255, 255, 255), 1)
            if g.staleness_s > 0.05:
                _text(view, "~%.1fs" % g.staleness_s, (c[0] - 12, c[1] + s / 2 + 12),
                      0.3, col)
            # approach-normal arrow (producer already signs it toward the camera)
            if g.normal_valid:
                d = float(np.clip(g.range_m * 0.35, 0.8, 4.0))
                tip = self._proj(np.asarray(g.pos_body) + d * np.asarray(g.normal_body))
                if tip is not None:
                    cv2.arrowedLine(view, tuple(np.int32(c)), tuple(np.int32(tip)),
                                    C_NORMAL, 2, cv2.LINE_AA, tipLength=0.3)
            # normalfuse (2026-08-02): the IPPE twin the accumulator REJECTED, thin and
            # magenta, plus the likelihood margin and the LOS baseline behind the verdict.
            # Without the loser drawn, a wrong-side normal looks like a confident one --
            # which is exactly how the act=12 wrong-side symptom went unnoticed.
            tr = getattr(prod, "tracks", {}).get(g.index) if prod is not None else None
            acc = getattr(tr, "nrm", None) if tr is not None else None
            if acc is not None and acc.mu[0] is not None:
                d = float(np.clip(g.range_m * 0.35, 0.8, 4.0))
                los = np.asarray(g.pos_body) / max(g.range_m, 1e-9)
                nl = acc.loser_body(los) if g.normal_valid else None
                if nl is not None:
                    tip = self._proj(np.asarray(g.pos_body) + d * nl)
                    if tip is not None:
                        cv2.arrowedLine(view, tuple(np.int32(c)), tuple(np.int32(tip)),
                                        (255, 0, 255), 1, cv2.LINE_AA, tipLength=0.3)
                _text(view, "L%.1f b%.0f %s" % (acc.llr, acc.baseline_deg,
                                                getattr(tr, "normal_src", "?")[:6]),
                      (c[0] - 28, c[1] + s / 2 + 24), 0.3,
                      C_NORMAL if g.normal_valid else C_NOTRUST)
        # ribbon direction samples
        rb = obs.ribbon
        if rb.valid:
            for b, e in zip(rb.bearings_rad, rb.elevs_rad):
                dir_body = np.array([math.cos(e) * math.cos(b), math.cos(e) * math.sin(b),
                                     -math.sin(e)])
                q = self._proj(dir_body * 5.0)
                if q is not None:
                    cv2.circle(view, tuple(np.int32(q)), 3, (200, 220, 60), -1)
        # attention target
        a = obs.attention
        tgt = self._proj(np.asarray(a.target_dir_body) * 5.0)
        if tgt is not None:
            cv2.drawMarker(view, tuple(np.int32(tgt)), C_DIAL, cv2.MARKER_TILTED_CROSS,
                           14, 2)
            _text(view, ["SEARCH", "CUR", "NEXT", "RIBBON"][int(a.kind)],
                  (tgt[0] + 8, tgt[1]), 0.34, C_DIAL)

    # -- compass ----------------------------------------------------------------------
    def _run_compass(self, frame, gravity):
        try:
            R_lb = SK.level_rotation(gravity)
            if R_lb is None:
                return None
            comps = SK.light_components(frame, R_lb)
            votes = []
            for contours, _b in comps:
                votes.extend(SK.segment_votes(contours, R_lb))
            if not votes:
                return {"comps": comps, "theta": None, "psi": None, "spread": None,
                        "n_seg": 0, "g_cam": SK._R_CB @ np.asarray(gravity, float)}
            theta, spread = SK._circular_median_mod90(votes)
            return {"comps": comps, "theta": theta, "psi": (-theta) % 90.0,
                    "spread": spread, "n_seg": len(votes),
                    "g_cam": SK._R_CB @ np.asarray(gravity, float)}
        except Exception:
            return None

    def _draw_compass(self, view, comp, ctx):
        H, W = view.shape[:2]
        # faint light-quad outlines
        for contours, _b in comp["comps"]:
            for c in contours:
                cv2.polylines(view, [np.int32(c)], True, C_LIGHT, 1, cv2.LINE_AA)
        # gravity horizon line: rays r with r . g_cam = 0
        g = comp["g_cam"]
        if abs(g[1]) > 1e-3:
            def v_at(u):
                return L.CY + L.FY * (-g[2] - (u - L.CX) / L.FX * g[0]) / g[1]
            cv2.line(view, (0, int(v_at(0))), (W, int(v_at(W))), C_HORIZON, 1, cv2.LINE_AA)
        # dial, top-right within the safe band (below teleop's top bar)
        cx, cy, rad = W - 44, 92, 30
        # branch health: persistently large corrections mean the gyro integral and the
        # compass disagree -- the branch is not to be trusted, so DIM the whole dial
        # (honest-uncertainty display) rather than pretend the heading is solid.
        sick = self._psi.corr_ema > 6.0
        dim = (lambda c: tuple(int(x * 0.45) for x in c)) if sick else (lambda c: c)
        cv2.circle(view, (cx, cy), rad, dim(C_DIAL), 1, cv2.LINE_AA)
        for k in range(4):                       # mod-90 -> 4 ticks
            a = math.radians(90 * k)
            cv2.line(view, (cx, cy),
                     (int(cx + rad * math.sin(a)), int(cy - rad * math.cos(a))),
                     (60, 90, 90), 1, cv2.LINE_AA)
        # heading: precomputed unwrapped CSV if the replay has one, else the live
        # gyro-bridged integral, else (before the first anchor) the raw mod-90 fix.
        # DRAWN FOUR-FOLD SYMMETRIC (2026-08-02, Claire: "always points vaguely
        # eastward", read as broken). The skylight compass measures heading only
        # MODULO 90 -- the ceiling light grid is four-fold symmetric and the quadrant
        # is unresolved -- so a single full-360 arrow asserted a direction we do not
        # have. Unwrapping (psi_deg / the gyro bridge) buys CONTINUITY, not a quadrant,
        # so the indicator is folded to mod 90 and every one of the four branches is
        # drawn identically: the display then states exactly what is known, an
        # orientation mod 90, and no arm can be mistaken for north.
        psi = ctx.get("psi_deg")
        if psi is None:
            psi = self._psi.psi
        bridged = psi is not None
        if psi is None:
            psi = comp["psi"]
        if psi is not None:
            spread = comp["spread"] or 99
            dcol = C_TRUST if (comp["n_seg"] >= 8 and spread <= 6) else \
                (C_CAUTION if comp["n_seg"] >= 4 else C_NOTRUST)
            dcol = dim(dcol)
            psi_m = psi % 90.0            # the whole of what is measurable
            for k in range(4):            # four indistinguishable branches
                a = math.radians(psi_m + 90.0 * k)
                cv2.arrowedLine(view, (cx, cy),
                                (int(cx + rad * math.sin(a)), int(cy - rad * math.cos(a))),
                                dcol, 2, cv2.LINE_AA, tipLength=0.35)
            cv2.circle(view, (cx, cy), 3, dcol, -1)
            # says it in words too, so the cross is not re-invented as a heading rose
            _text(view, "grid mod90", (cx - 30, cy - rad - 5), 0.32, dcol)
            # round BEFORE the fold, else 89.98 prints as "90"
            _text(view, "psi %.0f/90%s" % (round(psi) % 90.0, "" if bridged else "?"),
                  (cx - 30, cy + rad + 12), 0.34, dcol)
            _text(view, "n%d s%.0f c%.1f" % (comp["n_seg"], comp["spread"] or 0,
                  self._psi.corr_ema), (cx - 30, cy + rad + 24), 0.3, dcol)

    # -- map strip + minimap ----------------------------------------------------------
    def _draw_map(self, view, ctx):
        H, W = view.shape[:2]
        act = ctx.get("active_gate", -1)
        order = self._map["order"]
        n = len(order)
        # order ribbon along the bottom (above teleop's 20px bar)
        y = H - 30
        x0, bw = 8, min(18, (W - 60) // n)
        for i, g in enumerate(order):
            x = x0 + i * bw
            filled = (g == act)
            col = C_MAP_ACT if filled else (C_MAP_NEXT if g == act + 1 else C_MAP_OTH)
            cv2.rectangle(view, (x, y), (x + bw - 2, y + 10),
                          col, -1 if filled else 1)
            if g % 4 == 0:
                _text(view, str(g), (x, y - 2), 0.28, C_MAP_OTH, bg=False)
        _text(view, "race order  active=%s" % (act if act >= 0 else "-"),
              (x0, y - 12), 0.32, C_TXT)
        # top-down minimap, bottom-left
        self._draw_minimap(view, ctx, (8, H - 150), (120, 100))

    def _draw_minimap(self, view, ctx, org, size):
        ox, oy = org
        sw, sh = size
        xy = self._map["xy"]
        xs = [p[0] for p in xy.values()]
        ys = [p[1] for p in xy.values()]
        lo = np.array([min(xs), min(ys)])
        span = np.array([max(xs) - min(xs), max(ys) - min(ys)])
        span[span < 1e-6] = 1.0
        pad = 8

        def to_px(x, y):
            u = ox + pad + (x - lo[0]) / span[0] * (sw - 2 * pad)
            v = oy + sh - pad - (y - lo[1]) / span[1] * (sh - 2 * pad)   # y up
            return int(u), int(v)

        cv2.rectangle(view, (ox, oy), (ox + sw, oy + sh), C_PANEL, -1)
        cv2.rectangle(view, (ox, oy), (ox + sw, oy + sh), (60, 60, 60), 1)
        order = self._map["order"]
        pts = [to_px(*xy[g]) for g in order]
        for a, b in zip(pts, pts[1:]):
            cv2.line(view, a, b, (70, 70, 70), 1, cv2.LINE_AA)
        act = ctx.get("active_gate", -1)
        for g, p in zip(order, pts):
            col = C_MAP_ACT if g == act else (C_MAP_NEXT if g == act + 1 else C_MAP_OTH)
            cv2.circle(view, p, 3 if g in (act, act + 1) else 2, col, -1)
        # active -> next internal cue, drawn in the map's own frame (sign-safe)
        if 0 <= act < len(order) - 1:
            a, b = xy[order[act]], xy[order[act + 1]]
            cv2.arrowedLine(view, to_px(*a), to_px(*b), C_MAP_NEXT, 1, cv2.LINE_AA,
                            tipLength=0.3)
            # aircraft heading ray from compass psi (grid) rotated into the sketch frame.
            psi = ctx.get("psi_deg")
            if psi is None:
                psi = self._psi.psi            # gyro-bridged integral (continuous)
            if psi is None and self._comp_cache[0] is not None:
                psi = self._comp_cache[0].get("psi")
            if psi is not None:
                # same mod-90 honesty as the dial: four rays, not one heading ray.
                L2 = 0.25 * float(np.linalg.norm(span))
                for kq in range(4):
                    hs = math.radians(psi % 90.0 + 90.0 * kq + self._map["rot"])
                    tip = (a[0] + L2 * math.cos(hs), a[1] + L2 * math.sin(hs))
                    cv2.arrowedLine(view, to_px(*a), to_px(*tip), C_DIAL, 1, cv2.LINE_AA,
                                    tipLength=0.3)
        _text(view, "map (rel)", (ox + 2, oy + 10), 0.3, C_TXT, bg=False)

    # -- status + budget --------------------------------------------------------------
    def _degrade(self):
        tot = self.ema.get("total")
        if tot > BUDGET_MS:
            if self.net_stride < 4:
                self.net_stride += 1
            elif self.comp_stride < 4:
                self.comp_stride += 1
        elif tot < 0.55 * BUDGET_MS:
            if self.comp_stride > 1:
                self.comp_stride -= 1
            elif self.net_stride > 1:
                self.net_stride -= 1

    def _draw_status(self, view, ctx, t):
        H, W = view.shape[:2]
        lines = []
        e = self.ema
        if e.get("prod") > 0:
            lat = "prod%3.0f cmp%3.0f map%2.0f | %4.1fms" % (
                e.get("prod"), e.get("comp"), e.get("map"), e.get("total"))
        else:
            lat = "det%3.0f net%3.0f cmp%3.0f map%2.0f | %4.1fms" % (
                e.get("det"), e.get("net"), e.get("comp"), e.get("map"), e.get("total"))
        lines.append(lat)
        deg = []
        if self.net_stride > 1:
            deg.append("net 1/%d" % self.net_stride)
        if self.comp_stride > 1:
            deg.append("cmp 1/%d" % self.comp_stride)
        if not self.use_net:
            deg.append("NET OFF" + (":" + self._net_err if self._net_err else ""))
        if deg:
            lines.append("DEGRADED " + "  ".join(deg))
        gyro = ctx.get("gyro")
        if gyro is not None:
            g = np.asarray(gyro, float)
            lines.append("rate |w|%5.1f  r%+.0f p%+.0f y%+.0f deg/s" % (
                math.degrees(np.linalg.norm(g)), *np.degrees(g)))
        accel = ctx.get("accel")
        if accel is not None:
            mag = float(np.linalg.norm(accel))
            lines.append("|a|-g %+5.2f  (att %s)" % (
                mag - 9.81, "ok" if abs(mag - 9.81) < 1.0 else "dynamic"))
        if self._last_seen_t is not None:
            lines.append("gate stale %.1fs" % max(0.0, t - self._last_seen_t))
        if ctx.get("fps") is not None:
            lines.append("cam %.0f fps" % ctx["fps"])
        y = self._status_y0
        for ln in lines:
            _text(view, ln, (8, y), 0.36, C_TXT)
            y += 15

    def _draw_help(self, view):
        H, W = view.shape[:2]
        txt = ["1 contour  2 gatenet  3 pnp", "4 compass  5 map  6 status",
               "n net on/off   h help   esc quit"]
        y = H // 2 - 20
        for ln in txt:
            _text(view, ln, (W // 2 - 90, y), 0.4, C_DIAL)
            y += 18


# =====================================================================================
# Live wiring: a background worker so the control loop never blocks on perception.
# =====================================================================================

class LiveHUD:
    """Runs perception off the teleop control loop. The control thread only reads the
    most recent finished overlay via overlay(); this thread does all the work.

    Consumes the SAME telemetry teleop already has (VisionRX.take, Telemetry.snapshot),
    so it adds no new sockets and cannot perturb the flight path. Gyro/accel here are the
    raw HIGHRES_IMU values; the mirror (x -1 on the gyro) is applied here to hand the
    status readout canonical body rates, exactly as CONVENTIONS.md specifies."""

    def __init__(self, vision, tel, layers="123456", use_net=True, use_producer=False):
        self.vision = vision
        self.tel = tel
        self.hud = PerceptionHUD(layers=layers, use_net=use_net, adapt=True,
                                 draw_status=True)
        # Producer mode (--hud-producer): run the real pilot/producer.py in this worker
        # so the overlay shows the filled interface.Observation the control policy would
        # consume, plus a numeric readout of it. Guarded so an absent/broken producer just
        # falls back to the independent layer recompute.
        self.prod = None
        self._t0 = None
        if use_producer:
            try:
                import producer as PR
                self.prod = PR.Producer(use_net=use_net)
                print("LiveHUD: producer mode on -- overlay reflects producer.step()")
            except Exception as e:
                print("LiveHUD: producer mode unavailable (%s); layer recompute only." % e)
                self.prod = None
        self._overlay = None
        self._lock = threading.Lock()
        self._last_fid = None
        self._running = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _loop(self):
        self.hud._ensure_net()     # pay import+load+warmup off the control loop
        while self._running:
            latest = self.vision.take()
            if latest is None or latest[0] == self._last_fid:
                time.sleep(0.003)
                continue
            fid, _stime, img = latest
            self._last_fid = fid
            snap = self.tel.snapshot()
            accel = np.asarray(snap["accel"], float)
            mag = float(np.linalg.norm(accel))
            gravity = (-accel / mag) if mag > 1e-6 else None
            gyro = -np.asarray(snap["gyro"], float)      # mirror -> canonical
            ctx = {"gravity": gravity, "accel": accel, "gyro": gyro,
                   "active_gate": snap["active_gate"], "fps": self.vision.fps,
                   "t": time.time(), "frame_id": fid}
            # Producer mode: run the real perception stack on this frame and hand the
            # filled Observation to render(). Feed RAW IMU rows since the last frame (the
            # producer applies the gyro mirror itself). A monotonic-ish elapsed clock is
            # fine -- the producer uses t_s only for its own frame dt.
            if self.prod is not None:
                if self._t0 is None:
                    self._t0 = ctx["t"]
                rows = self.tel.drain_imu()
                race = {"active_gate_index": snap["active_gate"],
                        "race_time_s": getattr(self.tel, "race_time_s", 0.0),
                        "armed": getattr(self.tel, "armed", True),
                        "n_gates_total": 17}
                try:
                    obs = self.prod.step(img, rows, race, ctx["t"] - self._t0)
                    ctx["obs"] = obs
                    ctx["producer"] = self.prod
                    ctx["active_gate"] = obs.race.active_gate_index
                except Exception as e:
                    print("LiveHUD: producer.step failed (%s); recompute only." % e)
                    self.prod = None
            try:
                ov = self.hud.render(img, ctx)
            except Exception:
                ov = None
            if ov is not None:
                with self._lock:
                    self._overlay = ov

    def overlay(self):
        with self._lock:
            return self._overlay

    def on_key(self, code):
        self.hud.on_key(code)

    def stop(self):
        self._running = False
        self._th.join(timeout=1.0)


# =====================================================================================
# Replay
# =====================================================================================

def _race_stepper(session):
    """t_ns -> active_gate_index, from race.csv (last row at or before t)."""
    rows = L.load_csv(os.path.join(session, "race.csv"))
    ts = np.array([float(r["t_wall_ns"]) for r in rows])
    gi = np.array([int(r["active_gate_index"]) for r in rows])
    order = np.argsort(ts)
    ts, gi = ts[order], gi[order]

    def at(t_ns):
        i = int(np.searchsorted(ts, t_ns, side="right")) - 1
        return int(gi[max(0, i)]) if len(gi) else -1
    return at


def _psi_lookup(session):
    """frame_id -> psi_unwrapped from a precomputed skylight_<session>.csv, if present."""
    name = os.path.basename(os.path.normpath(session))
    path = os.path.join(PERC, "skylight_%s.csv" % name)
    if not os.path.exists(path):
        return {}
    out = {}
    for r in L.load_csv(path):
        if r.get("psi_unwrapped"):
            try:
                out[str(int(r["frame_id"]))] = float(r["psi_unwrapped"])
            except ValueError:
                pass
    return out


def replay(args):
    session = args.replay
    frames = [r for r in L.load_csv(os.path.join(session, "frames.csv")) if r["file"]]
    if not frames:
        sys.exit("no frames in %s" % session)
    imu = SK.Imu(session)
    race_at = _race_stepper(session)
    psi = _psi_lookup(session)
    hud = PerceptionHUD(layers=args.layers, use_net=not args.no_net,
                        adapt=False, draw_status=True)
    hud._ensure_net()          # pay import+load+warmup now, not inside a timed frame

    # optional: drive the layers from producer.py's filled Observation instead of this
    # module's independent recompute. Guarded so a broken/absent producer just falls back.
    prod = None
    imu_rows = imu_t_all = race_rows = race_t_all = None
    if args.use_producer:
        try:
            import producer as PR
            prod = PR.Producer(use_net=not args.no_net)
            imu_rows = L.load_csv(os.path.join(session, "imu.csv"))
            imu_t_all = np.array([float(r["t_wall_ns"]) for r in imu_rows])
            race_rows = L.load_csv(os.path.join(session, "race.csv"))
            race_t_all = np.array([float(r["t_wall_ns"]) for r in race_rows]) \
                if race_rows else np.array([0.0])
            print("producer mode: rendering producer.Producer.step() Observations")
        except Exception as e:
            print("WARNING: --use-producer unavailable (%s); rendering from assets." % e)
            prod = None

    lo = args.start
    hi = args.end if args.end > 0 else len(frames)
    sel = list(range(lo, min(hi, len(frames)), args.stride))

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, 30.0 / args.stride, (640, 360))
        if not writer.isOpened():
            print("VideoWriter failed to open (%s); falling back to --frames-out" %
                  args.out)
            writer = None
            args.frames_out = args.frames_out or (args.out + "_frames")
    if args.frames_out:
        os.makedirs(args.frames_out, exist_ok=True)
    preview_at = set()
    if args.preview > 0:
        preview_at = set(np.linspace(0, len(sel) - 1, args.preview).astype(int).tolist())
        os.makedirs(os.path.join(PERC, "vercheck"), exist_ok=True)

    t0_ns = float(frames[sel[0]]["t_recv_wall_ns"]) if sel else 0.0
    imu_cur = int(np.searchsorted(imu_t_all, t0_ns)) if prod is not None else 0
    lat_acc = {}
    n = 0
    for k, idx in enumerate(sel):
        rec = frames[idx]
        img = cv2.imread(os.path.join(session, "frames", rec["file"]))
        if img is None:
            continue
        t_ns = float(rec["t_recv_wall_ns"])
        gravity, gmag = imu.gravity(t_ns)
        j = int(np.argmin(np.abs(imu.t - t_ns)))
        ctx = {"gravity": gravity, "accel": imu.a[j], "gyro": imu.w[j],
               "active_gate": race_at(t_ns), "fps": 30.0,
               "t": t_ns / 1e9, "frame_id": rec["frame_id"],
               "psi_deg": psi.get(str(rec["frame_id"]))}
        if prod is not None:
            jj = int(np.searchsorted(imu_t_all, t_ns))
            rows = imu_rows[imu_cur:jj]
            imu_cur = jj
            ri = max(0, int(np.searchsorted(race_t_all, t_ns)) - 1)
            active = int(race_rows[ri]["active_gate_index"]) if race_rows else 0
            rtime = 0.0
            if race_rows:
                rs = float(race_rows[ri]["race_start_boot_time_ms"])
                sb = float(race_rows[ri]["sim_boot_time_ms"])
                rtime = max(0.0, (sb - rs) / 1000.0) if rs >= 0 else 0.0
            try:
                obs = prod.step(img, rows, {"active_gate_index": active,
                                            "race_time_s": rtime, "armed": True,
                                            "n_gates_total": 17}, (t_ns - t0_ns) / 1e9)
                ctx["obs"] = obs
                ctx["producer"] = prod      # for the normalfuse twin/margin overlay
                ctx["active_gate"] = obs.race.active_gate_index
            except Exception as e:
                if n == 0:
                    print("producer.step failed (%s); rendering from assets." % e)
                prod = None
        view = hud.render(img, ctx)
        for kk in ("det", "net", "prod", "comp", "map", "total"):
            lat_acc.setdefault(kk, []).append(hud.ema.get(kk))
        if writer is not None:
            writer.write(view)
        if args.frames_out:
            cv2.imwrite(os.path.join(args.frames_out, rec["file"].replace(".jpg",
                        "") + "_hud.png"), view)
        if k in preview_at:
            tag = "%s_f%s_g%s" % (os.path.basename(os.path.normpath(session))[9:20],
                                  rec["frame_id"], ctx["active_gate"])
            out = os.path.join(PERC, "vercheck", "hud_preview_%02d_%s.png"
                               % (len(preview_at & set(range(k + 1))), tag))
            cv2.imwrite(out, view)
            print("preview ->", out)
        if args.show:
            cv2.imshow("perception HUD", view)
            key = cv2.waitKey(0 if args.step else 1) & 0xFF
            if key == 27:
                break
            hud.on_key(key)
        n += 1
        if n % 200 == 0:
            print("  %d/%d frames" % (n, len(sel)), flush=True)
    if writer is not None:
        writer.release()
        print("wrote", args.out)
    if args.show:
        cv2.destroyAllWindows()
    print("\nlatency (EMA px-ms, per stage):")
    for kk in ("det", "net", "prod", "comp", "map", "total"):
        v = lat_acc.get(kk, [0])
        print("  %-6s median %5.1f  p90 %5.1f  ms" %
              (kk, float(np.median(v)), float(np.percentile(v, 90))))
    print("rendered %d frames from %s" % (n, session))


def main():
    ap = argparse.ArgumentParser(description="perception HUD (replay renderer)")
    ap.add_argument("--replay", required=True, help="session dir under pilot/sessions/")
    ap.add_argument("--out", default=None, help="MP4 output path")
    ap.add_argument("--frames-out", default=None, help="dir for per-frame PNGs")
    ap.add_argument("--preview", type=int, default=0,
                    help="save N evenly-spaced annotated frames to perception/vercheck/")
    ap.add_argument("--show", action="store_true", help="interactive window")
    ap.add_argument("--step", action="store_true", help="with --show, wait for a key")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--layers", default="123456")
    ap.add_argument("--no-net", action="store_true")
    ap.add_argument("--use-producer", action="store_true",
                    help="render producer.py's filled Observation instead of this "
                         "module's independent recompute (optional; falls back if absent)")
    replay(ap.parse_args())


if __name__ == "__main__":
    main()
