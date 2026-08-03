"""Is the per-gate rate limited by CONTROL or by PERCEPTION?

    python pilot/control/evalsuite/_probe_noise_ceiling.py --ckpt vq2_ladder1_s48234496
    python pilot/control/evalsuite/_probe_noise_ceiling.py --policy baseline --seeds 0-99

`vq2_ladder1` sits at `grate` ~0.38 with 78% of episodes ending on a gate frame, floor
0.03 and ceiling 0.00. Altitude is solved; the whole remaining failure is lateral. The
question that decides what to do next is whether the policy is aiming badly or being lied
to, and the two have opposite fixes -- more training and a higher promote gate on one
side, a different noise model or a lowered `gate_rate_promote` on the other.

The arithmetic that motivates it: the clear half-aperture is 0.75 m and the bounding
sphere is 0.214 m, leaving **0.536 m** of usable margin. The surrogate's own reported
detection error is a median 1.46 m at 9.1 m range (`surrogate/selfcheck.py`, full-3D so
range-dominated), and the MEASURED pipeline puts bearing-plane error over 0.5 m on 34% of
detections (`STATE_PERCEPTION_FOR_CONTROL.md` Q2). If a third of sightings place the
perceived centre outside the entire margin, then a policy flying perfectly to that centre
still clips at about that rate, and no amount of training reaches 0.60.

So: fly the SAME checkpoint over the SAME seeds at `noise_scale` 1.0 and 0.0 and compare.
`noise.sample` interpolates every range from its perfect-sensor value (`_CLEAN`) toward
the configured one, so 0.0 is a genuinely noiseless detector while the physics, the
course and the plant are untouched. The gap between the two columns is the share of the
gate-frame failure that perception alone accounts for.

READ IT AS A BOUND, NOT A VERDICT. `noise_scale=0` is a sensor no pipeline will ever
have, so the clean column is an upper bound on what better perception could buy, not a
prediction. And the surrogate's noise is the hand-specified fallback (`noise.py` line 1),
optimistic on continuity and wrong in the sign of the range bias, so the noisy column is
not the real pipeline either. What the comparison establishes is which of the two is
worth spending the next month on.

Scored through `SingleSurrogate` + `RLPolicy` -- the deployment path, not the training
one -- so a divergence between them shows up here as a bonus.
"""

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pilot.control.surrogate.env import EnvConfig            # noqa: E402
from pilot.control.surrogate.single import SingleSurrogate   # noqa: E402

# Pad-only starts, matching training since the spawn change. Not EVAL_START_PROBS, which
# predates it.
PAD_ONLY = (1.0, 0.0, 0.0, 0.0, 0.0)


def parse_seeds(spec):
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def make_config(noise_scale, gates_per_episode, vq2_frac, difficulty, speed_cap):
    """The TRAINING config, not the eval default.

    `grate` here has to mean what it means in the training log or the comparison is
    against a different number: same difficulty, same speed_cap, same K, same pad-only
    spawns, same course mix.
    """
    import dataclasses
    cfg = EnvConfig.curriculum(float(difficulty), float(speed_cap))
    return dataclasses.replace(
        cfg, noise_scale=float(noise_scale), start_probs=PAD_ONLY,
        vq2_frac=float(vq2_frac), gates_per_episode=int(gates_per_episode))


def fly(policy, cfg, seed, max_steps):
    """One episode. Returns (attempts, passes, gates, cause) for the ACTIVE gate.

    Everything is read from the INFO dict of the terminating step, never from the env
    afterwards: `VecSurrogate` auto-resets on `done`, so `n_attempt`, `n_pass` and the
    three collision counters are already zeroed for the NEXT episode by the time `step`
    returns. Reading them off the env gave 0 attempts on every seed.
    """
    env = SingleSurrogate(cfg, seed)
    obs = env.reset()
    policy.reset()
    info = {}
    for _ in range(max_steps):
        obs, info = env.step(policy(obs))
        if info.get("done"):
            break

    def g(key):
        return int(info.get(key, 0) or 0)

    if g("collision_floor"):
        cause = "floor"
    elif g("collision_ceiling"):
        cause = "ceiling"
    elif g("collision_gate"):
        cause = "gate"
    elif info.get("finished"):
        cause = "finished"
    elif info.get("corridor_exit"):
        cause = "corridor"
    elif info.get("gate_timeout"):
        cause = "gate_timeout"
    else:
        cause = "timeout"
    return g("gate_attempts"), g("gate_passes"), g("gates_this_episode"), cause


def run(policy, seeds, noise_scale, args):
    cfg = make_config(noise_scale, args.gates_per_episode, args.vq2_frac,
                      args.difficulty, args.speed_cap)
    att = pas = gat = 0
    causes = {}
    for s in seeds:
        a, p, g, c = fly(policy, cfg, s, args.max_steps)
        att += a
        pas += p
        gat += g
        causes[c] = causes.get(c, 0) + 1
    n = max(len(seeds), 1)
    rate = (pas / att) if att else float("nan")
    # Binomial standard error over ATTEMPTS. Reported because the honest sample size here
    # is small: at K=3 a hundred seeds buy only a couple of hundred plane crossings, and
    # a 0.05 difference in `grate` is inside the noise of that.
    se = float(np.sqrt(rate * (1.0 - rate) / att)) if att else float("nan")
    return dict(grate=rate, se=se,
                attempts=att, passes=pas, gates=gat / n,
                causes={k: v / n for k, v in causes.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="rl", choices=("rl", "baseline"))
    ap.add_argument("--ckpt", default=None, help="checkpoint stem, e.g. vq2_ladder1_s48234496")
    ap.add_argument("--seeds", default="0-99")
    ap.add_argument("--gates-per-episode", type=int, default=3,
                    help="match the run's K -- the training log prints it")
    ap.add_argument("--vq2-frac", type=float, default=1.0)
    ap.add_argument("--difficulty", type=float, default=0.0)
    ap.add_argument("--speed-cap", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scales", default="1.0,0.5,0.0",
                    help="noise_scale values to compare")
    args = ap.parse_args()

    if args.policy == "rl":
        if not args.ckpt:
            ap.error("--policy rl needs --ckpt")
        from pilot.control.policies.rl import RLPolicy
        policy = RLPolicy(args.ckpt, device=args.device)
    else:
        from pilot.control.policies.baseline import BaselinePolicy
        policy = BaselinePolicy()

    seeds = parse_seeds(args.seeds)
    scales = [float(x) for x in args.scales.split(",") if x.strip()]

    print("policy=%s%s  seeds=%d  K=%d  vq2_frac=%.2f  difficulty=%.2f  speed_cap=%.2f"
          % (args.policy, (" " + args.ckpt) if args.ckpt else "", len(seeds),
             args.gates_per_episode, args.vq2_frac, args.difficulty, args.speed_cap))
    print("usable lateral margin 0.536 m = 0.75 half-aperture - 0.214 sphere\n")
    print("%-12s %8s %7s %9s %8s   %s"
          % ("noise_scale", "grate", "+-1se", "attempts", "gates", "endings"))

    rows = []
    for s in scales:
        r = run(policy, seeds, s, args)
        rows.append((s, r))
        ends = "  ".join("%s %.2f" % (k, v) for k, v in sorted(r["causes"].items()))
        print("%-12.2f %8.3f %7.3f %9d %8.2f   %s"
              % (s, r["grate"], r["se"], r["attempts"], r["gates"], ends))

    if len(rows) >= 2:
        noisy, clean = rows[0][1], rows[-1][1]
        # Difference of two proportions. `grate` is pooled over gate ATTEMPTS, so the
        # sample size is the attempt count, not the seed count -- at K=3 those differ by
        # more than an order of magnitude and using seeds would overstate confidence
        # roughly threefold.
        se = float(np.hypot(noisy["se"], clean["se"]))
        diff = clean["grate"] - noisy["grate"]
        print("\nclean - noisy = %+.3f  +-%.3f (1se)" % (diff, se))
        if abs(diff) < 2.0 * se:
            print("  NOT SIGNIFICANT at this sample size: |diff| < 2se. Re-run with more")
            print("  --seeds before reading the verdict below as anything.")
        noisy, clean = noisy["grate"], clean["grate"]
        # The interpretation, spelled out so the number is not read backwards.
        if np.isnan(noisy) or np.isnan(clean):
            print("  inconclusive: no active-gate plane crossings recorded")
        elif clean - noisy > 0.15:
            print("  PERCEPTION-LIMITED: most of the gate-frame failure is the sensor,")
            print("  not the controller. More training buys little; the levers are the")
            print("  noise model and gate_rate_promote (0.60 may be unreachable).")
        elif clean - noisy < 0.05:
            print("  CONTROL-LIMITED: clean detections barely help, so the policy is")
            print("  aiming badly with good information. There is real headroom in")
            print("  training, exploration and the approach geometry.")
        else:
            print("  MIXED: perception accounts for part of it. Worth splitting the")
            print("  remaining budget rather than committing to either lever.")


if __name__ == "__main__":
    raise SystemExit(main())
