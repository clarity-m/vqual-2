"""E5's decision-rate stress test: the same evaluation pinned to ~30 Hz.

    python pilot/control/evalsuite/stress.py --seeds 0-19 --difficulty 0.2
    python pilot/control/evalsuite/stress.py --policy rl --ckpt run3 --supervisor

Training runs at a per-episode randomized 45-65 Hz, which is what the VQ2 pilot
measured on an unloaded machine. The recordings also show a ~32 Hz tier on LOADED
sessions (NOTES.md: IMU rate is load-dependent, 62.9 Hz on one build against 47.7 on
another), and race day is a loaded machine. A policy whose gains only work at 55 Hz is
a policy that degrades exactly when the run counts.

So: same seeds, same config, `decision_hz_range` pinned to (28, 32), and the two runs
printed side by side. The number to look at is the completion delta. For the baseline
this is also a stability check on the proportional loops -- `k_roll * dt` grows as the
rate falls, and a gain that is deadbeat at 30 Hz is oscillatory at 25.
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pilot.control.evalsuite.run_policy import (                 # noqa: E402
    add_common_args, add_policy_args, build_config, config_dict, evaluate,
    make_policy, parse_pair, parse_seeds, print_results, summarize, write_json)

STRESS_HZ = (28.0, 32.0)


def compare(policy, name, seeds, difficulty, speed_cap, stress_hz=STRESS_HZ,
            nominal_hz=None, n_gates=None, time_penalty=None, max_steps=5000,
            quiet=False, vq2_frac=None):
    runs = {}
    for label, hz in (("nominal", nominal_hz), ("stress %g-%g Hz" % stress_hz, stress_hz)):
        cfg = build_config(difficulty, speed_cap, decision_hz=hz, n_gates=n_gates,
                           time_penalty=time_penalty, vq2_frac=vq2_frac)
        print("\n--- %s ---" % label)
        results = evaluate(policy, cfg, seeds, max_steps=max_steps)
        summary = summarize(results)
        print_results("%s @ %s" % (name, label), results, summary, per_seed=not quiet)
        runs[label] = dict(config=config_dict(cfg), results=results, summary=summary)
    return runs


def print_delta(runs):
    labels = list(runs)
    a, b = runs[labels[0]]["summary"], runs[labels[1]]["summary"]
    print("\n%-22s %10s %10s %9s" % ("", labels[0], labels[1], "delta"))
    for key, fmt in (("completion_rate", "%.3f"), ("gates_mean", "%.2f"),
                     ("course_frac_mean", "%.3f"), ("collisions_mean", "%.2f")):
        va, vb = a.get(key), b.get(key)
        print("%-22s %10s %10s %9s"
              % (key, fmt % va, fmt % vb, fmt % (vb - va)))
    ta, tb = a.get("time_mean_completed"), b.get("time_mean_completed")
    print("%-22s %10s %10s %9s"
          % ("time_mean_completed",
             "%.2f" % ta if ta else "--", "%.2f" % tb if tb else "--",
             "%.2f" % (tb - ta) if (ta and tb) else "--"))
    drop = (a.get("completion_rate", 0.0) - b.get("completion_rate", 0.0))
    print("\ncompletion drops %.1f points at ~30 Hz" % (100.0 * drop))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_policy_args(ap)
    add_common_args(ap)
    ap.add_argument("--stress-hz", default="28,32", help="the pinned decision rate")
    args = ap.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    policy, name = make_policy(args.policy, args.ckpt, args.gains, args.supervisor,
                               args.device)
    print("%s | difficulty %.2f speed_cap %.2f | %d seeds"
          % (name, args.difficulty, args.speed_cap, len(seeds)))

    runs = compare(policy, name, seeds, args.difficulty, args.speed_cap,
                   stress_hz=parse_pair(args.stress_hz),
                   nominal_hz=parse_pair(args.decision_hz),
                   n_gates=parse_pair(args.n_gates), time_penalty=args.time_penalty,
                   max_steps=args.max_steps, quiet=args.quiet)
    print_delta(runs)
    write_json(args.json, dict(policy=name, seeds=seeds, runs=runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
