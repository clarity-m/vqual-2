"""E5 model selection: rank candidates by completion under worst-case draws.

    python pilot/control/evalsuite/select.py --baseline --ckpt a.pt --ckpt b.pt \
        --seeds 0-19 --draws 3 --percentile 80 --json out/select.json

THE RANKING, and why it is shaped like this (TRAINING_ARCHITECTURE.md E5):

  primary   completion rate over a HIGH-PERCENTILE-DIFFICULTY subset of draws.
            Not the literal worst draw -- one pathological seed is noise, and ranking
            on a max means ranking on which candidate got the unluckiest single world.
  secondary MEAN-case time, over every completed run. Deliberately not worst-case
            time: selecting on that as well just breeds over-conservative policies,
            and the leaderboard counts completed runs first, fast runs second.

DIFFICULTY IS A PROPERTY OF THE DRAW, NOT OF THE CANDIDATE. Each candidate's own worst
draws are wherever it happens to be weak, and those subsets are not comparable across
candidates. So every candidate flies the SAME draw list, a draw's difficulty is the
fraction of candidates that failed it, and the hardest `100 - percentile` per cent of
draws by that measure become the scoring subset -- one subset, shared. With a single
candidate this degenerates to "the draws it failed", which is why selection wants at
least two.

A draw is a seed. `SingleSurrogate(config, seed)` seeds the course and the noise
parameters jointly, so this ranks over joint (course, noise) draws rather than over
noise alone -- the contract exposes one seed, not two. `--draws N` expands each seed
into N further seeds so a small held-out seed list can still span many noise draws.
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pilot.control.evalsuite.run_policy import (                 # noqa: E402
    add_batch_arg, add_common_args, build_config, config_dict, evaluate, make_policy,
    parse_pair, parse_seeds, run_batched, summarize, write_json)

DRAW_STRIDE = 997   # coprime with anything a human types into --seeds


def expand_draws(seeds, draws):
    out = []
    for s in seeds:
        for j in range(max(1, int(draws))):
            out.append(int(s) * DRAW_STRIDE + j if draws > 1 else int(s))
    return out


def hard_subset(per_candidate, percentile):
    """The shared high-difficulty draws: (seeds, per-draw failure fraction)."""
    by_name = {name: {r["seed"]: r for r in rs} for name, rs in per_candidate.items()}
    seeds = [r["seed"] for r in next(iter(per_candidate.values()))]
    fail = {}
    for s in seeds:
        done = [1.0 if by_name[n][s]["completed"] else 0.0
                for n in by_name if s in by_name[n]]
        fail[s] = 1.0 - (sum(done) / len(done) if done else 0.0)
    n_hard = max(1, int(round(len(seeds) * (1.0 - percentile / 100.0))))
    ordered = sorted(seeds, key=lambda s: (-fail[s], s))   # seed breaks ties: stable
    return ordered[:n_hard], fail


def score(per_candidate, hard, tie_tol=0.05):
    rows = []
    for name, results in per_candidate.items():
        by_seed = {r["seed"]: r for r in results}
        sub = [by_seed[s] for s in hard if s in by_seed]
        worst = sum(1 for r in sub if r["completed"]) / max(len(sub), 1)
        done = [r["time_s"] for r in results if r["completed"]]
        mean_time = sum(done) / len(done) if done else float("inf")
        rows.append(dict(name=name, worst_completion=round(worst, 4),
                         mean_time_completed=(round(mean_time, 3) if done else None),
                         **{k: v for k, v in summarize(results).items()
                            if k in ("completion_rate", "gates_mean",
                                     "collisions_mean", "n")}))
    # Bucket the primary score so that near-ties really are decided by mean time.
    rows.sort(key=lambda r: (-round(r["worst_completion"] / tie_tol),
                             r["mean_time_completed"]
                             if r["mean_time_completed"] is not None else float("inf")))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows


def print_table(rows, hard, fail, percentile):
    print("\nscoring subset: %d hardest of %d draws (>= %gth percentile difficulty)"
          % (len(hard), len(fail), percentile))
    print("  %s" % ", ".join("%d(%.0f%%fail)" % (s, 100 * fail[s]) for s in hard[:12])
          + (" ..." if len(hard) > 12 else ""))
    print("\n%-4s %-34s %10s %10s %9s %7s" % ("rank", "candidate", "worst-case",
                                              "overall", "mean time", "gates"))
    print("%-4s %-34s %10s %10s %9s %7s" % ("", "", "completion", "completion",
                                            "(done)", "mean"))
    for r in rows:
        print("%-4d %-34s %10.3f %10.3f %9s %7.2f"
              % (r["rank"], r["name"][:34], r["worst_completion"], r["completion_rate"],
                 "%.2f" % r["mean_time_completed"]
                 if r["mean_time_completed"] is not None else "--",
                 r["gates_mean"]))
    print("\nselected: %s" % (rows[0]["name"] if rows else "-- no candidates --"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", action="append", default=[],
                    help="candidate checkpoint; repeatable")
    ap.add_argument("--baseline", action="store_true", help="include the baseline")
    ap.add_argument("--gains", default="", help="baseline gains")
    ap.add_argument("--supervisor", action="store_true",
                    help="wrap every candidate in the recovery supervisor")
    ap.add_argument("--draws", type=int, default=1,
                    help="noise draws per seed (each becomes its own derived seed)")
    ap.add_argument("--percentile", type=float, default=80.0,
                    help="score on the hardest 100-P per cent of draws")
    ap.add_argument("--device", default="cpu")
    add_common_args(ap)
    add_batch_arg(ap)
    args = ap.parse_args(argv)

    if not args.baseline and not args.ckpt:
        raise SystemExit("nothing to rank: pass --baseline and/or --ckpt")

    seeds = expand_draws(parse_seeds(args.seeds), args.draws)
    cfg = build_config(args.difficulty, args.speed_cap,
                       decision_hz=parse_pair(args.decision_hz),
                       n_gates=parse_pair(args.n_gates),
                       time_penalty=args.time_penalty, vq2_frac=args.vq2_frac,
                       random_starts=args.random_starts)

    candidates = []
    if args.baseline:
        candidates.append(("baseline", None,
                           make_policy("baseline", gains=args.gains,
                                       supervisor=args.supervisor)))
    for c in args.ckpt:
        candidates.append(("rl", c, make_policy("rl", ckpt=c, supervisor=args.supervisor,
                                                device=args.device)))

    # Every candidate must fly the SAME draw list -- `hard_subset` calls a draw hard
    # because several candidates failed it, which is meaningless across different worlds.
    # Both paths hold that: `evaluate` replays the seed list, and `evaluate_batch` is
    # deterministic in (config, n_lanes, seed) and scores only the policy-independent
    # first episode of each lane.
    per_candidate = {}
    for kind, ckpt, (policy, name) in candidates:
        print("\nflying %s over %d draws ..." % (name, len(seeds)))
        if args.batch:
            shim = argparse.Namespace(policy=kind, ckpt=ckpt, gains=args.gains,
                                      supervisor=args.supervisor, device=args.device,
                                      max_steps=args.max_steps)
            results = run_batched(shim, cfg, policy, seeds)
        else:
            results = evaluate(policy, cfg, seeds, max_steps=args.max_steps)
        per_candidate[name] = results
        s = summarize(results)
        print("  completion %.3f | gates %.2f | time %s"
              % (s["completion_rate"], s["gates_mean"],
                 s["time_mean_completed"] if s["time_mean_completed"] else "--"))

    hard, fail = hard_subset(per_candidate, args.percentile)
    rows = score(per_candidate, hard)
    print_table(rows, hard, fail, args.percentile)
    write_json(args.json, dict(config=config_dict(cfg), seeds=seeds,
                               percentile=args.percentile, hard_seeds=hard,
                               draw_failure_fraction=fail, ranking=rows,
                               results=per_candidate))
    return 0


if __name__ == "__main__":
    sys.exit(main())
