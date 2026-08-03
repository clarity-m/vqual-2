# Cross-eval: score the OLD pre-filter checkpoint on the CURRENT (filtered) val split.
#
# Why: the filtered retrain reported corner med 1.60 px vs the 0.849 headline, but the
# filtering shifted every block boundary, so the val set itself changed (>=120 px bucket
# went 267 -> 108 with zero >=120 removals). Same model class, same cache, same split --
# the only variable left is the checkpoint. If the old model also scores ~1.6 here, the
# split got harder and there is no regression; if it scores ~1.0, the filtered training
# genuinely hurt and we train on unfiltered labels while keeping the filter for eval.
#
# Run INSIDE the running gatenet_colab.ipynb session (needs va_items/LOCAL_CACHE/meta):
#   exec(open(f"{DRIVE}/perception/crosseval_colab.py").read())

from torch.utils.data import DataLoader as _DL

_old_ck_path = f"{RUNS_DRIVE}/block/best.pt"   # overnight laptop run, unfiltered 13008
_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_old = G.GateNet(1.0).to(_dev)
_ck = torch.load(_old_ck_path, map_location=_dev, weights_only=False)
_old.load_state_dict(_ck["model"]); _old.eval()
print(f"OLD checkpoint: {_old_ck_path}")
print(f"  epoch {_ck['epoch']}, its own best val corner med {_ck['best']:.3f} px "
      f"(on the UNfiltered split -- not comparable, shown for identification only)")
print(f"NEW val set: {len(va_items)} instances, filtered labels, block split\n")

_va = _DL(P.CachedGateCrops(va_items, LOCAL_CACHE, meta, train=False),
          batch_size=256, shuffle=False,
          num_workers=WORKERS if "WORKERS" in dir() else 4)
_res = G.evaluate(_old, _va, _dev, va_items)
print(G.fmt_rows(G.breakdown(_res, va_items)))
print("\nRead-off: compare the ALL row above against the new run's 1.60 px on this "
      "same set. Close to 1.6 -> split got harder, keep the filtered model. "
      "Close to 1.0 -> real regression from filtered training.")
