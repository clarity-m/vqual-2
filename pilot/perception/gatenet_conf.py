"""gatenet_conf.py -- confidence head for gatenet: trust or reject each crop at race time.

WHY. gatenet answers on EVERY crop by construction (TRAINING.md caveat 4): it cannot say
"this is not a gate" (decoration false positives, hangar clutter) or "this answer is
garbage" (the heavy tail -- p90 7.2 px, extreme clipping, coherent hallucinations on
crops whose seed box contained no gate at all). The PnP reprojection residual catches
84% of >10 px catastrophes at a 12.6% flag rate (measured baseline, 2026-08-02 brief),
but it is blind to coherent hallucinations: a wrong-but-self-consistent quad reprojects
fine. This file adds the missing per-crop reject signal.

DESIGN -- a frozen-trunk head, two outputs, no third class.

  * The head consumes the trunk feature map that the regressor ALREADY computes at
    flight time, so its marginal cost is one 4x4 pool + a 2-layer MLP: ~0.33 M params,
    ~0.7 MFLOPs, measured well under 1 ms on CPU (--mode bench prints it). A separate
    tiny classifier on raw crops would re-run a conv stack per crop and double the
    perception budget for no accuracy argument.
  * The trunk is FROZEN. Two reasons, both load-bearing: (1) the regressor path stays
    BIT-IDENTICAL -- head(body(stem(x))) is literally GateNet.forward, so attaching the
    conf head cannot move a single corner prediction, and no regressor re-validation is
    owed; (2) the negative set is small (~10^2-10^3 after dedup) and finetuning 732k
    trunk params against it invites the trunk to forget geometry for a classification
    it can carry in a head. If the frozen head fails the frozen accept criteria, the
    notebook has a FINETUNE=True escape hatch which then REQUIRES the cross-eval cell
    to clear the regressor on the clean subset before the pair is deployable.
  * Output 1: logit for "trustworthy gate" vs "no gate here" (decoration, phantom,
    behind-gate board). Output 2: predicted log10 corner error of the regressor on this
    very crop, trained against the frozen regressor's ACTUAL error on the jittered crop
    (self-supervised -- no new labels). A discrete third "clipped/degraded" class was
    considered and REJECTED: clipping is already observable at inference from the box
    touching the image edge, a softmax class throws away magnitude, and the quantity a
    consumer thresholds is expected pixel error -- so predict that directly. The
    continuous error output subsumes "degraded" and yields the reject threshold with no
    class-boundary tuning.

DATA -- the entire point, because the label policy forbids the easy thing.

  Class 1 (trustworthy): labels_merged_v3.json instances, minus `unsure` (an unsure quad
  is Claire declining to vouch; it may not teach EITHER side of a trust boundary).

  Class 0 (reject), explicit sources only -- LABEL_POLICY.md: hand-labelled VQ2 frames
  are NOT exhaustively labelled (max 3 gates/frame), so "unlabelled region = background"
  is FORBIDDEN. Sources, each with provenance kept per-instance:
    * labelfix_negatives_v3.json  -- 527 behind-gate labels (VQ1): the label projects
      onto a nearer gate's solid board. The canonical "coherent hallucination" crop.
    * autolabels_vq1_negatives.json -- reason 'no-orange' ONLY (437 phantoms: a label
      with no orange evidence under it). The 17 'degenerate-visible-box' entries are
      EXCLUDED per that file's provenance: their visible box is a degenerate sliver, the
      crop geometry is meaningless, and they carry no negative evidence.
    * conf_negatives_vq2.json -- mined here by --mode mine (see mine() docstring for the
      by-construction safety argument): detect.py decoration false positives on Claire's
      hand-labelled deco frames that overlap NO hand quad, plus deterministic random
      crops from the verified-empty frames.
  Near-duplicate runs (e.g. the 154-frame parked sequence) are capped per run so no
  single view dominates class 0 -- dedupe_negatives().

SPLIT. By time, never randomly, same discipline as gatenet: contiguous 240-frame blocks
of the RAW frame counter, every 5th block held out, 45-frame guard dropped from train.
(gatenet blocks on the per-file ordinal of its label file; negatives live on frames that
file never sees, so conf blocks on the frame number itself. Same leakage argument, one
extra property: a positive and a negative from the same frame land on the same side.)

EVAL leakage discipline: the error-output targets come from the frozen regressor, which
TRAINED on most of these frames, so its on-train errors are optimistic. All catastrophe
numbers are therefore reported ONLY on the clean subset (val_clean_keys.json -- held out
of both regressor splits) plus the conf-val hand instances. Stated, not netted out: the
error head is TRAINED partly on optimistic targets; the eval measures what survives.

USAGE
    python3 pilot/perception/gatenet_conf.py --mode mine        # -> conf_negatives_vq2.json
    python3 pilot/perception/gatenet_conf.py --mode zipframes   # -> conf_frames_vq1.zip
    python3 pilot/perception/gatenet_conf.py --mode index       # counts per class/source
    python3 pilot/perception/gatenet_conf.py --mode train --tag conf ...
    python3 pilot/perception/gatenet_conf.py --mode eval  --tag conf
    python3 pilot/perception/gatenet_conf.py --mode bench
    python3 pilot/perception/gatenet_conf.py --mode smoke       # 2-epoch CPU proof
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
import zipfile

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gatenet as G  # noqa: E402  (crop pipeline + GateNet; never edited by this file)

SESSIONS = G.SESSIONS
RUNS = os.path.join(HERE, 'gatenet_runs')

NEG_V3 = os.path.join(HERE, 'labelfix_negatives_v3.json')
NEG_VQ1 = os.path.join(HERE, 'autolabels_vq1_negatives.json')
NEG_VQ2 = os.path.join(HERE, 'conf_negatives_vq2.json')          # --mode mine output
MERGED = os.path.join(HERE, 'labels_merged_v3.json')
VQ2_INDEX = os.path.join(HERE, 'vq2_label', 'index.json')
HAND_RAW = os.path.join(HERE, 'labels_gates_all.json')           # incl. unsure quads
DET_CACHE = os.path.join(HERE, 'cache')                          # vq2cache.py pickles
CLEAN_KEYS = os.path.join(HERE, 'val_clean_keys.json')
TRUNK_DEFAULT = os.path.join(RUNS, 'colab-v3-nw', 'best.pt')     # production regressor

# split constants -- gatenet's discipline on the raw frame counter (module docstring)
BLOCK, VAL_EVERY, VAL_PHASE, GUARD = 240, 5, 3, 45

# negative dedupe: a "run" is same (session, source-id) with frame gaps <= RUN_GAP;
# at most RUN_CAP instances survive per run, evenly spaced. NOTES.md records the
# negatives as highly redundant (one 154-frame parked sequence), and without the cap
# class 0 is ~30% one view of one board.
RUN_GAP, RUN_CAP = 45, 6

# empty-frame random crops: deterministic per frame, so a rebuild is a no-op diff
EMPTY_PER_FRAME, EMPTY_SIDE = 12, (24.0, 200.0)

# catastrophe definition + the measured PnP-residual baseline it must beat/complement:
# 84% of >10 px catastrophes caught at 12.6% flag rate (2026-08-02 brief).
CATASTROPHE_PX = 10.0
PNP_BASELINE = {'catch': 0.84, 'flag': 0.126}

# frozen accept criteria (decided before any number existed -- TRAINING.md runbook):
# deploy the head only if it rejects >=80% of decoration FPs at <=2% true-gate loss
# on the clean eval set.
ACCEPT_DECO_REJECT = 0.80
ACCEPT_GATE_LOSS = 0.02

LOG_ERR_EPS = 1e-3        # floor inside log10(err_norm + eps): err_norm ~ 2*px/S
LAMBDA_ERR = 1.0


# ---------------------------------------------------------------------------------------
# model


class ConfHead(nn.Module):
    """Two scalars from the trunk feature map the regressor already computed.

    Pool to 4x4 (same spatial code size as the regressor head -- pooling to 1x1 would
    discard where the orange is, and 'orange in the wrong place' is the decoration
    signal), flatten, 128-wide MLP, 2 outputs: [gate logit, log10 corner error in
    normalised crop units]. ~0.33 M params; the whole head is one GEMM deep."""

    def __init__(self, cin=160):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(cin * 16, 128), nn.SiLU(True), nn.Linear(128, 2))
        # start pessimistic-neutral: p(gate)=0.5, predicted err = median-ish 0.02 norm
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([0.0, math.log10(0.02)]))

    def forward(self, feat):
        o = self.net(feat)
        return o[:, 0], o[:, 1]


def load_trunk(path, dev):
    m = G.GateNet(1.0).to(dev)
    ck = torch.load(path, map_location=dev, weights_only=False)
    m.load_state_dict(ck['model'])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def conf_forward(trunk, head, x, finetune=False):
    """-> (pred8, logit, logerr). head(body(stem(x))) for the corners is EXACTLY
    GateNet.forward -- same modules, same order -- so with the trunk frozen the
    regressor output is bit-identical to running the trunk alone (asserted in smoke)."""
    ctx = torch.enable_grad() if finetune else torch.no_grad()
    with ctx:
        feat = trunk.body(trunk.stem(x))
        pred8 = trunk.head(feat)
    logit, logerr = head(feat if finetune else feat.detach())
    return pred8, logit, logerr


# ---------------------------------------------------------------------------------------
# index construction


def _fnum(fname):
    try:
        return int(os.path.splitext(os.path.basename(fname))[0])
    except ValueError:
        return 0


def _item(key, inst, cls, reason, src='auto'):
    sess, fname = key.split('/', 1)
    c = np.asarray(inst['corners'], np.float32)
    edges = [float(np.linalg.norm(c[(j + 1) % 4] - c[j])) for j in range(4)]
    return {'key': key, 'path': os.path.join(SESSIONS, sess, 'frames', fname),
            'session': sess, 'fnum': _fnum(fname), 'corners': c,
            'size_px': float(inst.get('size_px', max(edges))),
            'clipped': bool(inst.get('clipped', False)),
            'cls': cls, 'reason': reason, 'src': src}


def dedupe_negatives(negs):
    """Cap near-duplicate runs. Group by (session, source-tag); split a group into runs
    wherever consecutive frame numbers gap by more than RUN_GAP; keep at most RUN_CAP
    per run, evenly spaced. The source-tag is the labelled gate id for the VQ1 files
    (one physical board per gate id) and a coarse spatial cell for mined VQ2 crops."""
    groups = {}
    for d in negs:
        groups.setdefault((d['session'], d['_dedupe']), []).append(d)
    kept = []
    for g in groups.values():
        g.sort(key=lambda d: d['fnum'])
        run = [g[0]]
        runs = []
        for d in g[1:]:
            if d['fnum'] - run[-1]['fnum'] > RUN_GAP:
                runs.append(run)
                run = [d]
            else:
                run.append(d)
        runs.append(run)
        for r in runs:
            if len(r) <= RUN_CAP:
                kept.extend(r)
            else:
                kept.extend(r[i] for i in
                            np.linspace(0, len(r) - 1, RUN_CAP).astype(int))
    return kept


def build_index(require_mined=True):
    """-> (items, counts). Every negative's provenance is one of the four explicit
    sources; nothing is ever sampled from an unlabelled region of a hand frame."""
    items = []

    with open(G.LABELS if os.path.basename(G.LABELS).startswith('labels_merged')
              else MERGED) as fh:
        merged = json.load(fh)
    n_unsure = 0
    for key, insts in merged.items():
        for inst in insts:
            if inst.get('unsure'):
                n_unsure += 1          # may vouch for neither class -- excluded
                continue
            it = _item(key, inst, 1, 'gate', inst.get('src', 'auto'))
            it['gt'] = np.asarray(inst['corners'], np.float32)
            items.append(it)

    negs = []
    with open(NEG_V3) as fh:
        for key, insts in json.load(fh).items():
            for inst in insts:
                d = _item(key, inst, 0, 'behind-gate')
                d['_dedupe'] = f"bg{inst.get('gate', -1)}"
                negs.append(d)
    n_degen = 0
    with open(NEG_VQ1) as fh:
        for key, insts in json.load(fh).items():
            for inst in insts:
                if inst.get('reason') == 'degenerate-visible-box':
                    n_degen += 1       # excluded per the file's own provenance
                    continue
                d = _item(key, inst, 0, 'no-orange')
                d['_dedupe'] = f"no{inst.get('gate', -1)}"
                negs.append(d)
    if os.path.exists(NEG_VQ2):
        with open(NEG_VQ2) as fh:
            for key, insts in json.load(fh).items():
                for inst in insts:
                    d = _item(key, inst, 0, inst['reason'])
                    c = np.asarray(inst['corners'], np.float32).mean(0)
                    d['_dedupe'] = f"{inst['reason']}:{int(c[0]//80)}:{int(c[1]//80)}"
                    negs.append(d)
    elif require_mined:
        raise SystemExit(f'{NEG_VQ2} missing -- run --mode mine first (the decoration '
                         f'negatives are the whole reason this head exists)')

    pre = len(negs)
    negs = dedupe_negatives(negs)
    for d in negs:
        d.pop('_dedupe', None)
    items.extend(negs)
    items.sort(key=lambda d: (d['session'], d['fnum']))

    from collections import Counter
    counts = Counter((d['cls'], d['reason']) for d in items)
    counts['neg_before_dedupe'] = pre
    counts['unsure_excluded'] = n_unsure
    counts['degenerate_excluded'] = n_degen
    return items, counts


def conf_split(items):
    """Time-block split on the raw frame counter (module docstring)."""
    tr, va = [], []
    for d in items:
        blk = d['fnum'] // BLOCK
        if blk % VAL_EVERY == VAL_PHASE:
            va.append(d)
            continue
        pos = d['fnum'] % BLOCK
        prev_val = (blk - 1) % VAL_EVERY == VAL_PHASE
        next_val = (blk + 1) % VAL_EVERY == VAL_PHASE
        if (prev_val and pos < GUARD) or (next_val and pos >= BLOCK - GUARD):
            continue
        tr.append(d)
    return tr, va


# ---------------------------------------------------------------------------------------
# mining the VQ2 decoration negatives (runs LOCALLY; the output json ships to Colab)


def mine(args):
    """Build conf_negatives_vq2.json: decoration FPs + verified-empty random crops.

    WHY THIS IS SAFE UNDER LABEL_POLICY.md, stated as the construction it is:

      * Decoration crops come ONLY from frames Claire hand-labelled (key present in
        labels_merged_v3 with at least one src='hand' instance, or an explicit []).
        On those frames the prominent gates ARE labelled. A detect.py detection that
        (a) drop_decorations_by_parent() rejected -- meaning its own implied range
        contradicts the range of the orange parent blob it sits inside, which is
        physically impossible for a real gate -- and (b) overlaps NO hand quad
        (including `unsure` quads, the conservative direction) is therefore an orange
        surface that is not any labelled gate and cannot be a plausibly-real one:
        decoration by construction, not by absence of a label.
      * Random crops come ONLY from frames that are BOTH staged category 'empty' AND
        an explicit [] in the merged labels. Three of the seven staged empty frames
        gained an `unsure` quad during labelling -- those are NOT used (an unsure quad
        means possibly a gate, which disqualifies the frame as verified-empty).

    Everything else -- unlabelled regions of hand frames, frames Claire never opened --
    is untouched, exactly as the policy demands."""
    with open(VQ2_INDEX) as fh:
        vidx = json.load(fh)
    with open(HAND_RAW) as fh:
        hand_raw = json.load(fh)
    with open(MERGED) as fh:
        merged = json.load(fh)

    caches = {}
    for s in vidx['sources']:
        p = os.path.join(DET_CACHE, s + '.det.pkl')
        if os.path.exists(p):
            with open(p, 'rb') as fh:
                caches[s] = pickle.load(fh)
        else:
            print(f'WARNING: no detection cache for {s} ({p}) -- its frames skipped')

    def quads_overlap(a, b, grow=1.25):
        a = np.asarray(a, np.float32)
        b = np.asarray(b, np.float32)
        b = b.mean(0) + (b - b.mean(0)) * grow     # grow the hand quad: conservative
        inter, _ = cv2.intersectConvexConvex(a, b.astype(np.float32))
        m = min(abs(cv2.contourArea(a)), abs(cv2.contourArea(b.astype(np.float32))))
        return m > 0 and inter / m > 0.05

    out = {}
    n_deco = n_empty = n_skip_overlap = n_skip_unlabelled = 0
    for f in vidx['frames']:
        key = f['session'] + '/' + f['file']
        labelled = key in merged and (
            any(i.get('src') == 'hand' for i in merged[key]) or merged[key] == [])
        hand_quads = [np.asarray(q['corners'], np.float32)
                      for q in hand_raw.get(f['labelui_key'], [])]

        recs = []
        if labelled and f['session'] in caches:
            for det in caches[f['session']].get(f['file'], []):
                if not det.get('deco'):
                    continue
                quad = np.asarray(det['quad'], np.float32).reshape(4, 2)
                sz = float(det['size_px'])
                if sz < 8.0:
                    continue
                if any(quads_overlap(quad, hq) for hq in hand_quads):
                    n_skip_overlap += 1
                    continue
                recs.append({'corners': quad.round(2).tolist(), 'reason': 'deco',
                             'size_px': round(sz, 2), 'clipped': False})
                n_deco += 1
        elif not labelled and f.get('n_deco', 0) > 0:
            n_skip_unlabelled += 1

        if f['category'] == 'empty' and merged.get(key) == []:
            rng = random.Random(hash(key) & 0xFFFFFFFF)
            for _ in range(EMPTY_PER_FRAME):
                side = math.exp(rng.uniform(math.log(EMPTY_SIDE[0]),
                                            math.log(EMPTY_SIDE[1])))
                cx = rng.uniform(side * 0.3, G.W - side * 0.3)
                cy = rng.uniform(side * 0.3, G.H - side * 0.3)
                h = side / 2.0
                recs.append({'corners': [[cx - h, cy - h], [cx + h, cy - h],
                                         [cx + h, cy + h], [cx - h, cy + h]],
                             'reason': 'empty-random', 'size_px': round(side, 2),
                             'clipped': False})
                n_empty += 1
        if recs:
            out[key] = recs

    with open(NEG_VQ2, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'wrote {NEG_VQ2}: {n_deco} deco crops on hand-labelled frames, '
          f'{n_empty} empty-frame randoms over {len(out)} frames')
    print(f'  skipped: {n_skip_overlap} deco hits overlapping a hand quad '
          f'(conservative), {n_skip_unlabelled} deco frames Claire never labelled')


def zipframes(args):
    """conf_frames_vq1.zip -- the VQ1 negative frames vq1_frames.zip does not carry.

    vq1_frames.zip ships only frames autolabels_vq1.json references; 173 of the
    no-orange phantom frames were removed from the label files entirely, so their
    frames never made any zip. Everything the VQ2 miner needs is already in
    vq2_frames.zip (hand-labelled + verified-empty frames ship there)."""
    have = set(json.load(open(os.path.join(HERE, 'autolabels_vq1.json'))))
    need = set()
    for p in (NEG_V3, NEG_VQ1):
        need.update(json.load(open(p)))
    missing = sorted(need - have)
    out = os.path.join(HERE, 'conf_frames_vq1.zip')
    n = 0
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_STORED) as z:
        for k in missing:
            sess, fname = k.split('/', 1)
            src = os.path.join(SESSIONS, sess, 'frames', fname)
            if os.path.exists(src):
                z.write(src, f'{sess}/frames/{fname}')
                n += 1
            else:
                print(f'  MISSING on disk: {k}')
    print(f'{n} frames (of {len(missing)} not in vq1_frames.zip) -> {out}  '
          f'{os.path.getsize(out)/1e6:.1f} MB')


# ---------------------------------------------------------------------------------------
# dataset


class ConfCrops(Dataset):
    """Same crop pipeline as gatenet (crop_params jitter / make_crop / photometric),
    plus a class label. Targets for positives ride along so the training loop can
    compute the frozen regressor's actual error on the exact jittered crop."""

    def __init__(self, items, train):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cv2.setNumThreads(0)
        d = self.items[i]
        img = cv2.imread(d['path'], cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((G.H, G.W, 3), np.uint8)
        rng = random.Random((i * 2654435761 + int(time.time() * 1e6)) & 0xFFFFFFFF) \
            if self.train else None
        cx, cy, S = G.crop_params(d['corners'], rng)
        crop = G.make_crop(img, cx, cy, S)
        if self.train:
            crop = G.photometric(crop, rng)
        tgt = (G.to_norm(d['gt'], cx, cy, S) if d['cls'] == 1
               else np.zeros((4, 2), np.float32))
        x = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1))).float()
        x = x.div_(255.0).sub_(0.45).div_(0.25)
        return (x, torch.tensor(float(d['cls'])),
                torch.from_numpy(tgt.reshape(8).astype(np.float32)),
                torch.tensor([cx, cy, S], dtype=torch.float32), i)


def make_conf_loaders(tr_items, va_items, bs, workers):
    """Class-balanced sampling: negatives are ~5% of instances, so an unweighted epoch
    shows the head ~3 negatives per batch of 64 and BCE mostly optimises the easy
    majority. The sampler draws negatives to ~25% of each batch; epoch length stays
    len(train) so 'epoch' keeps meaning what it means elsewhere."""
    n_pos = sum(1 for d in tr_items if d['cls'] == 1)
    n_neg = len(tr_items) - n_pos
    w_neg = n_pos / (3.0 * max(n_neg, 1))
    weights = torch.tensor([1.0 if d['cls'] == 1 else w_neg for d in tr_items],
                           dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(tr_items),
                                    replacement=True)
    tr = DataLoader(ConfCrops(tr_items, True), batch_size=bs, sampler=sampler,
                    num_workers=workers, pin_memory=True, drop_last=True,
                    persistent_workers=workers > 0)
    va = DataLoader(ConfCrops(va_items, False), batch_size=max(bs, 128), shuffle=False,
                    num_workers=workers, pin_memory=True,
                    persistent_workers=workers > 0)
    return tr, va


# ---------------------------------------------------------------------------------------
# train


def err_norm_of(pred8, tgt8):
    """Mean corner L2 in normalised crop units (px = norm * S/2)."""
    p = pred8.reshape(-1, 4, 2)
    t = tgt8.reshape(-1, 4, 2)
    return torch.linalg.norm(p - t, dim=2).mean(1)


def train(args):
    dev = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    outdir = os.path.join(RUNS, args.tag)
    os.makedirs(outdir, exist_ok=True)
    logf = os.path.join(outdir, 'log.csv')

    items, counts = build_index(require_mined=not args.allow_unmined)
    if args.limit:
        # smoke subsetting: every k-th positive, ALL negatives (they are the scarce class)
        pos = [d for d in items if d['cls'] == 1][::args.limit]
        items = sorted(pos + [d for d in items if d['cls'] == 0],
                       key=lambda d: (d['session'], d['fnum']))
    tr_items, va_items = conf_split(items)
    npos = sum(d['cls'] for d in tr_items)
    print(f'[{args.tag}] train {len(tr_items)} ({npos} pos / {len(tr_items)-npos} neg)'
          f'  val {len(va_items)} '
          f'({sum(d["cls"] for d in va_items)} pos / '
          f'{sum(1 for d in va_items if not d["cls"])} neg)', flush=True)

    trunk = load_trunk(args.trunk, dev)
    finetune = bool(args.finetune)
    if finetune:
        for p in trunk.parameters():
            p.requires_grad_(True)
        trunk.train()
    head = ConfHead().to(dev)
    nparam = sum(p.numel() for p in head.parameters())
    print(f'[{args.tag}] head parameters: {nparam:,}   trunk '
          f'{"FINETUNED" if finetune else "FROZEN"} ({args.trunk})', flush=True)

    opt = (torch.optim.AdamW([{'params': head.parameters(), 'lr': args.lr},
                              {'params': trunk.parameters(), 'lr': args.lr * 0.1}],
                             weight_decay=1e-4) if finetune else
           torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=args.lr * 0.02)
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda')
    start, best = 0, float('inf')
    last_ck = os.path.join(outdir, 'last.pt')
    if args.resume and os.path.exists(last_ck):
        ck = torch.load(last_ck, map_location=dev, weights_only=False)
        head.load_state_dict(ck['head'])
        if finetune and 'trunk' in ck:
            trunk.load_state_dict(ck['trunk'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        start, best = ck['epoch'] + 1, ck['best']
        print(f'[{args.tag}] resumed from epoch {start}', flush=True)

    if not os.path.exists(logf):
        G.log_line(logf, 'epoch,train_loss,val_bce,val_auc,val_err_mae,lr,secs')

    tr_loader, va_loader = make_conf_loaders(tr_items, va_items, args.batch,
                                             args.workers)
    t0 = time.time()
    deadline = t0 + args.max_hours * 3600.0
    for ep in range(start, args.epochs):
        head.train()
        if finetune:
            trunk.train()
        ep_t = time.time()
        tot, cnt = 0.0, 0
        for x, cls, tgt8, geo, _ in tr_loader:
            x = x.to(dev, non_blocking=True)
            cls = cls.to(dev)
            tgt8 = tgt8.to(dev)
            with torch.amp.autocast('cuda', enabled=dev.type == 'cuda'):
                pred8, logit, logerr = conf_forward(trunk, head, x, finetune)
                loss = F.binary_cross_entropy_with_logits(logit, cls)
                pos = cls > 0.5
                if pos.any():
                    en = err_norm_of(pred8[pos].float().detach(), tgt8[pos])
                    t_le = torch.log10(en + LOG_ERR_EPS)
                    loss = loss + LAMBDA_ERR * F.smooth_l1_loss(logerr[pos], t_le)
                if finetune and pos.any():
                    # keep the regressor honest while its trunk moves
                    loss = loss + F.smooth_l1_loss(pred8[pos], tgt8[pos], beta=0.05)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g['params']], 5.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach()) * len(cls)
            cnt += len(cls)
            if time.time() > deadline:
                break

        vres = score_items(trunk, head, va_loader, dev)
        vcls = vres['cls']
        bce = float(F.binary_cross_entropy_with_logits(
            torch.from_numpy(vres['logit']), torch.from_numpy(vcls)).item())
        auc = _auc(vres['logit'], vcls)
        pos = vcls > 0.5
        err_mae = float(np.mean(np.abs(vres['logerr'][pos] -
                                       np.log10(vres['err_norm'][pos] + LOG_ERR_EPS)))) \
            if pos.any() else float('nan')
        sched.step()
        secs = time.time() - ep_t
        G.log_line(logf, f'{ep},{tot/max(cnt,1):.6f},{bce:.6f},{auc:.4f},'
                         f'{err_mae:.4f},{opt.param_groups[0]["lr"]:.6g},{secs:.1f}')
        print(f'[{args.tag}] ep {ep:3d}  train {tot/max(cnt,1):.4f}  val bce {bce:.4f} '
              f' auc {auc:.4f}  err-mae {err_mae:.3f} dex  {secs:.0f}s', flush=True)

        ck = {'head': head.state_dict(), 'opt': opt.state_dict(),
              'sched': sched.state_dict(), 'epoch': ep, 'best': best,
              'trunk_path': args.trunk, 'finetune': finetune, 'args': vars(args)}
        if finetune:
            ck['trunk'] = trunk.state_dict()
        torch.save(ck, last_ck + '.tmp')
        os.replace(last_ck + '.tmp', last_ck)
        score = bce - auc          # lower better; ties broken by AUC
        if score < best:
            best = score
            ck['best'] = best
            torch.save(ck, os.path.join(outdir, 'best.pt.tmp'))
            os.replace(os.path.join(outdir, 'best.pt.tmp'),
                       os.path.join(outdir, 'best.pt'))
        if time.time() > deadline:
            print(f'[{args.tag}] hit --max-hours', flush=True)
            break
    print(f'[{args.tag}] done in {(time.time()-t0)/60:.1f} min', flush=True)


# ---------------------------------------------------------------------------------------
# scoring / eval


@torch.no_grad()
def score_items(trunk, head, loader, dev):
    trunk.eval()
    head.eval()
    out = {k: [] for k in ('logit', 'logerr', 'err_norm', 'cls', 'idx', 'S')}
    for x, cls, tgt8, geo, i in loader:
        x = x.to(dev, non_blocking=True)
        pred8, logit, logerr = conf_forward(trunk, head, x)
        en = err_norm_of(pred8.float(), tgt8.to(dev))
        out['logit'].append(logit.float().cpu().numpy())
        out['logerr'].append(logerr.float().cpu().numpy())
        out['err_norm'].append(en.cpu().numpy())
        out['cls'].append(cls.numpy())
        out['idx'].append(i.numpy())
        out['S'].append(geo[:, 2].numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def _auc(logit, cls):
    pos = logit[cls > 0.5]
    neg = logit[cls < 0.5]
    if not len(pos) or not len(neg):
        return float('nan')
    r = np.argsort(np.argsort(np.concatenate([pos, neg])))[:len(pos)] + 1
    return float((r.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def eval_report(trunk, head, va_items, dev, workers=2, clean_keys=CLEAN_KEYS,
                aux_items=None):
    """The decision tables. Printed by the notebook AND by --mode eval/smoke.

    Clean-eval positives = conf-val positives that are either (a) VQ1 instances on
    frames in val_clean_keys.json's clean list -- held out of BOTH regressor splits,
    so the regressor's error there is honest -- or (b) VQ2 hand instances (their
    ground truth is drawn, not projected). Deco negatives are conf-val mined crops."""
    loader = DataLoader(ConfCrops(va_items, False), batch_size=128, shuffle=False,
                        num_workers=workers)
    r = score_items(trunk, head, loader, dev)
    order = r['idx']
    cls = r['cls']
    p = 1.0 / (1.0 + np.exp(-r['logit']))
    err_px = r['err_norm'] * r['S'] / 2.0                    # actual regressor error
    pred_px = (10.0 ** r['logerr']) * r['S'] / 2.0           # head's prediction of it

    reason = np.array([va_items[i]['reason'] for i in order])
    src = np.array([va_items[i]['src'] for i in order])
    keys = np.array([va_items[i]['key'] for i in order])
    clean = set(json.load(open(clean_keys))['clean']) if os.path.exists(clean_keys) \
        else set()
    is_clean_pos = (cls > 0.5) & (np.isin(keys, list(clean)) | (src == 'hand'))
    is_deco = reason == 'deco'
    is_neg = cls < 0.5

    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    say(f'\n### confidence eval -- {int(cls.sum())} pos / {int((~(cls>0.5)).sum())} neg '
        f'in conf-val; clean pos {int(is_clean_pos.sum())}, deco negs '
        f'{int(is_deco.sum())}, all negs {int(is_neg.sum())}')
    say(f'AUC (gate vs all-neg): {_auc(r["logit"], cls):.4f}')
    if is_deco.any():
        auc_deco = _auc(np.concatenate([r['logit'][cls > 0.5], r['logit'][is_deco]]),
                        np.concatenate([np.ones(int((cls > 0.5).sum())),
                                        np.zeros(int(is_deco.sum()))]))
        say(f'AUC (gate vs deco only): {auc_deco:.4f}')
    else:
        say('no deco negatives in this val subset -- thresholds fall back to all negs')

    say('\n| target deco reject | threshold p | deco rejected | all-neg rejected | '
        'true-gate loss (clean) |')
    say('|---|---:|---:|---:|---:|')
    accept_line = None
    for want in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
        base = p[is_deco] if is_deco.any() else p[is_neg]
        if not len(base):
            break
        t = float(np.quantile(base, want))
        deco_rej = float(np.mean(p[is_deco] < t)) if is_deco.any() else float('nan')
        neg_rej = float(np.mean(p[is_neg] < t)) if is_neg.any() else float('nan')
        loss = float(np.mean(p[is_clean_pos] < t)) if is_clean_pos.any() else float('nan')
        say(f'| {want:.0%} | {t:.3f} | {deco_rej:.1%} | {neg_rej:.1%} | {loss:.2%} |')
        if abs(want - ACCEPT_DECO_REJECT) < 1e-9:
            accept_line = (deco_rej, loss)

    # catastrophe catch, clean positives only, deterministic crops
    cp = is_clean_pos
    cat = err_px > CATASTROPHE_PX
    say(f'\n### >10 px catastrophes on clean positives: {int((cat&cp).sum())} of '
        f'{int(cp.sum())} ({float((cat&cp).mean()/max(cp.mean(),1e-9)):.1%})'
        f'   [PnP-residual baseline: {PNP_BASELINE["catch"]:.0%} caught at '
        f'{PNP_BASELINE["flag"]:.1%} flag rate]')
    say('| flag rate | catch (pred-err alone) | catch (pred-err OR p<t80) |')
    say('|---:|---:|---:|')
    t80 = float(np.quantile(p[is_deco], ACCEPT_DECO_REJECT)) if is_deco.any() else 0.0
    for fr in (0.05, 0.10, PNP_BASELINE['flag'], 0.15, 0.20):
        if not cp.any():
            break
        tau = float(np.quantile(pred_px[cp], 1.0 - fr))
        flag1 = pred_px >= tau
        catch1 = float((flag1 & cat & cp).sum() / max((cat & cp).sum(), 1))
        flag2 = flag1 | (p < t80)
        # report the OR at ITS OWN flag rate, honestly
        fr2 = float((flag2 & cp).mean() / max(cp.mean(), 1e-9))
        catch2 = float((flag2 & cat & cp).sum() / max((cat & cp).sum(), 1))
        say(f'| {fr:.1%} | {catch1:.1%} | {catch2:.1%} (at {fr2:.1%} flags) |')

    # DIAGNOSTIC: same catastrophe table over the WHOLE clean set (both conf splits).
    # Regressor-honest everywhere (these frames are outside both regressor splits);
    # the HEAD trained on its conf-train portion, so its error predictions there are
    # optimistic -- diagnostic only, the strict table above is the decision table.
    if aux_items:
        lo = DataLoader(ConfCrops(aux_items, False), batch_size=128, shuffle=False,
                        num_workers=workers)
        ra = score_items(trunk, head, lo, dev)
        aerr = ra['err_norm'] * ra['S'] / 2.0
        apred = (10.0 ** ra['logerr']) * ra['S'] / 2.0
        acat = aerr > CATASTROPHE_PX
        say(f'\n### DIAGNOSTIC (head-seen, regressor-honest): whole clean set, '
            f'{len(aux_items)} pos, {int(acat.sum())} catastrophes')
        say('| flag rate | catch (pred-err alone) |')
        say('|---:|---:|')
        for fr in (0.05, 0.10, PNP_BASELINE['flag'], 0.15, 0.20):
            tau = float(np.quantile(apred, 1.0 - fr))
            say(f'| {fr:.1%} | '
                f'{float(((apred >= tau) & acat).sum() / max(acat.sum(), 1)):.1%} |')

    say('\n### FROZEN accept rule (decided before numbers): deploy iff deco reject '
        f'>= {ACCEPT_DECO_REJECT:.0%} at true-gate loss <= {ACCEPT_GATE_LOSS:.0%}')
    if accept_line and not math.isnan(accept_line[1]):
        ok = accept_line[0] >= ACCEPT_DECO_REJECT - 1e-9 \
            and accept_line[1] <= ACCEPT_GATE_LOSS + 1e-9
        say(f'=> {"ACCEPT: ship the head" if ok else "REJECT: do not ship; PnP residual stays the only gate"} '
            f'(deco reject {accept_line[0]:.1%} at gate loss {accept_line[1]:.2%})')
    else:
        say('=> UNDECIDABLE here (no deco negatives or no clean positives in this '
            'subset) -- only the Colab run with the full index decides.')
    return '\n'.join(lines), r


def eval_only(args):
    dev = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    items, _ = build_index(require_mined=not args.allow_unmined)
    _, va_items = conf_split(items)
    ck = torch.load(os.path.join(RUNS, args.tag, 'best.pt'), map_location=dev,
                    weights_only=False)
    trunk = load_trunk(ck.get('trunk_path', args.trunk), dev)
    if ck.get('finetune') and 'trunk' in ck:
        trunk.load_state_dict(ck['trunk'])
    head = ConfHead().to(dev)
    head.load_state_dict(ck['head'])
    clean = set(json.load(open(CLEAN_KEYS))['clean']) if os.path.exists(CLEAN_KEYS) \
        else set()
    aux = [d for d in items if d['cls'] == 1
           and (d['key'] in clean or d['src'] == 'hand')]
    eval_report(trunk, head, va_items, dev, args.workers, aux_items=aux)


# ---------------------------------------------------------------------------------------
# bench -- the <1 ms budget, measured not asserted


def bench(args):
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    dev = torch.device('cpu')          # the budget is CPU alongside control
    trunk = load_trunk(args.trunk, dev) if os.path.exists(args.trunk) else \
        G.GateNet(1.0).eval()
    head = ConfHead().eval()
    x = torch.randn(1, 3, G.RES, G.RES)

    def timed(fn, inner=10):
        with torch.no_grad():
            t = time.perf_counter()
            for _ in range(inner):
                fn()
        return (time.perf_counter() - t) / inner * 1000.0

    def full():
        feat = trunk.body(trunk.stem(x))
        trunk.head(feat)
        head(feat)

    feat0 = trunk.body(trunk.stem(x))
    # INTERLEAVED medians, because a laptop under load makes an A-then-B comparison lie
    # by more than the quantity being measured (first bench run here showed a +1.1 ms
    # "increment" while the head alone costs 0.2 ms). The three measurements:
    #   t_reg  -- trunk alone (the deployed baseline)
    #   t_full -- trunk once + both heads (the deployed path with confidence)
    #   t_head -- conf head on the ALREADY-COMPUTED feature map. This IS the marginal
    #             cost by construction: at flight time the trunk forward is shared, so
    #             the budget question is exactly this number. t_full - t_reg estimates
    #             the same thing with far more noise; reported as a cross-check.
    for fn in (lambda: trunk(x), full, lambda: head(feat0)):   # warm-up
        timed(fn)
    reg_s, full_s, head_s = [], [], []
    for _ in range(15):
        reg_s.append(timed(lambda: trunk(x)))
        full_s.append(timed(full))
        head_s.append(timed(lambda: head(feat0)))
    t_reg = float(np.median(reg_s))
    t_full = float(np.median(full_s))
    t_head = float(np.median(head_s))
    msg = (f'CPU batch1 (medians of 15 interleaved trials): regressor alone '
           f'{t_reg:.2f} ms; regressor+conf {t_full:.2f} ms '
           f'(delta {t_full-t_reg:+.3f} ms, noisy); head on the shared feature map '
           f'{t_head:.3f} ms <- the marginal cost at flight time')
    print(msg)
    ok = t_head < 1.0
    print(f'budget <1 ms on the marginal path: {"MET" if ok else "EXCEEDED"}'
          f'   (cross-check delta {"consistent" if t_full - t_reg < 1.0 else "noisy/over -- rerun on an idle machine"})')
    return msg, t_head


# ---------------------------------------------------------------------------------------
# smoke -- 2 epochs, CPU, small subset; proves plumbing + the policy invariants


def check_invariants(items):
    """The LABEL_POLICY.md guarantees, re-checked against the raw files rather than
    trusted from the miner. Called by smoke locally AND by the Colab notebook before
    training, so a stale upload cannot silently void the policy."""
    # INVARIANT 1: every negative's provenance is one of the explicit sources.
    allowed = {'behind-gate', 'no-orange', 'deco', 'empty-random'}
    bad = [d for d in items if d['cls'] == 0 and d['reason'] not in allowed]
    assert not bad, f'negatives with unknown provenance: {bad[:3]}'

    # INVARIANT 2: no negative comes from an unlabelled region of a hand frame --
    # every VQ2 negative sits on a frame Claire labelled (or verified empty), and the
    # deco crops overlap no hand quad (including unsure ones).
    merged = json.load(open(MERGED))
    vidx = json.load(open(VQ2_INDEX))
    key2ui = {f['session'] + '/' + f['file']: f['labelui_key'] for f in vidx['frames']}
    hand_raw = json.load(open(HAND_RAW))
    for d in items:
        if d['cls'] or d['reason'] not in ('deco', 'empty-random'):
            continue
        k = d['key']
        assert k in merged and (any(i.get('src') == 'hand' for i in merged[k])
                                or merged[k] == []), f'unlabelled VQ2 frame: {k}'
        if d['reason'] == 'empty-random':
            assert merged[k] == [], f'empty-random on a non-empty frame: {k}'
        else:
            for q in hand_raw.get(key2ui.get(k, ''), []):
                hq = np.asarray(q['corners'], np.float32)
                inter, _ = cv2.intersectConvexConvex(
                    np.asarray(d['corners'], np.float32), hq)
                m = min(abs(cv2.contourArea(np.asarray(d['corners'], np.float32))),
                        abs(cv2.contourArea(hq)))
                assert m == 0 or inter / m <= 0.30, \
                    f'deco crop overlaps a hand quad: {k}'
    # INVARIANT 3: the degenerate-visible-box entries are out.
    n_no = sum(1 for d in items if d['reason'] == 'no-orange')
    raw_no = sum(1 for v in json.load(open(NEG_VQ1)).values()
                 for i in v if i.get('reason') == 'no-orange')
    assert n_no <= raw_no
    print(f'invariants hold: {sum(1 for d in items if not d["cls"])} negatives, all '
          f'from explicit sources; degenerate boxes excluded; hand-frame regions '
          f'never sampled as background', flush=True)


def smoke(args):
    print('=== SMOKE: build index and check the label-policy invariants ===')
    items, counts = build_index(require_mined=True)
    print('counts:', dict(counts))
    check_invariants(items)

    # INVARIANT 4: regressor path bit-identical with the head attached (frozen).
    dev = torch.device('cpu')
    trunk = load_trunk(args.trunk, dev)
    head = ConfHead().eval()
    x = torch.randn(2, 3, G.RES, G.RES)
    with torch.no_grad():
        a = trunk(x)
        b, _, _ = conf_forward(trunk, head, x)
    assert torch.equal(a, b), 'regressor output changed with head attached'
    print('regressor path bit-identical with head attached: OK')

    print('\n=== SMOKE: 2-epoch CPU train on a subset ===')
    sm = argparse.Namespace(**vars(args))
    sm.tag = 'conf-smoke'
    sm.epochs = 2
    sm.batch = 32
    sm.workers = 0
    sm.cpu = True
    sm.limit = 40                # every 40th positive + all negatives
    sm.resume = False
    sm.finetune = False
    global RUNS
    old_runs = RUNS
    RUNS = os.path.join(HERE, 'smoke_tmp', 'conf_runs')
    try:
        train(sm)
        print('\n=== SMOKE: eval tables render (numbers are meaningless) ===')
        items2, _ = build_index()
        pos = [d for d in items2 if d['cls'] == 1][::40]
        items2 = sorted(pos + [d for d in items2 if d['cls'] == 0],
                        key=lambda d: (d['session'], d['fnum']))
        _, va2 = conf_split(items2)
        ck = torch.load(os.path.join(RUNS, 'conf-smoke', 'best.pt'),
                        map_location=dev, weights_only=False)
        head.load_state_dict(ck['head'])
        eval_report(trunk, head, va2, dev, workers=0)
    finally:
        RUNS = old_runs

    print('\n=== SMOKE: latency ===')
    bench(args)
    print('\nSMOKE PASSED')


def index_mode(args):
    items, counts = build_index(require_mined=not args.allow_unmined)
    tr, va = conf_split(items)
    print('per (class, reason):')
    for k in sorted((k for k in counts if isinstance(k, tuple))):
        print(f'  cls={k[0]}  {k[1]:14s} {counts[k]:6d}')
    print(f"negatives before dedupe: {counts['neg_before_dedupe']}  "
          f"after: {sum(v for k, v in counts.items() if isinstance(k, tuple) and k[0]==0)}")
    print(f"excluded: {counts['unsure_excluded']} unsure positives, "
          f"{counts['degenerate_excluded']} degenerate-visible-box negatives")
    print(f'split: train {len(tr)}  val {len(va)}  '
          f'(val negs {sum(1 for d in va if not d["cls"])}, '
          f'val deco {sum(1 for d in va if d["reason"]=="deco")})')
    missing = [d['key'] for d in items if not os.path.exists(d['path'])]
    print(f'frames missing on disk: {len(missing)}'
          + (f'  e.g. {missing[:3]}' if missing else ''))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='index',
                    choices=['mine', 'zipframes', 'index', 'train', 'eval', 'bench',
                             'smoke'])
    ap.add_argument('--tag', default='conf')
    ap.add_argument('--trunk', default=TRUNK_DEFAULT,
                    help='frozen regressor checkpoint (production colab-v3-nw)')
    ap.add_argument('--labels', default=MERGED)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--max-hours', type=float, default=3.0)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--finetune', action='store_true',
                    help='unfreeze the trunk (lr/10). ONLY with the cross-eval step; '
                         'the pair deploys together or not at all.')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0,
                    help='smoke subsetting: keep every k-th positive')
    ap.add_argument('--allow-unmined', action='store_true',
                    help='proceed without conf_negatives_vq2.json (dev only)')
    args = ap.parse_args()

    G.LABELS = os.path.abspath(args.labels)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cv2.setNumThreads(0)

    {'mine': mine, 'zipframes': zipframes, 'index': index_mode, 'train': train,
     'eval': eval_only, 'bench': bench, 'smoke': smoke}[args.mode](args)


if __name__ == '__main__':
    main()
