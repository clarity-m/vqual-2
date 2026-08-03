"""Build vq2_frames.zip -- the VQ2 frames a merged label file references.

Companion to `_mkframezip.py` (which ships the VQ1 frames for the Colab cache rebuild).
After `mergelabels.py` folds Claire's hand labels into the training set, the merged file
references frames from VQ2 sessions that `vq1_frames.zip` does not contain; without them
`packcrops.py --mode pack` on Colab would warp black tiles for every hand instance and say
so only via `missing_frames` in meta.json. This zips exactly the frames whose keys carry at
least one hand-labelled instance (verified-negative empty arrays reference nothing the
cache reads -- negatives stay unused until there is a confidence head).

    python3 pilot/perception/_mkframezip_vq2.py [labels_merged_v3.json]

Layout matches _mkframezip.py / gatenet.load_index():  <session>/frames/<file>.jpg
Stored, not deflated: JPEGs.
"""
import json
import os
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SESSIONS = os.path.join(ROOT, 'pilot', 'sessions')
OUT = os.path.join(HERE, 'vq2_frames.zip')

labels = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'labels_merged_v3.json')
raw = json.load(open(labels))
# Hand-labelled frames, INCLUDING the verified-empty ones: an empty array is a real
# label ("no gate here") and the frame is the evidence for it. Training reads none of
# them (no instances -> no crops), but a negatives pass would, and a zip that silently
# omits them would make those negatives unusable later for no visible reason.
VQ2_SESSIONS = ('20260801-',)
keys = sorted(k for k, v in raw.items()
              if any(i.get('src') == 'hand' for i in v)
              or (not v and k.startswith(VQ2_SESSIONS)))
if not keys:
    sys.exit(f'no hand-labelled instances in {labels} -- nothing to zip')

t0 = time.time()
missing = []
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_STORED, allowZip64=True) as z:
    for k in keys:
        sess, fname = k.split('/', 1)
        src = os.path.join(SESSIONS, sess, 'frames', fname)
        if not os.path.exists(src):
            missing.append(k)
            continue
        z.write(src, f'{sess}/frames/{fname}')
if missing:
    print(f'WARNING: {len(missing)} referenced frames not on disk, e.g. {missing[:3]}')
print(f'{len(keys) - len(missing)} frames -> {OUT}  '
      f'{os.path.getsize(OUT)/1e6:.1f} MB  {time.time()-t0:.0f}s')
