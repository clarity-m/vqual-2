"""Stage everything a Colab run needs into one folder, laid out like Drive.

Why a script instead of a folder you maintain by hand: a stale copy is invisible.
Uploading last week's gatenet.py with this week's labels produces a run that looks
fine and answers nothing, and the cache fingerprint would not catch it (it hashes
the LABELS file, not the code). So this rebuilds the staging folder from the
canonical files every time and prints what changed.

    python3 pilot/perception/mkcolab.py            # stage + report
    python3 pilot/perception/mkcolab.py --check    # report only, change nothing

Layout produced under colab_upload/ mirrors Drive, so uploading is drag-per-folder:

    colab_upload/perception/   ->  MyDrive/vqual2/perception/
    colab_upload/drive_root/   ->  MyDrive/vqual2/

Optional files (the merged hand labels, the VQ2 frame zip) are staged when they
exist and reported as MISSING when they do not -- missing is normal before Claire's
labelling lands, and is not an error.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
STAGE = os.path.join(HERE, 'colab_upload')

# (source path, staged subdir, required)   -- source is relative to pilot/perception
FILES = [
    ('gatenet.py',                     'perception', True),
    ('packcrops.py',                   'perception', True),
    ('crosseval_clean.py',             'perception', True),
    ('val_clean_keys.json',            'perception', True),
    ('autolabels_vq1_v3.json',         'perception', True),
    ('labels_merged_v3.json',          'perception', False),   # after mergelabels.py
    ('gatenet_colab_v3_vq1only.ipynb', 'drive_root', True),
    ('gatenet_colab_v3.ipynb',         'drive_root', True),
    ('vq2_frames.zip',                 'drive_root', False),   # after _mkframezip_vq2.py
    # confidence-head run (gatenet_conf_colab.ipynb, TAG colab-conf)
    ('gatenet_conf.py',                'perception', True),
    ('conf_negatives_vq2.json',        'perception', True),    # gatenet_conf --mode mine
    ('labelfix_negatives_v3.json',     'perception', True),
    ('autolabels_vq1_negatives.json',  'perception', True),
    ('labels_gates_all.json',          'perception', True),    # invariant re-check input
    ('vq2_label_index.json',           'perception', True),    # staged copy, see SRC_AS
    ('gatenet_conf_colab.ipynb',       'drive_root', True),
    ('conf_frames_vq1.zip',            'drive_root', True),    # gatenet_conf --mode zipframes
]

# staged name -> actual source path (relative to pilot/perception), for files whose
# on-disk home is a subdirectory but whose Drive copy must sit flat in perception/
SRC_AS = {
    'vq2_label_index.json': os.path.join('vq2_label', 'index.json'),
}

README = """Colab upload staging -- REBUILT BY mkcolab.py, do not edit by hand.

    colab_upload/perception/*   ->  Drive: MyDrive/vqual2/perception/
    colab_upload/drive_root/*   ->  Drive: MyDrive/vqual2/

Replace the existing files on Drive when names collide (gatenet.py, packcrops.py).
Already on Drive and NOT staged here because they are large and unchanged:
    vq1_frames.zip              (247 MB, VQ1 frames)
    gatenet_runs/block/best.pt  (production checkpoint, the cross-eval baseline)

Which notebook:
    gatenet_colab_v3_vq1only.ipynb   audited labels alone, no hand labels  (done)
    gatenet_colab_v3.ipynb           adds the merged VQ2 hand labels       (done)
    gatenet_conf_colab.ipynb         confidence head on the FROZEN
                                     colab-v3-nw trunk, TAG colab-conf     <- current

Run order historically: each changes ONE variable, so a regression is diagnosable.
"""


def sha(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true',
                    help='report only; do not copy anything')
    args = ap.parse_args()

    rows, missing = [], []
    for name, sub, required in FILES:
        src = os.path.join(HERE, SRC_AS.get(name, name))
        dst = os.path.join(STAGE, sub, name)
        if not os.path.isfile(src):
            (missing if required else rows).append(name)
            if not required:
                rows[-1] = (name, sub, 'absent (optional)', 0.0)
            continue

        s_hash = sha(src)
        if os.path.isfile(dst) and sha(dst) == s_hash:
            state = 'up to date'
        else:
            state = 'STALE -> restaged' if os.path.isfile(dst) else 'new'
            if not args.check:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
        rows.append((name, sub, state, os.path.getsize(src) / 1e6))

    if not args.check:
        os.makedirs(STAGE, exist_ok=True)
        with open(os.path.join(STAGE, 'README.txt'), 'w') as fh:
            fh.write(README)

    print(f'staging -> {STAGE}')
    for r in rows:
        if isinstance(r, tuple) and len(r) == 4:
            name, sub, state, mb = r
            size = f'{mb:8.1f} MB' if mb else ' ' * 11
            print(f'  {sub:10s} {name:32s} {size}  {state}')
    for name in missing:
        print(f'  {"":10s} {name:32s} {"":11s}  MISSING (required)')
    if missing:
        raise SystemExit('required file(s) missing -- nothing to upload yet')

    # a staged file older than its source can only mean an edit landed after staging
    stale = [n for n, sub, *_ in [r for r in rows if len(r) == 4]
             if os.path.isfile(os.path.join(STAGE, sub, n))
             and os.path.isfile(os.path.join(HERE, SRC_AS.get(n, n)))
             and os.path.getmtime(os.path.join(HERE, SRC_AS.get(n, n)))
             > os.path.getmtime(os.path.join(STAGE, sub, n)) + 1]
    print('\nupload:  colab_upload/perception/ -> MyDrive/vqual2/perception/'
          '   (replace on collision)')
    print('         colab_upload/drive_root/ -> MyDrive/vqual2/')
    print('already on Drive, do not re-upload: vq1_frames.zip, gatenet_runs/block/best.pt')
    if stale:
        print('\nWARNING, source newer than staged copy:', ', '.join(stale))
    print(f'\nstaged at {time.strftime("%Y-%m-%d %H:%M")}. Re-run this before every upload.')


if __name__ == '__main__':
    main()
