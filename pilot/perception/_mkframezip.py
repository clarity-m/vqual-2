"""Build vq1_frames.zip -- only the 6598 frames autolabels_vq1.json actually references.

The Colab notebook can rebuild the 4.4 GB crop cache from these in a couple of minutes, so
this ~247 MB zip is what moves to Drive instead of the cache. Stored (not deflated): the
members are JPEGs, deflate would spend minutes to save ~1%.

Layout inside the zip mirrors what gatenet.load_index() expects under G.SESSIONS:
    <session>/frames/<file>.jpg
"""
import json
import os
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SESSIONS = os.path.join(ROOT, 'pilot', 'sessions')
OUT = os.path.join(HERE, 'vq1_frames.zip')

keys = sorted(json.load(open(os.path.join(HERE, 'autolabels_vq1.json'))))
t0 = time.time()
n = 0
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_STORED, allowZip64=True) as z:
    for k in keys:
        sess, fname = k.split('/', 1)
        src = os.path.join(SESSIONS, sess, 'frames', fname)
        z.write(src, f'{sess}/frames/{fname}')
        n += 1
        if n % 1000 == 0:
            print(f'  {n}/{len(keys)}  {time.time()-t0:.0f}s', flush=True)
print(f'{n} frames -> {OUT}  {os.path.getsize(OUT)/1e6:.1f} MB  {time.time()-t0:.0f}s')
