"""Full-session scan for real two-gate frames on 2-3-c, decoration-resistant:
pair = two detections, each size>=26 px, range 8-35 m, centres >=150 px apart.
GOLD if both inner+unclipped, ONECLIP if one clipped, BOTHCLIP else."""
import os, sys
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import detect as D, label as L
SESS = os.path.join(HERE, '..', 'sessions', '20260802-010914-vq2-strafe-2-3-c')
W,H=640,360
def clipped(d):
    if d['source']=='outer': return True
    q=d['quad']
    return bool((q[:,0]<3).any() or (q[:,0]>W-4).any() or (q[:,1]<3).any() or (q[:,1]>H-4).any())
frames=[r for r in L.load_csv(os.path.join(SESS,'frames.csv')) if r['file']]
t0=float(frames[0]['t_recv_wall_ns'])
gold=[]; oneclip=[]; bothclip=[]
for i in range(0,len(frames),4):
    img=cv2.imread(os.path.join(SESS,'frames',frames[i]['file']))
    if img is None: continue
    ds=[d for d in D.detections(img) if d['size_px']>=26 and 8.0<=d['range_m']<=35.0]
    best=None
    for a in range(len(ds)):
        for b in range(a+1,len(ds)):
            sep=abs(ds[a]['centre'][0]-ds[b]['centre'][0])
            if sep<150: continue
            nc=clipped(ds[a])+clipped(ds[b])
            innerok=sum(1 for d in (ds[a],ds[b]) if d['source']=='inner' and not clipped(d))
            cat = 0 if innerok==2 else (1 if innerok==1 else 2)
            if best is None or cat<best[0]: best=(cat,sep,ds[a]['range_m'],ds[b]['range_m'])
    if best is not None:
        ts=(float(frames[i]['t_recv_wall_ns'])-t0)/1e9
        [gold,oneclip,bothclip][best[0]].append((ts,frames[i]['file'],best[1],best[2],best[3]))
    if i%400==0: print(f'{i}/{len(frames)}', flush=True)
for name,lst in (('GOLD',gold),('ONECLIP',oneclip),('BOTHCLIP',bothclip)):
    print(f'\n{name}: {len(lst)} frames (stride 4)')
    for ts,f,sep,r1,r2 in lst[:60]:
        print(f'  t={ts:6.1f}s {f} sep={sep:.0f}px ranges {r1:.1f}/{r2:.1f} m')
