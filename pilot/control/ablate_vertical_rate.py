"""Where does the vertical-rate signal live in the 73-D observation?

Rescued from the vq2-course-variations worktree before it was deleted. This is the
evidence behind the architecture session's note that `k_clear` is doing work a missing
OBSERVATION created: the policy cannot perceive its own vertical rate, so the clearance
potential is compensating for an unobservable. Do not tune that term without reading
this first -- the fix belongs on the observation side.

Originally pointed at an absolute path inside a sibling worktree; now resolves relative
to itself so it survives.
"""
import os, sys, numpy as np
CTRL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CTRL)
sys.path.insert(0, os.path.dirname(os.path.dirname(CTRL)))
from surrogate.env import VecSurrogate, EnvConfig
N,T=96,900; HOVER=0.27
env=VecSurrogate(N,EnvConfig(vq2_frac=1.0,difficulty=0.3,speed_cap=0.8),seed=0)
dt=float(env.dt_dec[0]); rng=np.random.default_rng(0)
OBS,VZ=[],[]; rate=np.zeros((N,2)); thr=np.full(N,HOVER)
for t in range(T):
    rate=np.clip(0.92*rate+rng.normal(0,0.35,(N,2)),-2.0,2.0)
    thr=np.clip(0.95*thr+0.05*HOVER+rng.normal(0,0.012,N),0.12,0.55)
    o,_,_,_=env.step(np.stack([rate[:,0],rate[:,1],thr],axis=1).astype(np.float32))
    OBS.append(o.copy()); VZ.append(env.v[:,2].copy())
OBS=np.array(OBS,np.float32); VZ=np.array(VZ,np.float32)
mu=OBS.reshape(-1,73).mean(0); sd=OBS.reshape(-1,73).std(0)+1e-6; OBSn=(OBS-mu)/sd
GROUPS={"gates 0-32":range(0,33),"ribbon 33-47":range(33,48),"gyro 48-50":range(48,51),
        "accel 51-53":range(51,54),"roll/pitch 54-56":range(54,57),
        "vel bearing 57-59":range(57,60),"speed 60-61":range(60,62),
        "race 62-65":range(62,66),"attn 66-72":range(66,73)}
def run(cols,offsets,ridge=10.0):
    lag=max(offsets); idx=np.arange(lag,T)
    X=np.concatenate([OBSn[idx-o][:,:,cols].reshape(len(idx)*N,-1) for o in offsets],axis=1)
    y=VZ[idx].reshape(-1); n=len(y); p=np.random.default_rng(1).permutation(n); cut=int(0.7*n)
    tr,te=p[:cut],p[cut:]
    Xtr=np.c_[X[tr],np.ones(cut)]
    w=np.linalg.solve(Xtr.T@Xtr+ridge*np.eye(Xtr.shape[1]),Xtr.T@y[tr])
    r=y[te]-np.c_[X[te],np.ones(len(te))]@w
    return 1-r.var()/y[te].var(), r.std()
D=[0,2,4,8,16,32]
print("WHERE DOES THE VERTICAL-RATE SIGNAL COME FROM?  (dilated stack, %.2f s)"%(32*dt))
print("  channel group          R2      RMSE")
allc=list(range(73))
for name,g in GROUPS.items():
    r2,rm=run(list(g),D); print("   %-20s %+.3f   %.2f m/s"%(name,r2,rm))
print()
r2,rm=run(allc,D); print("   ALL 73               %+.3f   %.2f m/s"%(r2,rm))
vis=list(range(0,48)); own=list(range(48,62))
r2,rm=run(vis,D); print("   vision only (0-47)   %+.3f   %.2f m/s"%(r2,rm))
r2,rm=run(own,D); print("   own-state (48-61)    %+.3f   %.2f m/s"%(r2,rm))
print()
print("  signal: sigma(vz) = %.2f m/s"%VZ.std())
