"""Re-implementation of the large-network retrieval simulation of
Samsonovich & Ascoli (2005) Learning & Memory 12:193-208, doi:10.1101/lm.85205 (Methods,
'Simulation of the hippocampal module as a large two-layer neural network'; Fig. 8).
N CA3 + N CA1 units; each CA3 unit has n=10 random efferent CA3->CA3 links (matrix J).
Exploration: for each afferent link of each unit g, one backward epoch of M steps along J;
every CA3 unit active in the epoch is associated with CA1 unit g with W = max(W_old, 1/(t-t_j)); W_ii = 2.
Retrieval: at each step every efferent of the current unit is probed (phase-precession sweep);
the CA3 packet is 1.0 on the probed unit and 0.2 on its 10 efferents; the probe that most excites
goal CA1 unit G is taken. If no probe excites G, a random efferent is taken.
Unreachable (start, goal) pairs (goal with zero in-degree) are skipped.
Noise model is an approximation: Gaussian with sd = noise * 2.0 added to non-zero probe scores.
"""
import numpy as np, scipy.sparse as sp, json
from collections import deque
def build(N,n=10,M=5,seed=0):
    rng=np.random.default_rng(seed)
    succ=np.stack([rng.choice(N-1,n,replace=False) for _ in range(N)]); succ=succ+(succ>=np.arange(N)[:,None])  # no self
    preds=[[] for _ in range(N)]
    for u in range(N):
        for v in succ[u]: preds[v].append(u)
    rows,cols,vals=[],[],[]
    # one backward epoch of M steps per afferent connection of each unit g; associate active CA3 units with CA1 unit g, weight 1/(t-tj)
    for g in range(N):
        best={}
        for p in preds[g]:
            cur=p; k=1
            while True:
                best[cur]=max(best.get(cur,0),1.0/k)
                if k==M or not preds[cur]: break
                cur=preds[cur][rng.integers(len(preds[cur]))]; k+=1
        best.pop(g,None)
        for u,w in best.items(): rows.append(g); cols.append(u); vals.append(w)
    W=sp.csr_matrix((vals,(rows,cols)),shape=(N,N))
    return succ,preds,W

def bfs(succ,s,g):
    if s==g: return 0
    seen={s}; q=deque([(s,0)])
    while q:
        u,d=q.popleft()
        for v in succ[u]:
            if v==g: return d+1
            if v not in seen: seen.add(v); q.append((v,d+1))
    return None

def retrieve2(succ,W,s,g,rng,noise=0.0,maxsteps=2000,avoid=False):
    wg=np.asarray(W[g].todense()).ravel(); wg[g]=2.0
    cur=s; steps=0; prev=-1
    while cur!=g and steps<maxsteps:
        cands=succ[cur]
        sc=wg[cands]+0.2*wg[succ[cands]].sum(1)
        if noise>0: sc=sc+rng.normal(0,noise*2.0,len(sc))*(sc!=0)
        if avoid: sc=np.where(cands==prev,-np.inf,sc)
        nxt = cands[rng.integers(len(cands))] if np.max(sc)<=0 else cands[np.argmax(sc)]
        prev=cur; cur=nxt; steps+=1
    return steps

def retrieve3(succ,W,s,g,rng,mode,maxsteps=2000):
    wg=np.asarray(W[g].todense()).ravel(); wg[g]=2.0
    cur=s; steps=0
    while cur!=g and steps<maxsteps:
        cands=succ[cur]
        if mode=='random': sc=np.zeros(len(cands))
        elif mode=='no_lookahead': sc=wg[cands]
        sc=sc if mode!='full' else wg[cands]+0.2*wg[succ[cands]].sum(1)
        cur = cands[rng.integers(len(cands))] if np.max(sc)<=0 else cands[np.argmax(sc)]
        steps+=1
    return steps

def run2(N,M=5,sessions=10000,noise=0.0,seed=0,avoid=False,G=None):
    succ,preds,W=G if G else build(N,M=M,seed=seed); rng=np.random.default_rng(seed+1)
    R=[];S=[]; unreach=0
    while len(R)<sessions:
        s,g=rng.integers(N,size=2)
        if g==s: continue
        d=bfs(succ,s,g)
        if d is None: unreach+=1; continue
        R.append(retrieve2(succ,W,s,g,rng,noise,avoid=avoid)); S.append(d)
    R=np.array(R); ok=R<2000
    return dict(N=N,M=M,noise=noise,avoid=avoid,unreachable_skipped=unreach,lost=float((~ok).mean()),ret_mean_found=round(float(R[ok].mean()),2),ret_sd_found=round(float(R[ok].std()),2),
                short_mean=round(float(np.mean(S)),2),short_sd=round(float(np.std(S)),2),frac_lt20=float((R<20).mean()),frac_lt60=float((R<60).mean()))

def run3(G,N,mode,sessions,seed=0):
    succ,preds,W=G; rng=np.random.default_rng(seed+7); R=[];S=[]
    while len(R)<sessions:
        s,g=rng.integers(N,size=2)
        if g==s: continue
        d=bfs(succ,s,g)
        if d is None: continue
        R.append(retrieve3(succ,W,s,g,rng,mode)); S.append(d)
    R=np.array(R); ok=R<2000
    return dict(N=N,mode=mode,sessions=sessions,reached_within_2000=float(ok.mean()),ret_mean_found=round(float(R[ok].mean()),2) if ok.any() else None,ret_sd_found=round(float(R[ok].std()),2) if ok.any() else None,short_mean=round(float(np.mean(S)),2))

if __name__ == "__main__":
    G1 = build(10000, M=5, seed=0); res = []
    for avoid in [False, True]:
        for noise in [0.0, 0.025]:
            res.append(run2(10000, 5, 10000, noise, avoid=avoid, G=G1))
    res.append(run2(10000, 2, 10000, 0.0, G=build(10000, M=2, seed=0)))
    res.append(run2(20000, 5, 10000, 0.0, G=build(20000, M=5, seed=0)))
    abl = [run3(G1, 10000, "no_lookahead", 10000), run3(G1, 10000, "random", 300)]
    json.dump(dict(replication=res, ablations=abl), open("sa2005_replication.json", "w"), indent=1, default=float)
