"""Small-N Thomson basin probe only. Not a 960-point rational certificate."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def unit(x):
    return x/np.linalg.norm(x,axis=1,keepdims=True)
def energy_and_gradient(v):
    x=unit(v.reshape(-1,3))
    d=x[:,None,:]-x[None,:,:]
    d2=np.sum(d*d,axis=2);np.fill_diagonal(d2,np.inf)
    inv=np.where(np.isfinite(d2),1./np.sqrt(d2),0.)
    E=float(np.sum(inv)*.5)
    g=-np.sum(d*inv[:,:,None]**3,axis=1)
    raw=v.reshape(-1,3); norm=np.linalg.norm(raw,axis=1,keepdims=True)
    g=(g-np.sum(g*x,axis=1,keepdims=True)*x)/norm
    return E,g.ravel()
def relax(pts,maxiter=100):
    res=minimize(energy_and_gradient,pts.ravel(),jac=True,method='L-BFGS-B',
                 options={'maxiter':maxiter,'ftol':1e-11,'gtol':1e-7,'maxls':15})
    ans=unit(res.x.reshape(-1,3))
    return ans,float(energy_and_gradient(ans.ravel())[0]),int(res.nfev)
def defect_indices(pts):
    hull=ConvexHull(pts)
    adj=[set() for _ in pts]
    for tri in hull.simplices:
        for i,j in ((tri[0],tri[1]),(tri[0],tri[2]),(tri[1],tri[2])):
            adj[i].add(int(j));adj[j].add(int(i))
    return np.array([i for i,a in enumerate(adj) if len(a)!=6],dtype=int)
def probe(n,seed,steps=12,maxiter=100):
    rng=np.random.default_rng(seed)
    start=unit(rng.normal(size=(n,3)));start,e0,ev0=relax(start,maxiter)
    rows=[]
    for method in ('random','defect'):
        local=np.random.default_rng(seed+3457)
        pts=start.copy();best=e0;ev=ev0;trajectory=[]
        for t in range(steps):
            candidate=pts.copy()
            if method=='defect':
                hot=defect_indices(pts)
                center=int(local.choice(hot if len(hot)>0 else np.arange(n)))
            else:center=int(local.integers(n))
            nearby=np.argsort(np.sum((pts-pts[center])**2,axis=1))[:max(2,round(n*.10))]
            candidate[nearby]+=local.normal(size=(len(nearby),3))*.16
            candidate,e,ne=relax(unit(candidate),maxiter);ev+=ne
            if e<best-1e-9:pts,best=candidate,e
            trajectory.append(round(best,10))
        rows.append({'n':n,'seed':seed,'method':method,'start_energy':round(e0,10),
                     'best_energy':round(best,10),'energy_drop':round(e0-best,10),
                     'objective_gradient_evaluations':ev,'trajectory':trajectory})
    return rows
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--lengths',nargs='+',type=int,default=[32,60])
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--steps',type=int,default=12)
    p.add_argument('--maxiter',type=int,default=100)
    p.add_argument('--output',type=Path,default=Path('results/p007.json'))
    a=p.parse_args()
    runs=[r for n in a.lengths for seed in range(a.seeds) for r in probe(n,seed,a.steps,a.maxiter)]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'protocol':'small-N, matched local-refinement calls; unequal objective evaluations','runs':runs},indent=2)+'\n')
    for n in a.lengths:
        for m in ('random','defect'):
            x=[r for r in runs if r['n']==n and r['method']==m]
            print('P007',n,m,'drops',[r['energy_drop'] for r in x])
if __name__=='__main__':main()
