"""Classical LABS reference: best admissible single-flip Tabu with exact integer deltas.

Not an exact reproduction of the 2025 GPU memetic-tabu method: this omits
population/recombination/GPU components. It is a checkable local search baseline.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np

def correlations(s: np.ndarray) -> np.ndarray:
    n=len(s)
    return np.array([np.dot(s[:n-k].astype(np.int32),s[k:].astype(np.int32))
                     for k in range(1,n)],dtype=np.int64)

def all_flip_energies(s: np.ndarray,c:np.ndarray) -> np.ndarray:
    """N exact candidate energies using O(N^2) vectorized integer operations."""
    n=len(s)
    j=np.arange(n)[:,None]
    k=np.arange(1,n)[None,:]
    right=np.where(j+k<n,s[np.minimum(j+k,n-1)],0)
    left=np.where(j-k>=0,s[np.maximum(j-k,0)],0)
    next_corr=c[None,:]-2*s[:,None]*(right+left)
    return np.sum(next_corr*next_corr,axis=1,dtype=np.int64)

def one_flip(s:np.ndarray,c:np.ndarray,j:int)->np.ndarray:
    """Update correlations exactly after flipping one bit."""
    k=np.arange(1,len(s))
    r=np.where(j+k<len(s),s[np.minimum(j+k,len(s)-1)],0)
    l=np.where(j-k>=0,s[np.maximum(j-k,0)],0)
    return c-2*s[j]*(r+l)

def tabu_search(n:int,seed:int,steps:int,start:str|None=None,
                tenure_low:float=.08,tenure_high:float=.15)->dict:
    rng=np.random.default_rng(seed)
    s=np.array([1 if v=='+' else -1 for v in start],dtype=np.int64) if start else rng.choice([-1,1],n).astype(np.int64)
    if len(s)!=n:raise ValueError('start length mismatch')
    corr=correlations(s)
    value=int(corr@corr)
    first=value;best=value;best_s=s.copy()
    tabu_until=np.zeros(n,dtype=np.int64)
    improv=0;t0=time.perf_counter()
    for t in range(steps):
        candidate=all_flip_energies(s,corr)
        legal=(tabu_until<=t)|(candidate<best)
        if not np.any(legal): legal[:]=True
        allowed=np.where(legal,candidate,np.iinfo(np.int64).max)
        indices=np.flatnonzero(allowed==np.min(allowed))
        j=int(rng.choice(indices))
        newcorr=one_flip(s,corr,j)
        s[j]*=-1
        corr=newcorr
        value=int(corr@corr)
        assert value==int(candidate[j]),'incremental energy mismatch'
        low=max(1,round(n*tenure_low))
        high=max(low+1,round(n*tenure_high)+1)
        tenure=int(rng.integers(low,high))
        tabu_until[j]=t+tenure
        if value<best:
            best=value;best_s=s.copy();improv+=1
    return {'n':n,'seed':seed,'steps':steps,'initial':first,
            'best_energy':best,
            'best_sequence':''.join('+' if v>0 else '-' for v in best_s),
            'moves_evaluated':int(n*steps),'improvements':improv,
            'elapsed_sec':round(time.perf_counter()-t0,3)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--lengths',nargs='+',type=int,default=[40,60,100])
    p.add_argument('--seeds',type=int,default=5)
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--output',type=Path,default=Path('results/p006_tabu.json'))
    a=p.parse_args()
    rows=[tabu_search(n,seed,a.steps) for n in a.lengths for seed in range(a.seeds)]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'algorithm':'best-admissible-flip tabu, no population','runs':rows},indent=2)+'\n')
    for n in a.lengths:
        scores=[r['best_energy'] for r in rows if r['n']==n]
        print('tabu',n,'best',min(scores),'mean',round(np.mean(scores),2),'scores',scores)
if __name__=='__main__':main()
