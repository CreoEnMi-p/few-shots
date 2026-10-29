"""Exploratory LABS search. Online 2-layer neural contextual policy versus baselines."""
import argparse,json
from pathlib import Path
import numpy as np

MOVES=(1,2,4,8)
def energy(s):
    n=len(s)
    c=np.correlate(s.astype(np.int32),s.astype(np.int32),'full')[n:].astype(np.int64)
    return int(c@c)

class TinyPolicy:
    """4->16->4 tanh-softmax REINFORCE policy with online moving baseline."""
    def __init__(self,seed):
        r=np.random.default_rng(seed+1089)
        self.w1=r.normal(0,.12,(4,16));self.b1=np.zeros(16)
        self.w2=r.normal(0,.08,(16,4));self.b2=np.zeros(4)
        self.baseline=0.
    def choose(self,f,rng):
        h=np.tanh(f@self.w1+self.b1); logits=h@self.w2+self.b2
        logits=np.clip(logits-logits.max(),-30,0)
        probs=np.exp(logits);probs=.03+.88*(probs/probs.sum())
        probs/=probs.sum()
        return int(rng.choice(4,p=probs)),h,probs
    def update(self,f,h,probs,arm,reward,lr=.009):
        adv=np.clip(reward-self.baseline,-1.5,1.5)
        grad=-probs.copy();grad[arm]+=1
        gw2=np.outer(h,grad)*adv;gb2=grad*adv
        dh=(self.w2@grad)*adv*(1-h*h)
        self.w2+=lr*gw2;self.b2+=lr*gb2
        self.w1+=lr*np.outer(f,dh);self.b1+=lr*dh
        self.baseline=.985*self.baseline+.015*reward

def run(n,budget,seed,method,warmstart=None):
    rng=np.random.default_rng(seed)
    s=np.array([1 if ch=='+' else -1 for ch in warmstart],dtype=np.int8) if warmstart else rng.choice(np.array([-1,1],dtype=np.int8),size=n)
    current=energy(s);initial=current;best=current;best_s=s.copy()
    policy=TinyPolicy(seed) if method=='neural' else None
    stagnation=0;accepted=0;improves=0;op_counts=np.zeros(4,dtype=int)
    for step in range(budget):
        features=np.array([min(current/n**2,1),min(stagnation/300,1),step/budget,accepted/max(1,step+1)])
        if method=='neural':arm,h,probs=policy.choose(features,rng)
        elif method=='uniform':arm=int(rng.integers(0,4))
        elif method=='single':arm=0
        else:raise ValueError(method)
        op_counts[arm]+=1
        candidate=s.copy()
        candidate[rng.choice(n,size=min(MOVES[arm],n),replace=False)]*=-1
        new=energy(candidate);delta=new-current
        temperature=3.+.6*n*(1-step/budget)**2
        take=delta<=0 or rng.random()<np.exp(-delta/temperature)
        if take:s=candidate;current=new;accepted+=1
        gained=new<best
        if gained:best=new;best_s=candidate.copy();improves+=1;stagnation=0
        else:stagnation+=1
        if policy:
            reward=np.clip(-delta/(3*n),-1,1)+(1. if gained else 0.)
            policy.update(features,h,probs,arm,reward)
    return {'n':n,'seed':seed,'method':method,'budget':budget,'initial_energy':initial,
            'best_energy':int(best),'improvements':int(improves),
            'acceptance_rate':round(accepted/budget,4),
            'operator_counts':op_counts.tolist(),
            'best_sequence':''.join('+' if x>0 else '-' for x in best_s)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--lengths',nargs='+',type=int,default=[40,60,100])
    p.add_argument('--seeds',type=int,default=6)
    p.add_argument('--budget',type=int,default=6000)
    p.add_argument('--output',type=Path,default=Path('results/p006.json'))
    p.add_argument('--warmstart',type=str,default='')
    a=p.parse_args()
    runs=[run(n,a.budget,seed,m,a.warmstart if n==100 else None)
          for n in a.lengths for seed in range(a.seeds) for m in ('single','uniform','neural')]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'protocol':'equal proposal count, same random-start seed per method; CPU','runs':runs},indent=2)+'\n')
    for n in a.lengths:
        for m in ('single','uniform','neural'):
            scores=[r['best_energy'] for r in runs if r['n']==n and r['method']==m]
            print('P006',n,m,'scores',scores,'mean',round(float(np.mean(scores)),2))
if __name__=='__main__':main()
