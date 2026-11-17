import sys, numpy as np, torch
from collections import Counter, defaultdict
from .analysis import *
from .train import load
s,l,w,c=load(sys.argv[1]); inf=Informant(s,l,w,c); rng=np.random.default_rng(0)
tr=[e for e in np.nonzero(w.partition==0)[0] if w.events[e,0]==2]
for x,y in (('A_type','P_type'),('pred','A_type'),('A_colour','P_colour')):
    tab=defaultdict(Counter)
    for i in rng.permutation(tr)[:600]:
        ctx=make_context(w,rng,int(i),[x,y,y] if rng.random()<.5 else [x,x,y])
        if ctx: sl=w.slots(i); tab[(int(sl[x]),int(sl[y]))][inf.say([ctx])[0]]+=1
    print(x,y)
    vx=sorted({k[0] for k in tab}); vy=sorted({k[1] for k in tab})
    for a in vx: print('  ', ' | '.join((' '.join(map(str,modal(tab[(a,b)])))+f'({share(tab[(a,b)]):.1f})').ljust(14) if (a,b) in tab else ' '*14 for b in vy))
