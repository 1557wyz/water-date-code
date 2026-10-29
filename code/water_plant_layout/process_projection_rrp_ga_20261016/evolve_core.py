"""GA initialization of the process-first decoder, with joint local repair.

Chromosomes carry actual centers and legal orientation bits. Feasibility is
measured on every original process connection, never on stage averages.
"""
import math
import time
import numpy as np
from geometric_repair import relax,initial_positions

def initialize_core(original,gaps,targets,head,seed=1201,population=24,generations=120,repair_steps=2000,warm=None,observer=None,groups=None):
    rng=np.random.default_rng(seed);n=len(original);pool=[];evaluations=0;history=[];begin=time.perf_counter()
    edge_a,edge_b=np.where(np.triu(targets>=0,1))
    def topology(r):
        d=r['centers'][edge_a]-r['centers'][edge_b]
        side=np.where(np.abs(d[:,0])>np.abs(d[:,1]),np.where(d[:,0]>0,0,1),np.where(d[:,1]>0,2,3))
        return bytes(r['rotations'])+bytes(side.astype(np.uint8))
    def diverse(candidates):
        candidates.sort(key=lambda r:r['loss']);selected=[];seen=set()
        for r in candidates:
            key=topology(r)
            if key not in seen:
                seen.add(key);selected.append(r)
            if len(selected)==population:return selected
        for r in candidates:
            if all(r is not x for x in selected):selected.append(r)
            if len(selected)==population:return selected
        return selected
    def assess(xy,rot):
        nonlocal evaluations
        sizes=original.copy();sizes[rot]=sizes[rot][:,::-1]
        xy,ok,it,loss,bad,count=relax(xy,sizes,gaps,targets,head,repair_steps,int(rng.integers(1,2**30)))
        evaluations+=1
        out={'centers':xy,'rotations':rot.copy(),'feasible':bool(ok),'loss':float(loss),'maximum_error_m':float(bad),'violations':int(count),'repair_iterations':int(it)}
        if observer:observer(out,evaluations)
        return out
    if warm:
        for w in warm:
            rot=np.array(w['rotations'],dtype=bool);sizes=original.copy();sizes[rot]=sizes[rot][:,::-1]
            xy=np.array(w['positions'])+sizes/2
            pool.append(assess(xy,rot))
            if pool[-1]['feasible']:return pool[-1],{'evaluations':evaluations,'generations':0,'elapsed_s':time.perf_counter()-begin,'history':[]}
    while len(pool)<population:
        rot=rng.integers(0,2,n).astype(bool);sizes=original.copy();sizes[rot]=sizes[rot][:,::-1]
        xy=initial_positions(targets,sizes,int(rng.integers(1,2**30)),len(pool)%3)
        r=assess(xy,rot);pool.append(r)
        if r['feasible']:return r,{'evaluations':evaluations,'generations':0,'elapsed_s':time.perf_counter()-begin,'history':[]}
    for generation in range(generations):
        pool.sort(key=lambda r:r['loss']);history.append(pool[0]['loss'])
        elite=pool[:max(4,population//3)];offspring=[elite[0]]
        for k in range(population-1):
            a=elite[int(rng.integers(len(elite)))];xy=a['centers'].copy();rot=a['rotations'].copy()
            action=int(rng.integers(6));anchor=int(rng.integers(n))
            if action==0:
                rot[anchor]=not rot[anchor]
            elif action==1:
                chosen=rng.choice(n,int(rng.integers(1,max(2,n//5))),replace=False)
                rot[chosen]=~rot[chosen];xy[chosen]+=rng.normal(0.,2.,(len(chosen),2))
            elif action==2:
                # Rotate a connected process block without breaking its internal
                # relative geometry; subsequent repair checks its external edges.
                block=[anchor]
                for _ in range(int(rng.integers(1,max(3,n//3)))):
                    options=[j for i in block for j in np.flatnonzero(targets[i]>=0) if j not in block]
                    if not options:break
                    block.append(int(rng.choice(options)))
                mid=xy[block].mean(axis=0);rel=xy[block]-mid
                xy[block]=np.column_stack((-rel[:,1],rel[:,0]))+mid;rot[block]=~rot[block]
            elif action==3:
                b=elite[int(rng.integers(len(elite)))];chosen=rng.random(n)<.25
                # Parents are centered; translate the donor block to its anchor.
                donor=b['centers']+(xy[anchor]-b['centers'][anchor])
                xy[chosen]=donor[chosen];rot[chosen]=b['rotations'][chosen]
            elif action==4:
                # Permute comparable parallel units, not two directly connected
                # stages. This changes train ordering around shared collectors.
                if groups is not None:candidates=np.flatnonzero((groups==groups[anchor])&(np.arange(n)!=anchor))
                else:
                    candidates=np.flatnonzero((targets[anchor]<0)&(np.arange(n)!=anchor)&
                        (np.max(original,axis=1)<np.max(original[anchor])*1.8)&
                        (np.max(original,axis=1)>np.max(original[anchor])/.8/2.))
                if len(candidates):
                    second=int(rng.choice(candidates));xy[[anchor,second]]=xy[[second,anchor]]
            else:xy+=rng.normal(0.,float(rng.choice([.4,1.,3.,6.])),xy.shape)
            r=assess(xy,rot);offspring.append(r)
            if r['feasible']:
                return r,{'evaluations':evaluations,'generations':generation+1,'elapsed_s':time.perf_counter()-begin,'history':history}
        pool=diverse(pool+offspring)
        if generation%15==14:
            # Retain the best, diversify near it to escape identical local minima.
            for k in range(population//2,population):
                base=pool[0];xy=base['centers']+rng.normal(0.,3.,base['centers'].shape)
                rot=base['rotations'].copy();change=rng.choice(n,max(1,n//8),replace=False);rot[change]=~rot[change]
                pool[k]=assess(xy,rot)
                if pool[k]['feasible']:return pool[k],{'evaluations':evaluations,'generations':generation+1,'elapsed_s':time.perf_counter()-begin,'history':history}
    pool.sort(key=lambda r:r['loss'])
    return pool[0],{'evaluations':evaluations,'generations':generations,'elapsed_s':time.perf_counter()-begin,'history':history}

if __name__=='__main__':
    import sys
    from probe_repair import problem
    from common import ROOT,load,save
    for name in sys.argv[1:] or ['A49','A42','B80']:
        d,fs,sizes,gaps,targets,head=problem(name)
        path=ROOT/f'pilot/repair_{name}.json';warm=[load(path)] if path.exists() else None
        best,stats=initialize_core(sizes,gaps,targets,head,warm=warm)
        oriented=sizes.copy();oriented[best['rotations']]=oriented[best['rotations']][:,::-1]
        result={**best,'centers':best['centers'].tolist(),'rotations':best['rotations'].tolist(),'positions':(best['centers']-oriented/2).tolist(),'sizes':oriented.tolist(),'stats':stats}
        save(ROOT/f'pilot/evolve_{name}.json',result)
        print(name,best['feasible'],round(best['loss'],6),best['maximum_error_m'],stats['evaluations'],round(stats['elapsed_s'],1),flush=True)
