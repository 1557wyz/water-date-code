"""Vectorized checks derived directly from input; no decoder/repair imports."""
import numpy as np

class Checker:
    def __init__(self,ins,variant,process_only=False):
        nodes={e[k] for e in ins['process_edges'] for k in ('source','target')}
        self.fs=[f for f in ins['facilities'] if not process_only or f['id'] in nodes]
        fs=self.fs;idx={f['id']:i for i,f in enumerate(fs)};self.n=len(fs)
        self.original=np.array([[f['length_m'],f['width_m']] for f in fs])
        self.i,self.j=np.triu_indices(len(fs),1)
        c=ins['constraint_versions'][variant];o=c.get('category_overrides',{})
        matrix={tuple(sorted(r['levels'])):r['distance_m'] for r in c['safety_matrix']}
        self.gaps=[]
        for a,b in zip(self.i,self.j):
            u,v=fs[a],fs[b];g=matrix[tuple(sorted((u['safety_level'],v['safety_level']))) ]
            if o.get(u['id'],u['category'])==o.get(v['id'],v['category']):g=max(1.,g-1.)
            self.gaps.append(g)
        self.gaps=np.array(self.gaps)
        edges=ins['process_edges'];self.ei=np.array([idx[e['source']] for e in edges]);self.ej=np.array([idx[e['target']] for e in edges])
        self.targets=np.array([e['target_gap_m'] for e in edges]);segments={(e['source'],e['target'],e['system']):e for e in ins['hydraulic_segments']}
        self.factor=[];self.terminal=[];self.slope=[];self.constant=[];self.available=[]
        for e in edges:
            h=segments[e['source'],e['target'],e['system']];a=ins['elevations'][e['source']];b=ins['elevations'][e['target']]
            factor,terminal={'main_water':(1.08,4.),'advanced_water':(1.10,4.),'sludge':(1.18,5.)}[e['system']]
            self.factor.append(factor);self.terminal.append(terminal);self.slope.append(h['parameters'][0]);self.constant.append(h['parameters'][1]+h['parameters'][2])
            self.available.append(h.get('source_outlet_elevation_m',a['water_out'])-h.get('target_inlet_elevation_m',b['ground']+(-.5 if b['type']=='pump' else .3)))
        for name in ('factor','terminal','slope','constant','available'):setattr(self,name,np.array(getattr(self,name)))

    def check(self,rectangles,rotations):
        xy=np.array(rectangles,dtype=float);rot=np.array(rotations,dtype=bool)
        if xy.shape!=(self.n,4) or rot.shape!=(self.n,):return {'joint':False,'malformed':True}
        x,y,w,h=xy.T;expected=self.original.copy();expected[rot]=expected[rot][:,::-1]
        dims=bool((np.abs(xy[:,2:]-expected)<1e-7).all());i,j=self.i,self.j
        dx=np.maximum(0.,np.maximum(x[i],x[j])-np.minimum(x[i]+w[i],x[j]+w[j]))
        dy=np.maximum(0.,np.maximum(y[i],y[j])-np.minimum(y[i]+h[i],y[j]+h[j]))
        overlap=(x[i]<x[j]+w[j]-1e-7)&(x[j]<x[i]+w[i]-1e-7)&(y[i]<y[j]+h[j]-1e-7)&(y[j]<y[i]+h[i]-1e-7)
        spatial=bool(dims and np.isfinite(xy).all() and (w>0).all() and (h>0).all() and (x>=-1e-7).all() and (y>=-1e-7).all() and not overlap.any() and (np.hypot(dx,dy)+1e-6>=self.gaps).all())
        a,b=self.ei,self.ej
        ex=np.maximum(0.,np.maximum(x[a],x[b])-np.minimum(x[a]+w[a],x[b]+w[b]))
        ey=np.maximum(0.,np.maximum(y[a],y[b])-np.minimum(y[a]+h[a],y[b]+h[b]))
        adjacent=bool((np.hypot(ex,ey)<=self.targets+1e-6).all())
        lengths=self.factor*(ex+ey+self.terminal+np.where((ex>0)&(ey>0),1.5,0.))
        margins=self.available-self.slope*lengths-self.constant;head=bool((margins>=-1e-7).all())
        return {'joint':spatial and adjacent and head,'spatial':spatial,'adjacent':adjacent,'head':head,
                'minimum_head_margin_m':float(margins.min()),'malformed':False}
