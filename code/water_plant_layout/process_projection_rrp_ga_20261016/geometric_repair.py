"""Joint geometric repair inside the process decoder; no external optimizer.

The objective is only constraint residual. Footprints, connectivity, target gaps
and levels are immutable. All movements are provisional until independent final
checks pass. The main algorithm's Euclidean edge clearance is used, including
valid diagonal clearances, instead of the preceding axis-only restriction.
"""
import math
import numpy as np
from numba import njit

@njit(cache=True)
def residual_gradient(centers,sizes,safety,targets,head_limits,slack=.02,safety_weight=1.):
    n=len(centers);gradient=np.zeros((n,2));loss=0.;maxbad=0.;violations=0
    for i in range(n):
        for j in range(i):
            vx=centers[i,0]-centers[j,0];vy=centers[i,1]-centers[j,1]
            sx=abs(vx)-(sizes[i,0]+sizes[j,0])/2
            sy=abs(vy)-(sizes[i,1]+sizes[j,1])/2
            px=max(0.,sx);py=max(0.,sy);norm=math.hypot(px,py)
            signx=1. if vx>=0 else -1.;signy=1. if vy>=0 else -1.
            if norm>1e-10:gx=px/norm*signx;gy=py/norm*signy;d=norm
            elif sx>=sy:gx=signx;gy=0.;d=sx
            else:gx=0.;gy=signy;d=sy
            g=safety[i,j]+slack
            error=0.
            if d<g:
                v=g-d;loss+=v*v;error=-2*safety_weight*v
                maxbad=max(maxbad,max(0.,safety[i,j]-d));violations+=int(d<safety[i,j]-1e-6)
            if targets[i,j]>=0 and d>targets[i,j]-slack:
                v=d-targets[i,j]+slack;loss+=v*v;error+=2*v
                maxbad=max(maxbad,max(0.,d-targets[i,j]));violations+=int(d>targets[i,j]+1e-6)
            gradient[i,0]+=error*gx;gradient[j,0]-=error*gx
            gradient[i,1]+=error*gy;gradient[j,1]-=error*gy
            if targets[i,j]>=0:
                # Same routed-length head inequality after eliminating constants.
                route_gap=px+py+(1.5 if px>1e-8 and py>1e-8 else 0.)
                if route_gap>head_limits[i,j]-slack:
                    v=route_gap-head_limits[i,j]+slack;loss+=v*v
                    hx=signx if px>0 else 0.;hy=signy if py>0 else 0.
                    gradient[i,0]+=2*v*hx;gradient[j,0]-=2*v*hx
                    gradient[i,1]+=2*v*hy;gradient[j,1]-=2*v*hy
                    maxbad=max(maxbad,max(0.,route_gap-head_limits[i,j]));violations+=int(route_gap>head_limits[i,j]+1e-6)
    return loss,gradient,maxbad,violations

@njit(cache=True)
def relax(initial,sizes,safety,targets,head_limits,steps=3000,seed=1):
    centers=initial.copy();m=np.zeros_like(centers);v=np.zeros_like(centers)
    best=centers.copy();bestloss=1e100;bestbad=1e100;bestcount=100000;lastimprove=0
    np.random.seed(seed)
    for t in range(steps):
        sw=1.
        if seed%3==0:sw=.15+2.*((t%900)/899.)
        elif seed%3==1:sw=2.-1.85*((t%900)/899.)
        loss,grad,bad,count=residual_gradient(centers,sizes,safety,targets,head_limits,safety_weight=sw)
        if loss<bestloss:
            bestloss=loss;best=centers.copy();bestbad=bad;bestcount=count;lastimprove=t
        if count==0:return centers,True,t+1,loss,bad,count
        lr=.25 if t<steps*.7 else .08
        for i in range(len(centers)):
            for k in range(2):
                q=grad[i,k]
                m[i,k]=.9*m[i,k]+.1*q;v[i,k]=.999*v[i,k]+.001*q*q
                centers[i,k]-=lr*m[i,k]/(math.sqrt(v[i,k])+1e-8)
        for k in range(2):
            average=np.mean(centers[:,k])
            for i in range(len(centers)):centers[i,k]-=average
        if t-lastimprove>350:
            centers=best.copy();m[:]=0.;v[:]=0.
            for i in range(len(centers)):
                centers[i,0]+=np.random.normal(0.,1.5)
                centers[i,1]+=np.random.normal(0.,1.5)
            lastimprove=t
    return best,bestcount==0,steps,bestloss,bestbad,bestcount

def spectral_embedding(targets,sizes):
    adjacency=(targets>=0).astype(float);np.fill_diagonal(adjacency,0.)
    degree=adjacency.sum(axis=1);lap=np.diag(degree)-adjacency
    vals,vecs=np.linalg.eigh(lap)
    xy=vecs[:,1:3].copy()
    i,j=np.where(np.triu(adjacency,1))
    desired=(np.minimum(sizes[i,0],sizes[i,1])+np.minimum(sizes[j,0],sizes[j,1]))/2+3.
    actual=np.linalg.norm(xy[i]-xy[j],axis=1)
    scale=float(np.median(desired/np.maximum(actual,.001)))
    return xy*scale

def distance_embedding(targets,sizes):
    n=len(sizes);dist=np.full((n,n),1e6);np.fill_diagonal(dist,0.)
    for i,j in zip(*np.where(targets>=0)):
        dist[i,j]=(min(sizes[i])+min(sizes[j]))/2+min(targets[i,j],3.)
    for k in range(n):dist=np.minimum(dist,dist[:,k,None]+dist[None,k,:])
    square=dist*dist;gram=-.5*(square-square.mean(axis=0)[None,:]-square.mean(axis=1)[:,None]+square.mean())
    vals,vecs=np.linalg.eigh(gram);order=np.argsort(vals)[-2:]
    return vecs[:,order]*np.sqrt(np.maximum(vals[order],0.))

def initial_positions(targets,sizes,seed,strategy=0):
    rng=np.random.default_rng(seed)
    xy=distance_embedding(targets,sizes) if strategy==0 else spectral_embedding(targets,sizes)
    if strategy==1:xy=xy[:,::-1].copy()
    if strategy==2:
        angles=rng.uniform(0.,2*np.pi,len(sizes));radius=math.sqrt(float(np.prod(sizes,axis=1).sum()))*.6
        xy=np.column_stack((np.cos(angles),np.sin(angles)))*radius
    xy+=rng.normal(0.,2.,xy.shape)
    return xy
