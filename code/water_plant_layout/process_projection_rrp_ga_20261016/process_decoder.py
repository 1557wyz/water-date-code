"""Graph-driven version of the native adjacent-placement construction.

No CP/MIP/LP or alternate global solver. The GA controls rotations, frontier tie
order and local strategy. Treatment nodes are constructed first, using the
native four-side aligned-neighbor idea, all already placed route neighbors,
bounded backtracking and lookahead. Actual residual rectangles fill auxiliaries.
Search failure is diagnostic, never a feasible plant or infeasibility proof.
"""
import math
import numpy as np
from numba import njit
from rrp_decoder import split_rectangles

@njit(cache=True)
def gap_at(x,y,w,h,p,q,a,b):
    dx=max(0.,x-p-a,p-x-w);dy=max(0.,y-q-b,q-y-h)
    return dx,dy

@njit(cache=True)
def geometry_ok(i,x,y,sizes,pos,active,safety,targets,head_limits,check_routes=True):
    w,h=sizes[i]
    for j in range(len(active)):
        if not active[j]:continue
        p,q=pos[j];a,b=sizes[j];g=safety[i,j]
        if not(x+w+g<=p+1e-7 or p+a+g<=x+1e-7 or y+h+g<=q+1e-7 or q+b+g<=y+1e-7):return False
        if check_routes and targets[i,j]>=0:
            dx,dy=gap_at(x,y,w,h,p,q,a,b)
            if dx*dx+dy*dy>targets[i,j]*targets[i,j]+1e-7:return False
            if dx+dy+(1.5 if dx>1e-8 and dy>1e-8 else 0)>head_limits[i,j]+1e-7:return False
    return True

@njit(cache=True)
def future_possible(sizes,pos,active,targets):
    """Necessary intersection of neighbor-expanded x/y domains, not a proof."""
    n=len(active)
    for i in range(n):
        if active[i]:continue
        left,bottom,right,top=-1e20,-1e20,1e20,1e20
        for j in range(n):
            if active[j] and targets[i,j]>=0:
                t=targets[i,j];p,q=pos[j];w,h=sizes[j]
                left=max(left,p-sizes[i,0]-t);right=min(right,p+w+t)
                bottom=max(bottom,q-sizes[i,1]-t);top=min(top,q+h+t)
        if left>right+1e-7 or bottom>top+1e-7:return False
    return True

@njit(cache=True)
def candidate_positions(i,sizes,pos,active,safety,targets,head_limits,strategy,keep):
    n=len(active);w,h=sizes[i]
    points=np.empty((n*(80+4*n)+1,2));count=0
    for j in range(n):
        if not active[j] or targets[i,j]<0:continue
        p,q=pos[j];a,b=sizes[j]
        # Preserve native side/center/corner candidates, with a bounded sliding
        # scan along the neighboring face to accommodate parallel units.
        for offset in (safety[i,j], targets[i,j]):
            for k in range(9):
                yy=q-h+(a*0.+b+h)*k/8
                xx=p-w+(a+w)*k/8
                for x,y in ((p+a+offset,yy),(p-w-offset,yy),(xx,q+b+offset),(xx,q-h-offset)):
                    points[count,0]=x;points[count,1]=y;count+=1
        # Corner positions of already placed rectangles allow a parallel unit
        # to sit directly after its sibling along a shared inlet/collector.
        for k in range(n):
            if not active[k]:continue
            ox,oy=pos[k];ow,oh=sizes[k];g=safety[i,k]
            for x,y in ((ox+ow+g,q+b+safety[i,j]),(ox-w-g,q+b+safety[i,j]),
                        (p+a+safety[i,j],oy+oh+g),(p+a+safety[i,j],oy-h-g)):
                if count>=len(points):break
                points[count,0]=x;points[count,1]=y;count+=1
    values=np.full(keep,np.inf);best=np.zeros((keep,2));total=0
    for k in range(count):
        x,y=points[k]
        if not geometry_ok(i,x,y,sizes,pos,active,safety,targets,head_limits):continue
        duplicate=False
        for z in range(min(total,keep)):
            if abs(best[z,0]-x)<1e-7 and abs(best[z,1]-y)<1e-7:duplicate=True;break
        if duplicate:continue
        pos[i,0]=x;pos[i,1]=y;active[i]=True
        possible=future_possible(sizes,pos,active,targets)
        active[i]=False
        if not possible:continue
        x0,y0,x1,y1=x,y,x+w,y+h;near=0.
        for j in range(n):
            if not active[j]:continue
            x0=min(x0,pos[j,0]);y0=min(y0,pos[j,1]);x1=max(x1,pos[j,0]+sizes[j,0]);y1=max(y1,pos[j,1]+sizes[j,1])
            if targets[i,j]>=0:
                dx,dy=gap_at(x,y,w,h,pos[j,0],pos[j,1],sizes[j,0],sizes[j,1]);near+=dx+dy
        # Equal geometric candidates differ by genotype strategy direction.
        directional=(x+y)*.001 if strategy==0 else ((x-y)*.001 if strategy==1 else (-x+y)*.001)
        score=(x1-x0)*(y1-y0)+near*4.+directional
        if score>=values[-1]:continue
        slot=keep-1
        while slot>0 and score<values[slot-1]:
            values[slot]=values[slot-1];best[slot]=best[slot-1];slot-=1
        values[slot]=score;best[slot,0]=x;best[slot,1]=y;total=min(keep,total+1)
    return best,values,total,count

@njit(cache=True)
def construct_core(order,sizes,safety,targets,head_limits,strategy,node_budget=192,keep=6):
    n=len(order);degree=np.zeros(n,np.int64);rank=np.empty(n,np.int64)
    for k in range(n):rank[order[k]]=k
    for i in range(n):
        for j in range(n):degree[i]+=int(targets[i,j]>=0)
    core_count=np.sum(degree>0)
    pos=np.zeros((n,2));active=np.zeros(n,np.bool_)
    chosen=np.full(n,-1,np.int64);options=np.zeros((n,keep,2));counts=np.zeros(n,np.int64);next_slot=np.zeros(n,np.int64)
    depth=0;visits=0;attempts=0;best_depth=0;best_pos=pos.copy();best_active=active.copy()
    while depth>=0 and depth<core_count and visits<node_budget:
        if chosen[depth]<0:
            idx=-1;key=-1e30;placed_neighbors=0
            for i in range(n):
                if active[i] or degree[i]==0:continue
                linked=0
                for j in range(n):linked+=int(active[j] and targets[i,j]>=0)
                if depth>0 and linked==0:continue
                score=linked*10000.+degree[i]*100.-rank[i]
                if strategy==1:score=linked*10000.+degree[i]*10.-rank[i]
                if depth==0 and strategy==1:score=-float(rank[i])
                if depth==0 and strategy==2:
                    score=-float(rank[i])
                    for j in range(n):
                        if targets[i,j]>=0:score+=targets[i,j]*100.
                if score>key:idx=i;key=score;placed_neighbors=linked
            if idx<0:
                for i in order:
                    if not active[i] and degree[i]>0:idx=i;break
            chosen[depth]=idx
            if depth==0:
                options[depth,0,0]=0.;options[depth,0,1]=0.;counts[depth]=1
            elif placed_neighbors==0:
                xmax=0.
                for j in range(n):
                    if active[j]:xmax=max(xmax,pos[j,0]+sizes[j,0])
                options[depth,0,0]=xmax+np.max(safety);options[depth,0,1]=0.;counts[depth]=1
            else:
                pts,vals,total,tried=candidate_positions(idx,sizes,pos,active,safety,targets,head_limits,strategy,keep)
                options[depth]=pts;counts[depth]=total;attempts+=tried
            next_slot[depth]=0
            if counts[depth]==0 and depth>0:
                # Return directly to a placed process neighbor whose geometry
                # constrains this dead end, rather than enumerating unrelated
                # tail placements. This is bounded heuristic backjumping;
                # completeness is not claimed.
                jump=-1
                for previous in range(depth):
                    if targets[idx,chosen[previous]]>=0:jump=previous
                if jump>=0:
                    chosen[depth]=-1
                    for previous in range(depth-1,jump,-1):
                        active[chosen[previous]]=False;chosen[previous]=-1
                    depth=jump;active[chosen[depth]]=False
                    continue
        idx=chosen[depth]
        if next_slot[depth]>=counts[depth]:
            chosen[depth]=-1;depth-=1
            if depth>=0:active[chosen[depth]]=False
            continue
        slot=next_slot[depth];next_slot[depth]+=1;visits+=1
        pos[idx]=options[depth,slot];active[idx]=True;depth+=1
        if depth>best_depth:
            best_depth=depth;best_pos=pos.copy();best_active=active.copy()
        if depth<core_count:chosen[depth]=-1
    success=depth==core_count
    if not success:pos=best_pos;active=best_active
    return pos,active,success,visits,attempts,int(best_depth),int(core_count)

@njit(cache=True)
def pack_aux(order,sizes,safety,contact_weights,target_gaps,categories,strategy,pos,active):
    n=len(order);minx=0.;miny=0.
    for i in range(n):
        if active[i]:minx=min(minx,pos[i,0]);miny=min(miny,pos[i,1])
    for i in range(n):
        if active[i]:pos[i,0]-=minx;pos[i,1]-=miny
    side=2*(np.sum(np.maximum(sizes[:,0],sizes[:,1]))+n*np.max(safety)+1.)
    maxx,maxy=0.,0.;splits=0;tries=0;fallbacks=0;peak=1
    for i in range(n):
        if active[i]:maxx=max(maxx,pos[i,0]+sizes[i,0]);maxy=max(maxy,pos[i,1]+sizes[i,1])
    for idx in order:
        if active[idx]:continue
        w,h=sizes[idx];free=np.array([[0.,0.,side,side]])
        for j in range(n):
            if not active[j]:continue
            g=safety[idx,j]
            free=split_rectangles(free,np.array([pos[j,0]-g,pos[j,1]-g,sizes[j,0]+2*g,sizes[j,1]+2*g]))
            splits+=1;peak=max(peak,len(free))
        scorebest=np.inf;bx,by=0.,0.
        for k in range(len(free)):
            x,y,fw,fh=free[k]
            if fw+1e-7<w or fh+1e-7<h:continue
            tries+=1
            near=0.;zone=0.
            for j in range(n):
                if not active[j]:continue
                dx,dy=gap_at(x,y,w,h,pos[j,0],pos[j,1],sizes[j,0],sizes[j,1])
                near+=contact_weights[idx,j]*max(0.,math.hypot(dx,dy)-target_gaps[idx,j])
                if categories[idx]==categories[j]:zone+=abs(x+w/2-pos[j,0]-sizes[j,0]/2)+abs(y+h/2-pos[j,1]-sizes[j,1]/2)
            score=max(maxx,x+w)*max(maxy,y+h)+near*2+zone*.1
            if score<scorebest:scorebest=score;bx=x;by=y
        if not math.isfinite(scorebest):bx=maxx+np.max(safety[idx]);by=0.;fallbacks+=1
        pos[idx,0]=bx;pos[idx,1]=by;active[idx]=True;maxx=max(maxx,bx+w);maxy=max(maxy,by+h)
    return pos,maxx,maxy,tries,splits,fallbacks,peak
