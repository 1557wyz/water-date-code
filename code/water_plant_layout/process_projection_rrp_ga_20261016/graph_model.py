"""ID-agnostic graph/RRP extension of the existing six-island optimizer.

Physical inputs are frozen old instances. All connections, targets and hydraulic
ports are data-driven. Facility IDs only join tables, never select behavior.
"""
import copy
import importlib.util
import math
import sys
from pathlib import Path
import numpy as np
from common import CORE, ROOT, load
from rrp_decoder import graph_order
from process_decoder import construct_core,pack_aux
from process_model import evaluate,PARAMS
from validation import serialize

OLD = ROOT.parent/'independent_cross_plant_validation_20261015'


def load_adapter():
    spec = importlib.util.spec_from_file_location('read_only_native_adapter', OLD/'native_adapter.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def make_optimizer(instance, variant='enhanced', seed=1201, population=36, generations=24,core_layout=None):
    adapter = load_adapter()
    module = adapter.load_core()
    facilities = adapter.configure(module, instance, variant)

    class GraphRRPGA(module.AdvancedGA):
        def __init__(self):
            super().__init__(facilities, population_size=population, generations=generations, random_seed=seed)
            self.categories = sorted(set(f.category for f in self.facilities))
            self.cat_to_idx = {c: i for i, c in enumerate(self.categories)}
            self.instance, self.variant = instance, variant
            self.process_seed=copy.deepcopy(core_layout)
            self.process_byid={r['id']:r for r in core_layout} if core_layout else {}
            self.index = {f.name: i for i, f in enumerate(self.facilities)}
            if len(self.index) != self.n:
                raise ValueError('duplicate facility ID')
            self.predecessors = [set() for _ in facilities]
            self.contact_weights = np.zeros((self.n, self.n))
            self.target_gaps = np.zeros((self.n, self.n))
            self.edges = []
            for e in instance['contacts']:
                a, b = self.index[e['source']], self.index[e['target']]
                if a == b or e['adjacency_weight'] <= 0 or e['target_gap_m'] < 0:
                    raise ValueError('invalid contact')
                self.edges.append((a, b, e))
                # Local heuristic aggregates multiple systems between two units.
                for i, j in ((a,b),(b,a)):
                    oldw = self.contact_weights[i,j]
                    self.target_gaps[i,j] = (oldw*self.target_gaps[i,j]+e['adjacency_weight']*e['target_gap_m'])/(oldw+e['adjacency_weight'])
                    self.contact_weights[i,j] += e['adjacency_weight']
                if e['system'] in ('main_water', 'advanced_water') and not e.get('recycle', False):
                    self.predecessors[b].add(a)
            self.edges.sort(key=lambda r: (r[0], r[1], r[2]['system']))
            self.gaps = np.array([[module.get_safety_distance(a,b) for b in facilities] for a in facilities],dtype=float)
            self.hard_targets=np.full((self.n,self.n),-1.)
            self.head_limits=np.full((self.n,self.n),np.inf)
            for e in instance['process_edges']:
                i,j=self.index[e['source']],self.index[e['target']]
                for a,b in ((i,j),(j,i)):
                    if self.hard_targets[a,b]<0:self.hard_targets[a,b]=e['target_gap_m']
                    else:self.hard_targets[a,b]=min(self.hard_targets[a,b],e['target_gap_m'])
            for e in instance['hydraulic_segments']:
                i,j=self.index[e['source']],self.index[e['target']]
                a,b=instance['elevations'][e['source']],instance['elevations'][e['target']]
                outlet=e.get('source_outlet_elevation_m',a['water_out'])
                inlet=e.get('target_inlet_elevation_m',b['ground']+(-.5 if b['type']=='pump' else .3))
                slope,local,margin,_=e['parameters'];factor,terminal,_=PARAMS[e['system']]
                limit=(outlet-inlet-local-margin)/(slope*factor)-terminal
                self.head_limits[i,j]=min(self.head_limits[i,j],limit)
                self.head_limits[j,i]=min(self.head_limits[j,i],limit)
            if not np.isfinite(self.gaps).all() or (self.gaps < 0).any():
                raise ValueError('invalid clearance')
            if any(not (math.isfinite(f.length) and math.isfinite(f.width) and f.length>0 and f.width>0) for f in facilities):
                raise ValueError('invalid facility size')
            for row in instance['hydraulic_segments']:
                for end in ('source','target'):
                    if row[end] not in self.index or row[end] not in instance['elevations']:
                        raise ValueError('missing hydraulic node/elevation')
            self.fitness_calls = 0
            self.unique_evaluations = 0
            self.decoder_calls = 0
            self.candidate_attempts = 0
            self.split_calls = 0
            self.fallback_count = 0
            self.peak_free = 0
            self.best_evaluated = None
            self.best_diagnostic = None
            self.feasible_evaluations = 0
            self.core_successes = 0
            self.backtracking_visits = 0
            self._decode_cache = {}
            self.observer = None
            self.last_trace = None
            # Reject process cycles immediately, before optimization.
            graph_order({'order': list(range(self.n)), 'zone_order': self.categories, 'strategy': 0},
                        self.predecessors, [f.category for f in facilities])

        def _repair_chromosome(self,chrom):
            c=super()._repair_chromosome(chrom)
            if getattr(self,'process_seed',None):
                fixed=[i for i,f in enumerate(self.facilities) if f.name in self.process_byid]
                c['order']=fixed+[i for i in c['order'] if i not in fixed]
                for i in fixed:c['rotations'][i]=self.process_byid[self.facilities[i].name]['rotated']
            return c

        def decode_chromosome(self, chrom, container_size):
            c = self._repair_chromosome(chrom)
            key=self._chromosome_key(c)
            if key in self._decode_cache:
                cached=self._decode_cache[key]
                return [f.copy() for f in cached[0]],cached[1],cached[2]
            cats = [f.category for f in self.facilities]
            actual_order=list(c['order'])
            if c['strategy']==1:
                zones={z:k for k,z in enumerate(c['zone_order'])}
                ranks={i:k for k,i in enumerate(actual_order)}
                actual_order.sort(key=lambda i:(zones[cats[i]],ranks[i]))
            order = np.asarray(actual_order,dtype=np.int64)
            sizes = np.array([(f.width,f.length) if c['rotations'][i] else (f.length,f.width)
                              for i,f in enumerate(self.facilities)],dtype=float)
            if self.process_seed:
                initial=np.zeros((self.n,2));active=np.zeros(self.n,dtype=np.bool_)
                width=max(r['x_m']+r['length_m'] for r in self.process_seed)
                height=max(r['y_m']+r['width_m'] for r in self.process_seed)
                for i,f in enumerate(self.facilities):
                    if f.name not in self.process_byid:continue
                    row=self.process_byid[f.name];x,y=row['x_m'],row['y_m'];w,h=row['length_m'],row['width_m']
                    if c['strategy']==1:x,y,w,h=y,x,h,w
                    elif c['strategy']==2:x,y=width-x-w,height-y-h
                    initial[i]=x,y;sizes[i]=w,h;active[i]=True
                success=True;visits=0;tried=0;depth=total=int(active.sum())
            else:
                initial,active,success,visits,tried,depth,total=construct_core(order,sizes,self.gaps,self.hard_targets,self.head_limits,c['strategy'])
            self.core_successes+=int(success)
            self.backtracking_visits+=visits
            values=pack_aux(order,sizes,self.gaps,self.contact_weights,self.target_gaps,
                            np.array([self.cat_to_idx[x] for x in cats]),c['strategy'],initial,active)
            positions,bw,bh,attempts,splits,fallbacks,peak=values
            attempts+=tried
            trace={'core_success':bool(success),'placed_core_nodes':depth,'core_nodes':total,'visits':visits}
            self.decoder_calls += 1
            self.candidate_attempts += attempts
            self.split_calls += splits
            self.fallback_count += fallbacks
            self.peak_free = max(self.peak_free,peak)
            self.last_trace = trace
            placed = []
            for i, original in enumerate(self.facilities):
                f = original.copy()
                f.rotated = c['rotations'][i]
                if f.name in self.process_byid and c['strategy']==1:f.rotated=not f.rotated
                f.x, f.y = map(float,positions[i])
                placed.append(f)
            self._decode_cache[key]=([f.copy() for f in placed],float(bw),float(bh))
            return placed,float(bw),float(bh)

        def _contact_details(self, placed):
            byid = {f.name:f for f in placed}
            rows = []
            for a,b,e in self.edges:
                fa,fb = byid[e['source']],byid[e['target']]
                dx,dy = module.facility_edge_gap(fa,fb)
                distance = math.hypot(dx,dy)
                route = module.calculate_pipeline_segment(fa,fb,e['system'])
                rows.append((e,distance,route))
            return rows

        def _calc_process_adjacency_strict(self, placed):
            # Native piecewise rewards/penalties, with data-driven route tiers.
            # Ancillary links remain soft; hard acceptance concerns three routes.
            required={(e['source'],e['target']):e['route_class'] for e in instance['process_edges']}
            total=0.
            for e,d,_ in self._contact_details(placed):
                route=required.get((e['source'],e['target']))
                if route in ('main_water','advanced_water'):
                    total+=(-2000 if d<=1+1e-7 else -1600 if d<=2+1e-7 else -1000 if d<=4+1e-7 else -400 if d<=8+1e-7 else d*100 if d<=12 else d*350)
                elif route=='sludge' or e['adjacency_weight']>=8:
                    total+=(-400 if d<=5+1e-7 else -100 if d<=10+1e-7 else d*40 if d>25 else 0)
                else:total+=(-150 if d<=5+1e-7 else d*15 if d>40 else 0)
            return total

        def _calc_adjacency_penalty_strict(self, placed):
            return self._calc_process_adjacency_strict(placed),0.0

        def _calc_pipeline_length(self, placed):
            return math.fsum(r['weighted_length_m'] for _,_,r in self._contact_details(placed))

        def hydraulic_details(self,placed):
            byid = {f.name:f for f in placed}
            rows=[]
            for e in self.instance['hydraulic_segments']:
                up,down=e['source'],e['target']
                a,b=self.instance['elevations'][up],self.instance['elevations'][down]
                # Explicit port overrides support sludge outlets without any ID special case.
                source=e.get('source_outlet_elevation_m',a['water_out'])
                target=e.get('target_inlet_elevation_m',b['ground']+(-.5 if b['type']=='pump' else .3))
                length=module.calculate_pipeline_segment(byid[up],byid[down],module.infer_pipeline_system(up,down))['routed_length_m']
                slope,local,margin,weight=e['parameters']
                violation=slope*length+local+margin-(source-target)
                rows.append((e,violation,weight))
            return rows

        def _calc_elevation_constraint(self, placed):
            total=0.0
            for _,v,w in self.hydraulic_details(placed):
                if v>0:
                    total+=w*v**1.2
                    if w>=2.5 and v>.5:
                        total+=w*(v-.5)*2
                else:
                    total-=w*min(-v,2.0)*.008
            return total

        def _get_elevation_report(self,placed):
            return {(e['source'],e['target']):{'shortage_m':max(0.,v),'feasible':v<=1e-7}
                    for e,v,_ in self.hydraulic_details(placed)}

        def _calc_dormitory_cluster_penalty(self, placed):
            # Role/category-based replacement; no assumed dorm count or facility ID.
            group=[f for f in placed if f.category=='living']
            if len(group)<2:
                return 0.0
            x0,y0=min(f.x for f in group),min(f.y for f in group)
            x1=max(f.x+f.get_dimensions()[0] for f in group)
            y1=max(f.y+f.get_dimensions()[1] for f in group)
            fill=sum(f.area for f in group)/((x1-x0)*(y1-y0))
            return max(0.0,.7-fill)*500

        def score_details(self,placed,bw=None,bh=None):
            # One declared occupied-envelope convention at every new call site.
            bw=max(f.x+f.get_dimensions()[0] for f in placed)-min(f.x for f in placed)
            bh=max(f.y+f.get_dimensions()[1] for f in placed)-min(f.y for f in placed)
            return adapter.components(self,placed,bw,bh)

        def calculate_fitness(self, chrom):
            self.fitness_calls+=1
            c=self._repair_chromosome(chrom)
            key=self._chromosome_key(c)
            if self.enable_cache and key in self._fitness_cache:
                return self._fitness_cache[key]
            placed,bw,bh=self.decode_chromosome(c,0.0)
            details=self.score_details(placed)
            contract=evaluate(instance,serialize(placed))
            # Independently check every pair before permitting a feasible elite.
            xy=np.array([(f.x,f.y,*f.get_dimensions()) for f in placed])
            xx,yy,ww,hh=xy.T;i,j=np.triu_indices(self.n,1)
            dx=np.maximum(0,np.maximum(xx[i],xx[j])-np.minimum(xx[i]+ww[i],xx[j]+ww[j]))
            dy=np.maximum(0,np.maximum(yy[i],yy[j])-np.minimum(yy[i]+hh[i],yy[j]+hh[j]))
            overlap=(xx[i]<xx[j]+ww[j]-1e-7)&(xx[j]<xx[i]+ww[i]-1e-7)&(yy[i]<yy[j]+hh[j]-1e-7)&(yy[j]<yy[i]+hh[i]-1e-7)
            spatial=bool(not overlap.any() and (np.hypot(dx,dy)+1e-6>=self.gaps[i,j]).all() and (xx>=-1e-7).all() and (yy>=-1e-7).all())
            feasible=spatial and contract['process_and_head_all']
            raw_score=details['score']
            # Feasibility-first selection cannot trade a failed process route
            # for a smaller envelope or a larger compactness reward.
            score=1e6*(.5+math.atan(raw_score/100000.)/math.pi) if feasible else (
                1e12+1e9*(int(not spatial)+contract['adjacency_violations']+contract['head_violations'])+
                1e6*min(999.,contract['adjacency_excess_m']+contract['head_shortage_m']))
            details['selection_fitness']=score;details['joint_feasible']=feasible
            details['process_contract']=contract;details['spatial_feasible']=spatial
            self.feasible_evaluations+=int(feasible)
            self.unique_evaluations+=1
            record={'score':raw_score,'selection_fitness':score,'chromosome':copy.deepcopy(c),
                    'layout':[f.copy() for f in placed],'width':bw,'height':bh,'components':details}
            if self.best_diagnostic is None or score<self.best_diagnostic['selection_fitness']:
                self.best_diagnostic=record
            if feasible and (self.best_evaluated is None or raw_score<self.best_evaluated['score']):
                self.best_evaluated={'score':raw_score,'chromosome':copy.deepcopy(c),
                    'layout':[f.copy() for f in placed],'width':bw,'height':bh,'components':details}
            if self.observer:
                self.observer(self,placed,c,details)
            if self.enable_cache:
                if len(self._fitness_cache)>=self._fitness_cache_max_size:
                    self._fitness_cache.clear()
                self._fitness_cache[key]=score
            return score

        def _calculate_multi_objective(self,chrom):
            placed,bw,bh=self.decode_chromosome(chrom,0.)
            return bw*bh,self._calc_pipeline_length(placed),self._calc_process_adjacency_strict(placed),self.total_area/(bw*bh),self._calc_elevation_constraint(placed)

        # Bypass legacy ID-specific beautification. No new geometry is silently introduced.
        def _beautify_layout_constrained(self,layout,max_w,max_h):
            return layout

        def _fix_overlaps_constrained(self,layout,max_w,max_h):
            return layout

        def optimize(self):
            super().optimize()  # Retain original islands/GA/DE/SA/RL/archive mechanisms.
            # Explicitly return best evaluated scalar solution, rather than area-based
            # legacy final override or a historical lock-frame score.
            best=self.best_evaluated
            if best is None:
                self.accepted=False
                return [],0.,0.
            self.accepted=True
            self.best_solution=copy.deepcopy(best['chromosome'])
            self.best_fitness=best['score']
            self.best_layout=[f.copy() for f in best['layout']]
            return self.best_layout,best['width'],best['height']

    return module,GraphRRPGA()


def rename_instance(instance,mapping):
    """Bijective ID change; same row order, dimensions, graph and semantics."""
    out=copy.deepcopy(instance)
    for f in out['facilities']:
        f['id']=mapping[f['id']]
        f['label']='neutral label'  # Labels must not be interpreted either.
    seen=set()
    for row in out['contacts']+out['hydraulic_segments']+out.get('process_edges',[]):
        if id(row) in seen:continue
        seen.add(id(row))
        row['source'],row['target']=mapping[row['source']],mapping[row['target']]
    out['elevations']={mapping[k]:v for k,v in out['elevations'].items()}
    for c in out['constraint_versions'].values():
        c['category_overrides']={mapping[k]:v for k,v in c.get('category_overrides',{}).items()}
    for row in out.get('reference_site',{}).get('geometry_witness',[]):
        row['id']=mapping[row['id']]
    return out
