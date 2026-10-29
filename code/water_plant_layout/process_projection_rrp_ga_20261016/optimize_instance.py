"""Generic input-driven entry point; application runs are outside study counts."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
os.environ['MPLBACKEND']='Agg'
import argparse
import contextlib
import csv
from pathlib import Path
import time
from common import ROOT,load,save,digest
from process_initializer import construct
from graph_model import make_optimizer
from validation import audit,serialize
from process_model import evaluate
from run_pure import ENGINES

def optimize(input_path,output,seed=5301,variant='enhanced'):
    input_path=Path(input_path).resolve();output=Path(output).resolve()
    if (output/'result.json').exists():raise RuntimeError('结果已存在，请使用新的输出目录；不会覆盖已有运行。')
    ins=load(input_path)
    for name in ('facilities','contacts','process_edges','hydraulic_segments','elevations','constraint_versions'):
        if name not in ins:raise ValueError('输入缺少显式字段：'+name)
    if variant not in ins['constraint_versions']:raise ValueError('不存在该约束版本：'+variant)
    output.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    with (output/'search.log').open('w',encoding='utf-8') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        core=construct(ins,variant,seed,24,120,2000,stages=3)
        save(output/'process_initializer.json',core)
        if not core['found']:
            result={'accepted':False,'final_layout':None,'reason':'本次工艺遗传搜索预算内未找到联合可行骨架，不构成数学无解证明',
                    'process_initializer':core['stats']}
        else:
            _,ga=make_optimizer(ins,variant,seed,36,24,core_layout=core['layout']);layout,_,_=ga.optimize()
            rows=serialize(layout);g=audit(ins,variant,rows);e=evaluate(ins,rows)
            accepted=g['spatial_feasible'] and e['process_and_head_all']
            if not accepted:raise RuntimeError('最终独立联合校核未通过，拒绝输出为可行布局。')
            result={'accepted':True,'final_layout':rows,'geometry':g,'three_routes_and_head':e,
                    'raw_score':ga.best_evaluated['score'],'process_initializer':core['stats'],'fitness_history':ga.fitness_history}
            with (output/'layout.csv').open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result.update(input_path=str(input_path),input_sha256=digest(input_path),seed=seed,variant=variant,
        runtime_s=time.perf_counter()-start,engine_hashes={n:digest(ROOT/n) for n in ENGINES},
        counted_in_formal_validation=False,witness_coordinates_supplied=False)
    save(output/'result.json',result)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='按显式设施、三线路与水头参数优化单个水厂布局；厂界为可变占用包围框。')
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True);parser.add_argument('--seed',type=int,default=5301);parser.add_argument('--variant',default='enhanced')
    args=parser.parse_args();result=optimize(args.input,args.output,args.seed,args.variant)
    print('联合可行：',result['accepted'],'；结果：',str(Path(args.output).resolve()/'result.json'))
