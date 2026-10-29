"""Verify small cases and run a bounded, auditable same-instance CP reference."""
import csv
import hashlib
import json
import time
from pathlib import Path
import ortools
from ortools.sat.python import cp_model
import water_plant_layout_v3 as legacy
from revision_exact_area import build_model, extract_layout
from revision_decoders import validate_layout
from revision_evaluation import evaluate_layout, EvaluationConfig


def verify():
    records = []
    for a,b,g,expected in [(2,2,0,8),(2,2,1,10),(4,2,1,18)]:
        fs = [legacy.Facility(str(i),str(i),a,b,'test') for i in range(2)]
        hint = [f.copy() for f in fs]
        hint[1].x = a+g
        gap = lambda x,y: g
        m,v,info = build_model(fs,hint,gap)
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = 10
        s.parameters.num_search_workers = 1
        status = s.solve(m)
        assert status == cp_model.OPTIMAL
        assert s.objective_value == s.best_objective_bound == expected
        assert not validate_layout(extract_layout(s,v,fs),fs,None,gap)
        records.append({'dimensions':[a,b],'gap':g,'optimum':expected})
    fs[0].length = 4.5
    try:
        build_model(fs,hint,gap)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid input accepted')
    return records


def main():
    root = Path(__file__).resolve().parent.parent
    out = root/'paper/revision_step09_cp'
    out.mkdir(exist_ok=False)
    tests = verify()
    fs = legacy.create_facilities()
    path = root/'paper/Supplementary Materials/List of Tables (manuscript)/Layout/facility_layout.csv'
    rows = {r['code']:r for r in csv.DictReader(path.open(encoding='utf-8-sig'))}
    hint = [f.copy() for f in fs]
    for f in hint:
        r = rows[f.name]
        f.x,f.y,f.rotated = float(r['x_m']),float(r['y_m']),bool(int(r['rotated']))
        assert f.get_dimensions() == (float(r['width_m']),float(r['height_m']))
    m,v,info = build_model(fs,hint,legacy.get_safety_distance)
    m.export_to_file(str(out/'model.pbtxt'))
    params = {'max_time_in_seconds':300,'num_search_workers':8,'random_seed':20260912}
    manifest = {'model':info,'parameters':params,'ortools_version':ortools.__version__,
                'small_case_tests':tests,'source_hashes':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__),Path(__file__).with_name('revision_exact_area.py'),path]},
                'historical_evaluation':evaluate_layout(hint,fs,EvaluationConfig(None))}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    solver = cp_model.CpSolver()
    for k,value in params.items():
        setattr(solver.parameters,k,value)
    solver.parameters.log_search_progress = True
    solver.parameters.log_to_stdout = False
    started = time.perf_counter()
    events = []
    class Callback(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            event = {'seconds':time.perf_counter()-started,'area':self.objective_value,'bound':self.best_objective_bound}
            events.append(event)
            print(json.dumps(event),flush=True)
    with (out/'solver.log').open('w',encoding='utf-8') as log:
        solver.log_callback = lambda message: log.write(message+'\n')
        status = solver.solve(m,Callback())
    result = {'status':solver.status_name(status),'wall_seconds':solver.wall_time,
              'solver_bound':solver.best_objective_bound,'events':events,'response_stats':solver.response_stats()}
    lower = max(info['analytical_lower_bound'],solver.best_objective_bound)
    result['lower_bound'] = lower
    result['upper_bound'] = info['verified_hint_upper_bound']
    if status in (cp_model.FEASIBLE,cp_model.OPTIMAL):
        layout = extract_layout(solver,v,fs)
        assert not validate_layout(layout,fs,None,legacy.get_safety_distance)
        result['evaluation'] = evaluate_layout(layout,fs,EvaluationConfig(None))
        assert result['evaluation']['metrics']['area_m2'] == solver.objective_value
        result['upper_bound'] = min(result['upper_bound'],solver.objective_value)
        result['layout'] = [{'name':f.name,'x':f.x,'y':f.y,'rotated':f.rotated} for f in layout]
    result['relative_gap'] = (result['upper_bound']-lower)/result['upper_bound']
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','wall_seconds','lower_bound','upper_bound','relative_gap']}),flush=True)


if __name__ == '__main__':
    main()
