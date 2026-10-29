"""Insert solver output into the local revision notes, without changing Word."""
import json
from pathlib import Path


def main():
    paper = Path(__file__).resolve().parent.parent/'paper'
    result = json.loads((paper/'revision_step09_cp/result.json').read_text(encoding='utf-8'))
    manifest = json.loads((paper/'revision_step09_cp/manifest.json').read_text(encoding='utf-8'))
    before = manifest['historical_evaluation']['metrics']
    after = result['evaluation']['metrics']
    lb,ub,gap = result['lower_bound'],result['upper_bound'],result['relative_gap']
    text = '\n## 本次实际求解结果\n\n'
    text += f"状态：{result['status']}；实际求解时间：{result['wall_seconds']:.2f}秒；有效下界：{lb:.0f}平方米；可行上界：{ub:.0f}平方米；相对间隙：{gap:.2%}。\n\n"
    text += '| 统一评价指标 | 历史布局 | CP面积参照 |\n|---|---:|---:|\n'
    for key in before:
        if isinstance(before[key],(float,int)) and key in after:
            text += f'| {key} | {before[key]:.6g} | {after[key]:.6g} |\n'
    text += f"\nCP几何面积相对历史布局减少{1-ub/before['area_m2']:.2%}，但目标不同，不能据此宣称完整算法优劣。保存布局已用独立几何检查及共同四目标评价器复算。下界到上界之间仍有未闭合区间，不将上界写成已证最优面积。\n"
    path = paper/'revision_step09_exact_reference.md'
    original = path.read_text(encoding='utf-8').split('\n## 本次实际求解结果')[0]
    path.write_text(original+text,encoding='utf-8')
    path = paper/'revision_methods_insertions_draft.md'
    original = path.read_text(encoding='utf-8').split('\n## 约简精确参照')[0]
    insertion = f'''
## 约简精确参照（新增方法小节与结果表）

An area-only reference was formulated for the same 34 facilities, retaining their dimensions, optional 90-degree rotations, and all pairwise axis-aligned safety-clearance constraints. The occupied bounding-box area was minimized without a prescribed site boundary. Pipeline, adjacency, and hydraulic objectives were excluded from this reference. With integral dimensions and clearances, fixed-orientation separation constraints admit integer longest-path coordinates without increasing the enclosing dimensions; the integer formulation therefore preserves the optimum of this reduced continuous-coordinate problem. This argument does not establish integer equivalence for the complete multiobjective formulation.

OR-Tools CP-SAT 9.14.6206 was run with a 300-second limit and eight workers, using the independently verified historical layout as a feasible hint. The run returned {result['status']}, with a feasible area of {ub:.0f} square metres and a valid lower bound of {lb:.0f} square metres. The relative bound gap, defined as (upper bound minus lower bound)/upper bound, was {gap:.2%}. The historical layout occupied {before['area_m2']:.0f} square metres under the same geometry evaluator. These bounds concern the reduced area problem and do not establish optimality of the complete multiobjective design or superiority of one algorithm under a common objective.

中文修改提醒：正文应同时放入本步报告中的统一指标对照表，披露面积压缩的其他指标代价；补充材料保留完整模型、整数等价论证、求解参数与日志。不要只插入面积降低百分比。
'''
    path.write_text(original+insertion,encoding='utf-8')
    path = paper/'revision_status.md'
    original = path.read_text(encoding='utf-8')
    original = original.replace('| 2 精确方法参照 | 已明确无固定边界的数学问题 | 同34设施约简MIP/CP模型、可行上界/有效下界/求解间隙或实例特定证据 |',f'| 2 精确方法参照 | 同34设施面积约简CP模型；上界{ub:.0f}、下界{lb:.0f}平方米，间隙{gap:.2%}；独立复算通过 | 将约简范围、整数等价证明、实际状态与指标代价纳入正文和补充材料；未证明完整问题最优 |')
    path.write_text(original,encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
