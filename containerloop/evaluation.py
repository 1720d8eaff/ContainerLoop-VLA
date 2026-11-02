"""Aggregate episode records, including paired initial-condition comparisons."""
import csv
from collections import defaultdict
import math
import json


def read_results(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    required = {'condition_id', 'method', 'success', 'drift_m'}
    if rows and not required.issubset(rows[0]):
        raise ValueError('CSV requires condition_id, method, success, drift_m')
    result = []
    seen = set()
    for row in rows:
        key = row['condition_id'], row['method']
        if not all(key) or key in seen or row['success'] not in ('0', '1'):
            raise ValueError('invalid or duplicate episode key/success')
        drift = float(row['drift_m'])
        if not math.isfinite(drift) or drift < 0:
            raise ValueError('drift must be finite and nonnegative')
        seen.add(key)
        result.append({'condition_id': key[0], 'method': key[1], 'success': int(row['success']), 'drift_m': drift})
    return result


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['method']].append(row)
    return {method: {'episodes': len(group), 'successes': sum(r['success'] for r in group), 'success_rate': sum(r['success'] for r in group) / len(group), 'mean_drift_m': sum(r['drift_m'] for r in group) / len(group)} for method, group in grouped.items()}


def paired(rows, method_a, method_b):
    a = {r['condition_id']: r for r in rows if r['method'] == method_a}
    b = {r['condition_id']: r for r in rows if r['method'] == method_b}
    ids = sorted(a.keys() & b.keys())
    if not ids:
        raise ValueError('no matched conditions')
    return {'matched_conditions': len(ids), 'success_a': sum(a[i]['success'] for i in ids), 'success_b': sum(b[i]['success'] for i in ids), 'mean_drift_difference_b_minus_a_m': sum(b[i]['drift_m'] - a[i]['drift_m'] for i in ids) / len(ids)}


def summarize_jsonl(path):
    """Summarize measured episode records, keeping absent measurements null."""
    with open(path, encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['method']].append(row)
    def mean(rows, name):
        values = [r[name] for r in rows if r.get(name) is not None]
        if any(not math.isfinite(v) for v in values):
            raise ValueError('nonfinite metric')
        return sum(values) / len(values) if values else None
    return {method: {'episodes': len(group), 'success_rate': mean(group, 'success'),
                     'grasp_drift_m': mean(group, 'grasp_drift_m'),
                     'constraint_violations': sum(r.get('constraint_violations', 0) for r in group),
                     'execution_time_s': mean(group, 'execution_time_s'),
                     'failure_reasons': dict(__import__('collections').Counter(r['failure_reason'] for r in group if r.get('failure_reason')))}
            for method, group in grouped.items()}
SUCCESS_PREDICATE = {'max_height_gt_m': .14, 'drift_lt_m': .045, 'final_z_gt_m': .025, 'final_z_lt_m': .07}
SUCCESS_DEFINITION = 'grasp_and_place_within_0.045m_xy_and_final_z_in_0.025_0.07m'
PAIR_PROTOCOL = 'paired_initial_conditions_matched_to_simulation_conditions'


def read_episode_jsonl(path):
    """Read per-episode records and verify every success flag against the predicate.

    A record whose flag disagrees with the predicate is a data error rather than
    a rounding difference, so this raises instead of averaging over it.
    """
    rows, seen = [], set()
    with open(path, encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            condition = record.get('condition') or {}
            cid, method = condition.get('id', record.get('condition_id')), record.get('method')
            if not cid or not method:
                raise ValueError('record requires condition.id and method')
            key = (cid, method)
            if key in seen:
                raise ValueError(f'duplicate episode key {key}')
            seen.add(key)
            drift = float(record['drift_m'])
            z = float(record['final_position_m'][2])
            expected = (float(record['max_height_m']) > SUCCESS_PREDICATE['max_height_gt_m']
                        and drift < SUCCESS_PREDICATE['drift_lt_m']
                        and SUCCESS_PREDICATE['final_z_gt_m'] < z < SUCCESS_PREDICATE['final_z_lt_m'])
            if expected != bool(record['success']):
                raise ValueError(f"{record.get('episode_id')}: success flag disagrees with the predicate")
            grasp = record.get('grasp_drift_m')
            rows.append({'condition_id': cid, 'method': method,
                         'success': int(bool(record['success'])), 'drift_m': drift,
                         'grasp_drift_m': None if grasp is None else float(grasp)})
    if not rows:
        raise ValueError('no episode records')
    return rows


def summarize_episodes(rows):
    """Per-method counts, drift, and the paired comparison of the first two methods."""
    order, grouped = [], defaultdict(list)
    for row in rows:
        if row['method'] not in grouped:
            order.append(row['method'])
        grouped[row['method']].append(row)
    results = {}
    for method in order:
        group = grouped[method]
        measured = [r['grasp_drift_m'] for r in group if r['grasp_drift_m'] is not None]
        successes = sum(r['success'] for r in group)
        results[method] = {'episodes': len(group), 'successes': successes,
                           'success_rate': successes / len(group),
                           'mean_grasp_drift_m_all_attempts': sum(measured) / len(measured) if measured else None,
                           'mean_placement_xy_error_m_all_attempts': sum(r['drift_m'] for r in group) / len(group)}
    summary = {'results': results}
    if len(order) >= 2:
        a = {r['condition_id']: r for r in grouped[order[0]]}
        b = {r['condition_id']: r for r in grouped[order[1]]}
        ids = sorted(a.keys() & b.keys())
        if not ids:
            raise ValueError('no matched conditions')
        summary['paired'] = {'methods': [order[0], order[1]], 'matched_conditions': len(ids),
                             'both_success': sum(1 for i in ids if a[i]['success'] and b[i]['success']),
                             'both_failure': sum(1 for i in ids if not a[i]['success'] and not b[i]['success']),
                             'a_only_success': sum(1 for i in ids if a[i]['success'] and not b[i]['success']),
                             'b_only_success': sum(1 for i in ids if not a[i]['success'] and b[i]['success'])}
    return summary


def build_report(episodes_path):
    """Rebuild the committed evaluation summary from the per-episode records."""
    rows = read_episode_jsonl(episodes_path)
    report = {'notice': '本文件由 containerloop.evaluation 从逐条记录重算，不手工编辑；'
                        '仿真部分见 experiments/g1d_pouring_adaptive/main_run/summary.json。',
              'source': str(episodes_path), 'data_source': 'real_robot',
              'protocol': PAIR_PROTOCOL, 'success_definition': SUCCESS_DEFINITION}
    report.update(summarize_episodes(rows))
    return report


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Rebuild the evaluation summary from per-episode JSONL records.')
    parser.add_argument('episodes', nargs='?', default='artifacts/contact/real_robot_episodes.jsonl')
    parser.add_argument('-o', '--output', default=None, help='default: real_robot_evaluation.json next to the input')
    args = parser.parse_args()
    text = json.dumps(build_report(args.episodes), ensure_ascii=False, indent=2) + '\n'
    out = Path(args.output) if args.output else Path(args.episodes).with_name('real_robot_evaluation.json')
    out.write_text(text, encoding='utf-8')
    print(text)
