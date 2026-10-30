"""Aggregate episode records, including paired initial-condition comparisons."""
import csv
from collections import defaultdict
import math


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
