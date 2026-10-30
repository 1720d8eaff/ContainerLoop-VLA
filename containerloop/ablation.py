"""Unified ablation harness: inject the actual simulator or robot evaluator."""
import argparse
import importlib
import json
from pathlib import Path

CONFIGS = {
    'direct': {'smoothing': False, 'geometry': False, 'feedback': False},
    'smoothing': {'smoothing': True, 'geometry': False, 'feedback': False},
    'geometry': {'smoothing': False, 'geometry': True, 'feedback': False},
    'feedback': {'smoothing': False, 'geometry': False, 'feedback': True},
    'full': {'smoothing': True, 'geometry': True, 'feedback': True},
}


def run(evaluator, conditions, output):
    """Evaluator receives (condition, flags), returns real episode metrics."""
    with Path(output).open('w', encoding='utf-8') as stream:
        for condition in conditions:
            for method, config in CONFIGS.items():
                result = dict(evaluator(condition, dict(config)))
                result.update(condition_id=condition, method=method)
                stream.write(json.dumps(result, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evaluator', required=True, help='module:function')
    p.add_argument('--conditions', required=True, help='JSON array of condition IDs')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    module, name = a.evaluator.split(':', 1)
    run(getattr(importlib.import_module(module), name), json.loads(Path(a.conditions).read_text(encoding='utf-8')), a.output)


if __name__ == '__main__':
    main()
