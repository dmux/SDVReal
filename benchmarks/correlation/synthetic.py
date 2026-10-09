"""Inter-table correlation experiment on data with known correlations.

Fits the ``IndependentSynthesizer`` with every combination of the correlation options (each
technique isolated and accumulated) on ``tests.utils.generate_correlated_parent_child`` and
compares the inter-table metrics of the synthetic data with the real data.

Usage:
    python benchmarks/correlation/synthetic.py --num-parents 20000 --seeds 0 1 2 \
        --output benchmarks/correlation/results_synthetic.jsonl
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from scipy.stats import ks_2samp  # noqa: E402
from tests.utils import generate_correlated_parent_child  # noqa: E402

import metrics  # noqa: E402
from sdv.multi_table import IndependentSynthesizer  # noqa: E402

T2 = {'model_cardinality': True}
T3 = {'context_columns': {'child': {'parent': ['size', 'segment', 'tier']}}}
T3_PROCESSED = {**T3, 'categorical_context': 'processed'}
T5_LATENT = {'sibling_correlation': True, 'sibling_categorical': 'latent'}
T4 = {'sibling_order': {'child': {'by': 'kind', 'ascending': False}}}
T4_RULES = {'group_rules': {'child': {'column': 'kind', 'first': 'HQ', 'others': 'BRANCH'}}}
T5 = {'sibling_correlation': True}
CONFIGS = {
    'baseline': {},
    'T2': T2,
    'T3': T3,
    'T3(processed)': T3_PROCESSED,
    'T4': T4,
    'T4+rules': {**T4, **T4_RULES},
    'T5': T5,
    'T5(latent)': T5_LATENT,
    'T2+T3': {**T2, **T3},
    'T2+T3+T4': {**T2, **T3, **T4},
    'T2+T3+T4+rules': {**T2, **T3, **T4, **T4_RULES},
    'T2..T5': {**T2, **T3, **T4, **T4_RULES, **T5},
    'T2..T5(plan)': {**T2, **T3_PROCESSED, **T4, **T4_RULES, **T5_LATENT},
}


def evaluate(data, metadata):
    """Compute the inter-table metrics of a parent/child dataset."""
    parent, child = data['parent'], data['child']
    counts = metrics.children_per_parent(parent['parent_id'], child['parent_id'])
    joined = child.merge(parent, on='parent_id')
    return {
        'spearman_size_children': metrics.spearman(parent['size'], counts),
        'spearman_size_amount': metrics.spearman(joined['size'], joined['amount']),
        'spearman_segment_amount': metrics.spearman(
            joined['segment'].map({'small': 0, 'mid': 1, 'large': 2}), joined['amount']
        ),
        'eta2_tier_discount': metrics.correlation_ratio(joined['tier'], joined['discount']),
        'pct_one_hq': metrics.pct_one_first(child['parent_id'], child['kind'] == 'HQ'),
        'icc_score': metrics.sibling_icc(child['parent_id'], child['score']),
        'sibling_same_region': metrics.sibling_same_value(child['parent_id'], child['region']),
        'counts': counts,
    }


def run(config_name, data, metadata, real_metrics):
    """Fit and sample one configuration and compare its metrics with the real ones."""
    synthesizer = IndependentSynthesizer(metadata, verbose=False, **CONFIGS[config_name])
    start = time.perf_counter()
    synthesizer.fit(data)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    synthetic = synthesizer.sample('parent', len(data['parent']))
    sample_seconds = time.perf_counter() - start
    try:
        metadata.validate_data(synthetic)
        integrity = True
    except Exception as error:
        integrity = str(error)[:200]

    synthetic_metrics = evaluate(synthetic, metadata)
    result = {
        'config': config_name,
        'fit_seconds': round(fit_seconds, 3),
        'sample_seconds': round(sample_seconds, 3),
        'integrity': integrity,
        'ks_children': round(
            float(ks_2samp(real_metrics['counts'], synthetic_metrics['counts']).statistic), 4
        ),
        'rows_child': len(synthetic['child']),
    }
    for name, value in synthetic_metrics.items():
        if name != 'counts':
            result[name] = round(value, 4)

    result['quality'] = metrics.quality_report(data, synthetic, metadata)
    return result


def main():
    """Run every configuration for every seed and append the results as JSON lines."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--num-parents', type=int, default=20_000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--configs', nargs='+', default=list(CONFIGS))
    parser.add_argument('--output', default=os.path.join(HERE, 'results_synthetic.jsonl'))
    args = parser.parse_args()
    warnings.simplefilter('ignore')
    for seed in args.seeds:
        data, metadata = generate_correlated_parent_child(args.num_parents, seed=seed)
        real_metrics = evaluate(data, metadata)
        real = {name: round(value, 4) for name, value in real_metrics.items() if name != 'counts'}
        real.update({'config': 'real', 'seed': seed, 'num_parents': args.num_parents})
        real['rows_child'] = len(data['child'])
        with open(args.output, 'a') as file:
            file.write(json.dumps(real) + '\n')

        print(json.dumps(real), flush=True)  # noqa: T201
        for config_name in args.configs:
            np.random.seed(seed)
            result = run(config_name, data, metadata, real_metrics)
            result.update({'seed': seed, 'num_parents': args.num_parents})
            with open(args.output, 'a') as file:
                file.write(json.dumps(result) + '\n')

            print(json.dumps(result), flush=True)  # noqa: T201


if __name__ == '__main__':
    main()
