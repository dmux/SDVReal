"""Scale benchmark of SDV multi-table synthesizers on the Brazilian CNPJ open data.

Every (synthesizer, variant, fraction) runs in its own subprocess. The parent samples the
child's resident memory every 0.25s, records the peak, and kills the child when it crosses the
memory limit or the timeout. The child reports the time of every phase and quality metrics as a
JSON line.

Usage:
    python benchmarks/cnpj/run.py --sample-dir ~/.cache/sdv-cnpj/2026-09/sample_0.05 \
        --synthesizer independent --variant full --fractions 0.001 0.005 0.01 0.025 0.05
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
import warnings

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT_CONTEXT = ['porte_empresa', 'natureza_juridica', 'capital_social']
CORRELATION_OPTIONS = {
    'T1': lambda variant: {'lookup_tables': DOMAIN_TABLES} if variant == 'full' else {},
    'T2': lambda variant: {'model_cardinality': True},
    'T2S': lambda variant: {
        'cardinality_by': {
            child: ['porte_empresa'] for child in ('estabelecimentos', 'socios', 'simples')
        }
    },
    'T3': lambda variant: {
        'context_columns': {
            child: {'empresas': PARENT_CONTEXT}
            for child in ('estabelecimentos', 'socios', 'simples')
        }
    },
    'T4': lambda variant: {
        'sibling_order': {
            'estabelecimentos': {'by': 'identificador_matriz_filial', 'ascending': True}
        }
    },
    'T4R': lambda variant: {
        'group_rules': {
            'estabelecimentos': {
                'column': 'identificador_matriz_filial',
                'first': '1',
                'others': '2',
            }
        }
    },
    'T5': lambda variant: {'sibling_correlation': True},
}
PLAN_ENCODING = {'categorical_context': 'processed', 'sibling_categorical': 'latent'}
DOMAIN_TABLES = ['cnaes', 'motivos', 'municipios', 'naturezas', 'paises', 'qualificacoes']
# (parent column, child table, child column) pairs whose association is compared
ASSOCIATION_PAIRS = [
    ('porte_empresa', 'simples', 'opcao_simples'),
    ('porte_empresa', 'simples', 'opcao_mei'),
    ('porte_empresa', 'estabelecimentos', 'situacao_cadastral'),
    ('natureza_juridica', 'estabelecimentos', 'situacao_cadastral'),
    ('natureza_juridica', 'socios', 'identificador_socio'),
    ('natureza_juridica', 'socios', 'qualificacao_socio'),
]


def correlation_kwargs(tokens, variant, encoding):
    """Build the ``IndependentSynthesizer`` arguments for the correlation ``tokens``."""
    kwargs = {}
    for token in tokens:
        kwargs.update(CORRELATION_OPTIONS[token](variant))

    if encoding == 'plan' and tokens:
        kwargs.update(PLAN_ENCODING)

    return kwargs


class _RUsageInfoV4(ctypes.Structure):
    _fields_ = [('uuid', ctypes.c_uint8 * 16), ('values', ctypes.c_uint64 * 40)]


def _peak_footprint(pid):
    """Return the lifetime peak physical footprint of ``pid`` in bytes.

    On macOS the resident set size excludes compressed pages, so it underestimates the memory
    a process needs. The physical footprint (``ri_lifetime_max_phys_footprint``) is what the
    Activity Monitor shows. On other platforms the resident set size is used.
    """
    if sys.platform != 'darwin':
        return psutil.Process(pid).memory_info().rss

    info = _RUsageInfoV4()
    libproc = ctypes.CDLL('/usr/lib/libproc.dylib')
    if libproc.proc_pid_rusage(pid, 4, ctypes.byref(info)) != 0:
        raise psutil.NoSuchProcess(pid)

    return info.values[28]


def _children_per_parent(parent_keys, child_keys):
    import pandas as pd

    return child_keys.value_counts().reindex(pd.Index(parent_keys), fill_value=0).to_numpy()


def _cramers_v(left, right):
    import numpy as np
    import pandas as pd

    table = pd.crosstab(pd.Series(left).fillna('NA'), pd.Series(right).fillna('NA')).to_numpy()
    total = table.sum()
    if total == 0 or min(table.shape) < 2:
        return float('nan')

    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / total
    chi2 = ((table - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi2 / total / (min(table.shape) - 1)))


def _contingency_similarity(real_pairs, synthetic_pairs):

    real = real_pairs.fillna('NA').value_counts(normalize=True)
    synthetic = synthetic_pairs.fillna('NA').value_counts(normalize=True)
    real, synthetic = real.align(synthetic, fill_value=0)
    return float(1 - (real - synthetic).abs().sum() / 2)


def _correlation_metrics(real, synthetic):
    """Inter-table correlation metrics of section 8 of the correlation plan."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'correlation'))
    import metrics as correlation_metrics

    results = {}
    joined = {}
    for name, data in (('real', real), ('synthetic', synthetic)):
        companies = data['empresas']
        for child in ('estabelecimentos', 'socios'):
            counts = _children_per_parent(companies['cnpj_basico'], data[child]['cnpj_basico'])
            results[f'spearman_capital_n_{child}_{name}'] = correlation_metrics.spearman(
                companies['capital_social'], counts
            )
            results[f'spearman_porte_n_{child}_{name}'] = correlation_metrics.spearman(
                correlation_metrics.ordinal(companies['porte_empresa']), counts
            )

        establishments = data['estabelecimentos']
        for column in ('cnae_fiscal_principal', 'uf', 'municipio', 'situacao_cadastral'):
            results[f'sibling_same_{column}_{name}'] = correlation_metrics.sibling_same_value(
                establishments['cnpj_basico'], establishments[column].astype(str)
            )
            results[f'sibling_same_per_empresa_{column}_{name}'] = (
                correlation_metrics.sibling_same_value_per_parent(
                    establishments['cnpj_basico'], establishments[column].astype(str)
                )
            )

        parent_columns = companies[['cnpj_basico', 'porte_empresa', 'natureza_juridica']]
        joined[name] = {
            child: data[child].merge(parent_columns, on='cnpj_basico', how='inner')
            for child in ('estabelecimentos', 'socios', 'simples')
        }

    real_pairs = set(
        zip(
            real['estabelecimentos']['uf'].astype(str),
            real['estabelecimentos']['municipio'].astype(str),
        )
    )
    synthetic_pairs = list(
        zip(
            synthetic['estabelecimentos']['uf'].astype(str),
            synthetic['estabelecimentos']['municipio'].astype(str),
        )
    )
    results['pct_uf_municipio_consistent_synthetic'] = sum(
        pair in real_pairs for pair in synthetic_pairs
    ) / max(len(synthetic_pairs), 1)
    for parent_column, child, child_column in ASSOCIATION_PAIRS:
        key = f'{parent_column}~{child}.{child_column}'
        for name in ('real', 'synthetic'):
            table = joined[name][child]
            results[f'cramers_v_{key}_{name}'] = _cramers_v(
                table[parent_column], table[child_column]
            )

        results[f'contingency_{key}'] = _contingency_similarity(
            joined['real'][child][[parent_column, child_column]].astype(str),
            joined['synthetic'][child][[parent_column, child_column]].astype(str),
        )

    return {name: round(float(value), 4) for name, value in results.items()}


def _sdmetrics_quality(real, synthetic, metadata, max_parents=2000):
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'correlation'))
    import metrics as correlation_metrics

    return correlation_metrics.quality_report(
        real, synthetic, metadata, max_parents=max_parents, root='empresas'
    )


def _quality(real, synthetic):
    """Cardinality KS tests and the one-headquarters-per-company rule."""
    from scipy.stats import ks_2samp

    metrics = {}
    for child in ('estabelecimentos', 'socios'):
        real_counts = _children_per_parent(
            real['empresas']['cnpj_basico'], real[child]['cnpj_basico']
        )
        synthetic_counts = _children_per_parent(
            synthetic['empresas']['cnpj_basico'], synthetic[child]['cnpj_basico']
        )
        result = ks_2samp(real_counts, synthetic_counts)
        metrics[f'ks_{child}_per_empresa'] = round(float(result.statistic), 4)
        metrics[f'mean_{child}_per_empresa_real'] = round(float(real_counts.mean()), 3)
        metrics[f'mean_{child}_per_empresa_synthetic'] = round(float(synthetic_counts.mean()), 3)
        metrics[f'max_{child}_per_empresa_real'] = int(real_counts.max())
        metrics[f'max_{child}_per_empresa_synthetic'] = int(synthetic_counts.max())

    for name, data in (('real', real), ('synthetic', synthetic)):
        establishments = data['estabelecimentos']
        is_headquarters = establishments['identificador_matriz_filial'].astype(str) == '1'
        headquarters = is_headquarters.groupby(establishments['cnpj_basico']).sum()
        metrics[f'pct_empresas_one_matriz_{name}'] = round(float((headquarters == 1).mean()), 4)

    return metrics


def worker(config):
    """Run one benchmark configuration and print the result as a JSON line."""
    sys.path.insert(0, HERE)
    warnings.simplefilter('ignore')
    import dataset

    from sdv.multi_table import HMASynthesizer, IndependentSynthesizer

    synthesizer_class = {'independent': IndependentSynthesizer, 'hma': HMASynthesizer}
    phases = {}
    peak_after = {}

    def timed(phase, function, *args, **kwargs):
        start = time.perf_counter()
        result = function(*args, **kwargs)
        phases[phase] = round(time.perf_counter() - start, 2)
        peak_after[phase] = round(_peak_footprint(os.getpid()) / 2**30, 2)
        print(  # noqa: T201
            f'PHASE {phase} {phases[phase]}s peak={peak_after[phase]}GB',
            file=sys.stderr,
            flush=True,
        )
        return result

    data, metadata, nulled = timed(
        'load', dataset.load, config['sample_dir'], config['fraction'], config['variant']
    )
    rows = {table: len(table_data) for table, table_data in data.items()}
    kwargs = {}
    if config['synthesizer'] == 'independent':
        kwargs = correlation_kwargs(
            config.get('correlation', []), config['variant'], config.get('encoding', 'improved')
        )

    synthesizer = timed(
        'init',
        synthesizer_class[config['synthesizer']],
        metadata,
        locales=['pt_BR'],
        verbose=False,
        **kwargs,
    )
    processed = timed('preprocess', synthesizer.preprocess, data)
    timed('fit', synthesizer.fit_processed_data, processed)
    del processed
    synthetic = timed('sample', synthesizer.sample, 'empresas', rows['empresas'])
    try:
        timed('validate', metadata.validate_data, synthetic)
        integrity = True
    except Exception as error:
        integrity = str(error)[:300]

    quality = timed('quality', _quality, data, synthetic)
    quality.update(timed('correlation', _correlation_metrics, data, synthetic))
    if config.get('sdmetrics'):
        quality['sdmetrics'] = timed('sdmetrics', _sdmetrics_quality, data, synthetic, metadata)
    print(  # noqa: T201
        'RESULT '
        + json.dumps({
            'rows': rows,
            'total_rows': sum(rows.values()),
            'phases': phases,
            'peak_memory_after_gb': peak_after,
            'integrity': integrity,
            'nulled_foreign_keys': nulled,
            **quality,
        }),
        flush=True,
    )


def run(config, memory_limit, timeout):
    """Run ``config`` in a subprocess, tracking its peak memory."""
    process = subprocess.Popen(
        [sys.executable, __file__, 'worker', json.dumps(config)],
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
    )
    peak_memory = 0
    status = 'ok'
    start = time.perf_counter()
    while process.poll() is None:
        try:
            peak_memory = max(peak_memory, _peak_footprint(process.pid))
        except psutil.NoSuchProcess:
            break

        available = psutil.virtual_memory().available
        if peak_memory > memory_limit or available < 512 * 2**20:
            status = f'killed: memory (footprint={peak_memory / 2**30:.1f} GB)'
            process.kill()
        elif time.perf_counter() - start > timeout:
            status = f'killed: timeout ({timeout}s)'
            process.kill()

        time.sleep(0.25)

    stdout = process.stdout.read()
    result = {**config, 'status': status, 'wall_seconds': round(time.perf_counter() - start, 1)}
    result['peak_memory_gb'] = round(peak_memory / 2**30, 2)
    for line in stdout.splitlines():
        if line.startswith('RESULT '):
            result.update(json.loads(line[len('RESULT ') :]))

    if status == 'ok' and 'phases' not in result:
        result['status'] = f'failed: exit code {process.returncode}'

    return result


def main():
    """Run the benchmark ladder, or a single configuration in worker mode."""
    if len(sys.argv) > 2 and sys.argv[1] == 'worker':
        worker(json.loads(sys.argv[2]))
        return

    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--sample-dir', required=True)
    parser.add_argument('--synthesizer', choices=['independent', 'hma'], default='independent')
    parser.add_argument('--variant', choices=['core', 'full'], default='full')
    parser.add_argument('--fractions', type=float, nargs='+', default=[0.001, 0.005, 0.01])
    parser.add_argument('--memory-limit-gb', type=float, default=None)
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--output', default=None)
    parser.add_argument(
        '--correlation',
        nargs='*',
        default=[],
        choices=list(CORRELATION_OPTIONS),
        help='Inter-table correlation techniques of the IndependentSynthesizer.',
    )
    parser.add_argument(
        '--encoding',
        choices=['improved', 'plan'],
        default='improved',
        help="'plan' uses the encodings of the original plan for categorical context/siblings.",
    )
    parser.add_argument('--sdmetrics', action='store_true', help='Run the sdmetrics report.')
    parser.add_argument('--label', default=None)
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()

    sample_dir = os.path.expanduser(args.sample_dir)
    output = args.output or os.path.join(sample_dir, 'results.jsonl')
    total = psutil.virtual_memory().total
    memory_limit = (args.memory_limit_gb or total / 2**30 * 0.8) * 2**30
    print(  # noqa: T201
        f'RAM {total / 2**30:.0f} GB, limit {memory_limit / 2**30:.1f} GB, '
        f'available {psutil.virtual_memory().available / 2**30:.1f} GB',
        flush=True,
    )
    for fraction in [fraction for fraction in args.fractions for _ in range(args.repeat)]:
        config = {
            'sample_dir': sample_dir,
            'synthesizer': args.synthesizer,
            'variant': args.variant,
            'fraction': fraction,
            'correlation': args.correlation,
            'encoding': args.encoding,
            'sdmetrics': args.sdmetrics,
            'label': args.label or '+'.join(args.correlation) or 'baseline',
        }
        result = run(config, memory_limit, args.timeout)
        with open(output, 'a') as file:
            file.write(json.dumps(result) + '\n')

        print(  # noqa: T201
            f'{config["label"]:16s} {args.synthesizer:11s} {args.variant:4s} {fraction:<7} '
            f'rows={result.get("total_rows", "-"):>10} {result["status"]:<32} '
            f'wall={result["wall_seconds"]}s peak={result["peak_memory_gb"]}GB '
            f'phases={result.get("phases")}',
            flush=True,
        )
        if result['status'] != 'ok':
            break


if __name__ == '__main__':
    main()
