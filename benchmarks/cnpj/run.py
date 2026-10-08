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
    synthesizer = timed(
        'init',
        synthesizer_class[config['synthesizer']],
        metadata,
        locales=['pt_BR'],
        verbose=False,
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
    for fraction in args.fractions:
        config = {
            'sample_dir': sample_dir,
            'synthesizer': args.synthesizer,
            'variant': args.variant,
            'fraction': fraction,
        }
        result = run(config, memory_limit, args.timeout)
        with open(output, 'a') as file:
            file.write(json.dumps(result) + '\n')

        print(  # noqa: T201
            f'{args.synthesizer:11s} {args.variant:4s} {fraction:<7} '
            f'rows={result.get("total_rows", "-"):>10} {result["status"]:<32} '
            f'wall={result["wall_seconds"]}s peak={result["peak_memory_gb"]}GB '
            f'phases={result.get("phases")}',
            flush=True,
        )
        if result['status'] != 'ok':
            break


if __name__ == '__main__':
    main()
