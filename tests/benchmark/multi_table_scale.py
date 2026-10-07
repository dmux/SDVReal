"""Scale benchmark for the ``IndependentSynthesizer``.

Run with ``invoke scale`` or ``python -m pytest tests/benchmark/multi_table_scale.py -m scale -s``.
"""

import time
import tracemalloc
import warnings

import pandas as pd
import pytest

from sdv.errors import SynthesizerInputError
from sdv.multi_table import HMASynthesizer, IndependentSynthesizer
from tests.utils import generate_multi_table_schema

pytestmark = pytest.mark.scale

# Linear growth means a 10x larger input takes ~10x longer. The margin absorbs noise and
# the fixed costs that make small inputs relatively slower, not quadratic growth (100x).
MAX_RATIO_PER_10X = 15


def _run(num_tables, depth, rows_per_table):
    """Fit and sample a generated schema, returning the elapsed time and the peak memory."""
    data, metadata = generate_multi_table_schema(
        num_tables, depth, rows_per_table, null_foreign_key_rate=0.05
    )
    tracemalloc.start()
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        synthesizer = IndependentSynthesizer(metadata, verbose=False)
        synthesizer.fit(data)
        synthetic_data = synthesizer.sample('table_0', rows_per_table)

    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    metadata.validate_data(synthetic_data)
    assert all(len(table) == rows_per_table for table in synthetic_data.values())
    return {
        'tables': num_tables,
        'depth': depth,
        'total_rows': num_tables * rows_per_table,
        'seconds': round(elapsed, 2),
        'peak_mb': round(peak / 2**20, 1),
    }


def _print(title, results):
    print(f'\n{title}\n{pd.DataFrame(results).to_string(index=False)}')  # noqa: T201


def test_scale_with_rows():
    """Fit+sample time grows linearly with the number of rows (10k -> 100k -> 1M)."""
    results = [_run(10, 4, rows_per_table) for rows_per_table in (1_000, 10_000, 100_000)]
    _print('Scaling with rows (10 tables, depth 4)', results)

    for smaller, larger in zip(results, results[1:]):
        assert larger['seconds'] / smaller['seconds'] < MAX_RATIO_PER_10X


def test_scale_with_tables():
    """Fit+sample time grows linearly with the number of tables (10 -> 25 -> 50)."""
    results = [_run(num_tables, 4, 2_000) for num_tables in (10, 25, 50)]
    _print('Scaling with tables (2k rows per table, depth 4)', results)

    for smaller, larger in zip(results, results[1:]):
        growth = larger['tables'] / smaller['tables']
        assert larger['seconds'] / smaller['seconds'] < growth * MAX_RATIO_PER_10X / 10


def test_scale_with_depth():
    """Memory does not explode with depth, unlike HMA's column expansion."""
    results = [_run(depth, depth, 20_000) for depth in (2, 4, 8)]
    _print('Scaling with depth (chain of tables, 20k rows per table)', results)

    per_table = [result['peak_mb'] / result['tables'] for result in results]
    assert max(per_table) / min(per_table) < 2


def test_hma_rejects_benchmark_schemas():
    """The schemas used in this benchmark are rejected by ``HMASynthesizer``."""
    for num_tables, depth in ((10, 4), (50, 4), (8, 8)):
        _, metadata = generate_multi_table_schema(num_tables, depth, 10)
        with pytest.raises(SynthesizerInputError):
            HMASynthesizer(metadata)
