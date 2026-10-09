"""Utils for testing."""

import contextlib
from copy import deepcopy
from functools import lru_cache

import numpy as np
import pandas as pd
from rdt.transformers.utils import learn_rounding_digits

from sdv.datasets.demo import download_demo
from sdv.logging import get_sdv_logger
from sdv.metadata.metadata import Metadata
from sdv.multi_table import HMASynthesizer
from sdv.single_table import GaussianCopulaSynthesizer

RANGE_KEYS = {
    'range_is_nullable',
    'range_min',
    'range_max',
    'range_values',
    'decimal_places',
    'high_cardinality',
}


class DataFrameMatcher:
    """Match a given Pandas DataFrame in a mock function call."""

    def __init__(self, df):
        self.df = df

    def __eq__(self, other):
        pd.testing.assert_frame_equal(self.df, other)
        return True


class DataFrameDictMatcher:
    """Match a given dictionary of pandas DataFrames in a mock function call."""

    def __init__(self, data):
        self.data = data

    def __eq__(self, other):
        """Assert the data keys match, then use pandas to assert the values are equal."""
        assert self.data.keys() == other.keys()
        for key in self.data:
            pd.testing.assert_frame_equal(self.data[key], other[key])

        return True


class SeriesMatcher:
    """Match a given Pandas Series in a mock function call."""

    def __init__(self, series):
        self.series = series

    def __eq__(self, other):
        pd.testing.assert_series_equal(self.series, other)
        return True


def get_multi_table_metadata():
    """Return a multi-table ``Metadata`` object to be used with tests."""
    dict_metadata = {
        'tables': {
            'nesreca': {
                'primary_key': 'id_nesreca',
                'columns': {
                    'upravna_enota': {'sdtype': 'id'},
                    'id_nesreca': {'sdtype': 'id'},
                    'nesreca_val': {'sdtype': 'numerical'},
                },
            },
            'oseba': {
                'columns': {
                    'upravna_enota': {'sdtype': 'id'},
                    'id_nesreca': {'sdtype': 'id'},
                    'oseba_val': {'sdtype': 'numerical'},
                }
            },
            'upravna_enota': {
                'primary_key': 'id_upravna_enota',
                'columns': {
                    'id_upravna_enota': {'sdtype': 'id'},
                    'upravna_val': {'sdtype': 'numerical'},
                },
            },
        },
        'relationships': [
            {
                'parent_table_name': 'upravna_enota',
                'parent_primary_key': 'id_upravna_enota',
                'child_table_name': 'nesreca',
                'child_foreign_key': 'upravna_enota',
            },
            {
                'parent_table_name': 'upravna_enota',
                'parent_primary_key': 'id_upravna_enota',
                'child_table_name': 'oseba',
                'child_foreign_key': 'upravna_enota',
            },
            {
                'parent_table_name': 'nesreca',
                'parent_primary_key': 'id_nesreca',
                'child_table_name': 'oseba',
                'child_foreign_key': 'id_nesreca',
            },
        ],
        'METADATA_SPEC_VERSION': 'V1',
    }

    return Metadata.load_from_dict(dict_metadata)


def get_simplified_multi_table_metadata():
    """Return a simplified ``Metadata`` object to be used with HMA tests."""
    dict_metadata = {
        'tables': {
            'nesreca': {
                'primary_key': 'id_nesreca',
                'columns': {
                    'upravna_enota': {'sdtype': 'id'},
                    'id_nesreca': {'sdtype': 'id'},
                    'nesreca_val': {'sdtype': 'numerical'},
                },
            },
            'oseba': {
                'columns': {
                    'upravna_enota': {'sdtype': 'id'},
                    'id_nesreca': {'sdtype': 'id'},
                    'oseba_val': {'sdtype': 'numerical'},
                }
            },
            'upravna_enota': {
                'primary_key': 'id_upravna_enota',
                'columns': {
                    'id_upravna_enota': {'sdtype': 'id'},
                    'upravna_val': {'sdtype': 'numerical'},
                },
            },
        },
        'relationships': [
            {
                'parent_table_name': 'upravna_enota',
                'parent_primary_key': 'id_upravna_enota',
                'child_table_name': 'oseba',
                'child_foreign_key': 'upravna_enota',
            },
            {
                'parent_table_name': 'nesreca',
                'parent_primary_key': 'id_nesreca',
                'child_table_name': 'oseba',
                'child_foreign_key': 'id_nesreca',
            },
        ],
        'METADATA_SPEC_VERSION': 'V1',
    }

    return Metadata.load_from_dict(dict_metadata)


def get_multi_table_data():
    """Return a dictionary containing some data for multi table."""
    data = {
        'nesreca': pd.DataFrame({
            'id_nesreca': list(range(4)),
            'upravna_enota': list(range(4)),
            'nesreca_val': list(range(4)),
        }),
        'oseba': pd.DataFrame({
            'upravna_enota': list(range(4)),
            'id_nesreca': list(range(4)),
            'oseba_val': list(range(4)),
        }),
        'upravna_enota': pd.DataFrame({
            'id_upravna_enota': list(range(4)),
            'upravna_val': list(range(4)),
        }),
    }

    return data


@contextlib.contextmanager
def catch_sdv_logs(caplog, level, logger):
    """Context manager to capture logs from an SDV logger."""
    logger = get_sdv_logger(logger)
    orig_level = logger.level
    logger.setLevel(level)
    logger.addHandler(caplog.handler)
    try:
        yield
    finally:
        logger.setLevel(orig_level)
        logger.removeHandler(caplog.handler)


def run_constraint(constraint, data, metadata):
    """Run a constraint."""
    constraint.validate(data, metadata)
    updated_metadata = constraint.get_updated_metadata(metadata)
    constraint.fit(data, metadata)
    transformed = constraint.transform(data)
    reverse_transformed = constraint.reverse_transform(transformed)

    return updated_metadata, transformed, reverse_transformed


def run_copula(data, metadata, constraints=None):
    synthesizer = GaussianCopulaSynthesizer(metadata)
    if constraints:
        synthesizer.add_constraints(constraints=constraints)
    synthesizer.fit(data)

    return synthesizer


def run_hma(data, metadata, constraints=None):
    synthesizer = HMASynthesizer(metadata)
    if constraints:
        synthesizer.add_constraints(constraints=constraints)
    synthesizer.fit(data)

    return synthesizer


@lru_cache
def _download_demo(modality, dataset_name):
    return download_demo(modality, dataset_name)


def download_test_demo(modality, dataset_name):
    """Download demo datasets with caching.

    Args:
        modality:
            The modality of the dataset: 'single_table', 'multi_table', 'sequential'.
        dataset_name:
            Name of the dataset to download.
    """
    data, metadata = _download_demo(modality, dataset_name)
    return deepcopy(data), deepcopy(metadata)


def compare_metadata(metadata, expected_metadata):
    """Compare metadata, allowing detected range fields to be omitted from expected metadata."""
    actual = metadata.to_dict() if isinstance(metadata, Metadata) else deepcopy(metadata)
    expected = (
        expected_metadata.to_dict()
        if isinstance(expected_metadata, Metadata)
        else deepcopy(expected_metadata)
    )

    for table_name, table in actual['tables'].items():
        for column_name, column in table['columns'].items():
            expected_column = expected['tables'][table_name]['columns'][column_name]
            for key in RANGE_KEYS:
                if key not in expected_column:
                    column.pop(key, None)

    assert actual == expected


def compare_ranges(metadata, data):
    """Check that detected ranges are consistent with the source data."""
    metadata = metadata.to_dict() if isinstance(metadata, Metadata) else metadata
    for table_name, table in metadata['tables'].items():
        primary_key = table.get('primary_key')
        primary_keys = {primary_key} if isinstance(primary_key, str) else set(primary_key or [])
        for column_name, column in table['columns'].items():
            sdtype = column.get('sdtype')
            range_keys = set(column) & RANGE_KEYS
            if column_name in primary_keys or sdtype == 'unknown':
                assert not range_keys
                continue

            column_data = data[table_name][column_name]
            clean_data = column_data.dropna()

            if 'range_is_nullable' in column:
                assert column['range_is_nullable'] == column_data.isna().any()

            if 'range_values' in column:
                assert set(column['range_values']) == set(clean_data)

            if 'range_min' in column:
                if column['sdtype'] == 'datetime':
                    assert pd.to_datetime(column['range_min']) == pd.to_datetime(clean_data).min()
                    assert pd.to_datetime(column['range_max']) == pd.to_datetime(clean_data).max()
                else:
                    assert column['range_min'] == clean_data.min()
                    assert column['range_max'] == clean_data.max()

            if 'decimal_places' in column:
                assert column['decimal_places'] == learn_rounding_digits(column_data)


def generate_multi_table_schema(
    num_tables, depth, rows_per_table, num_columns=2, null_foreign_key_rate=0.0, seed=0
):
    """Generate random data and metadata for a tree shaped multi-table schema.

    The first ``depth`` tables form a chain (``table_0`` is the root) and the remaining tables
    are attached to random tables, without exceeding ``depth``. Every non-root table has a
    foreign key to its parent, assigned uniformly at random.

    Args:
        num_tables (int):
            Number of tables in the schema.
        depth (int):
            Number of tables in the longest relationship chain, as computed by
            ``Metadata._get_max_schema_depth``.
        rows_per_table (int):
            Number of rows of every table.
        num_columns (int):
            Number of numerical columns of every table. A categorical column is also added.
        null_foreign_key_rate (float):
            Fraction of foreign keys that are null in every child table.
        seed (int):
            Seed for the random generator.

    Returns:
        tuple[dict, Metadata]:
            The data and the metadata.
    """
    rng = np.random.default_rng(seed)
    table_depths = {'table_0': 1}
    parents = {}
    for index in range(1, num_tables):
        table_name = f'table_{index}'
        if index < depth:
            parent_name = f'table_{index - 1}'
        else:
            candidates = [name for name, level in table_depths.items() if level < depth]
            parent_name = candidates[rng.integers(len(candidates))]

        parents[table_name] = parent_name
        table_depths[table_name] = table_depths[parent_name] + 1

    data = {}
    tables = {}
    relationships = []
    for table_name in table_depths:
        columns = {'id': {'sdtype': 'id'}, 'category': {'sdtype': 'categorical'}}
        table_data = {
            'id': np.arange(rows_per_table),
            'category': rng.choice(['A', 'B', 'C', 'D'], rows_per_table),
        }
        for column_index in range(num_columns):
            columns[f'value_{column_index}'] = {'sdtype': 'numerical'}
            table_data[f'value_{column_index}'] = rng.normal(column_index, 1, rows_per_table)

        if table_name in parents:
            columns['parent_id'] = {'sdtype': 'id'}
            foreign_keys = rng.integers(0, rows_per_table, rows_per_table).astype(float)
            foreign_keys[rng.random(rows_per_table) < null_foreign_key_rate] = np.nan
            table_data['parent_id'] = foreign_keys
            relationships.append({
                'parent_table_name': parents[table_name],
                'child_table_name': table_name,
                'parent_primary_key': 'id',
                'child_foreign_key': 'parent_id',
            })

        data[table_name] = pd.DataFrame(table_data)
        tables[table_name] = {'primary_key': 'id', 'columns': columns}

    metadata = Metadata.load_from_dict({'tables': tables, 'relationships': relationships})
    return data, metadata


def generate_correlated_parent_child(num_parents=2000, seed=0):
    """Generate a parent/child dataset with known correlations between the tables.

    * ``parent.size`` (numerical, log-normal) drives ``parent.segment`` (categorical), the number
      of children of every parent and ``child.amount``.
    * ``parent.tier`` (categorical, independent of ``size``) shifts ``child.discount`` with a
      non-monotonic effect (``bronze < silver < gold`` with ``silver`` the most frequent).
    * The first child of every parent has ``kind == 'HQ'`` and the others ``'BRANCH'``.
    * Children of the same parent share a random effect in ``child.score`` and usually share
      ``child.region``.
    * About 5% of the children have a null foreign key.

    Args:
        num_parents (int):
            Number of rows of the parent table.
        seed (int):
            Seed for the random generator.

    Returns:
        tuple[dict, Metadata]:
            The data and the metadata.
    """
    rng = np.random.default_rng(seed)
    log_size = rng.normal(0, 1, num_parents)
    parent = pd.DataFrame({
        'parent_id': np.arange(num_parents),
        'size': np.round(np.exp(log_size) * 100, 2),
        'segment': np.where(log_size > 0.5, 'large', np.where(log_size > -0.5, 'mid', 'small')),
        'tier': rng.choice(['silver', 'bronze', 'gold'], num_parents, p=[0.5, 0.3, 0.2]),
    })
    tier_effect = parent['tier'].map({'bronze': -1.5, 'silver': 0.0, 'gold': 1.5}).to_numpy()
    counts = rng.poisson(np.exp(0.8 * log_size))
    parent_index = np.repeat(np.arange(num_parents), counts)
    num_children = len(parent_index)
    position = pd.Series(parent_index).groupby(parent_index).cumcount().to_numpy()
    group_effect = rng.normal(0, 1, num_parents)
    regions = np.array(['N', 'S', 'E', 'W', 'C'])
    parent_region = rng.integers(0, len(regions), num_parents)
    same_region = rng.random(num_children) < 0.8
    child_region = np.where(
        same_region, parent_region[parent_index], rng.integers(0, len(regions), num_children)
    )
    foreign_keys = parent_index.astype(float)
    foreign_keys[rng.random(num_children) < 0.05] = np.nan
    child = pd.DataFrame({
        'child_id': np.arange(num_children),
        'parent_id': foreign_keys,
        'amount': np.round(
            np.exp(0.7 * log_size[parent_index] + rng.normal(0, 0.7, num_children)) * 10, 2
        ),
        'score': np.round(
            0.8 * group_effect[parent_index] + 0.6 * rng.normal(0, 1, num_children), 3
        ),
        'discount': np.round(tier_effect[parent_index] + rng.normal(0, 1, num_children), 3),
        'region': regions[child_region],
        'kind': np.where(position == 0, 'HQ', 'BRANCH'),
    })
    # shuffle the children so the order of the rows carries no information
    child = child.sample(frac=1, random_state=seed).reset_index(drop=True)
    metadata = Metadata.load_from_dict({
        'tables': {
            'parent': {
                'primary_key': 'parent_id',
                'columns': {
                    'parent_id': {'sdtype': 'id'},
                    'size': {'sdtype': 'numerical'},
                    'segment': {'sdtype': 'categorical'},
                    'tier': {'sdtype': 'categorical'},
                },
            },
            'child': {
                'primary_key': 'child_id',
                'columns': {
                    'child_id': {'sdtype': 'id'},
                    'parent_id': {'sdtype': 'id'},
                    'amount': {'sdtype': 'numerical'},
                    'score': {'sdtype': 'numerical'},
                    'discount': {'sdtype': 'numerical'},
                    'region': {'sdtype': 'categorical'},
                    'kind': {'sdtype': 'categorical'},
                },
            },
        },
        'relationships': [
            {
                'parent_table_name': 'parent',
                'child_table_name': 'child',
                'parent_primary_key': 'parent_id',
                'child_foreign_key': 'parent_id',
            }
        ],
    })
    return {'parent': parent, 'child': child}, metadata
