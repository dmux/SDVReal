import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from sdv.multi_table import IndependentSynthesizer
from sdv.utils import load_synthesizer
from tests.utils import generate_multi_table_schema


def _children_per_parent(data, metadata):
    """Return the number of children per parent for every relationship."""
    counts = {}
    for relationship in metadata.relationships:
        parent = data[relationship['parent_table_name']][relationship['parent_primary_key']]
        child = data[relationship['child_table_name']][relationship['child_foreign_key']]
        key = (relationship['child_table_name'], relationship['child_foreign_key'])
        counts[key] = child.value_counts().reindex(parent, fill_value=0).to_numpy()

    return counts


def test_fake_hotels(fake_hotels):
    """Test the end to end workflow on the ``fake_hotels`` demo dataset."""
    # Setup
    data, metadata = fake_hotels
    synthesizer = IndependentSynthesizer(metadata, verbose=False)

    # Run
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample('hotels', len(data['hotels']))

    # Assert
    metadata.validate_data(synthetic_data)
    for table_name, table_data in data.items():
        assert len(synthetic_data[table_name]) == len(table_data)
        assert list(synthetic_data[table_name].columns) == list(table_data.columns)


def test_multiple_foreign_keys(data_metadata_multiple_foreign_keys):
    """Test a child table with foreign keys to two different parents."""
    # Setup
    data, metadata = data_metadata_multiple_foreign_keys
    synthesizer = IndependentSynthesizer(metadata, verbose=False)

    # Run
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample('parent', 10)

    # Assert
    metadata.validate_data(synthetic_data)
    child = synthetic_data['child']
    assert child['parent_1_id'].is_unique
    assert child['parent_1_id'].isin(synthetic_data['parent']['parent_id']).all()
    assert child['parent_2_id'].isin(synthetic_data['second_parent']['parent_id']).all()


def test_schema_rejected_by_hma():
    """Test a schema with 8 tables and depth 4, which ``HMASynthesizer`` rejects."""
    # Setup
    data, metadata = generate_multi_table_schema(
        num_tables=8, depth=4, rows_per_table=1000, null_foreign_key_rate=0.1
    )
    synthesizer = IndependentSynthesizer(metadata, verbose=False)

    # Run
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample('table_0', 1500)

    # Assert
    # HMASynthesizer._validate_schema_complexity rejects > 5 tables or depth > 2
    assert len(metadata.tables) > 5
    assert metadata._get_max_schema_depth() > 2
    metadata.validate_data(synthetic_data)
    assert all(len(table) == 1500 for table in synthetic_data.values())
    for table_name in list(metadata.tables)[1:]:
        null_rate = synthetic_data[table_name]['parent_id'].isna().mean()
        assert null_rate == pytest.approx(data[table_name]['parent_id'].isna().mean(), abs=0.01)


def test_children_per_parent_distribution():
    """Test that the children-per-parent distribution matches the real data."""
    # Setup
    data, metadata = generate_multi_table_schema(num_tables=4, depth=3, rows_per_table=3000)
    rng = np.random.default_rng(0)
    skewed_parents = rng.zipf(1.5, 3000) % 3000
    data['table_1']['parent_id'] = skewed_parents.astype(float)
    synthesizer = IndependentSynthesizer(metadata, verbose=False)

    # Run
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample('table_0', 3000)

    # Assert
    real_counts = _children_per_parent(data, metadata)
    synthetic_counts = _children_per_parent(synthetic_data, metadata)
    for key, counts in real_counts.items():
        assert ks_2samp(counts, synthetic_counts[key]).pvalue > 0.01, key


def test_scale():
    """Test that ``scale`` is applied to every table."""
    # Setup
    data, metadata = generate_multi_table_schema(num_tables=6, depth=3, rows_per_table=200)
    synthesizer = IndependentSynthesizer(metadata, verbose=False)
    synthesizer.fit(data)

    # Run
    synthetic_data = synthesizer.sample('table_0', 500)

    # Assert
    metadata.validate_data(synthetic_data)
    assert {len(table) for table in synthetic_data.values()} == {500}


def test_save_and_load(tmp_path):
    """Test that a saved synthesizer samples the same data after being loaded."""
    # Setup
    data, metadata = generate_multi_table_schema(num_tables=6, depth=3, rows_per_table=200)
    synthesizer = IndependentSynthesizer(metadata, verbose=False)
    synthesizer.fit(data)
    filepath = tmp_path / 'synthesizer.pkl'

    # Run
    synthesizer.save(filepath)
    loaded = load_synthesizer(filepath)
    synthesizer.reset_sampling()
    expected = synthesizer.sample('table_0', 200)
    result = loaded.sample('table_0', 200)

    # Assert
    for table_name, table_data in expected.items():
        pd.testing.assert_frame_equal(result[table_name], table_data)
