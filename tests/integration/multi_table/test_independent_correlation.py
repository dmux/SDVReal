"""Recovery of known inter-table correlations by the ``IndependentSynthesizer``."""

import numpy as np
import pandas as pd
import pytest

from sdv.multi_table import IndependentSynthesizer
from sdv.multi_table._group_features import intraclass_correlation
from tests.utils import generate_correlated_parent_child

ALL_OPTIONS = {
    'model_cardinality': True,
    'context_columns': {'child': {'parent': ['size', 'tier']}},
    'sibling_order': {'child': {'by': 'kind', 'ascending': False}},
    'group_rules': {'child': {'column': 'kind', 'first': 'HQ', 'others': 'BRANCH'}},
    'sibling_correlation': True,
}


def _metrics(data):
    parent, child = data['parent'], data['child']
    counts = child['parent_id'].value_counts().reindex(parent['parent_id'], fill_value=0)
    joined = child.merge(parent, on='parent_id')
    with_parent = child.dropna(subset=['parent_id'])
    headquarters = (with_parent['kind'] == 'HQ').groupby(with_parent['parent_id']).sum()
    codes = with_parent['parent_id'].astype('category').cat.codes.to_numpy()
    tier_means = joined.groupby('tier')['discount'].mean()
    return {
        'size_children': pd.Series(parent['size'].to_numpy()).corr(
            pd.Series(counts.to_numpy()), method='spearman'
        ),
        'size_amount': joined['size'].corr(joined['amount'], method='spearman'),
        'gold_minus_bronze': tier_means['gold'] - tier_means['bronze'],
        'one_hq': (headquarters == 1).mean(),
        'icc_score': intraclass_correlation(with_parent[['score']].to_numpy(), codes)[0],
    }


@pytest.fixture(scope='module')
def real_and_synthetic():
    data, metadata = generate_correlated_parent_child(4000, seed=1)
    baseline = IndependentSynthesizer(metadata, verbose=False)
    baseline.fit(data)
    correlated = IndependentSynthesizer(metadata, verbose=False, **ALL_OPTIONS)
    correlated.fit(data)
    return (
        metadata,
        _metrics(data),
        baseline.sample('parent', 4000),
        correlated.sample('parent', 4000),
    )


def test_integrity(real_and_synthetic):
    """Test that the synthetic data with all the options is valid."""
    metadata, _, baseline, correlated = real_and_synthetic

    # Run and Assert
    metadata.validate_data(baseline)
    metadata.validate_data(correlated)


def test_recovers_inter_table_correlations(real_and_synthetic):
    """Test that the options recover the correlations that the default ignores."""
    _, real, baseline, correlated = real_and_synthetic

    # Run
    baseline_metrics = _metrics(baseline)
    correlated_metrics = _metrics(correlated)

    # Assert
    assert abs(baseline_metrics['size_children']) < 0.1
    assert correlated_metrics['size_children'] == pytest.approx(real['size_children'], abs=0.1)
    assert abs(baseline_metrics['size_amount']) < 0.1
    assert correlated_metrics['size_amount'] > 0.4
    assert abs(baseline_metrics['gold_minus_bronze']) < 0.3
    assert correlated_metrics['gold_minus_bronze'] > 0.5 * real['gold_minus_bronze']
    assert correlated_metrics['one_hq'] == 1.0
    assert correlated_metrics['icc_score'] == pytest.approx(real['icc_score'], abs=0.1)


def test_reproducible():
    """Test that sampling twice after a reset gives the same data."""
    # Setup
    data, metadata = generate_correlated_parent_child(300)
    instance = IndependentSynthesizer(metadata, verbose=False, **ALL_OPTIONS)
    instance.fit(data)

    # Run
    first = instance.sample('parent', 300)
    instance.reset_sampling()
    second = instance.sample('parent', 300)

    # Assert
    for table_name in first:
        pd.testing.assert_frame_equal(first[table_name], second[table_name])


def test_scale():
    """Test that scaling keeps the children-per-parent mean and the integrity."""
    # Setup
    data, metadata = generate_correlated_parent_child(1000)
    instance = IndependentSynthesizer(metadata, verbose=False, **ALL_OPTIONS)
    instance.fit(data)

    # Run
    synthetic = instance.sample('parent', 2500)

    # Assert
    metadata.validate_data(synthetic)
    real_mean = data['child']['parent_id'].notna().sum() / len(data['parent'])
    fake_mean = synthetic['child']['parent_id'].notna().sum() / len(synthetic['parent'])
    assert fake_mean == pytest.approx(real_mean, rel=0.05)
    assert np.isclose(synthetic['child']['parent_id'].isna().mean(), 0.05, atol=0.02)
