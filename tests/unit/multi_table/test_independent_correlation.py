import numpy as np
import pandas as pd
import pytest
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import GaussianUnivariate
from scipy.stats import ks_2samp

from sdv.errors import SynthesizerInputError
from sdv.metadata.metadata import Metadata
from sdv.multi_table._conditional import ConditionalCopulaSampler
from sdv.multi_table._group_features import (
    apply_group_rule,
    intraclass_correlation,
    positions_from_assignment,
    sibling_positions,
)
from sdv.multi_table.independent import IndependentSynthesizer, detect_lookup_tables
from tests.utils import generate_correlated_parent_child


def _get_lookup_metadata():
    return Metadata.load_from_dict({
        'tables': {
            'cities': {
                'primary_key': 'code',
                'columns': {'code': {'sdtype': 'id'}, 'name': {'sdtype': 'unknown'}},
            },
            'people': {
                'primary_key': 'id',
                'columns': {
                    'id': {'sdtype': 'id'},
                    'city': {'sdtype': 'id'},
                    'state': {'sdtype': 'categorical'},
                },
            },
        },
        'relationships': [
            {
                'parent_table_name': 'cities',
                'child_table_name': 'people',
                'parent_primary_key': 'code',
                'child_foreign_key': 'city',
            }
        ],
    })


def _get_lookup_data(num_rows=200):
    rng = np.random.default_rng(0)
    cities = pd.DataFrame({'code': ['A1', 'A2', 'B1', 'B2'], 'name': ['a', 'b', 'c', 'd']})
    city = rng.choice(cities['code'], num_rows)
    people = pd.DataFrame({'id': np.arange(num_rows), 'city': city, 'state': [c[0] for c in city]})
    return {'cities': cities, 'people': people}


def _fit_gaussian_copula(data):
    model = GaussianMultivariate(distribution=GaussianUnivariate)
    model.fit(data)
    return model


class TestGroupFeatures:
    def test_sibling_positions(self):
        """Test that positions are computed per group, sorted and with nulls at 0."""
        # Setup
        foreign_keys = np.array([1, 1, 2, np.nan, 1, 2])
        order = np.array(['2', '1', '1', '1', '2', '2'])

        # Run
        unordered = sibling_positions(foreign_keys)
        ordered = sibling_positions(foreign_keys, order)

        # Assert
        np.testing.assert_array_equal(unordered, [0, 1, 0, 0, 2, 1])
        np.testing.assert_array_equal(ordered, [1, 0, 0, 0, 2, 1])

    def test_positions_from_assignment(self):
        """Test the positions of sorted parent assignments, with ``-1`` for no parent."""
        # Run
        positions = positions_from_assignment(np.array([0, 0, 0, 2, 3, 3, -1, -1]))

        # Assert
        np.testing.assert_array_equal(positions, [0, 1, 2, 0, 0, 1, 0, 0])

    def test_apply_group_rule(self):
        """Test that the first child gets ``first`` and the others ``others``."""
        # Setup
        table = pd.DataFrame({'kind': ['2', '2', '1', '1', '1']})
        assignment = np.array([0, 0, 1, 1, -1])
        positions = positions_from_assignment(assignment)

        # Run
        apply_group_rule(
            table, positions, assignment, {'column': 'kind', 'first': '1', 'others': '2'}
        )

        # Assert
        assert table['kind'].tolist() == ['1', '2', '1', '2', '1']

    def test_intraclass_correlation(self):
        """Test that a known intraclass correlation is recovered."""
        # Setup
        rng = np.random.default_rng(0)
        groups = np.repeat(np.arange(3000), 4)
        effect = rng.normal(0, 1, 3000)[groups]
        values = np.column_stack([
            np.sqrt(0.6) * effect + np.sqrt(0.4) * rng.normal(0, 1, len(groups)),
            rng.normal(0, 1, len(groups)),
        ])

        # Run
        rho = intraclass_correlation(values, groups)

        # Assert
        assert rho[0] == pytest.approx(0.6, abs=0.05)
        assert rho[1] == pytest.approx(0.0, abs=0.05)

    def test_intraclass_correlation_single_row_groups(self):
        """Test that groups of one row carry no information."""
        # Run
        rho = intraclass_correlation(np.arange(10.0)[:, None], np.arange(10))

        # Assert
        np.testing.assert_array_equal(rho, [0.0])


class TestConditionalCopulaSampler:
    def _get_model(self, num_rows=20_000):
        rng = np.random.default_rng(0)
        context = rng.normal(0, 1, num_rows)
        data = pd.DataFrame({
            'a': 0.8 * context + 0.6 * rng.normal(0, 1, num_rows),
            'b': rng.normal(0, 1, num_rows),
            'ctx': context,
        })
        return _fit_gaussian_copula(data), data

    def test_matches_copulas_conditional_distribution(self):
        """Test the equivalence with ``copulas`` for a single condition."""
        # Setup
        model, _ = self._get_model()
        sampler = ConditionalCopulaSampler(model, ['ctx'], ridge=0)
        condition = pd.Series({'ctx': 1.3})

        # Run
        means, covariance, columns = model._get_conditional_distribution(condition)

        # Assert
        assert list(columns) == sampler.own_columns
        np.testing.assert_allclose(sampler.coefficients[:, 0] * 1.3, means)
        np.testing.assert_allclose(sampler.cholesky @ sampler.cholesky.T, covariance, atol=1e-8)

    def test_sample_conditional_mean(self):
        """Test that every row follows the conditional mean of its own context."""
        # Setup
        model, data = self._get_model()
        sampler = ConditionalCopulaSampler(model, ['ctx'])
        context = data[['ctx']]
        np.random.seed(0)

        # Run
        sampled = sampler.sample(context)

        # Assert
        slope = np.polyfit(context['ctx'], sampled['a'], 1)[0]
        assert slope == pytest.approx(0.8 * data['a'].std(), abs=0.05)
        assert abs(np.corrcoef(context['ctx'], sampled['b'])[0, 1]) < 0.03
        np.testing.assert_array_equal(sampled['ctx'], context['ctx'])
        assert list(sampled.columns) == ['a', 'b', 'ctx']

    def test_sample_is_reproducible(self):
        """Test that the same numpy seed gives the same rows."""
        # Setup
        model, data = self._get_model(1000)
        sampler = ConditionalCopulaSampler(model, ['ctx'])

        # Run
        np.random.seed(1)
        first = sampler.sample(data[['ctx']])
        np.random.seed(1)
        second = sampler.sample(data[['ctx']])

        # Assert
        pd.testing.assert_frame_equal(first, second)

    def test_sample_sibling_effect(self):
        """Test that the random effect makes rows of the same group correlated."""
        # Setup
        model, _ = self._get_model(5000)
        rho = pd.Series({'a': 0.0, 'b': 0.7})
        sampler = ConditionalCopulaSampler(model, [], sibling_rho=rho)
        groups = np.repeat(np.arange(5000), 3)
        np.random.seed(0)

        # Run
        sampled = sampler.sample(pd.DataFrame(index=range(len(groups))), groups)

        # Assert
        estimated = intraclass_correlation(sampled[['a', 'b']].to_numpy(), groups)
        assert estimated[0] == pytest.approx(0.0, abs=0.05)
        assert estimated[1] == pytest.approx(0.7, abs=0.05)
        assert sampled['b'].std() == pytest.approx(1.0, abs=0.05)

    def test_standardized_columns(self):
        """Test conditioning on a column that is not modeled by the copula."""
        # Setup
        model, data = self._get_model(1000)
        correlation = model.correlation.copy()
        correlation['dummy'] = [0.5, 0.0, 0.0]
        correlation.loc['dummy'] = [0.5, 0.0, 0.0, 1.0]
        sampler = ConditionalCopulaSampler(
            model, ['dummy'], correlation=correlation, standardized={'dummy': (0.5, 0.5)}
        )
        context = pd.DataFrame({'dummy': np.repeat([0.0, 1.0], 5000)})
        np.random.seed(0)

        # Run
        sampled = sampler.sample(context)

        # Assert
        assert sampler.own_columns == ['a', 'b', 'ctx']
        difference = sampled['a'][5000:].mean() - sampled['a'][:5000].mean()
        assert difference == pytest.approx(2 * 0.5 * data['a'].std(), abs=0.1)

    def test_rejects_other_models(self):
        """Test that only Gaussian copulas are supported."""
        # Run and Assert
        with pytest.raises(TypeError, match='GaussianCopulaSynthesizer'):
            ConditionalCopulaSampler(object(), [])


class TestIndependentSynthesizerCorrelation:
    def test_lookup_tables(self):
        """Test that lookup tables are copied and their keys modeled as categorical columns."""
        # Setup
        data = _get_lookup_data()
        metadata = _get_lookup_metadata()
        instance = IndependentSynthesizer(metadata, verbose=False, lookup_tables=['cities'])

        # Run
        instance.fit(data)
        synthetic = instance.sample('people', 300)

        # Assert
        modified = instance._modified_multi_table_metadata.tables['people'].columns
        assert modified['city'] == {'sdtype': 'categorical'}
        assert metadata.tables['people'].columns['city']['sdtype'] == 'id'
        pd.testing.assert_frame_equal(synthetic['cities'], data['cities'])
        assert set(synthetic['people']['city']) <= set(data['cities']['code'])
        metadata.validate_data(synthetic)

    def test_lookup_tables_invalid(self):
        """Test that a lookup table must exist and have no parents."""
        # Setup
        metadata = _get_lookup_metadata()

        # Run and Assert
        with pytest.raises(SynthesizerInputError, match='not in the metadata'):
            IndependentSynthesizer(metadata, lookup_tables=['towns'])

        with pytest.raises(SynthesizerInputError, match='cannot have parent tables'):
            IndependentSynthesizer(metadata, lookup_tables=['people'])

    def test_detect_lookup_tables(self):
        """Test that only small descriptive parent tables are suggested."""
        # Run
        lookup_tables = detect_lookup_tables(_get_lookup_data(), _get_lookup_metadata())

        # Assert
        assert lookup_tables == ['cities']

    def test_context_columns_invalid(self):
        """Test that the context must come from the primary parent and existing columns."""
        # Setup
        _, metadata = generate_correlated_parent_child(10)

        # Run and Assert
        with pytest.raises(SynthesizerInputError, match='not a child table'):
            IndependentSynthesizer(metadata, context_columns={'parent': {'child': ['amount']}})

        with pytest.raises(SynthesizerInputError, match='not in table'):
            IndependentSynthesizer(metadata, context_columns={'child': {'parent': ['nope']}})

    def test_invalid_options(self):
        """Test the validation of the encoding options."""
        # Setup
        _, metadata = generate_correlated_parent_child(10)

        # Run and Assert
        with pytest.raises(SynthesizerInputError, match='categorical_context'):
            IndependentSynthesizer(metadata, categorical_context='nope')

        with pytest.raises(SynthesizerInputError, match='sibling_categorical'):
            IndependentSynthesizer(metadata, sibling_categorical='nope')

    def test_rank_mapping_keeps_exact_counts(self):
        """Test that ``model_cardinality`` keeps the real counts and follows the parent."""
        # Setup
        data, metadata = generate_correlated_parent_child(3000)
        instance = IndependentSynthesizer(metadata, verbose=False, model_cardinality=True)

        # Run
        instance.fit(data)
        synthetic = instance.sample('parent', 3000)

        # Assert
        real = data['child']['parent_id'].value_counts().reindex(data['parent']['parent_id'])
        fake = synthetic['child']['parent_id'].value_counts()
        fake = fake.reindex(synthetic['parent']['parent_id'])
        assert ks_2samp(real.fillna(0), fake.fillna(0)).statistic == 0
        spearman = pd.Series(synthetic['parent']['size'].to_numpy()).corr(
            pd.Series(fake.fillna(0).to_numpy()), method='spearman'
        )
        assert spearman > 0.4
        metadata.validate_data(synthetic)

    def test_one_to_one_cardinality(self):
        """Test that one-to-one children follow the modeled score of the parent."""
        # Setup
        rng = np.random.default_rng(0)
        size = rng.normal(0, 1, 2000)
        parent = pd.DataFrame({'id': np.arange(2000), 'size': size})
        child = pd.DataFrame({'id': np.flatnonzero(size > 0.3), 'flag': 'x'})
        metadata = Metadata.load_from_dict({
            'tables': {
                'parent': {
                    'primary_key': 'id',
                    'columns': {'id': {'sdtype': 'id'}, 'size': {'sdtype': 'numerical'}},
                },
                'child': {
                    'primary_key': 'id',
                    'columns': {'id': {'sdtype': 'id'}, 'flag': {'sdtype': 'categorical'}},
                },
            },
            'relationships': [
                {
                    'parent_table_name': 'parent',
                    'child_table_name': 'child',
                    'parent_primary_key': 'id',
                    'child_foreign_key': 'id',
                }
            ],
        })
        instance = IndependentSynthesizer(metadata, verbose=False, model_cardinality=True)

        # Run
        instance.fit({'parent': parent, 'child': child})
        synthetic = instance.sample('parent', 2000)

        # Assert
        has_child = synthetic['parent']['id'].isin(synthetic['child']['id'])
        assert len(synthetic['child']) == len(child)
        assert synthetic['parent']['size'][has_child].mean() > 0.5
        metadata.validate_data(synthetic)

    def test_options_off_keep_default_sampling(self):
        """Test that the options are off by default and the default path is used."""
        # Setup
        data, metadata = generate_correlated_parent_child(200)
        instance = IndependentSynthesizer(metadata, verbose=False)

        # Run
        instance.fit(data)

        # Assert
        assert instance._uses_conditional_sampling() is False
        assert set(instance._table_synthesizers['parent']._model.columns) == {
            'size',
            'segment',
            'tier',
        }
