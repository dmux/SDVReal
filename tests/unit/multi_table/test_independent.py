from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from sdv.metadata.metadata import Metadata
from sdv.multi_table.independent import IndependentSynthesizer
from sdv.single_table.copulas import GaussianCopulaSynthesizer
from tests.utils import generate_multi_table_schema


def _get_parent_child_metadata(child_primary_key='child_id'):
    child_columns = {
        'child_id': {'sdtype': 'id'},
        'parent_id': {'sdtype': 'id'},
        'value': {'sdtype': 'numerical'},
    }
    return Metadata.load_from_dict({
        'tables': {
            'parent': {
                'primary_key': 'parent_id',
                'columns': {'parent_id': {'sdtype': 'id'}, 'value': {'sdtype': 'numerical'}},
            },
            'child': {'primary_key': child_primary_key, 'columns': child_columns},
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


class TestIndependentSynthesizer:
    def test___init__(self):
        """Test the default initialization of the ``IndependentSynthesizer``."""
        # Setup
        metadata = _get_parent_child_metadata()

        # Run
        instance = IndependentSynthesizer(metadata)

        # Assert
        assert instance.metadata == metadata
        assert instance.verbose is True
        assert instance._table_sizes == {}
        assert instance._cardinality == {}
        assert instance._null_foreign_key_rates == {}
        assert isinstance(instance._table_synthesizers['parent'], GaussianCopulaSynthesizer)
        assert isinstance(instance._table_synthesizers['child'], GaussianCopulaSynthesizer)

    def test___init__complex_schema(self):
        """Test that schemas rejected by ``HMASynthesizer`` are accepted."""
        # Setup
        _, metadata = generate_multi_table_schema(num_tables=8, depth=4, rows_per_table=5)

        # Run
        instance = IndependentSynthesizer(metadata)

        # Assert
        assert set(instance._table_synthesizers) == set(metadata.tables)
        # HMASynthesizer._validate_schema_complexity rejects > 5 tables or depth > 2
        assert len(metadata.tables) > 5
        assert metadata._get_max_schema_depth() > 2

    def test__augment_tables(self):
        """Test that table sizes, cardinality and null foreign key rates are learned."""
        # Setup
        instance = IndependentSynthesizer(_get_parent_child_metadata())
        parent = pd.DataFrame({'value': [1.0, 2.0, 3.0]}, index=pd.Index([0, 1, 2], name='id'))
        child = pd.DataFrame(
            {'parent_id': [0, 0, 1, np.nan], 'value': [1.0, 2.0, 3.0, 4.0]},
            index=pd.Index([10, 11, 12, 13], name='child_id'),
        )
        processed_data = {'parent': parent, 'child': child}

        # Run
        result = instance._augment_tables(processed_data)

        # Assert
        assert result is processed_data
        assert instance._table_sizes == {'parent': 3, 'child': 4}
        key = '__parent__child__parent_id'
        np.testing.assert_array_equal(instance._cardinality[key], [1, 1, 1])
        assert instance._null_foreign_key_rates[key] == 0.25

    def test__model_tables(self):
        """Test that every table is fitted without its foreign keys."""
        # Setup
        instance = IndependentSynthesizer(_get_parent_child_metadata())
        parent_synthesizer = Mock()
        child_synthesizer = Mock()
        instance._table_synthesizers = {'parent': parent_synthesizer, 'child': child_synthesizer}
        parent = pd.DataFrame({'value': [1.0, 2.0]})
        child = pd.DataFrame({'parent_id': [0, 1], 'value': [3.0, 4.0]})

        # Run
        instance._model_tables({'parent': parent, 'child': child})

        # Assert
        parent_data = parent_synthesizer.fit_processed_data.call_args[0][0]['parent']
        child_data = child_synthesizer.fit_processed_data.call_args[0][0]['child']
        pd.testing.assert_frame_equal(parent_data, parent)
        pd.testing.assert_frame_equal(child_data, pd.DataFrame({'value': [3.0, 4.0]}))
        assert 'parent_id' in child.columns

    def test__sample_foreign_key_values(self):
        """Test that the sampled keys reference the parents and have the requested length."""
        # Setup
        np.random.seed(0)
        parent_keys = np.arange(100)
        cardinality = np.array([10, 0, 5, 1])

        # Run
        values = IndependentSynthesizer._sample_foreign_key_values(cardinality, parent_keys, 170)

        # Assert
        assert len(values) == 170
        assert np.isin(values, parent_keys).all()

    def test__sample_foreign_key_values_keeps_distribution(self):
        """Test that the children-per-parent distribution is preserved."""
        # Setup
        np.random.seed(0)
        parent_keys = np.arange(10_000)
        cardinality = np.array([5000, 0, 0, 0, 5000])

        # Run
        values = IndependentSynthesizer._sample_foreign_key_values(cardinality, parent_keys, 20_000)

        # Assert
        counts = pd.Series(values).value_counts().reindex(parent_keys, fill_value=0)
        assert counts.isin([0, 4]).mean() > 0.95

    def test__sample_foreign_key_values_no_children(self):
        """Test that no keys are returned if there are no children to assign."""
        # Run
        values = IndependentSynthesizer._sample_foreign_key_values(
            np.array([0, 3]), np.arange(5), 0
        )

        # Assert
        assert len(values) == 0

    def test__sample_foreign_key_values_all_zero_cardinality(self):
        """Test that the children are spread over the parents if the parents had no children."""
        # Run
        values = IndependentSynthesizer._sample_foreign_key_values(np.array([4]), np.arange(4), 8)

        # Assert
        assert len(values) == 8
        assert np.isin(values, np.arange(4)).all()

    def test__add_foreign_key_columns(self):
        """Test that the foreign keys reference the parent and respect the null rate."""
        # Setup
        np.random.seed(0)
        instance = IndependentSynthesizer(_get_parent_child_metadata())
        key = '__parent__child__parent_id'
        instance._cardinality[key] = np.array([2, 5, 3])
        instance._null_foreign_key_rates[key] = 0.1
        parent = pd.DataFrame({'parent_id': np.arange(50), 'value': np.zeros(50)})
        child = pd.DataFrame({'child_id': np.arange(200), 'value': np.zeros(200)})

        # Run
        instance._add_foreign_key_columns(child, parent, 'child', 'parent')

        # Assert
        foreign_keys = child['parent_id']
        assert foreign_keys.isna().sum() == 20
        assert foreign_keys.dropna().isin(parent['parent_id']).all()

    def test__add_foreign_key_columns_is_reproducible(self):
        """Test that the same seed produces the same foreign keys."""
        # Setup
        instance = IndependentSynthesizer(_get_parent_child_metadata())
        key = '__parent__child__parent_id'
        instance._cardinality[key] = np.array([1, 2, 1])
        parent = pd.DataFrame({'parent_id': np.arange(20)})
        first = pd.DataFrame({'child_id': np.arange(30)})
        second = pd.DataFrame({'child_id': np.arange(30)})

        # Run
        np.random.seed(1)
        instance._add_foreign_key_columns(first, parent, 'child', 'parent')
        np.random.seed(1)
        instance._add_foreign_key_columns(second, parent, 'child', 'parent')

        # Assert
        pd.testing.assert_series_equal(first['parent_id'], second['parent_id'])

    def test__add_foreign_key_columns_one_to_one(self):
        """Test that a foreign key which is also the primary key stays unique."""
        # Setup
        instance = IndependentSynthesizer(_get_parent_child_metadata('parent_id'))
        parent = pd.DataFrame({'parent_id': np.arange(10)})
        child = pd.DataFrame({'value': np.zeros(8)})

        # Run
        instance._add_foreign_key_columns(child, parent, 'child', 'parent')

        # Assert
        assert child['parent_id'].is_unique
        assert child['parent_id'].isin(parent['parent_id']).all()

    def test__add_foreign_key_columns_one_to_one_more_children(self):
        """Test that a warning is shown if a one-to-one child has more rows than its parent."""
        # Setup
        instance = IndependentSynthesizer(_get_parent_child_metadata('parent_id'))
        parent = pd.DataFrame({'parent_id': np.arange(3)})
        child = pd.DataFrame({'value': np.zeros(5)})

        # Run
        with pytest.warns(UserWarning, match='one-to-one relationship'):
            instance._add_foreign_key_columns(child, parent, 'child', 'parent')

        # Assert
        assert child['parent_id'].isin(parent['parent_id']).all()
