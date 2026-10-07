"""Independent multi-table synthesizer."""

import logging
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from sdv.multi_table.base import BaseMultiTableSynthesizer
from sdv.sampling import BaseIndependentSampler

LOGGER = logging.getLogger(__name__)


class IndependentSynthesizer(BaseIndependentSampler, BaseMultiTableSynthesizer):
    """Multi-table synthesizer that models every table independently.

    Each table is modeled by its own single-table synthesizer (``GaussianCopulaSynthesizer``
    by default) on its non foreign key columns. For every relationship, the empirical
    distribution of the number of children per parent and the rate of null foreign keys are
    learned. At sampling time, every table is sampled independently and the foreign keys are
    assigned in a vectorized way so that the children-per-parent distribution is preserved.

    The cost of fitting and sampling grows linearly with the number of rows and tables, and
    there is no limit on the number of tables or the depth of the schema. The trade-off is that
    correlations between columns of different tables are not modeled, only the cardinality of
    each relationship.

    Args:
        metadata (sdv.metadata.Metadata):
            Metadata representing the data tables that this synthesizer will be used for.
        locales (list or str):
            The default locale(s) to use for AnonymizedFaker transformers.
            Defaults to ``['en_US']``.
        verbose (bool):
            Whether to print progress for fitting or not.
    """

    def __init__(self, metadata, locales=['en_US'], verbose=True):
        BaseMultiTableSynthesizer.__init__(self, metadata, locales=locales)
        self._table_sizes = {}
        self._cardinality = {}
        self._null_foreign_key_rates = {}
        self.verbose = verbose
        BaseIndependentSampler.__init__(
            self, self.metadata, self._table_synthesizers, self._table_sizes
        )

    @staticmethod
    def _get_relationship_key(parent_name, child_name, foreign_key):
        return f'__{parent_name}__{child_name}__{foreign_key}'

    def _get_parent_keys(self, parent_name, parent_table):
        primary_key = self.metadata.tables[parent_name].primary_key
        if primary_key in parent_table.columns:
            return parent_table[primary_key].to_numpy()

        return parent_table.index.to_numpy()

    def _augment_tables(self, processed_data):
        """Learn the table sizes and the cardinality of every relationship.

        Args:
            processed_data (dict):
                Dictionary mapping each table name to a preprocessed ``pandas.DataFrame``.

        Returns:
            dict:
                The same ``processed_data``, since no columns are added to the tables.
        """
        for table_name, table_data in processed_data.items():
            self._table_sizes[table_name] = len(table_data)

        for relationship in self.metadata.relationships:
            parent_name = relationship['parent_table_name']
            child_name = relationship['child_table_name']
            foreign_key = relationship['child_foreign_key']
            child_table = processed_data[child_name]
            if foreign_key in child_table.columns:
                foreign_key_values = child_table[foreign_key]
            else:
                # the foreign key is the primary key of the child, which is used as index
                foreign_key_values = child_table.index.to_series()

            parent_keys = self._get_parent_keys(parent_name, processed_data[parent_name])
            counts = foreign_key_values.value_counts().reindex(parent_keys, fill_value=0)
            key = self._get_relationship_key(parent_name, child_name, foreign_key)
            self._cardinality[key] = np.bincount(counts.to_numpy().astype(int))
            self._null_foreign_key_rates[key] = (
                float(foreign_key_values.isna().mean()) if len(foreign_key_values) else 0.0
            )

        return processed_data

    def _model_tables(self, augmented_data):
        """Fit a single-table synthesizer for every table, without its foreign keys.

        Args:
            augmented_data (dict):
                Dictionary mapping each table name to a preprocessed ``pandas.DataFrame``.
        """
        pbar_args = self._get_pbar_args(desc='Modeling Tables')
        for table_name, table_data in tqdm(list(augmented_data.items()), **pbar_args):
            primary_key = self.metadata.tables[table_name].primary_key
            foreign_keys = [
                foreign_key
                for foreign_key in self.metadata._get_all_foreign_keys(table_name)
                if foreign_key != primary_key and foreign_key in table_data.columns
            ]
            table_data = table_data.drop(columns=foreign_keys)
            LOGGER.info('Fitting table %s; shape: %s', table_name, table_data.shape)
            if not table_data.empty:
                self._table_synthesizers[table_name].fit_processed_data({table_name: table_data})

    @staticmethod
    def _sample_foreign_key_values(cardinality, parent_keys, num_children):
        """Sample ``num_children`` parent keys following the children-per-parent distribution.

        The number of children of every parent is sampled from ``cardinality``. Since the sum
        of the sampled counts is close to, but not exactly, ``num_children``, the surplus is
        removed at random and the deficit is filled by repeating already assigned parents.
        """
        if len(parent_keys) == 0 or num_children == 0:
            return parent_keys[:0]

        total = cardinality.sum()
        if total == 0:
            counts = np.zeros(len(parent_keys), dtype=int)
        else:
            counts = np.random.choice(
                len(cardinality), size=len(parent_keys), p=cardinality / total
            )

        values = np.repeat(parent_keys, counts)
        if len(values) > num_children:
            values = np.random.choice(values, num_children, replace=False)
        elif len(values) < num_children:
            pool = values if len(values) else parent_keys
            extra = np.random.choice(pool, num_children - len(values), replace=True)
            values = np.concatenate([values, extra])

        return values

    def _add_foreign_key_columns(self, child_table, parent_table, child_name, parent_name):
        """Assign the foreign keys of ``child_table`` that reference ``parent_table``.

        The number of children of every parent is sampled from the learned children-per-parent
        distribution, and the null foreign keys follow the learned null rate.

        Args:
            child_table (pd.DataFrame):
                The table containing data sampled for the child. Modified in place.
            parent_table (pd.DataFrame):
                The table containing data sampled for the parent.
            child_name (str):
                The name of the child table.
            parent_name (str):
                The name of the parent table.
        """
        parent_keys = self._get_parent_keys(parent_name, parent_table)
        child_primary_key = self.metadata.tables[child_name].primary_key
        num_rows = len(child_table)
        for foreign_key in self.metadata._get_foreign_keys(parent_name, child_name):
            key = self._get_relationship_key(parent_name, child_name, foreign_key)
            if foreign_key == child_primary_key:
                child_table[foreign_key] = self._get_one_to_one_keys(
                    parent_keys, num_rows, child_name
                )
                continue

            num_nulls = round(num_rows * self._null_foreign_key_rates.get(key, 0.0))
            values = pd.Series(
                self._sample_foreign_key_values(
                    self._cardinality[key], parent_keys, num_rows - num_nulls
                )
            )
            if num_nulls:
                values = pd.concat([values, pd.Series([np.nan] * num_nulls)], ignore_index=True)

            child_table[foreign_key] = values.sample(frac=1).to_numpy()

    def _get_one_to_one_keys(self, parent_keys, num_rows, child_name):
        if num_rows > len(parent_keys):
            warnings.warn(
                f"Table '{child_name}' has more rows ({num_rows}) than its parent "
                f'({len(parent_keys)}) in a one-to-one relationship. Parent keys will repeat.'
            )
            return np.random.choice(parent_keys, num_rows, replace=True)

        return np.random.choice(parent_keys, num_rows, replace=False)
