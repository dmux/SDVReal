"""Independent multi-table synthesizer."""

import logging
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from sdv.errors import SynthesizerInputError
from sdv.multi_table._conditional import ConditionalCopulaSampler
from sdv.multi_table._group_features import (
    apply_group_rule,
    intraclass_correlation,
    positions_from_assignment,
    sibling_positions,
)
from sdv.multi_table.base import BaseMultiTableSynthesizer
from sdv.sampling import BaseIndependentSampler
from sdv.single_table.base import FIXED_RNG_SEED

LOGGER = logging.getLogger(__name__)
POSITION_COLUMN = '__ctx__position'
DESCRIPTIVE_SDTYPES = {'unknown', 'text'}


def detect_lookup_tables(data, metadata, max_rows=10_000):
    """Suggest the tables that can be passed as ``lookup_tables``.

    A lookup table is a small reference table (codes and descriptions) that is a parent and has
    no parents, and whose columns other than the primary key are descriptive only.

    Args:
        data (dict):
            Dictionary mapping each table name to a ``pandas.DataFrame``.
        metadata (sdv.metadata.Metadata):
            The metadata of ``data``.
        max_rows (int):
            Maximum number of rows of a lookup table. Defaults to 10,000.

    Returns:
        list:
            The names of the suggested lookup tables.
    """
    parent_map = metadata._get_parent_map()
    child_map = metadata._get_child_map()
    lookup_tables = []
    for table_name, table_metadata in metadata.tables.items():
        if parent_map[table_name] or not child_map[table_name]:
            continue

        if len(data[table_name]) > max_rows:
            continue

        primary_key = table_metadata.primary_key
        sdtypes = {
            column_metadata['sdtype']
            for column, column_metadata in table_metadata.columns.items()
            if column != primary_key
        }
        if sdtypes <= DESCRIPTIVE_SDTYPES:
            lookup_tables.append(table_name)

    return lookup_tables


class IndependentSynthesizer(BaseIndependentSampler, BaseMultiTableSynthesizer):
    """Multi-table synthesizer that models every table independently.

    Each table is modeled by its own single-table synthesizer (``GaussianCopulaSynthesizer``
    by default) on its non foreign key columns. For every relationship, the empirical
    distribution of the number of children per parent and the rate of null foreign keys are
    learned. At sampling time, every table is sampled independently and the foreign keys are
    assigned in a vectorized way so that the children-per-parent distribution is preserved.

    The cost of fitting and sampling grows linearly with the number of rows and tables, and
    there is no limit on the number of tables or the depth of the schema. By default,
    correlations between columns of different tables are not modeled, only the cardinality of
    each relationship. The optional arguments below recover the main inter-table correlations
    while keeping one model per table:

    * ``lookup_tables``: reference tables are copied from the real data and their foreign keys
      are modeled as categorical columns of the child, so they correlate with the other columns.
    * ``model_cardinality``: the number of children is modeled as a column of the parent, so it
      correlates with the parent's columns. The real distribution of the counts is kept exactly.
    * ``context_columns``: columns of the parent are joined to the child while fitting, and every
      child is sampled conditioned on the columns of its own synthetic parent.
    * ``sibling_order`` and ``group_rules``: the position of a child among its siblings is
      modeled as context, and rules such as "the first child is the headquarters" are enforced.
    * ``sibling_correlation``: siblings share a random effect, so they look alike.

    When any of the last four options is enabled, the tables are sampled parents first and the
    foreign keys are assigned before the children are sampled. Only the first relationship of
    every child table (its primary parent) is used for that; other foreign keys are assigned as
    without the options.

    Args:
        metadata (sdv.metadata.Metadata):
            Metadata representing the data tables that this synthesizer will be used for.
        locales (list or str):
            The default locale(s) to use for AnonymizedFaker transformers.
            Defaults to ``['en_US']``.
        verbose (bool):
            Whether to print progress for fitting or not.
        lookup_tables (list or None):
            Names of the reference tables to copy and model as categorical columns.
        model_cardinality (bool):
            Whether to model the number of children as a column of the parent.
        context_columns (dict or None):
            ``{child: {parent: [columns]}}`` with the columns of the primary parent to condition
            every child on.
        sibling_order (dict or None):
            ``{child: {'by': column, 'ascending': bool}}`` to model the position of the child
            among its siblings, sorted by ``column``.
        group_rules (dict or None):
            ``{child: {'column': column, 'first': value, 'others': value}}``.
        sibling_correlation (bool or list):
            Whether to add a random effect shared by siblings, for all the child tables or
            for the given list of tables.
        max_sibling_position (int):
            Positions greater than this value are modeled as the same position. Defaults to 3.
        categorical_context (str):
            How categorical context columns are given to the child model. ``'one_hot'`` adds one
            indicator per category (up to ``max_context_categories``), so the child can learn any
            effect of each category. ``'processed'`` uses the processed column of the parent,
            whose category order is arbitrary, so only some effects are learned.
            Defaults to ``'one_hot'``.
        max_context_categories (int):
            Maximum number of indicators of a one-hot context column (the most frequent
            categories). Defaults to 10.
        sibling_categorical (str):
            How ``sibling_correlation`` handles categorical columns of the child. ``'copy'``
            copies the value of the first sibling with the probability needed to reach the real
            share of siblings that have the same value as the first one. ``'latent'`` only uses
            the shared random effect of the copula. Defaults to ``'copy'``.
        cardinality_by (dict or None):
            ``{child: [parent columns]}``. The real children-per-parent counts are stored per
            stratum of these columns of the primary parent (categories, and quantile bins of
            numerical columns) and every synthetic parent receives a count resampled from its
            own stratum. Unlike ``model_cardinality``, it does not depend on the copula of the
            parent capturing the relationship. Both can be combined: the copula then orders
            the counts inside every stratum.
        cardinality_bins (int):
            Number of quantile bins of the numerical ``cardinality_by`` columns. Defaults to 10.
    """

    def __init__(
        self,
        metadata,
        locales=['en_US'],
        verbose=True,
        lookup_tables=None,
        model_cardinality=False,
        context_columns=None,
        sibling_order=None,
        group_rules=None,
        sibling_correlation=False,
        max_sibling_position=3,
        categorical_context='one_hot',
        max_context_categories=10,
        sibling_categorical='copy',
        cardinality_by=None,
        cardinality_bins=10,
    ):
        BaseMultiTableSynthesizer.__init__(self, metadata, locales=locales)
        self._table_sizes = {}
        self._cardinality = {}
        self._null_foreign_key_rates = {}
        self.verbose = verbose
        self.lookup_tables = list(lookup_tables or [])
        self.model_cardinality = model_cardinality
        self.context_columns = context_columns or {}
        self.sibling_order = sibling_order or {}
        self.group_rules = group_rules or {}
        self.sibling_correlation = sibling_correlation
        self.max_sibling_position = max_sibling_position
        self.categorical_context = categorical_context
        self.max_context_categories = max_context_categories
        self.sibling_categorical = sibling_categorical
        self._raw_context_values = defaultdict(dict)
        self._context_categories = defaultdict(dict)
        self._sibling_match_rates = {}
        self.cardinality_by = cardinality_by or {}
        self.cardinality_bins = cardinality_bins
        self._strata_specs = {}
        self._strata_cardinality = {}
        self._fitted_context_columns = {}
        self._dummy_columns = defaultdict(list)
        self._dummy_stats = {}
        self._context_correlation = {}
        self._lookup_data = {}
        self._sibling_positions = {}
        self._context_model_columns = defaultdict(dict)
        self._context_fill_values = defaultdict(dict)
        self._sibling_rho = {}
        self._primary_relationships = self._get_primary_relationships()
        self._validate_correlation_parameters()
        if self.categorical_context not in ('one_hot', 'processed'):
            raise SynthesizerInputError("'categorical_context' must be 'one_hot' or 'processed'.")

        if self.sibling_categorical not in ('copy', 'latent'):
            raise SynthesizerInputError("'sibling_categorical' must be 'copy' or 'latent'.")

        if self.lookup_tables:
            self._use_categorical_lookup_keys()

        BaseIndependentSampler.__init__(
            self, self.metadata, self._table_synthesizers, self._table_sizes
        )

    def _is_lookup_relationship(self, relationship):
        return relationship['parent_table_name'] in self.lookup_tables

    def _get_primary_relationships(self):
        """Map every child table to ``(parent, foreign_key)`` of its first relationship."""
        primary = {}
        for relationship in self.metadata.relationships:
            child_name = relationship['child_table_name']
            if child_name not in primary and not self._is_lookup_relationship(relationship):
                primary[child_name] = (
                    relationship['parent_table_name'],
                    relationship['child_foreign_key'],
                )

        return primary

    def _get_sibling_correlation_tables(self):
        if self.sibling_correlation is True:
            return set(self._primary_relationships)

        return set(self.sibling_correlation or [])

    def _uses_conditional_sampling(self):
        return bool(
            self.model_cardinality
            or self.context_columns
            or self.sibling_order
            or self.group_rules
            or self._get_sibling_correlation_tables()
            or self.cardinality_by
        )

    def _validate_correlation_parameters(self):
        tables = set(self.metadata.tables)
        for table_name in self.lookup_tables:
            if table_name not in tables:
                raise SynthesizerInputError(f"Lookup table '{table_name}' is not in the metadata.")

            if self.metadata._get_parent_map()[table_name]:
                raise SynthesizerInputError(
                    f"Lookup table '{table_name}' cannot have parent tables."
                )

        for argument in ('context_columns', 'sibling_order', 'group_rules', 'cardinality_by'):
            for child_name in getattr(self, argument):
                if child_name not in self._primary_relationships:
                    raise SynthesizerInputError(
                        f"Table '{child_name}' in '{argument}' is not a child table."
                    )

        for child_name, parents in self.context_columns.items():
            primary_parent = self._primary_relationships[child_name][0]
            for parent_name, columns in parents.items():
                if parent_name != primary_parent:
                    raise SynthesizerInputError(
                        f"Only the primary parent '{primary_parent}' of '{child_name}' can be "
                        f"used as context, not '{parent_name}'."
                    )

                unknown = set(columns) - set(self.metadata.tables[parent_name].columns)
                if unknown:
                    raise SynthesizerInputError(
                        f"Columns {sorted(unknown)} are not in table '{parent_name}'."
                    )

    def _use_categorical_lookup_keys(self):
        """Model the foreign keys to lookup tables as categorical columns of the child."""
        for relationship in self.metadata.relationships:
            if self._is_lookup_relationship(relationship):
                child_name = relationship['child_table_name']
                foreign_key = relationship['child_foreign_key']
                columns = self._modified_multi_table_metadata.tables[child_name].columns
                columns[foreign_key] = {'sdtype': 'categorical'}

        self._initialize_models()

    def _get_modeled_out_foreign_keys(self, table_name):
        """Foreign keys of ``table_name`` that are not modeled (all but the lookup ones)."""
        primary_key = self.metadata.tables[table_name].primary_key
        return [
            relationship['child_foreign_key']
            for relationship in self.metadata.relationships
            if relationship['child_table_name'] == table_name
            and not self._is_lookup_relationship(relationship)
            and relationship['child_foreign_key'] != primary_key
        ]

    def _assign_table_transformers(self, synthesizer, table_name, table_data):
        """Ignore the foreign keys while preprocessing, except those to lookup tables."""
        synthesizer._auto_assign_transformers({table_name: table_data})
        synthesizer.update_transformers({
            column: None for column in self._get_modeled_out_foreign_keys(table_name)
        })

    def preprocess(self, data):
        """Transform the raw data and store what the correlation options need from it."""
        processed_data = super().preprocess(data)
        self._lookup_data = {name: data[name].copy() for name in self.lookup_tables}
        self._sibling_positions = {}
        for child_name, order in self.sibling_order.items():
            foreign_key = self._primary_relationships[child_name][1]
            by = order.get('by')
            self._sibling_positions[child_name] = sibling_positions(
                data[child_name][foreign_key],
                None if by is None else data[child_name][by],
                order.get('ascending', True),
            )

        self._raw_context_values = defaultdict(dict)
        for child_name, parents in self.context_columns.items():
            for parent_name, columns in parents.items():
                for column in columns:
                    if self._is_categorical_context(parent_name, column):
                        values = data[parent_name][column].to_numpy()
                        self._raw_context_values[child_name][column] = values

        self._fit_cardinality_strata(data)
        self._sibling_match_rates = {}
        if self.sibling_categorical == 'copy':
            for child_name in self._get_sibling_correlation_tables():
                self._sibling_match_rates[child_name] = self._get_sibling_match_rates(
                    child_name, data[child_name]
                )

        return processed_data

    def _fit_cardinality_strata(self, data):
        """Learn the children-per-parent counts of every stratum of ``cardinality_by``."""
        self._strata_specs = {}
        self._strata_cardinality = {}
        for child_name, columns in self.cardinality_by.items():
            parent_name, foreign_key = self._primary_relationships[child_name]
            parent_table = data[parent_name]
            specs = []
            for column in columns:
                values = parent_table[column]
                sdtype = self.metadata.tables[parent_name].columns[column]['sdtype']
                if sdtype in ('numerical', 'datetime'):
                    numbers = pd.to_numeric(values, errors='coerce').astype(float)
                    quantiles = np.linspace(0, 1, self.cardinality_bins + 1)[1:-1]
                    edges = np.unique(np.nanquantile(numbers, quantiles))
                    specs.append((column, 'bins', edges))
                else:
                    categories = values.value_counts().index[:20]
                    specs.append((column, 'categories', list(categories)))

            self._strata_specs[child_name] = specs
            primary_key = self.metadata.tables[parent_name].primary_key
            parent_keys = parent_table[primary_key] if primary_key else parent_table.index
            counts = (
                data[child_name][foreign_key]
                .value_counts()
                .reindex(pd.Index(parent_keys), fill_value=0)
            )
            strata = self._get_strata(child_name, parent_table)
            self._strata_cardinality[child_name] = {
                stratum: np.bincount(stratum_counts.astype(int))
                for stratum, stratum_counts in pd.Series(counts.to_numpy()).groupby(strata)
            }

    def _get_strata(self, child_name, parent_table):
        """Code of the ``cardinality_by`` stratum of every row of ``parent_table``."""
        codes = np.zeros(len(parent_table), dtype=np.int64)
        for column, kind, spec in self._strata_specs[child_name]:
            values = parent_table[column]
            if kind == 'bins':
                numbers = pd.to_numeric(values, errors='coerce').astype(float).to_numpy()
                column_codes = np.where(
                    np.isnan(numbers), len(spec) + 1, np.searchsorted(spec, numbers, side='right')
                )
                size = len(spec) + 2
            else:
                column_codes = pd.Categorical(values, categories=spec).codes.astype(np.int64)
                column_codes[column_codes < 0] = len(spec)
                size = len(spec) + 1

            codes = codes * size + column_codes

        return codes

    def _is_categorical_context(self, parent_name, column):
        sdtype = self.metadata.tables[parent_name].columns[column]['sdtype']
        return self.categorical_context == 'one_hot' and sdtype in ('categorical', 'boolean')

    def _get_categorical_columns(self, table_name):
        # the original metadata, since constraints such as FixedCombinations merge columns
        columns = self._original_metadata.tables[table_name].columns
        lookup_keys = {
            relationship['child_foreign_key']
            for relationship in self.metadata.relationships
            if relationship['child_table_name'] == table_name
            and self._is_lookup_relationship(relationship)
        }
        rule_column = self.group_rules.get(table_name, {}).get('column')
        return [
            name
            for name, column_metadata in columns.items()
            if (column_metadata['sdtype'] in ('categorical', 'boolean') or name in lookup_keys)
            and name != rule_column
        ]

    def _get_sibling_match_rates(self, child_name, child_data):
        """Share of the non-first siblings that have the same value as the first sibling.

        The share is computed for every parent and then averaged over the parents.
        """
        foreign_key = self._primary_relationships[child_name][1]
        if foreign_key == self.metadata.tables[child_name].primary_key:
            return {}

        if child_name in self._sibling_positions:
            positions = self._sibling_positions[child_name]
        else:
            positions = sibling_positions(child_data[foreign_key])

        keys = child_data[foreign_key].to_numpy()
        has_key = pd.notna(keys)
        rates = {}
        for column in self._get_categorical_columns(child_name):
            values = child_data[column].astype(object).where(child_data[column].notna(), '__nan__')
            frame = pd.DataFrame({'key': keys, 'value': values.to_numpy(), 'pos': positions})
            frame = frame[has_key]
            first = frame[frame['pos'] == 0].drop_duplicates('key').set_index('key')['value']
            others = frame[frame['pos'] > 0]
            if len(others):
                matches = others['value'].to_numpy() == first.reindex(others['key']).to_numpy()
                # averaged per parent, so a parent with thousands of children does not dominate
                per_parent = pd.Series(matches).groupby(others['key'].to_numpy()).mean()
                rates[column] = float(per_parent.mean())

        return rates

    @staticmethod
    def _get_relationship_key(parent_name, child_name, foreign_key):
        return f'__{parent_name}__{child_name}__{foreign_key}'

    @staticmethod
    def _get_cardinality_column(child_name, foreign_key):
        return f'__{child_name}__{foreign_key}__num_children'

    def _get_parent_keys(self, parent_name, parent_table):
        primary_key = self.metadata.tables[parent_name].primary_key
        if primary_key in parent_table.columns:
            return parent_table[primary_key].to_numpy()

        return parent_table.index.to_numpy()

    def _augment_tables(self, processed_data):
        """Learn the table sizes and the cardinality of every relationship.

        With the correlation options enabled, also add the number of children as a column of
        the parent and the parent context and sibling position as columns of the child.

        Args:
            processed_data (dict):
                Dictionary mapping each table name to a preprocessed ``pandas.DataFrame``.

        Returns:
            dict:
                The ``processed_data`` with the extra columns.
        """
        rng = np.random.default_rng(FIXED_RNG_SEED)
        for table_name, table_data in processed_data.items():
            self._table_sizes[table_name] = len(table_data)

        self._fit_parent_indices = {}
        for relationship in self.metadata.relationships:
            if self._is_lookup_relationship(relationship):
                continue

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
            if self._primary_relationships.get(child_name) != (parent_name, foreign_key):
                continue

            self._fit_parent_indices[child_name] = pd.Index(parent_keys).get_indexer(
                foreign_key_values.to_numpy()
            )
            if self.model_cardinality:
                # the jitter breaks the ties between equal counts so the copula sees a
                # continuous variable whose ranks follow the counts
                column = self._get_cardinality_column(child_name, foreign_key)
                jitter = rng.random(len(counts))
                processed_data[parent_name][column] = np.log1p(counts.to_numpy() + jitter)

        for child_name in self._primary_relationships:
            self._add_context_columns(child_name, processed_data, rng)

        return processed_data

    def _get_processed_context_columns(self, parent_name, column, processed_parent):
        processed_columns = [
            name
            for name in processed_parent.columns
            if name == column or name.startswith(f'{column}.')
        ]
        if not processed_columns:
            raise SynthesizerInputError(
                f"Column '{column}' of table '{parent_name}' is not modeled, so it cannot be "
                'used as context.'
            )

        return processed_columns

    def _add_context_columns(self, child_name, processed_data, rng):
        parent_name, _ = self._primary_relationships[child_name]
        child_table = processed_data[child_name]
        parent_table = processed_data[parent_name]
        parent_indices = self._fit_parent_indices[child_name]
        has_parent = parent_indices >= 0
        context_columns = []
        self._dummy_columns[child_name] = []
        self._context_model_columns[child_name] = {}
        self._context_categories[child_name] = {}
        for column in self.context_columns.get(child_name, {}).get(parent_name, []):
            if column in self._raw_context_values.get(child_name, {}):
                raw_values = pd.Series(self._raw_context_values[child_name][column])
                frequencies = raw_values.value_counts()
                categories = list(frequencies.index[: self.max_context_categories])
                if frequencies.loc[categories].sum() == len(raw_values):
                    # the indicators of a full partition are collinear, drop the rarest one
                    categories = categories[:-1]

                self._context_categories[child_name][column] = categories
                parent_values = raw_values.to_numpy()
                for index, category in enumerate(categories):
                    context_column = f'__ctx__{parent_name}__{column}=={index}'
                    indicator = np.zeros(len(child_table))
                    indicator[has_parent] = parent_values[parent_indices[has_parent]] == category
                    child_table[context_column] = indicator
                    context_columns.append(context_column)
                    self._dummy_columns[child_name].append(context_column)

                continue

            for parent_column in self._get_processed_context_columns(
                parent_name, column, parent_table
            ):
                context_column = f'__ctx__{parent_name}__{parent_column}'
                parent_values = parent_table[parent_column].to_numpy(dtype=float)
                fill_value = float(np.nanmedian(parent_values)) if len(parent_values) else 0.0
                values = np.full(len(child_table), fill_value)
                values[has_parent] = parent_values[parent_indices[has_parent]]
                child_table[context_column] = values
                context_columns.append(context_column)
                self._context_model_columns[child_name][context_column] = parent_column
                self._context_fill_values[child_name][context_column] = fill_value

        if child_name in self._sibling_positions:
            positions = np.minimum(self._sibling_positions[child_name], self.max_sibling_position)
            positions = np.where(has_parent, positions, 0)
            for column, values in self._get_position_columns(positions, rng).items():
                child_table[column] = values
                context_columns.append(column)
                if self.categorical_context == 'one_hot':
                    self._dummy_columns[child_name].append(column)

        self._fitted_context_columns[child_name] = context_columns

    def _get_position_columns(self, positions, rng):
        """Encode the capped sibling positions as context columns.

        With ``categorical_context='one_hot'`` every position below the cap is an indicator,
        otherwise the position is a single column with a uniform jitter inside every position.
        """
        if self.categorical_context == 'one_hot':
            return {
                f'{POSITION_COLUMN}=={position}': (positions == position).astype(float)
                for position in range(self.max_sibling_position)
            }

        return {POSITION_COLUMN: positions + rng.random(len(positions))}

    def _get_context_columns(self, child_name):
        return list(self._fitted_context_columns.get(child_name, []))

    def _fit_context_correlation(self, table_name, model, table_data, chunk_size=500_000):
        """Correlation of the copula columns extended with the standardized indicators.

        Indicators (one-hot categories and positions) are kept out of the copula: their
        latent value would be spread uniformly inside every category, which attenuates their
        effect. Using the standardized indicator instead makes the conditional mean of the
        child an exact linear regression on the indicators.
        """
        dummy_columns = self._dummy_columns[table_name]
        dummies = table_data[dummy_columns].to_numpy(dtype=float)
        means = dummies.mean(axis=0)
        stds = dummies.std(axis=0)
        stds[stds == 0] = 1.0
        self._dummy_stats[table_name] = {
            column: (float(mean), float(std))
            for column, mean, std in zip(dummy_columns, means, stds)
        }
        num_columns = len(model.columns) + len(dummy_columns)
        total = np.zeros(num_columns)
        cross = np.zeros((num_columns, num_columns))
        for start in range(0, len(table_data), chunk_size):
            chunk = table_data.iloc[start : start + chunk_size]
            latent = model._transform_to_normal(chunk[model.columns])
            standardized = (dummies[start : start + chunk_size] - means) / stds
            values = np.hstack([latent, standardized])
            total += values.sum(axis=0)
            cross += values.T @ values

        num_rows = len(table_data)
        covariance = cross / num_rows - np.outer(total, total) / num_rows**2
        scale = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
        correlation = np.nan_to_num(covariance / np.outer(scale, scale))
        np.fill_diagonal(correlation, 1.0)
        columns = list(model.columns) + dummy_columns
        self._context_correlation[table_name] = pd.DataFrame(
            correlation, index=columns, columns=columns
        )

    def _get_conditional_sampler(self, table_name, model, context_columns, sibling_rho=None):
        return ConditionalCopulaSampler(
            model,
            context_columns,
            sibling_rho,
            correlation=self._context_correlation.get(table_name),
            standardized=self._dummy_stats.get(table_name),
        )

    def _model_tables(self, augmented_data):
        """Fit a single-table synthesizer for every table, without its foreign keys.

        Args:
            augmented_data (dict):
                Dictionary mapping each table name to a preprocessed ``pandas.DataFrame``.
        """
        pbar_args = self._get_pbar_args(desc='Modeling Tables')
        sibling_tables = self._get_sibling_correlation_tables()
        for table_name, table_data in tqdm(list(augmented_data.items()), **pbar_args):
            if table_name in self.lookup_tables:
                continue

            foreign_keys = [
                foreign_key
                for foreign_key in self._get_modeled_out_foreign_keys(table_name)
                if foreign_key in table_data.columns
            ]
            table_data = table_data.drop(columns=foreign_keys)
            LOGGER.info('Fitting table %s; shape: %s', table_name, table_data.shape)
            if table_data.empty:
                continue

            synthesizer = self._table_synthesizers[table_name]
            dummy_columns = self._dummy_columns.get(table_name, [])
            synthesizer.fit_processed_data({table_name: table_data.drop(columns=dummy_columns)})
            if dummy_columns:
                self._fit_context_correlation(table_name, synthesizer._model, table_data)

            if table_name in sibling_tables and table_name in self._fit_parent_indices:
                sampler = self._get_conditional_sampler(
                    table_name, synthesizer._model, self._get_context_columns(table_name)
                )
                residuals = sampler.residuals(table_data)
                rho = intraclass_correlation(residuals, self._fit_parent_indices[table_name])
                rho = pd.Series(rho, index=sampler.own_columns)
                copied = list(self._sibling_match_rates.get(table_name, {}))
                rho[rho.index.isin(copied)] = 0.0
                self._sibling_rho[table_name] = rho

        self._fit_parent_indices = {}

    @staticmethod
    def _resample_counts(cardinality, num_parents):
        """Resample the real children-per-parent counts without replacement.

        The whole set of real counts is repeated as many times as needed, so heavy tails such
        as a single parent with thousands of children are preserved.
        """
        real_counts = np.repeat(np.arange(len(cardinality)), cardinality)
        if len(real_counts) == 0:
            return np.zeros(num_parents, dtype=int)

        repeats, remainder = divmod(num_parents, len(real_counts))
        counts = np.concatenate([
            np.tile(real_counts, repeats),
            np.random.choice(real_counts, remainder, replace=False),
        ])
        return np.random.permutation(counts)

    @classmethod
    def _sample_foreign_key_values(cls, cardinality, parent_keys, num_children):
        """Sample ``num_children`` parent keys following the children-per-parent distribution.

        The real children-per-parent counts are resampled without replacement (repeating the
        whole set of counts as many times as needed), so heavy tails such as a single parent
        with thousands of children are preserved instead of being drawn or missed at random.
        Since the sum of the sampled counts is close to, but not exactly, ``num_children``, the
        surplus is removed at random and the deficit is filled by repeating assigned parents.
        """
        if len(parent_keys) == 0 or num_children == 0:
            return parent_keys[:0]

        counts = cls._resample_counts(cardinality, len(parent_keys))
        values = np.repeat(parent_keys, counts)
        if len(values) > num_children:
            values = np.random.choice(values, num_children, replace=False)
        elif len(values) < num_children:
            pool = values if len(values) else parent_keys
            extra = np.random.choice(pool, num_children - len(values), replace=True)
            values = np.concatenate([values, extra])

        return values

    def _sample_table(self, synthesizer, table_name, num_rows, sampled_data, **kwargs):
        if table_name in self.lookup_tables:
            sampled_data[table_name] = self._lookup_data[table_name].copy()
            return

        super()._sample_table(synthesizer, table_name, num_rows, sampled_data, **kwargs)

    def _add_foreign_key_columns(self, child_table, parent_table, child_name, parent_name):
        """Assign the foreign keys of ``child_table`` that reference ``parent_table``.

        The number of children of every parent is sampled from the learned children-per-parent
        distribution, and the null foreign keys follow the learned null rate. The foreign keys
        to lookup tables were sampled as categorical columns, so they are left untouched.

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
        if parent_name in self.lookup_tables:
            return

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
            values = self._sample_foreign_key_values(
                self._cardinality[key], parent_keys, num_rows - num_nulls
            )
            child_table[foreign_key] = np.random.permutation(
                self._with_nulls(values, parent_keys, num_nulls)
            )

    @staticmethod
    def _with_nulls(values, parent_keys, num_nulls):
        if not num_nulls:
            return values

        # keep string keys as object so they can still be merged with the parent keys
        dtype = object if parent_keys.dtype == object else float
        nulls = np.full(num_nulls, np.nan, dtype=dtype)
        return np.concatenate([values.astype(dtype), nulls])

    def _get_one_to_one_keys(self, parent_keys, num_rows, child_name):
        if num_rows > len(parent_keys):
            warnings.warn(
                f"Table '{child_name}' has more rows ({num_rows}) than its parent "
                f'({len(parent_keys)}) in a one-to-one relationship. Parent keys will repeat.'
            )
            return np.random.choice(parent_keys, num_rows, replace=True)

        return np.random.choice(parent_keys, num_rows, replace=False)

    def _get_sampling_order(self):
        """Tables sorted so that every primary parent comes before its children."""
        order = []
        remaining = list(self.metadata.tables)
        while remaining:
            for table_name in remaining:
                parent = self._primary_relationships.get(table_name, (None, None))[0]
                if parent is None or parent in order:
                    order.append(table_name)
                    remaining.remove(table_name)
                    break

        return order

    def _assign_parents(
        self, child_name, num_rows, parent_processed, num_parents, parent_table=None
    ):
        """Choose the parent index of every child before sampling it.

        Returns:
            numpy.ndarray:
                Index of the parent of every child, sorted, followed by ``-1`` for the children
                with a null foreign key.
        """
        parent_name, foreign_key = self._primary_relationships[child_name]
        column = self._get_cardinality_column(child_name, foreign_key)
        is_one_to_one = foreign_key == self.metadata.tables[child_name].primary_key
        has_scores = parent_processed is not None and column in parent_processed.columns
        has_strata = child_name in self._strata_cardinality and parent_table is not None
        if is_one_to_one and not has_scores and not has_strata:
            if num_rows > num_parents:
                warnings.warn(
                    f"Table '{child_name}' has more rows ({num_rows}) than its parent "
                    f'({num_parents}) in a one-to-one relationship. Parent keys will repeat.'
                )
                return np.sort(np.random.choice(num_parents, num_rows, replace=True))

            return np.sort(np.random.choice(num_parents, num_rows, replace=False))

        key = self._get_relationship_key(parent_name, child_name, foreign_key)
        scores = parent_processed[column].to_numpy() if has_scores else None
        if has_strata:
            counts = np.zeros(num_parents, dtype=int)
            strata = self._get_strata(child_name, parent_table)
            for stratum, indices in pd.Series(np.arange(num_parents)).groupby(strata):
                indices = indices.to_numpy()
                cardinality = self._strata_cardinality[child_name].get(
                    stratum, self._cardinality[key]
                )
                counts[indices] = self._rank_counts(
                    self._resample_counts(cardinality, len(indices)),
                    None if scores is None else scores[indices],
                )
        else:
            counts = self._rank_counts(
                self._resample_counts(self._cardinality[key], num_parents), scores
            )

        assignment = np.repeat(np.arange(num_parents), counts)
        if is_one_to_one:
            return assignment

        null_rate = self._null_foreign_key_rates.get(key, 0.0)
        if null_rate >= 1:
            num_nulls = num_rows
        else:
            num_nulls = round(len(assignment) * null_rate / (1 - null_rate))

        return np.concatenate([assignment, np.full(num_nulls, -1)])

    @staticmethod
    def _rank_counts(counts, scores):
        """Rank mapping: the parents with the highest modeled score get the largest counts."""
        if scores is None:
            return counts

        ranked = np.empty_like(counts)
        ranked[np.argsort(scores, kind='stable')] = np.sort(counts)
        return ranked

    def _sample_processed_rows(self, table_name, num_rows, context=None, groups=None):
        """Sample ``num_rows`` rows, returning them in the processed and in the final space."""
        synthesizer = self._table_synthesizers[table_name]
        if not synthesizer._fitted or synthesizer._model is None:
            sampled = self._sample_in_batches(synthesizer, num_rows, None, 100)
            return None, sampled

        if not synthesizer._random_state_set:
            synthesizer._set_random_state(FIXED_RNG_SEED)

        sibling_rho = self._sibling_rho.get(table_name)
        if context is None and sibling_rho is None:
            raw_sampled = synthesizer._sample(num_rows)
        else:
            if context is None:
                context = pd.DataFrame(index=range(num_rows))

            sampler = self._get_conditional_sampler(
                table_name, synthesizer._model, list(context.columns), sibling_rho
            )
            raw_sampled = sampler.sample(context, groups)

        sampled = synthesizer._data_processor.reverse_transform(raw_sampled)
        sampled = synthesizer.reverse_transform_constraints(sampled)
        if len(sampled) != num_rows:
            raise NotImplementedError(
                f"Table '{table_name}' has constraints that reject rows, which is not supported "
                'together with the correlation options of the IndependentSynthesizer.'
            )

        input_columns = set(synthesizer._data_processor._hyper_transformer._input_columns)
        extra_columns = [
            column
            for column in raw_sampled.columns
            if column not in input_columns and column not in sampled.columns
        ]
        sampled = sampled.reset_index(drop=True)
        for column in extra_columns:
            sampled[column] = raw_sampled[column].to_numpy()

        return raw_sampled, sampled

    def _build_context(self, child_name, assignment, parent_processed, parent_table):
        context_columns = self._get_context_columns(child_name)
        if not context_columns:
            return None

        has_parent = assignment >= 0
        context = {}
        parent_name = self._primary_relationships[child_name][0]
        for column, categories in self._context_categories.get(child_name, {}).items():
            parent_values = parent_table[column].to_numpy()
            for index, category in enumerate(categories):
                indicator = np.zeros(len(assignment))
                indicator[has_parent] = parent_values[assignment[has_parent]] == category
                context[f'__ctx__{parent_name}__{column}=={index}'] = indicator

        for context_column, parent_column in self._context_model_columns[child_name].items():
            values = np.full(len(assignment), self._context_fill_values[child_name][context_column])
            values[has_parent] = parent_processed[parent_column].to_numpy()[assignment[has_parent]]
            context[context_column] = values

        if child_name in self._sibling_positions:
            positions = positions_from_assignment(assignment)
            positions = np.minimum(positions, self.max_sibling_position)
            context.update(self._get_position_columns(positions, np.random))

        return pd.DataFrame(context)[context_columns]

    def _sample(self, scale=1.0, batch_size=None, max_tries_per_batch=100):
        """Sample the entire dataset.

        Without the correlation options, every table is sampled independently and connected
        afterwards. With them, the tables are sampled parents first: the children of every
        parent are counted and assigned before the child table is sampled, conditioned on the
        context of its parent.
        """
        if not self._uses_conditional_sampling():
            return super()._sample(
                scale=scale, batch_size=batch_size, max_tries_per_batch=max_tries_per_batch
            )

        sampled_data = {}
        processed = {}
        assignments = {}
        for table_name in self._get_sampling_order():
            if table_name in self.lookup_tables:
                sampled_data[table_name] = self._lookup_data[table_name].copy()
                continue

            num_rows = max(1, round(self._table_sizes[table_name] * scale))
            if table_name not in self._primary_relationships:
                processed[table_name], sampled_data[table_name] = self._sample_processed_rows(
                    table_name, num_rows
                )
                continue

            parent_name, foreign_key = self._primary_relationships[table_name]
            parent_table = sampled_data[parent_name]
            assignment = self._assign_parents(
                table_name, num_rows, processed.get(parent_name), len(parent_table), parent_table
            )
            context = self._build_context(
                table_name, assignment, processed.get(parent_name), parent_table
            )
            groups = assignment if table_name in self._sibling_rho else None
            raw_sampled, sampled = self._sample_processed_rows(
                table_name, len(assignment), context, groups
            )
            parent_keys = self._get_parent_keys(parent_name, parent_table)
            has_parent = assignment >= 0
            values = self._with_nulls(
                parent_keys[assignment[has_parent]], parent_keys, int((~has_parent).sum())
            )
            sampled[foreign_key] = values
            if self._sibling_match_rates.get(table_name):
                self._copy_sibling_values(sampled, assignment, table_name)

            if table_name in self.group_rules:
                positions = positions_from_assignment(assignment)
                apply_group_rule(sampled, positions, assignment, self.group_rules[table_name])

            processed[table_name] = raw_sampled
            sampled_data[table_name] = sampled
            assignments[table_name] = assignment

        for relationship in self.metadata.relationships:
            parent_name = relationship['parent_table_name']
            child_name = relationship['child_table_name']
            foreign_key = relationship['child_foreign_key']
            if self._is_lookup_relationship(relationship):
                continue

            if self._primary_relationships[child_name] == (parent_name, foreign_key):
                continue

            self._add_secondary_foreign_key(
                sampled_data[child_name],
                sampled_data[parent_name],
                child_name,
                parent_name,
                foreign_key,
            )

        sampled_data = self._reverse_transform_constraints(sampled_data)
        return self._finalize(sampled_data)

    def _copy_sibling_values(self, sampled, assignment, table_name):
        """Copy categorical values from the first sibling to reach the real match rate."""
        has_parent = assignment >= 0
        positions = positions_from_assignment(assignment)
        # the children of every parent are contiguous, so the first sibling is the group start
        starts = np.flatnonzero(np.r_[True, assignment[1:] != assignment[:-1]])
        first = np.repeat(starts, np.diff(np.r_[starts, len(assignment)]))
        others = has_parent & (positions > 0)
        if not others.any():
            return

        # one draw per row shared by all the columns: a row that copies a rarely shared column
        # also copies the more often shared ones, which keeps pairs such as city and state
        # consistent with each other
        draws = np.random.random(len(assignment))
        masks = {}
        for column, real_rate in self._sibling_match_rates[table_name].items():
            values = sampled[column].to_numpy(dtype=object)
            is_null = pd.isna(values)
            matches = (values == values[first]) | (is_null & is_null[first])
            synthetic_rate = pd.Series(matches[others]).groupby(assignment[others]).mean().mean()
            if synthetic_rate < real_rate and synthetic_rate < 1:
                probability = (real_rate - synthetic_rate) / (1 - synthetic_rate)
                masks[column] = others & (draws < probability)

        # columns tied by a FixedCombinations constraint are copied together
        for group in self._get_fixed_combination_groups(table_name):
            group_masks = [masks[column] for column in group if column in masks]
            if group_masks:
                union = np.logical_or.reduce(group_masks)
                for column in group:
                    if column in sampled.columns:
                        masks[column] = union

        for column, copy in masks.items():
            values = sampled[column].to_numpy(dtype=object, copy=True)
            values[copy] = values[first[copy]]
            sampled[column] = pd.Series(values, index=sampled.index).astype(sampled[column].dtype)

    def _get_fixed_combination_groups(self, table_name):
        groups = []
        for constraint in self._get_all_constraints_list():
            column_names = getattr(constraint, 'column_names', None)
            if type(constraint).__name__ == 'FixedCombinations' and column_names:
                if getattr(constraint, 'table_name', None) in (None, table_name):
                    groups.append(list(column_names))

        return groups

    def _add_secondary_foreign_key(
        self, child_table, parent_table, child_name, parent_name, foreign_key
    ):
        parent_keys = self._get_parent_keys(parent_name, parent_table)
        num_rows = len(child_table)
        if foreign_key == self.metadata.tables[child_name].primary_key:
            child_table[foreign_key] = self._get_one_to_one_keys(parent_keys, num_rows, child_name)
            return

        key = self._get_relationship_key(parent_name, child_name, foreign_key)
        num_nulls = round(num_rows * self._null_foreign_key_rates.get(key, 0.0))
        values = self._sample_foreign_key_values(
            self._cardinality[key], parent_keys, num_rows - num_nulls
        )
        child_table[foreign_key] = np.random.permutation(
            self._with_nulls(values, parent_keys, num_nulls)
        )
