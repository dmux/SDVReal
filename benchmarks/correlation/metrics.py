"""Inter-table correlation metrics shared by the correlation experiments."""

import numpy as np
import pandas as pd

from sdv.multi_table._group_features import intraclass_correlation


def children_per_parent(parent_keys, child_keys):
    """Number of children of every parent, aligned with ``parent_keys``."""
    return child_keys.value_counts().reindex(pd.Index(parent_keys), fill_value=0).to_numpy()


def spearman(left, right):
    """Spearman correlation between two aligned arrays (``nan`` if undefined)."""
    left = pd.Series(np.asarray(left))
    right = pd.Series(np.asarray(right))
    if left.nunique() < 2 or right.nunique() < 2:
        return float('nan')

    return float(left.corr(right, method='spearman'))


def correlation_ratio(categories, values):
    """Share of the variance of ``values`` explained by ``categories`` (eta squared)."""
    frame = pd.DataFrame({'category': np.asarray(categories), 'value': np.asarray(values)})
    frame = frame.dropna()
    if frame['value'].var() == 0 or len(frame) < 2:
        return float('nan')

    means = frame.groupby('category')['value'].transform('mean')
    return float(
        ((means - frame['value'].mean()) ** 2).sum()
        / ((frame['value'] - frame['value'].mean()) ** 2).sum()
    )


def ordinal(values):
    """Encode ``values`` as numbers, parsing numbers when possible and ranking otherwise."""
    numbers = pd.to_numeric(pd.Series(values), errors='coerce')
    if numbers.notna().mean() > 0.5:
        return numbers.to_numpy()

    return pd.Series(values).astype('category').cat.codes.astype(float).to_numpy()


def pct_one_first(child_keys, flags):
    """Share of the parents with children that have exactly one child flagged."""
    has_key = pd.Series(child_keys).notna().to_numpy()
    per_parent = (
        pd
        .Series(np.asarray(flags)[has_key])
        .groupby(pd.Series(child_keys)[has_key].to_numpy())
        .sum()
    )
    return float((per_parent == 1).mean()) if len(per_parent) else float('nan')


def sibling_same_value(child_keys, values):
    """Share of the pairs of siblings that have the same value."""
    frame = pd.DataFrame({'key': np.asarray(child_keys), 'value': np.asarray(values)}).dropna()
    group_sizes = frame.groupby('key').size()
    pairs = (group_sizes * (group_sizes - 1) / 2).sum()
    if pairs == 0:
        return float('nan')

    value_sizes = frame.groupby(['key', 'value']).size()
    return float((value_sizes * (value_sizes - 1) / 2).sum() / pairs)


def sibling_icc(child_keys, values):
    """Intraclass correlation of a numerical column between siblings."""
    keys = pd.Series(child_keys)
    codes = keys.astype('category').cat.codes.to_numpy()
    numbers = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy()
    valid = ~np.isnan(numbers)
    return float(intraclass_correlation(numbers[valid, None], codes[valid])[0])


def quality_report(real, synthetic, metadata, max_parents=None, root=None, seed=0):
    """Run the sdmetrics multi-table ``QualityReport`` and return the property scores.

    With ``max_parents``, the report runs on a random subset of ``root`` rows and all their
    descendants in the real and the synthetic data, to keep the cost bounded.
    """
    from sdmetrics.reports import QualityReport

    if max_parents:
        real = subset(real, metadata, root, max_parents, seed)
        synthetic = subset(synthetic, metadata, root, max_parents, seed)

    report = QualityReport()
    report.generate(real, synthetic, metadata.to_dict(), verbose=False)
    properties = report.get_properties()
    scores = dict(zip(properties['Property'], properties['Score']))
    scores['Overall'] = report.get_score()
    return {name: round(float(score), 4) for name, score in scores.items()}


def subset(data, metadata, root, max_parents, seed=0):
    """Keep ``max_parents`` random rows of ``root`` and the rows that reference them."""
    data = dict(data)
    rng = np.random.default_rng(seed)
    root_table = data[root]
    if len(root_table) > max_parents:
        data[root] = root_table.iloc[rng.choice(len(root_table), max_parents, replace=False)]

    queue = [root]
    done = {root}
    while queue:
        parent = queue.pop(0)
        primary_key = metadata.tables[parent].primary_key
        keys = set(data[parent][primary_key])
        for relationship in metadata.relationships:
            child = relationship['child_table_name']
            if relationship['parent_table_name'] != parent or child in done:
                continue

            foreign_key = relationship['child_foreign_key']
            data[child] = data[child][data[child][foreign_key].isin(keys)]
            done.add(child)
            queue.append(child)

    return {name: table.reset_index(drop=True) for name, table in data.items()}
