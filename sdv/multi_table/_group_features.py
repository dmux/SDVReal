"""Features computed over the children of the same parent (siblings).

Used by the ``IndependentSynthesizer`` to model the position of a child inside its parent's
group, to enforce rules between siblings and to estimate how similar siblings are.
"""

import numpy as np
import pandas as pd


def sibling_positions(foreign_keys, order_values=None, ascending=True):
    """Return the 0-based position of every row inside the group of its foreign key.

    Args:
        foreign_keys (array-like):
            Foreign key of every row. Rows with a null foreign key get position 0.
        order_values (array-like or None):
            Values used to sort the rows inside each group. If ``None``, the row order is kept.
        ascending (bool):
            Whether ``order_values`` are sorted in ascending order.

    Returns:
        numpy.ndarray:
            The position of every row, aligned with ``foreign_keys``.
    """
    frame = pd.DataFrame({'key': np.asarray(foreign_keys)})
    if order_values is not None:
        frame['order'] = np.asarray(order_values)
        frame = frame.sort_values('order', ascending=ascending, kind='stable', na_position='last')

    positions = frame.groupby('key', sort=False, dropna=True).cumcount()
    return positions.reindex(range(len(frame))).fillna(0).to_numpy(dtype=int)


def positions_from_assignment(assignment):
    """Return the position of every child inside its parent from the assigned parent indices.

    Args:
        assignment (numpy.ndarray):
            Index of the parent of every child, ``-1`` for children without a parent.

    Returns:
        numpy.ndarray:
            The position of every child inside its parent (0 for children without a parent).
    """
    assignment = np.asarray(assignment)
    positions = pd.Series(assignment).groupby(assignment).cumcount().to_numpy()
    positions[assignment < 0] = 0
    return positions


def apply_group_rule(table, positions, assignment, rule):
    """Enforce a rule between the siblings of every parent, in place.

    The first child of every parent receives ``rule['first']`` in ``rule['column']`` and the
    other children receive ``rule['others']``. Children without a parent are not changed.

    Args:
        table (pandas.DataFrame):
            The sampled child table, aligned with ``positions`` and ``assignment``.
        positions (numpy.ndarray):
            Position of every child inside its parent.
        assignment (numpy.ndarray):
            Index of the parent of every child, ``-1`` for children without a parent.
        rule (dict):
            Dictionary with the keys ``column``, ``first`` and ``others``.
    """
    column = rule['column']
    has_parent = np.asarray(assignment) >= 0
    values = table[column].to_numpy(dtype=object, copy=True)
    values[has_parent & (positions == 0)] = rule['first']
    if 'others' in rule:
        values[has_parent & (positions > 0)] = rule['others']

    table[column] = values


def intraclass_correlation(values, groups, max_rho=0.99):
    """Estimate the intraclass correlation of every column with a one-way ANOVA.

    Only the groups with at least two rows are used, since single-row groups carry no
    information about the similarity between siblings.

    Args:
        values (numpy.ndarray):
            Matrix of shape ``(num_rows, num_columns)``.
        groups (numpy.ndarray):
            Group code of every row, ``-1`` for rows without a group.
        max_rho (float):
            Upper bound of the estimated correlation.

    Returns:
        numpy.ndarray:
            The intraclass correlation of every column, in ``[0, max_rho]``.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    num_columns = values.shape[1]
    valid = groups >= 0
    _, codes, sizes = np.unique(groups[valid], return_inverse=True, return_counts=True)
    keep = sizes[codes] >= 2
    if keep.sum() == 0:
        return np.zeros(num_columns)

    _, codes, sizes = np.unique(codes[keep], return_inverse=True, return_counts=True)
    values = values[valid][keep]
    num_groups = len(sizes)
    num_rows = len(codes)
    if num_groups < 2 or num_rows <= num_groups:
        return np.zeros(num_columns)

    rho = np.zeros(num_columns)
    n0 = (num_rows - (sizes**2).sum() / num_rows) / (num_groups - 1)
    for index in range(num_columns):
        column = values[:, index]
        means = np.bincount(codes, weights=column, minlength=num_groups) / sizes
        grand_mean = column.mean()
        between = (sizes * (means - grand_mean) ** 2).sum() / (num_groups - 1)
        within = ((column - means[codes]) ** 2).sum() / (num_rows - num_groups)
        denominator = between + (n0 - 1) * within
        if denominator > 0:
            rho[index] = (between - within) / denominator

    return np.clip(np.nan_to_num(rho), 0.0, max_rho)
