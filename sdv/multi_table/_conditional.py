"""Per-row conditional sampling of a Gaussian copula.

``copulas.multivariate.GaussianMultivariate`` can condition on a single vector of values per
call. The ``IndependentSynthesizer`` needs every child row to be conditioned on the context of
its own parent, so this module computes the conditional distribution once and samples all the
rows with a couple of matrix multiplications.
"""

import numpy as np
import pandas as pd
from copulas.multivariate import GaussianMultivariate
from copulas.utils import EPSILON
from scipy import stats


def _safe_cholesky(covariance, epsilon=1e-9):
    """Cholesky factor of ``covariance``, clipping negative eigenvalues if needed."""
    size = len(covariance)
    try:
        return np.linalg.cholesky(covariance + np.eye(size) * epsilon)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2)
        return eigenvectors * np.sqrt(np.clip(eigenvalues, 0, None))


class ConditionalCopulaSampler:
    """Sample a fitted ``GaussianMultivariate`` conditioned on context columns, row by row.

    With the correlation matrix split in the own columns (``1``) and the context columns
    (``2``), the conditional distribution of every row is ``N(A z2, S)`` with
    ``A = S12 S22^-1`` and ``S = S11 - A S21``. ``A`` and the Cholesky factor of ``S`` are
    computed once, so sampling ``n`` rows costs one ``n x k`` matrix multiplication.

    Optionally, a random effect shared by the rows of the same group (the siblings of a parent)
    is mixed in the noise: ``z1 = A z2 + sqrt(rho) * u_group + sqrt(1 - rho) * e_row``, which
    keeps the variance of every column and makes siblings correlated by ``rho``.

    Args:
        model (copulas.multivariate.GaussianMultivariate):
            The fitted copula.
        context_columns (list):
            Names of the columns of ``model`` to condition on.
        sibling_rho (pandas.Series or None):
            Intraclass correlation of the own columns, indexed by column name.
        ridge (float):
            Regularization added to the diagonal of ``S22``.
        correlation (pandas.DataFrame or None):
            Correlation matrix to use instead of ``model.correlation``. It may include columns
            that are not modeled by the copula, which must be in ``standardized``.
        standardized (dict or None):
            ``{column: (mean, std)}`` of the context columns that are not modeled by the copula
            and are mapped to the normal space by standardizing them.
    """

    def __init__(
        self,
        model,
        context_columns,
        sibling_rho=None,
        ridge=1e-6,
        correlation=None,
        standardized=None,
    ):
        if not isinstance(model, GaussianMultivariate):
            raise TypeError(
                'Conditioning on the parent context requires a GaussianCopulaSynthesizer '
                f'for the child table, not {type(model).__name__}.'
            )

        self._standardized = dict(standardized or {})
        model_columns = list(model.columns)
        columns = model_columns + [
            column for column in context_columns if column in self._standardized
        ]
        missing = set(context_columns) - set(columns)
        if missing:
            raise ValueError(f'Context columns {sorted(missing)} are not modeled.')

        self.columns = columns
        self.context_columns = list(context_columns)
        self.own_columns = [
            column for column in model_columns if column not in set(context_columns)
        ]
        self._univariates = dict(zip(model_columns, model.univariates))
        correlation = model.correlation if correlation is None else correlation
        correlation = correlation.loc[columns, columns]
        sigma11 = correlation.loc[self.own_columns, self.own_columns].to_numpy()
        if self.context_columns:
            sigma12 = correlation.loc[self.own_columns, self.context_columns].to_numpy()
            sigma22 = correlation.loc[self.context_columns, self.context_columns].to_numpy()
            sigma22 = sigma22 + np.eye(len(sigma22)) * ridge
            self.coefficients = np.linalg.solve(sigma22, sigma12.T).T
            covariance = sigma11 - self.coefficients @ sigma12.T
        else:
            self.coefficients = np.zeros((len(self.own_columns), 0))
            covariance = sigma11

        self.cholesky = _safe_cholesky(covariance)
        self.rho = None
        if sibling_rho is not None:
            self.rho = sibling_rho.reindex(self.own_columns).fillna(0.0).to_numpy()

    def to_normal(self, data, columns):
        """Map ``columns`` of ``data`` to the standard normal space of the copula."""
        if not columns:
            return np.zeros((len(data), 0))

        normal = []
        for column in columns:
            values = data[column].to_numpy()
            if column in self._standardized:
                mean, std = self._standardized[column]
                normal.append((values - mean) / std)
            else:
                uniform = self._univariates[column].cdf(values).clip(EPSILON, 1 - EPSILON)
                normal.append(stats.norm.ppf(uniform))

        return np.column_stack(normal)

    def residuals(self, data, columns=None):
        """Return own ``columns`` of ``data`` in normal space minus their conditional mean."""
        columns = self.own_columns if columns is None else list(columns)
        indices = [self.own_columns.index(column) for column in columns]
        normal_own = self.to_normal(data, columns)
        normal_context = self.to_normal(data, self.context_columns)
        return normal_own - normal_context @ self.coefficients[indices].T

    def sample(self, context, groups=None, include_context=True):
        """Sample one row per row of ``context``.

        Args:
            context (pandas.DataFrame):
                Values of the context columns, in the processed space of the model.
            groups (numpy.ndarray or None):
                Group of every row (``-1`` for none), used by the sibling random effect.
            include_context (bool):
                Whether to add the context columns to the output. Defaults to ``True``.

        Returns:
            pandas.DataFrame:
                The sampled own columns and the given context columns, in the column order of
                the model.
        """
        num_rows = len(context)
        num_own = len(self.own_columns)
        normal = np.random.standard_normal((num_rows, num_own)) @ self.cholesky.T
        if self.rho is not None and groups is not None:
            groups = np.asarray(groups)
            has_group = groups >= 0
            _, codes = np.unique(groups[has_group], return_inverse=True)
            shared = np.random.standard_normal((codes.max() + 1 if len(codes) else 0, num_own))
            shared = shared @ self.cholesky.T
            normal[has_group] = (
                np.sqrt(self.rho) * shared[codes] + np.sqrt(1 - self.rho) * normal[has_group]
            )

        if self.context_columns:
            normal = normal + self.to_normal(context, self.context_columns) @ self.coefficients.T

        output = {}
        for index, column in enumerate(self.own_columns):
            uniform = stats.norm.cdf(normal[:, index])
            output[column] = self._univariates[column].percent_point(uniform)

        if not include_context:
            return pd.DataFrame(output, columns=self.own_columns)

        for column in self.context_columns:
            output[column] = context[column].to_numpy()

        return pd.DataFrame(output, columns=self.columns)
