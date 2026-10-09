"""Summarize the correlation experiment results as Markdown tables (mean ± std over runs)."""

import argparse
import json

import pandas as pd


def _flatten(row):
    flat = {}
    for key, value in row.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                flat[f'{key}.{inner_key}'] = inner_value
        else:
            flat[key] = value

    return flat


def load(path):
    """Load a JSON lines file of results into a flat ``DataFrame``."""
    with open(path) as file:
        return pd.DataFrame([_flatten(json.loads(line)) for line in file if line.strip()])


def table(frame, group, columns, digits=3):
    """Mean (± std when there are repetitions) of ``columns`` grouped by ``group``."""
    order = list(dict.fromkeys(frame[group]))
    numeric = frame[[group, *columns]].copy()
    for column in columns:
        numeric[column] = pd.to_numeric(numeric[column], errors='coerce')

    grouped = numeric.groupby(group, sort=False)
    means = grouped.mean().reindex(order)
    stds = grouped.std().reindex(order)
    counts = grouped.size().reindex(order)
    lines = ['| ' + ' | '.join([group, *columns]) + ' |', '|' + '---|' * (len(columns) + 1)]
    for name in order:
        cells = []
        for column in columns:
            mean = means.loc[name, column]
            std = stds.loc[name, column]
            if pd.isna(mean):
                cells.append('—')
            elif counts[name] > 1 and std > 10**-digits:
                cells.append(f'{mean:.{digits}f} ± {std:.{digits}f}')
            else:
                cells.append(f'{mean:.{digits}f}')

        lines.append('| ' + ' | '.join([str(name), *cells]) + ' |')

    return '\n'.join(lines)


def main():
    """Print the summary table of a results file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path')
    parser.add_argument('--group', default='config')
    parser.add_argument('--columns', nargs='+', required=True)
    parser.add_argument('--query', default=None)
    parser.add_argument('--digits', type=int, default=3)
    args = parser.parse_args()
    frame = load(args.path)
    if args.query:
        frame = frame.query(args.query)

    print(table(frame, args.group, args.columns, args.digits))  # noqa: T201


if __name__ == '__main__':
    main()
