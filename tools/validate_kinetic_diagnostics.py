#!/usr/bin/env python3
"""Validate CSV output from the DGFS kinetic diagnostics plugin."""

import argparse
import csv
import math
import sys


REQUIRED_SUMMARY_COLUMNS = {
    't', 'step', 'f_min', 'negative_dof_fraction',
    'negative_l1_fraction', 'entropy_positive_part', 'mass',
    'momentum_x', 'momentum_y', 'momentum_z', 'energy',
    'troubled_cells', 'total_cells', 'troubled_fraction',
    'max_distribution_modal_sensor', 'max_density_modal_sensor',
    'max_temperature_modal_sensor', 'max_modal_sensor'
}


def fail(message):
    print('FAIL: ' + message, file=sys.stderr)
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('summary_csv')
    parser.add_argument('--minimum-samples', type=int, default=2)
    parser.add_argument('--require-troubled-cells', action='store_true')
    args = parser.parse_args()

    with open(args.summary_csv, newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        missing = REQUIRED_SUMMARY_COLUMNS - set(reader.fieldnames or [])
        if missing:
            return fail('missing columns: ' + ', '.join(sorted(missing)))
        rows = list(reader)

    if len(rows) < args.minimum_samples:
        return fail('found {0} samples; expected at least {1}'.format(
            len(rows), args.minimum_samples
        ))

    numeric_columns = REQUIRED_SUMMARY_COLUMNS - {'step'}
    for row_number, row in enumerate(rows, 2):
        for column in numeric_columns:
            try:
                value = float(row[column])
            except ValueError:
                return fail('row {0}, {1} is not numeric'.format(
                    row_number, column
                ))
            if not math.isfinite(value):
                return fail('row {0}, {1} is not finite'.format(
                    row_number, column
                ))

        negative_fraction = float(row['negative_dof_fraction'])
        negative_l1_fraction = float(row['negative_l1_fraction'])
        troubled_fraction = float(row['troubled_fraction'])
        if not 0.0 <= negative_fraction <= 1.0:
            return fail('negative_dof_fraction is outside [0, 1]')
        if not 0.0 <= negative_l1_fraction <= 1.0:
            return fail('negative_l1_fraction is outside [0, 1]')
        if not 0.0 <= troubled_fraction <= 1.0:
            return fail('troubled_fraction is outside [0, 1]')

    troubled = max(int(float(row['troubled_cells'])) for row in rows)
    if args.require_troubled_cells and troubled == 0:
        return fail('no troubled cells were detected')

    final = rows[-1]
    print('PASS: {0} samples, final t={1}, f_min={2}, negL1={3}, '
          'troubled={4}/{5}'.format(
              len(rows), final['t'], final['f_min'],
              final['negative_l1_fraction'], final['troubled_cells'],
              final['total_cells']
          ))
    return 0


if __name__ == '__main__':
    sys.exit(main())
