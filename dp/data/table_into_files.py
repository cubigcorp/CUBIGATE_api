import pandas as pd
import os
import argparse

def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--text_file',
        required=True,
        type=str
    )
    parser.add_argument(
        '--result_dir',
        required=True,
        type=str
    )
    parser.add_argument(
        '--train',
        action='store_true',
        required=False,
        default=False
    )
    parser.add_argument(
        '--test',
        action='store_true',
        default=False,
        required=False
    )
    parser.add_argument(
        '--cols',
        required=True,
        type=str,
        help="list of columns to use and to be split by comma"
    )
    parser.add_argument(
        '--label_col',
        required=False,
        default=None,
        type=str,
        help="name of the column to use as label"
    )
    parser.add_argument(
        '--target',
        type=str,
        required=False
    )
    args = parser.parse_args()
    assert args.train ^ args.test
    args.target = 'train' if args.train else 'test'
    args.result_dir = os.path.join(args.result_dir, args.target)
    os.makedirs(args.result_dir, exist_ok=True)
    return args

if __name__ == '__main__':
    args = argument()
    cols = args.cols.split(',')
    assert args.label_col in cols, "Label column is specified but is not in the list of columns"
    label_idx = cols.index(args.label_col) if args.label_col is not None else None
    df = pd.read_csv(args.text_file, usecols=cols)
    for row in df.itertuples(name=None):
        text = '\n\n'.join(row[1:])
        label = row[label_idx] if label_idx is not None else "UNCOND"
        with open(os.path.join(args.result_dir, f'{label}_{row[0]}.txt'), 'w') as f:
            f.write(text)