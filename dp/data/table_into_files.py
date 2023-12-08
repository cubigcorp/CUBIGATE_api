import pandas as pd
import os
import argparse
import numpy as np

def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--table_file',
        required=True,
        type=str,
        help="Path of the target table file."
    )
    parser.add_argument(
        '--result_dir',
        required=True,
        type=str,
        help="Path of the result directory. The final path will be result_dir/[test|train]"
    )
    parser.add_argument(
        '--train',
        action='store_true',
        default=True,
        help="Whether it is train dataset"
    )
    parser.add_argument(
        '--test',
        action='store_true',
        required=False,
        help="Whether it is a test dataset"
    )
    parser.add_argument(
        '--concat_n_split',
        type=float,
        required=False,
        help="concatenate before split"
    )
    parser.add_argument(
        '--cols',
        required=False,
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
    args = parser.parse_args()
    assert args.train ^ args.test
    if args.concat_n_split is not None:
        test = os.path.join(args.result_dir, 'test')
        train = os.path.join(args.result_dir, 'train')
        os.makedirs(test, exist_ok=True)
        os.makedirs(train, exist_ok=True)
        args.result_dir = [test, train]
    else:
        target = 'train' if args.train else 'test'
        args.result_dir = [os.path.join(args.result_dir, target)]
        os.makedirs(args.result_dir, exist_ok=True)
    return args

def write_file(df: pd.DataFrame, label_idx: int, dir: str):
    for row in df.itertuples(name=None):
        items = [str(i) for i in row[1:]]
        text = ' '.join(items) 
        label = row[label_idx + 1] if label_idx is not None else "UNCOND"
        with open(os.path.join(dir, f'{label}_{row[0]}.txt'), 'w') as f:
            f.write(text)

if __name__ == '__main__':
    args = argument()
    if args.cols:
        cols = args.cols.split(',')
        df = pd.read_csv(args.table_file, usecols=cols)
    else:
        df=pd.read_csv(args.table_file)

    if args.concat_n_split is not None:
        total = df.shape[0]
        train = int(total * args.concat_n_split)
        indices = np.random.choice(total - 1, train, replace=False)
        print(f"{len(indices)} samples for train")
        train_df = df.loc[indices]
        test_df = df.loc[df.index.isin(indices)]
        data = [train_df, test_df]
    else:
        data = [df]

    label_idx = cols.index(args.label_col) if args.label_col is not None else None

    for df, dir in zip(data, args.result_dir):
        write_file(df, label_idx, dir)