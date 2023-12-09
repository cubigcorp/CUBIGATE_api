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
        help="Path of the result directory. The final path will be result_dir/[private|public]"
    )
    parser.add_argument(
        '--public',
        action='store_true',
        default=True,
        help="Whether it is public dataset"
    )
    parser.add_argument(
        '--private',
        action='store_true',
        required=False,
        help="Whether it is a private dataset"
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
    assert args.public ^ args.private
    if args.concat_n_split is not None:
        private = os.path.join(args.result_dir, 'private')
        public = os.path.join(args.result_dir, 'public')
        os.makedirs(private, exist_ok=True)
        os.makedirs(public, exist_ok=True)
        args.result_dir = [private, public]
    else:
        target = 'public' if args.public else 'private'
        args.result_dir = [os.path.join(args.result_dir, target)]
        os.makedirs(args.result_dir, exist_ok=True)
    return args

def write_file(df: pd.DataFrame, label_idx: int, dir: str):
    for row in df.itertuples(name=None):
        items = [str(row[i]) for i in range(1, len(row)) if i != label_idx + 1]
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
        public = int(total * args.concat_n_split)
        indices = np.random.choice(total - 1, public, replace=False)
        print(f"{len(indices)} samples for public")
        public_df = df.loc[indices]
        private_df = df.loc[df.index.isin(indices)]
        data = [public_df, private_df]
    else:
        data = [df]

    label_idx = cols.index(args.label_col) if args.label_col is not None else None

    for df, dir in zip(data, args.result_dir):
        write_file(df, label_idx, dir)