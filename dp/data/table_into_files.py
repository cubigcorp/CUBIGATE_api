import pandas as pd
import os
import argparse

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
        required=False,
        default=True,
        help="Whether it is train dataset"
    )
    parser.add_argument(
        '--test',
        action='store_true',
        default=False,
        required=False,
        help="Whether it is a test dataset"
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
    parser.add_argument(
        '--target',
        type=str,
        required=False,
        help="Not needed to be specified."
    )
    args = parser.parse_args()
    assert args.train ^ args.test
    args.target = 'train' if args.train else 'test'
    args.result_dir = os.path.join(args.result_dir, args.target)
    os.makedirs(args.result_dir, exist_ok=True)
    return args

if __name__ == '__main__':
    args = argument()
    if args.cols:
        cols = args.cols.split(',')
        df = pd.read_csv(args.table_file, usecols=cols)
    else:
        df=pd.read_csv(args.table_file)
        
    #assert args.label_col in cols, "Label column is specified but is not in the list of columns"
    label_idx = cols.index(args.label_col) if args.label_col is not None else None
   
    for row in df.itertuples(name=None):
        items = [str(i) for i in row[1:]]
        text = ' '.join(items) 
        label = row[label_idx] if label_idx is not None else "UNCOND"
        with open(os.path.join(args.result_dir, f'{label}_{row[0]}.txt'), 'w') as f:
            f.write(text)