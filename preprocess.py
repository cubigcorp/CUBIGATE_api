import os
import argparse
import numpy as np

def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True
    )
    parser.add_argument(
    '--class_name',
    required=False,
    type=str
    )
    parser.add_argument(
        '--split',
        action='store_true',
    )
    args = parser.parse_args()
    return args

def add_prefix(dir: str, prefix: str) -> str:
    for file in os.listdir(dir):
        if not file[0].isdigit():
            continue
        if os.path.isdir(os.path.join(dir, file)):
            continue
        original = os.path.join(dir, file)
        modified = os.path.join(dir, f"{prefix}_{file}")
        os.replace(original, modified)

def split(dir: str):
    files = np.array([file for file in os.listdir(dir) if file.split('.')[-1] in ['jpg', 'png']])
    rng = np.random.default_rng(2022)
    train_indices = rng.choice(len(files) - 1, 100)
    
    for idx in range(len(files)):
        if idx in train_indices:
            dir = 'train'
        else:
            dir = 'test'
            
        original = os.path.join(args.data_dir, files[idx])
        modified = os.path.join(args.data_dir, dir, files[idx])
        os.replace(original, modified)

if __name__ == '__main__':
    args = argument()
    
    if args.split:
        split(args.data_dir)
    else:
        add_prefix(args.data_dir, args.class_name)


    
    