import os
import argparse
import numpy as np
from typing import List
import json

EXTENSIONS={
    'image':
        ['jpg', 'jpeg', 'png', 'gif'],
    'text':
        ['txt']
    }

def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True
    )
    parser.add_argument(
        '--org_dir',
        type=str,
        required=True
    )
    parser.add_argument(
        '--num',
        type=int,
        required=False,
        default=100
    )
    parser.add_argument(
        '--modality',
        type=str,
        required=True
    )
    parser.add_argument(
        '--condition',
        action='store_true'
    )
    parser.add_argument(
        '--class_name',
        required=False,
        type=str,
        default='UNCOND'
    )
    parser.add_argument(
        '--split',
        action='store_true',
    )
    parser.add_argument(
        '--move',
        action='store_true',
    )
    parser.add_argument(
        '--labels',
        type=str,
        required=False,
        default='',
        help="Name of labels, each of which identified by a comma without a blank"
    )
    parser.add_argument(
        '--label_texts',
        type=str,
        required=False,
        help="Texts for clip image zero-shot prediction"
    )
    parser.add_argument(
        '--base_text',
        type=str,
        required=False,
        help="Texts for clip image zero-shot prediction"
    )
    args = parser.parse_args()
    args.labels = args.labels.split(',')
    if len(args.labels) > 1:
        assert args.dataset is not None
    if args.label_texts is not None:
        args.label_texts = args.label_texts.split(',')
    if args.move:
        assert args.org_dir is not None
    return args

def move_data(org: str, tgt: str, num: int, modality: str):
    idx = 0
    for file in os.listdir(org):
        if file.split('.')[-1] not in EXTENSIONS[modality]:
            continue
        full = os.path.join(org, file)
        modified = os.path.join(tgt, file)
        os.replace(full, modified)
        idx += 1
        if num > 0 and idx == num:
            break

def add_prefix(dir: str, prefix: str) -> str:
    """
    Add class name as prefix to the names of file in dir
    """
    for file in os.listdir(dir):
        if not file.split('_')[1][0].isdigit():
            continue
        if os.path.isdir(os.path.join(dir, file)):
            continue
        original = os.path.join(dir, file)
        modified = os.path.join(dir, f"{prefix}_{file}")
        os.replace(original, modified)

def split(dir: str, modality: str):
    """
    Split all the data in dir into train/test
    """
    os.makedirs(os.path.join(dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dir, 'test'), exist_ok=True)
    files = np.array([file for file in os.listdir(dir) if (file.split('.')[-1] in EXTENSIONS[modality]) and (not file.startswith('visualize'))])
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

def make_config(dir: str, labels: List[str], base_text: str, label_texts: List[str]):
    config = {'labels': labels}
    if base_text is not None:
        config['texts'] = {'base': base_text,
                        'labels': label_texts}
    file = os.path.join(dir, 'config')
    with open(file, 'w') as f:
        json.dump(config, f)
        

if __name__ == '__main__':
    args = argument()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    if args.move:
        move_data(args.org_dir, args.data_dir, args.num, args.modality)

    if args.split:
        split(args.data_dir, args.modality)

    if args.condition:
        add_prefix(args.data_dir, args.class_name)

    if len(args.labels) > 1:
        make_config(args.data_dir, args.labels, args.base_text, args.label_texts)


    
    