import argparse
from typing import List
import numpy as np


def str2bool(v):
    # From:
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def split_args(args: List):
    indices = []
    np_args = np.asanyarray(args)

    for i in range(0, len(args), 2):
        if 'scheduler' in args[i]:
            indices.extend([i, i+1])

    scheduler_args = np_args[indices].tolist()
    api_args = np.delete(np_args, indices).tolist()
    return api_args, scheduler_args