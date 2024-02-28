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
    s_indices = []
    p_indices = []
    np_args = np.asanyarray(args)

    for i in range(0, len(args), 2):
        if 'scheduler' in args[i]:
            s_indices.extend([i, i+1])
        elif 'prompt' in args[i]:
            p_indices.extend([i, i+1])

    scheduler_args = np_args[s_indices].tolist()
    prompt_args = np_args[p_indices].tolist()
    api_args = np.delete(np_args, s_indices + p_indices).tolist()
    return api_args, scheduler_args, prompt_args



def split_schedulers_args(args: List):
    indices = []
    np_args = np.asanyarray(args)

    for i in range(0, len(args), 2):
        if 'degree' in args[i]:
            indices.extend([i, i+1])
        arg_name = '_'.join(np_args[i].split('_')[2:])
        np_args[i] = f"--{arg_name}"

    scheduler_args = np_args[indices].tolist()
    weight_args = np.delete(np_args, indices).tolist()
    return weight_args, scheduler_args


def slice_scheduler_args(args: List):
    for i in range(0, len(args), 2):
        args[i] = f"--{args[i].split('_')[-1]}"
    return args