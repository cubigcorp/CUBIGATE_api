import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict
import json
import argparse

def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--non_dp',
        type=str,
        required=False
    )
    parser.add_argument(
        '--dp',
        type=str,
        required=False
    )
    parser.add_argument(
        '--task',
        type=str,
        choices=['age'],
        required=True
    )
    parser.add_argument(
        '--result_dir',
        type=str,
        default='result',
        required=False
    )
    args = parser.parse_args()

    if args.dp is None:
        args.dp = f'result/zero-shot_{args.task}_dp.json'
    if args.non_dp is None:
        args.non_dp = f'result/zero-shot_{args.task}_non_dp.json'
    return args

def draw(x, dist, dp_dist, path):
    fig, ax = plt.subplots()
    bar_width = 0.20
    index = np.arange(len(x))
    actual = np.array([1/len(x)] * len(x))
    ax.bar(x, dp_dist, bar_width, label='dp_dist', color='b', alpha=0.5)
    ax.bar(index + bar_width, dist, bar_width, label='dist', color='g', alpha=0.5, align='edge')
    ax.plot(x, actual, 'r--o', label='actual')

    ax.set_xticks(index)
    ax.set_xticklabels(x)
    plt.title('Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Relative Frequency')
    plt.legend(loc=0)
    plt.savefig(path)

def get_value(name: str, data: Dict, relative: bool = False) -> np.ndarray:
    if name == 'distribution':
        dist = np.array(data['predicted_distribution'])
        if not relative:
            return dist[:-1]
        rdist = dist[:-1] / dist[-1]
        return rdist
    elif name == 'confidence':
        return np.array(data['confidence'][:-1])
    elif name == 'accuracy':
        return np.array(data['accuracy'])
    else:
        raise Exception("Wrong name")

def get_labels(task: str) -> np.array:
    LABELS = {
        'age': ['18-20', '21-30', '31-40', '41-50', '51-60']
    }
    return np.array(LABELS[task])

if __name__ == '__main__':
    args = argument()

    with open(args.non_dp, 'r') as f:
        non_dp = json.load(f)

    with open(args.dp, 'r') as f:
        dp = json.load(f)

    x = get_labels(args.task)
    non_dp_dist = get_value('distribution', non_dp, True)
    dp_dist = get_value('distribution', dp, True)
    non_dp_conf = get_value('confidence', non_dp)
    dp_conf = get_value('confidence', dp)
    non_dp_acc = get_value('accuracy', non_dp)
    dp_acc = get_value('accuracy', dp)
    
    draw(x, non_dp_dist, dp_dist, f'{args.result_dir}/dist.png')
    