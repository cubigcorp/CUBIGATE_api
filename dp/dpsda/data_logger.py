import os
import imageio
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import wandb

def log_samples(samples, folder: str, save_each_sample: bool, modality: str=None, save_npz=True, additional_info=None, prefix: str=''):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_npz:
        np.savez(
            os.path.join(folder, f'{prefix}_samples.npz'),
            samples=samples,
            additional_info=additional_info)
    if save_each_sample:
        for i in range(samples.shape[0]):
            if modality == 'image':
                imageio.imwrite(os.path.join(folder, f'{prefix}_{i}.png'), samples[i])
            elif modality == 'text' or modality == 'time-series' or modality=="tabular":
                with open(os.path.join(folder, f"{prefix}_{i}.txt"), 'w', encoding='utf-8') as f:
                    f.write(samples[i])
            else:
                raise Exception(f'Unknown modality {modality}')


def log_count(count: np.ndarray, clean_count: Optional[np.ndarray], loser_filter: Optional[np.ndarray], path: str):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    np.savez(path, count=count, clean_count=clean_count, losers = loser_filter)


def plot_count(clean: np.ndarray, noisy: np.ndarray, dir: str, step: int, threshold: float):
    x = np.arange(len(clean))
    fig = plt.figure(facecolor='white')
    plt.scatter(x, clean, c='blue', marker='o', label='Clean')
    plt.scatter(x, noisy, c='red', marker='*', label='Noisy')
    plt.axhline(y=threshold, color='green', linestyle='--', label='Threshold')
    plt.xlabel('Sample Index')
    plt.ylabel('Count')
    plt.title(f'Comparison of Clean and Noisy Count at step {step}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(dir, f"{step}_count.png"))
    wandb.log({'count' : wandb.Image(fig), 't': step})
    plt.close()