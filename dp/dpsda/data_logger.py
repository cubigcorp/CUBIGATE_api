import os
import imageio
import numpy as np
from typing import Optional
from matplotlib import use
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
import wandb

use('Agg') 

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



def log_fid(folder, fid, t):
    with open(os.path.join(folder, 'fid.csv'), 'a') as f:
        f.write(f'{t} {fid}\n')



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



def round_to_uint8(image):
    return np.around(np.clip(image, a_min=0, a_max=255)).astype(np.uint8)



def visualize(samples: np.ndarray, count: np.ndarray, folder: str, packed_samples: Optional[np.ndarray] = None, suffix='', n_row: int = 10):
    if not os.path.exists(folder):
        os.makedirs(folder)
    samples = samples.transpose((0, 3, 1, 2))
    if packed_samples is None:
        prefix = ''
        row = samples.shape[0] // n_row
    else:
        prefix = 'candidates_'
        packed_samples = packed_samples.transpose((0, 1, 4, 2, 3))
        row = packed_samples.shape[1] + 1

    ids = np.argsort(count)[::-1][:5]
    if packed_samples is not None:
        vis_samples = []
        for i in range(len(ids)):
            vis_samples.append(samples[ids[i]])
            for j in range(packed_samples.shape[1]):
                vis_samples.append(packed_samples[ids[i]][j])
        vis_samples = np.stack(vis_samples)
    else:
        vis_samples = samples
    vis_samples = make_grid(
        torch.Tensor(vis_samples),
        nrow=row).numpy().transpose((1, 2, 0))
    vis_samples = round_to_uint8(vis_samples)
    imageio.imsave(
        os.path.join(folder, f'{prefix}top_samples_{suffix}.png'), vis_samples)

    ids = np.argsort(count)[:5]
    if packed_samples is not None:
        vis_samples = []
        for i in range(len(ids)):
            vis_samples.append(samples[ids[i]])
            for j in range(packed_samples.shape[1]):
                vis_samples.append(packed_samples[ids[i]][j])
        vis_samples = np.stack(vis_samples)
    else:
        vis_samples = samples
    vis_samples = make_grid(
        torch.Tensor(vis_samples),
        nrow=row).numpy().transpose((1, 2, 0))
    vis_samples = round_to_uint8(vis_samples)
    imageio.imsave(
        os.path.join(folder, f'{prefix}bottom_samples_{suffix}.png'), vis_samples)
