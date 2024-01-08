import os
import imageio
import numpy as np
from typing import Optional

def log_samples(samples, folder: str, plot_samples: bool, modality: str=None, save_npz=True, additional_info=None, prefix: str=''):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_npz:
        np.savez(
            os.path.join(folder, f'{prefix}_samples.npz'),
            samples=samples,
            additional_info=additional_info)
    if plot_samples:
        for i in range(samples.shape[0]):
            if modality == 'image':
                imageio.imwrite(os.path.join(folder, f'{prefix}_{i}.png'), samples[i])
            elif modality == 'text' or modality == 'time-series':
                with open(os.path.join(folder, f"{prefix}_{i}.txt"), 'w', encoding='utf-8') as f:
                    f.write(samples[i])
            else:
                raise Exception(f'Unknown modality {modality}')


def log_count(count: np.ndarray, clean_count: Optional[np.ndarray], loser_filter: Optional[np.ndarray], path: str):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    np.savez(path, count=count, clean_count=clean_count, losers = loser_filter)