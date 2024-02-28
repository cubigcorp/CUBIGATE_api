import os
import imageio
from torchvision.utils import make_grid
import numpy as np
import torch
from cubigate.dp.utils.round import round_to_uint8

def log_samples(samples, folder: str, plot_samples: bool, save_npz=True, prefix: str=''):
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_npz:
        np.savez(
            os.path.join(folder, f'{prefix}_samples.npz'),
            samples=samples)
    if plot_samples:
        for i in range(samples.shape[0]):
            imageio.imwrite(os.path.join(folder, f'{prefix}_{i}.png'), samples[i])


def log_count(count, path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    np.savez(path, count=count)

def visualize(samples, packed_samples, count, folder, suffix=''):
    if not os.path.exists(folder):
        os.makedirs(folder)
    samples = samples.transpose((0, 3, 1, 2))
    packed_samples = packed_samples.transpose((0, 1, 4, 2, 3))

    ids = np.argsort(count)[::-1][:5]
    print(count[ids])
    vis_samples = []
    for i in range(len(ids)):
        vis_samples.append(samples[ids[i]])
        for j in range(packed_samples.shape[1]):
            vis_samples.append(packed_samples[ids[i]][j])
    vis_samples = np.stack(vis_samples)
    vis_samples = make_grid(
        torch.Tensor(vis_samples),
        nrow=packed_samples.shape[1] + 1).numpy().transpose((1, 2, 0))
    vis_samples = round_to_uint8(vis_samples)
    imageio.imsave(
        os.path.join(folder, f'visualize_top_{suffix}.png'), vis_samples)

    ids = np.argsort(count)[:5]
    print(count[ids])
    vis_samples = []
    for i in range(len(ids)):
        vis_samples.append(samples[ids[i]])
        for j in range(packed_samples.shape[1]):
            vis_samples.append(packed_samples[ids[i]][j])
    vis_samples = np.stack(vis_samples)
    vis_samples = make_grid(
        torch.Tensor(vis_samples),
        nrow=packed_samples.shape[1] + 1).numpy().transpose((1, 2, 0))
    vis_samples = round_to_uint8(vis_samples)
    imageio.imsave(
        os.path.join(folder, f'visualize_bottom_{suffix}.png'), vis_samples)