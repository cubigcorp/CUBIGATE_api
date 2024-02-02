import os
import imageio
import numpy as np
from typing import Optional
from matplotlib import use
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sklearn.manifold import TSNE
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



def visualize(samples: np.ndarray,count: np.ndarray, folder: str, t: int,
              packed_samples: Optional[np.ndarray] = None, suffix='', n_row: int = 10):
    folder = f'{folder}/{t}'
    if not os.path.exists(folder):
        os.makedirs(folder)
    samples = samples.transpose((0, 3, 1, 2))
    if packed_samples is None:
        prefix = 'samples'
        row = samples.shape[0] // n_row
    else:
        prefix = 'candidates'
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
    path = os.path.join(folder, f'{prefix}_top_{suffix}.png')
    imageio.imsave(
        path, vis_samples)
    wandb.log({f'{prefix}_top': wandb.Image(path), 't': t})

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
    path = os.path.join(folder, f'{prefix}_bottom_{suffix}.png')
    imageio.imsave(
        path, vis_samples)
    wandb.log({f'{prefix}_bottom': wandb.Image(path), 't': t})



def log_plot(private_samples: np.ndarray, synthetic_samples: np.ndarray, size: str, step: int, dir: str, margin: int = 0.05) -> None:
    COLORS = {-1: 'blue', 0: 'green', 1: 'yellow', 2: 'purple', 3: 'teal', 4: 'olive', 
          5: 'peru', 6: 'crimson', 7: 'orange', 8: 'black', 9: 'darkgreen'}

    colors = np.array_split(synthetic_samples, 2, axis=1)[1].flatten()
    other_color_idx = np.where(colors != -1)[0]
    if len(other_color_idx) != len(synthetic_samples):
        blue_color_idx = np.array([idx for idx in range(synthetic_samples.shape[0]) if idx not in other_color_idx])
        blue_samples = synthetic_samples[blue_color_idx]

    x_syn_in_prv = np.where((private_samples[:, 0].min() - margin <= synthetic_samples[:, 0]) & (synthetic_samples[:, 0] <= private_samples[:, 0].max() + margin))[0]
    y_syn_in_prv = np.where((private_samples[:, 1].min() - margin <= synthetic_samples[:, 1]) & (synthetic_samples[:, 1] <= private_samples[:, 1].max() + margin))[0]
    syn_in_prv = len(np.intersect1d(x_syn_in_prv, y_syn_in_prv))
    
    fig = plt.figure()
    plt.scatter(private_samples[:, 0], private_samples[:, 1], color='red', label='Private')
    if 'blue_samples' in vars():
        plt.scatter(blue_samples[:, 0], blue_samples[:, 1], color='blue', label='Synthetic')
    for idx in other_color_idx:
        plt.scatter(synthetic_samples[idx, 0], synthetic_samples[idx, 1], color=COLORS[synthetic_samples[idx, 2]], marker="*")
    plt.title(f"Private vs Synthetic at step {step}")
    plt.suptitle(f"#Syn in Prv: {syn_in_prv}")
    # 범례 추가
    plt.legend()

    # x축과 y축에 실선 추가
    plt.axhline(y=0, color='black', linewidth=1)
    plt.axvline(x=0, color='black', linewidth=1)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.savefig(os.path.join(dir, f"{step}_plot.png"))
    wandb.log({"syn_in_prv": syn_in_prv, 't': step})
    wandb.log({'plot' : wandb.Image(fig), 't': step})




def t_sne(private_samples: np.ndarray, synthetic_samples: np.ndarray,
          private_labels: np.ndarray, synthetic_labels: np.ndarray, t: int, dir: str, **kwargs):
    num_private = len(private_samples)
    private_samples = private_samples.reshape((num_private, -1))
    synthetic_samples = synthetic_samples.reshape((len(synthetic_samples), -1))
    combined = np.vstack((private_samples, synthetic_samples))

    tsne = TSNE(n_components=2, **kwargs)
    X = tsne.fit_transform(combined)
    X_prv = X[:num_private]
    X_syn = X[num_private:]

    labels = np.unique(private_labels)
    assert np.unique(synthetic_labels) == labels
    MARKERS = ['o', '^', 's', 'X']

    fig = plt.figure(facecolor='white')
    for label in labels:
        plt.scatter(X_prv[private_labels == label, 0], X_prv[private_labels == label, 1], c='red', marker=MARKERS[label], label='Private')
        plt.scatter(X_syn[synthetic_labels == label, 0], X_syn[synthetic_labels == label, 1], c='blue', marker=MARKERS[label], label='Synthetic')

    plt.axhline(y=0, color='black', linewidth=1)
    plt.axvline(x=0, color='black', linewidth=1)
    plt.legend()
    plt.title(f't-SNE at step {t}')
    plt.savefig(f"{dir}/{t}_t-SNE.png")
    