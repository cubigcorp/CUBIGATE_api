import os
import imageio
import numpy as np
from typing import Optional
from matplotlib import use
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
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


def create_title_image(title, shape, font_path='arial.ttf', font_size=20):
    width, height, _ = shape
    image = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)
    text_width, text_height = draw.textsize(title, font=font)
    draw.text(((width - text_width) / 2, (height - text_height) / 2), title, fill=(0, 0, 0), font=font)
    return np.array(image)


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
        # titles = ['Base'] + [f'Variation{i + 1}' for i in range(packed_samples.shape[1])]
        # samples[0] = create_title_image('Selected', samples[0].shape)
        # packed_images = []
        # for col in range(row):
        #     col_img = packed_samples[col * 5: (col + 1) * 5]
        #     if col_img:
        #         titled = create_title_image(title=titles[col], shape=col_img[0].shape)
        #         col_images_with_title = [titled] + col_img  # 제목 이미지를 컬럼의 첫 이미지로 추가
        #         col_combined = np.vstack(col_images_with_title)  # 컬럼 내 이미지들을 세로로 병합
        #         packed_images.append(col_combined)
        # packed_images = np.hstack(packed_images)

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



def log_plot(private_samples: np.ndarray, synthetic_samples: np.ndarray, dir: str, step: int = -1, margin: int = 0.05) -> None:
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
    if step >= 0:
        wandb.log({"syn_in_prv": syn_in_prv, 't': step})
        wandb.log({'plot' : wandb.Image(fig), 't': step})
    plt.close()




def prv_syn_comp(method: str, private_samples: np.ndarray, synthetic_samples: np.ndarray,
          private_labels: np.ndarray, synthetic_labels: np.ndarray, t: int, dir: str, **kwargs):
    method = method.lower()
    assert method in ['t_sne', 'pca'], "Available methods are: t_SNE, PCA"
    num_private = len(private_samples)
    private_samples = private_samples.reshape((num_private, -1))
    synthetic_samples = synthetic_samples.reshape((len(synthetic_samples), -1))
    combined = np.vstack((private_samples, synthetic_samples))

    if method == 't_sne':
        from sklearn.manifold import TSNE
        f = TSNE(n_components=2, **kwargs)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        f = PCA(n_components=2, **kwargs)
    else:
        raise ValueError("Unknown method")
    X = f.fit_transform(combined)
    X_prv = X[:num_private]
    X_syn = X[num_private:]

    labels = np.unique(private_labels)
    assert np.unique(synthetic_labels) == labels

    fig = plt.figure(facecolor='white')
    for label in labels:
        plt.scatter(X_prv[private_labels == label, 0], X_prv[private_labels == label, 1], c='red', marker='o', label='Private')
        plt.scatter(X_syn[synthetic_labels == label, 0], X_syn[synthetic_labels == label, 1], color='blue', marker='*', label='Synthetic')

    plt.axhline(y=0, color='black', linewidth=1)
    plt.axvline(x=0, color='black', linewidth=1)
    plt.legend()
    plt.title(f'{method} at step {t}')
    plt.savefig(f"{dir}/{t}_prv_syn_comp.png")
    plt.close()
    