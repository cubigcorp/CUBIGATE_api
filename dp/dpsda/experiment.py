from matplotlib import pyplot as plt
import os
from typing import Literal, Optional, Tuple, Union
import numpy as np
import wandb

COLORS = {-1: 'blue', 0: 'green', 1: 'yellow', 2: 'purple', 3: 'teal', 4: 'olive', 
          5: 'peru', 6: 'crimson', 7: 'orange', 8: 'black', 9: 'darkgreen'}


def get_xy(position: str, ratio: float) -> float:
    if position == 'right' or position == 'upper':
        p = 1 - ratio
    elif position == 'left' or position == 'lower':
        p = 0
    elif position == 'middle':
        p = (1 - ratio) / 2
    else:
        raise ValueError(f"Unknown position: {position}")
    return p


def get_samples_out_bounding(shape: str, rng: np.random.Generator, num_data: int, bounding: Tuple, size: str, ratio: float, distinctive: int = 10) -> np.ndarray:
    x_dim, y_dim = list(map(int, size.split('x')))
    (x, y, w, h) = bounding
    x =  x_dim * ratio if x == 0 else 0
    y = y_dim * ratio if y == 0 else 0
    samples = get_samples(shape, rng, num_data, distinctive, (x, y, w, h))
    return samples


def get_samples(shape: str, rng: np.random.Generator, num_data: int, distinctive: int = 10, bounding: Optional[Tuple] = None, size: Optional[str] = None) -> np.ndarray:
    assert (bounding is None) ^ (size is None)
    if bounding is not None:
        x, y, w, h = bounding
    else:
        x = y = 0
        w, h = list(map(int, size.split('x')))

    other_color_idx = rng.choice(num_data, size=distinctive).tolist()
    colors = np.array([other_color_idx.index(idx) if idx in other_color_idx else -1 for idx in range(num_data)]).reshape((-1, 1))
    if shape == 'square':
        x_values = rng.uniform(low=x, high=x + w, size=num_data)
        y_values = rng.uniform(low=y, high=y + h, size=num_data)
        samples = [[x_val, y_val] for x_val, y_val in zip(x_values, y_values)]
    elif shape == 'circle':
        center_x = x + w / 2
        center_y = y + h / 2
        r = min(w, h) / 2
        samples = []
        for _ in range(num_data):
            angle = rng.uniform(0, 2 * np.pi)
            l = rng.uniform(0, r)
            sample = [center_x + l * np.cos(angle), center_y + l * np.sin(angle)]
            samples.append(sample)
    else:
        raise ValueError(f"Unknown shape: {shape}")

    samples = np.concatenate((np.stack(samples), colors), axis=1)
    return samples


def get_bounding(y_position: str, x_position: str, size: str, ratio: float) -> Tuple:
    x_dim, y_dim = list(map(int, size.split('x')))
    y = int(get_xy(y_position, ratio) * y_dim)
    x = int(get_xy(x_position, ratio) * x_dim)
    w = int(ratio * x_dim)
    h = int(ratio * y_dim)
    return (x, y, w, h)


def normalize(samples: np.ndarray, size: str) -> np.ndarray:
    x_dim, y_dim = list(map(int, size.split('x')))
    samples[:,0] = samples[:, 0] / x_dim * 2 - 1
    samples[:, 1] = samples[:, 1] / y_dim * 2 -1
    return samples
        

def get_toy_data(shape: Literal['square', 'circle'], y_position: str, x_position: str, num_data: int, num_labels: int, ratio: float, rng: np.random.Generator, size: str) -> np.ndarray:
    assert 0 <= ratio <= 1
    bounding = get_bounding(y_position, x_position, size, ratio)
    labels = np.zeros(shape=(num_data))
    samples = get_samples(shape, rng, num_data, bounding=bounding)
    samples = normalize(samples, size)
    return samples, labels
   


def log_plot(private_samples: np.ndarray, synthetic_samples: np.ndarray, size: str, step: int, dir: str, margin: int = 0.05) -> None:
    colors = np.array_split(synthetic_samples, 2, axis=1)[1].flatten()
    other_color_idx = np.where(colors != -1)[0]
    blue_color_idx = np.array([idx for idx in range(synthetic_samples.shape[0]) if idx not in other_color_idx])
    blue_samples = synthetic_samples[blue_color_idx]

    x_syn_in_prv = np.where((private_samples[:, 0].min() - margin <= synthetic_samples[:, 0]) & (synthetic_samples[:, 0] <= private_samples[:, 0].max() + margin))[0]
    y_syn_in_prv = np.where((private_samples[:, 1].min() - margin <= synthetic_samples[:, 1]) & (synthetic_samples[:, 1] <= private_samples[:, 1].max() + margin))[0]
    syn_in_prv = len(np.intersect1d(x_syn_in_prv, y_syn_in_prv))
    
    fig = plt.figure()
    plt.scatter(private_samples[:, 0], private_samples[:, 1], color='red', label='Private')
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
    