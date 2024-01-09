from matplotlib import pyplot as plt
import os
from typing import Literal, Optional, Tuple
import numpy as np
import wandb


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


def get_samples(shape: str, rng: np.random.Generator, num_data: int, bounding: Optional[Tuple] = None, size: Optional[str] = None) -> np.ndarray:
    assert (bounding is None) ^ (size is None)
    if bounding is not None:
        x, y, w, h = bounding
    else:
        x = y = 0
        w, h = list(map(int, size.split('x')))
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
    samples = np.round(np.stack(samples))
    return samples
            
        

def get_toy_data(shape: Literal['square', 'circle'], y_position: str, x_position: str, num_data: int, num_labels: int, ratio: float, rng: np.random.Generator, size: str) -> np.ndarray:
    assert 0 <= ratio <= 1
    x_dim, y_dim = list(map(int, size.split('x')))
    y = int(get_xy(y_position, ratio) * y_dim)
    x = int(get_xy(x_position, ratio) * x_dim)
    w = int(ratio * x_dim)
    h = int(ratio * y_dim)

    
    labels = np.zeros(shape=(num_data))
    samples = get_samples(shape, rng, num_data, (x, y, w, h))
    return samples, labels


def log_plot(private_samples: np.ndarray, synthetic_samples: np.ndarray, size: str, step: int, dir: str) -> None:
    x_dim, y_dim = list(map(int, size.split('x')))
    fig = plt.figure()
    plt.scatter(private_samples[:, 0], private_samples[:, 1], color='red', label='Private')
    plt.scatter(synthetic_samples[:, 0], synthetic_samples[:, 1], color='blue', label='Synthetic')
    plt.title(f"Private vs Synthetic at step {step}")
    # 범례 추가
    plt.legend()

    # x축과 y축에 실선 추가
    plt.axhline(y=0, color='black', linewidth=1)
    plt.axvline(x=0, color='black', linewidth=1)
    plt.xlim(0, x_dim)
    plt.ylim(0, y_dim)
    plt.savefig(os.path.join(dir, f"{step}_plot.png"))
    wandb.log({'plot' : wandb.Image(fig), 't': step})
    