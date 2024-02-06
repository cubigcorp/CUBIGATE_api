from typing import Literal, Optional, Tuple
import logging
import numpy as np


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



def get_filter(arr: np.ndarray, lower: float, upper: float, tgt: int) -> np.ndarray:
    idx, xy = np.where((lower > arr) | (arr > upper))
    xy_filter = [idx[i] for i in range(len(idx)) if xy[i] == tgt]
    return xy_filter



def get_samples_out_bounding(shape: str, seed: int, num_data: int, bounding: Tuple, size: str, distinctive: int = 10) -> np.ndarray:
    (x, y, w, h) = bounding
    samples = []
    diff = num_data
    while diff > 0:
        temp = get_samples(shape, seed, diff, size=size)
        filter_x = get_filter(temp, x, x + w, 0)
        filter_y = get_filter(temp, y, y + h, 1)
        filter_xy = np.intersect1d(filter_x, filter_y)[:diff]
        if len(filter_xy) == 0:
            continue
        diff -= len(filter_xy)
        temp = temp[filter_xy]
        samples.append(temp)
    samples = np.concatenate(samples).reshape((-1, 2))
    samples = add_colors(samples, num_data, seed, distinctive)
    return normalize(samples, size)



def get_samples_in_bounding(shape: str, seed: int, num_data: int, bounding: Tuple, size: str, distinctive: int = 10) -> np.ndarray:
    samples = get_samples(shape, seed, num_data, bounding=bounding)
    samples = add_colors(samples, num_data, seed, distinctive)
    return normalize(samples, size)



def add_colors(samples: np.ndarray, num_data: int, seed: int, distinctive: int = 10) -> np.ndarray:
    if distinctive > num_data:
        distinctive = num_data
    rng = np.random.default_rng(seed)
    other_color_idx = rng.choice(num_data, size=distinctive, replace=False).tolist()
    colors = np.array([other_color_idx.index(idx) if idx in other_color_idx else -1 for idx in range(num_data)]).reshape((-1, 1))
    logging.info(other_color_idx)
    return np.concatenate((samples, colors), axis=1)




def get_samples(shape: str, seed: int, num_data: int, bounding: Optional[Tuple] = None, size: Optional[str] = None) -> np.ndarray:
    assert (bounding is None) ^ (size is None)
    if bounding is not None:
        x, y, w, h = bounding
    else:
        x = y = 0
        w, h = list(map(int, size.split('x')))
    rng = np.random.default_rng(seed)
    
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
    return np.stack(samples)


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
        

def get_toy_data(shape: Literal['square', 'circle'], y_position: str, x_position: str, num_data: int, num_labels: int, ratio: float, seed: int, size: str) -> np.ndarray:
    assert 0 <= ratio <= 1
    if y_position == 'multi':
        samples = []
        labels = []
        for position in ['upper', 'lower']:
            if position == 'lower' and x_position == 'left':
                continue
            sub_samples, sub_labels = get_toy_data(shape=shape, y_position=position, x_position=x_position, num_data=num_data//3, num_labels=num_labels, ratio=ratio, seed=seed, size=size)
            samples.extend(sub_samples)
            labels.append(sub_labels)
        samples = np.stack(samples)
        labels = np.concatenate(labels)
        return samples, labels
    if x_position == 'multi':
        samples = []
        labels = []
        for position in ['left', 'right']:
            if position == 'left' and y_position == 'lower':
                continue
            sub_samples, sub_labels = get_toy_data(shape=shape, y_position=y_position, x_position=position, num_data=num_data//3, num_labels=num_labels, ratio=ratio, seed=seed, size=size)
            samples.extend(sub_samples)
            labels.append(sub_labels)
        samples = np.stack(samples)
        labels = np.concatenate(labels)
        return samples, labels
    bounding = get_bounding(y_position, x_position, size, ratio)
    labels = np.zeros(shape=(num_data), dtype=int)
    samples = get_samples(shape, seed, num_data, bounding=bounding)
    samples = normalize(samples, size)
    return samples, labels
   