import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
import logging
import blobfile as bf
from PIL import Image
from typing import Tuple, Optional

from .dataset import ImageDataset, TextDataset, EXTENSIONS, list_files_recursively


def load_private_data(data_dir, batch_size, image_size, class_cond,
              num_private_samples, modality: str, model=None):

    if modality == 'image':
        transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
        dataset = ImageDataset(folder=data_dir, transform=transform)
    elif modality == 'text' or modality == 'time-series':
        dataset = TextDataset(folder=data_dir, model=model)

    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=10,
                        pin_memory=torch.cuda.is_available(), drop_last=False)
    all_samples = []
    all_labels = []
    cnt = 0
    for batch, cond in loader:
        all_samples.append(batch.cpu().numpy())
        if class_cond:
            all_labels.append(cond.cpu().numpy())

        cnt += batch.shape[0]

        logging.info(f'loaded {cnt} samples')
        if batch.shape[0] < batch_size:
            logging.info('WARNING: containing incomplete batch. Please check'
                         'num_private_samples')

        if cnt >= num_private_samples:
            break
    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = all_samples[:num_private_samples]
    if modality == 'image':
        all_samples = np.around(np.clip(
            all_samples * 255, a_min=0, a_max=255)).astype(np.uint8)
        all_samples = np.transpose(all_samples, (0, 2, 3, 1))
    elif modality == 'text' or modality == 'time-series':
        all_samples = all_samples.astype(np.int32)
    if class_cond:
        all_labels = np.concatenate(all_labels, axis=0)
        all_labels = all_labels[:num_private_samples]
    else:
        all_labels = np.zeros(shape=all_samples.shape[0], dtype=np.int64)
    return all_samples, all_labels


def load_samples(path):
    data = np.load(path)
    samples = data['samples']
    additional_info = data['additional_info']
    return samples, additional_info


def load_public_data(data_folder: str, modality: str, num_public_samples: int, prompt: str) -> np.ndarray:
    files = list_files_recursively(data_folder, modality)
    additional_info = []
    samples = []
    for path in files:
        if modality == 'image':  # Not tested
            with bf.BlobFile(path, 'rb') as f:
                sample = Image.open(f)
                sample.load()
        elif modality == 'text' or modality == 'time-series':
            with bf.BlobFile(path, 'r') as f:
                sample = f.read()
        samples.append(sample)
        if len(samples) == num_public_samples:
            break
    additional_info.extend([prompt[0].replace('BATCH', " ")] * len(samples))
    return np.array(samples), np.array(additional_info)


def load_count(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    npz = np.load(path)
    count = npz['count']
    losers = npz['losers'] if 'losers' in npz.files else None
    return (count, losers)