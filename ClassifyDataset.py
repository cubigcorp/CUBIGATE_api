import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms as T
from typing import Tuple
import numpy as np
import os

class ClassifyDataset(Dataset):
    def __init__(self, path:str, size: Tuple, num_classes: int, limit: int=None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.image_paths = []
        for root, dirs, files in os.walk(path):
            self.image_paths.extend(np.array([os.path.join(root, file) for file in files]))
        self.image_paths = np.array(self.image_paths)
        
        labels = [name.split('_')[0] for name in self.image_paths]
        sorted_labels = {x: i for i, x in enumerate(sorted(set(labels)))}
        self.labels = np.array([sorted_labels[x] for x in labels])
        self.transform = T.Compose(
            [
                T.Resize(size=[256, 256], interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                T.CenterCrop(size=size),
                T.ConvertImageDtype(torch.float32)
            ])

        if limit is not None:
            print(limit)
            rng = np.random.default_rng(2022)
            indices = rng.choice(len(self.image_paths) - 1, limit, replace=False)
            self.image_paths = self.image_paths[indices]
            self.labels = self.labels[indices]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index) -> torch.Tensor:
        image = read_image(self.image_paths[index])
        image = self.transform(image)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        # label = torch.nn.functional.one_hot(label, num_classes = self.num_classes)
        # label = label.to(torch.float32)
        return image, label
        