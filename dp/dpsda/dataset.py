from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import Dataset
import clip
from transformers import AutoTokenizer, AutoModel
import torch


EXTENSIONS={
    'image':
        ['jpg', 'jpeg', 'png', 'gif'],
    'text':
        ['txt']
}

def _list_files_recursively(data_dir, modality):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split('.')[-1]
        if "." in entry and ext.lower() in EXTENSIONS[modality]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_files_recursively(full_path, modality))
    return results


class ImageDataset(Dataset):
    def __init__(self, folder, transform):
        super().__init__()
        self.folder = folder
        self.transform = transform
        
        self.local_images = _list_files_recursively(folder, 'image')
        class_names = [bf.basename(path).split('_')[0]
                       for path in self.local_images]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        self.local_classes = [sorted_classes[x] for x in class_names]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, 'rb') as f:
            pil_image = Image.open(f)
            pil_image.load()

        arr = self.transform(pil_image)

        label = self.local_classes[idx]
        return arr, label

class TextDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.local_texts = _list_files_recursively(folder, 'text')
        class_name = [bf.basename(path).split('_')[0]
                      for path in self.local_texts]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_name)))}
        self.local_classes = [sorted_classes[x] for x in class_name]

    def __len__(self):
        return len(self.local_texts)

    def __getitem__(self, idx):
        path = self.local_texts[idx]
        with bf.BlobFile(path, 'r') as f:
            lines = f.read()
        if type=="bert":
            tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
            arr=tokenizer.encode_plus(lines, truncation=True)
        else:
            arr = clip.tokenize(lines, truncate=True).numpy().squeeze()
        label = self.local_classes[idx]
        return arr, label