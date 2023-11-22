from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
import torch
import blobfile as bf

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split('.')[-1]
        if "." in entry and ext.lower() in ['jpg', 'jpeg', 'png', 'gif']:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

class LoraDataset(Dataset):
    def __init__(self, dir, image_processor, tokenizer, num=100) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        local_images = _list_image_files_recursively(dir)
        PROMPT = {"NORMAL": "A photo of normal chest xray",
                  "PNEUMONIA": "A photo of chest xray with pneumonia"}
        self.image = []
        self.text = []
        for image in local_images:
            class_name = bf.basename(image).split('_')[0]
            prompt = PROMPT[class_name]
            self.image.append(image)
            self.text.append(prompt)
        rng = np.random.default_rng(2023)
        self.image = rng.choice(self.image, num, replace=False)
        self.text = rng.choice(self.text, num, replace=False)
        
        
    def __len__(self) -> int:
        return len(self.image)
    
    def __getitem__(self, index):
        img = read_image(self.image[index])
        img = self.image_processor(images=img)['pixel_values']
        img = np.array(img).squeeze()
        img = torch.tensor(img, dtype=torch.float32)
        txt = self.tokenizer(text=self.text[index],  max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True)['input_ids']
        txt = np.array(txt).squeeze()
        txt = torch.tensor(txt, dtype=torch.long)
        return img, txt