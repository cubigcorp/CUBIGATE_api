from PIL import Image
import blobfile as bf
from torch.utils.data import Dataset
import random
import os



def _list_files_recursively(data_dir):
    results = []
    file_list=os.listdir(data_dir)
    random.shuffle(file_list)
    print(file_list)
    for entry in file_list:
        full_path = bf.join(data_dir, entry)
        ext = entry.split('.')[-1]
        if "." in entry and ext.lower() in ['jpg', 'jpeg', 'png', 'gif']:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, folder, transform):
        super().__init__()
        self.folder = folder
        self.transform = transform
        
        self.local_images = _list_files_recursively(folder)
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

