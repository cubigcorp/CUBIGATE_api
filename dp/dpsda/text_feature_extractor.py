import torch
import clip
from cleanfid.fid import get_batch_features
import zipfile
import numpy as np
from tqdm import tqdm
import clip
from sentence_transformers import SentenceTransformer


def get_files_features(l_files, model=None, num_workers=12,
                       batch_size=128, device=torch.device("cuda"),
                       description="", fdir=None, verbose=True):
    # wrap the images in a dataloader for parallelizing the resize operation
    dataset = ResizeDataset(l_files, fdir=fdir)

    dataloader = torch.utils.data.DataLoader(dataset,
                    batch_size=batch_size, shuffle=False,
                    drop_last=False, num_workers=num_workers)

    # collect all inception features
    l_feats = []
    if verbose:
        pbar = tqdm(dataloader, desc=description)
    else:
        pbar = dataloader
    
    for batch in pbar:
        l_feats.append(get_batch_features(batch, model, device))
    np_feats = np.concatenate(l_feats)
    return np_feats

class CLIP_fx_txt():
    def __init__(self, name="ViT-B/32", device="cuda"):
        self.model, _ = clip.load(name, device=device)
        self.model.eval()
        self.name = "clip_"+name.lower().replace("-","_").replace("/","_")
    
    def __call__(self, txt):
        with torch.no_grad():
            z = self.model.encode_text(txt)
        return z
    
class BERT_fx_txt():
    def __init__(self, name="base-nli-mean-tokens", device="cuda"):
        self.model, _ = SentenceTransformer("bert-base-nli-mean-tokens")
        self.model.eval()
        self.name = "bert_"+name.lower().replace("-","_").replace("/","_")
    
    def __call__(self, txt):
        with torch.no_grad():
            z = self.model.encode(txt)
        return z


class ResizeDataset(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, fdir=None):
        self.files = files
        self.fdir = fdir
        self._zipfile = None

    def _get_zipfile(self):
        assert self.fdir is not None and '.zip' in self.fdir
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self.fdir)
        return self._zipfile

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = str(self.files[i])
        if self.fdir is not None and '.zip' in self.fdir:
            with self._get_zipfile().open(path, 'r') as f:
                txt_str = f.read()
        elif ".npy" in path:
            txt_str = np.load(path)
        else:
            with open(path, 'r') as f:
                txt_str = f.read()

        txt_np = clip.tokenize(txt_str, truncate=True).numpy().squeeze()

        return txt_np
