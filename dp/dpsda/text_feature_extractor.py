import torch
import clip
from dpsda.dataset import ResizeDataset, EXTENSIONS
import zipfile
from glob import glob
import os
import numpy as np
from tqdm.auto import tqdm
import random
import clip
from sentence_transformers import SentenceTransformer
from dpsda.tokenizer import detokenize

def get_batch_features(batch, model, device):
    with torch.no_grad():
        feat = model(batch.to(device))
    if isinstance(feat, torch.Tensor):
        return feat.detach().cpu().numpy()
    else:
        return feat

def get_folder_features(fdir, modality: str, model=None, num_workers=12, num=None,
                        shuffle=False, seed=0, batch_size=128, device=torch.device("cuda"),
                        description="", verbose=True):
    # get all relevant files in the dataset
    if ".zip" in fdir:
        files = list(set(zipfile.ZipFile(fdir).namelist()))
        # remove the non-image files inside the zip
        files = [x for x in files if os.path.splitext(x)[1].lower()[1:] in EXTENSIONS[modality]]
    else:
        files = sorted([file for ext in EXTENSIONS[modality]
                    for file in glob(os.path.join(fdir, f"**/*.{ext}"), recursive=True)])
    if verbose:
        print(f"Found {len(files)} {modality}s in the folder {fdir}")
    # use a subset number of files if needed
    if num is not None:
        if shuffle:
            random.seed(seed)
            random.shuffle(files)
        files = files[:num]
    np_feats = get_files_features(files, model, num_workers=num_workers,
                                  batch_size=batch_size, device=device,
                                  description=description, fdir=fdir, verbose=verbose)
    return np_feats

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
        if isinstance(batch, list):
            batch = torch.tensor(batch)
        l_feats.append(get_batch_features(batch, model, device))
    np_feats = np.concatenate(l_feats)
    return np_feats

class CLIP_fx_txt():
    def __init__(self, name="ViT-B/32", device="cuda"):
        self.model, _ = clip.load(name, device=device)
        self.model.eval()
    
    def __call__(self, txt):

        with torch.no_grad():
            z = self.model.encode_text(txt)
        return z
    
class Sentence_fx_txt():
    def __init__(self, name="base-nli-mean-tokens", device="cuda"):
        self.model = SentenceTransformer(name)
        self.model.to(device)
        self.model.eval()
        self.name = name

    def __call__(self, token):
        txt = detokenize(self.name, token)
        with torch.no_grad():
            z = self.model.encode(txt)
        return z


