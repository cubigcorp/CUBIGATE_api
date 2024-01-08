import os
import shutil
import numpy as np
import imageio
from cleanfid import fid
from cleanfid.resize import make_resizer
import torch
import cleanfid
from cleanfid.features import build_feature_extractor, get_reference_statistics
from tqdm import tqdm

def round_to_uint8(image):
    return np.around(np.clip(image, a_min=0, a_max=255)).astype(np.uint8)



def cleanfid_make_custom_stats(
        name, fdir, split, res, num=None, modality: str='image', mode="clean",
        model_name="inception_v3", num_workers=0, batch_size=64,
        device=torch.device("cuda"), verbose=True, metric="FID"):
    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    os.makedirs(stats_folder, exist_ok=True)
    if model_name == "inception_v3":
        model_modifier = ""
    else:
        model_modifier = "_" + model_name
    outf = os.path.join(
        stats_folder,
        f"{name}_{mode}{model_modifier}_{split}_{res}.npz".lower())
    # if the custom stat file already exists
    if os.path.exists(outf):
        return
    if modality == 'image':
        from cleanfid.fid import get_folder_features
        if model_name == "inception_v3":
            feat_model = build_feature_extractor(mode, device)
            custom_fn_resize = None
            custom_image_tranform = None
        elif model_name == "clip_vit_b_32":
            from cleanfid.clip_features import CLIP_fx, img_preprocess_clip
            clip_fx = CLIP_fx("ViT-B/32")
            feat_model = clip_fx
            custom_fn_resize = img_preprocess_clip
            custom_image_tranform = None
        else:
            raise ValueError(
                f"The entered model name - {model_name} was not recognized.")

        # get all inception features for folder images
        np_feats = get_folder_features(
            fdir, feat_model, num_workers=num_workers, num=num,
            batch_size=batch_size, device=device, verbose=verbose,
            mode=mode, description=f"custom stats: {os.path.basename(fdir)} : ",
            custom_image_tranform=custom_image_tranform,
            custom_fn_resize=custom_fn_resize)

    elif modality == 'text' or modality == 'time-series':
        from dpsda.text_feature_extractor import get_folder_features
        if model_name == 'clip_vit_b_32':
            from dpsda.text_feature_extractor import CLIP_fx_txt
            feat_model = CLIP_fx_txt("ViT-B/32", device=device)
        else:
            raise ValueError(
                f"The entered model name - {model_name} was not recognized.")

        # get all features for folder samples
        np_feats = get_folder_features(
            fdir, model=feat_model, num_workers=num_workers, num=num,
            batch_size=batch_size, device=device, verbose=verbose,
            description=f"custom stats: {os.path.basename(fdir)} : ", modality=modality)
    

    if metric == "FID":
        mu = np.mean(np_feats, axis=0)
        sigma = np.cov(np_feats, rowvar=False)
        np.savez_compressed(outf, mu=mu, sigma=sigma)
    elif metric == "KID":
        outf = os.path.join(stats_folder, f"{name}_{mode}{model_modifier}_{split}_{res}_kid.npz".lower())
        np.savez_compressed(outf, feats=np_feats)
    else:
        raise Exception(f"Unknown metric: {metric}")
    print(f"saving custom {metric} stats to {outf}")
    


def make_fid_stats(samples, dataset, dataset_res, dataset_split, modality: str,
                   tmp_folder='tmp_fid', batch_size=5000,
                   model_name='inception_v3', device=torch.device("cuda"), metric="FID"):
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    else:
        shutil.rmtree(tmp_folder, ignore_errors=False, onerror=None)
        os.makedirs(tmp_folder)

    if modality == 'image':
        assert samples.dtype == np.uint8
        resizer = make_resizer(
            library='PIL', quantize_after=False, filter='bicubic',
            output_size=(dataset_res, dataset_res))
        # TODO: in memory processing for computing stats.
        for i in tqdm(range(samples.shape[0])):
            image = round_to_uint8(resizer(samples[i]))
            imageio.imsave(os.path.join(tmp_folder, f'{i}.png'), image)
    elif modality == 'text' or modality == 'time-series':
        for i in tqdm(range(samples.shape[0])):
            with open(os.path.join(tmp_folder, f'{i}.txt'), 'w') as f:
                item = [str(s) for s in samples[i]]
                text = ','.join(item)
                f.write(text)
    cleanfid_make_custom_stats(
        name=dataset, fdir=tmp_folder, split=dataset_split, res=dataset_res,
        batch_size=batch_size, model_name=model_name, modality=modality, device=device, metric=metric)


def compute_metric(samples, modality: str, tmp_folder='tmp_fid', dataset='cifar10',
                dataset_res=32, dataset_split='train', batch_size=5000,
                num_fid_samples=10000, model_name='inception_v3', device = torch.device("cuda"), metric="FID"):
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    else:
        shutil.rmtree(tmp_folder, ignore_errors=False, onerror=None)
        os.makedirs(tmp_folder)
    if num_fid_samples == samples.shape[0]:
        ids = np.arange(samples.shape[0])
    elif num_fid_samples < samples.shape[0]:
        ids = np.random.choice(
            samples.shape[0], size=num_fid_samples, replace=False)
    else:
        ids = np.random.choice(
            samples.shape[0], size=num_fid_samples, replace=True)
    samples = samples[ids]

    if modality == 'image':
        assert samples.dtype == np.uint8
        resizer = make_resizer(
            library='PIL', quantize_after=False, filter='bicubic',
            output_size=(dataset_res, dataset_res))
        # TODO: in memory processing for computing FID.
        for i in range(samples.shape[0]):
            image = round_to_uint8(resizer(samples[i]))
            imageio.imsave(os.path.join(tmp_folder, f'{i}.png'), image)
        score = fid.compute_fid(
            tmp_folder, dataset_name=dataset, dataset_split=dataset_split,
            dataset_res=dataset_res, batch_size=batch_size, model_name=model_name)

    elif modality == 'text' or modality == 'time-series':
        for i in tqdm(range(samples.shape[0])):
            with open(os.path.join(tmp_folder, f'{i}.txt'), 'w') as f:
                item = [str(s) for s in samples[i]]
                text = ','.join(item)
                f.write(text)
            if model_name == 'clip_vit_b_32':
                from dpsda.text_feature_extractor import CLIP_fx_txt
                feat_model = CLIP_fx_txt("ViT-B/32", device=device)
        
        score = score_folder(tmp_folder, model=feat_model, dataset_name=dataset, dataset_split=dataset_split,
                           dataset_res=dataset_res, batch_size=batch_size, model_name=model_name, device=device, metric=metric)

    
    return score

def score_folder(fdir, dataset_name, dataset_res, dataset_split,
               model=None, mode="clean", model_name="inception_v3", num_workers=12,
               batch_size=128, device=torch.device("cuda"), verbose=True, metric='FID'):
    # Load reference statistics (download if needed)
    if metric == "FID":
        ref_mu, ref_sigma = get_reference_statistics(dataset_name, dataset_res,
                            mode=mode, model_name=model_name, seed=0, split=dataset_split, metric=metric)
    else:
        ref_feats = get_reference_statistics(dataset_name, dataset_res,
                            mode=mode, model_name=model_name, seed=0, split=dataset_split, metric=metric)
    fbname = os.path.basename(fdir)
    # get all features for folder samples
    from dpsda.text_feature_extractor import get_folder_features
    np_feats = get_folder_features(fdir, model=model, num_workers=num_workers,
                                    batch_size=batch_size, device=device,
                                    description=f"{metric} {fbname} : ", verbose=verbose, modality='text')

    if metric == 'FID':
        mu = np.mean(np_feats, axis=0)
        sigma = np.cov(np_feats, rowvar=False)
        score = fid.frechet_distance(mu, sigma, ref_mu, ref_sigma)
    elif metric == 'KID':
        score = fid.kernel_distance(ref_feats, np_feats)
    else:
        raise Exception(f"Unknown metric: {metric}")
    return score
