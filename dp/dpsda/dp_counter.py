import faiss
import logging
import numpy as np
from collections import Counter
import torch
from dpsda.agm import get_sigma

def sanity_check(counts: np.ndarray) -> np.ndarray:
    for i in range(counts.shape[0]):
        if np.all(counts[i] == 0):
            counts[i] = np.ones_like(counts[i])
    return counts

def get_count(ids: np.ndarray, num_candidate: int) -> np.ndarray:
    counter = Counter(list(ids.flatten()))
    count = np.zeros(shape=num_candidate)
    for k in counter:
        count[k % num_candidate] += counter[k]
        logging.debug(f"count[{k}]: {count[k]}")
    logging.info(f'Clean count sum: {np.sum(count)}')
    logging.info(f'Clean count num>0: {np.sum(count > 0)}')
    logging.info(f'Largest clean counters: {sorted(count)[::-1][:50]}')
    count = np.asarray(count)
    return count


def add_noise(counts: np.ndarray, epsilon: float, delta: float, num_nearest_neighbor: int, noise_multiplier: float, dim: int = 0) -> np.ndarray:
    if dim > 0:
        counts = counts.flatten()

    if epsilon is not None:
        sigma = get_sigma(epsilon=epsilon, delta=delta, GS=1)
        logging.info(f'calculated sigma: {sigma}')
        counts += (np.random.normal(scale=sigma, size=len(counts))) * np.sqrt(num_nearest_neighbor)
    else:
        counts += (np.random.normal(size=len(counts)) * np.sqrt(num_nearest_neighbor)
                * noise_multiplier)

    if dim > 0 :
        counts = counts.reshape((-1, dim))

    return counts


def dp_nn_histogram(public_features, private_features, epsilon: float, delta: float, 
                    noise_multiplier, num_packing=1, num_nearest_neighbor=1, mode='L2',
                    threshold=0.0, t=None, result_folder: str=None, direct_variate: bool=False):
    # public_features shape: (Nsyn, lookahead, embedding) if direct_variate
    #                        (Nsyn, embedding) otherwise
    np.set_printoptions(100)
    assert public_features.shape[0] % num_packing == 0
    num_true_public_features = public_features.shape[0] // num_packing
    faiss_res = faiss.StandardGpuResources()
    if mode == 'L2':
        index = faiss.IndexFlatL2(public_features.shape[-1])
    elif mode == 'IP':
        index = faiss.IndexFlatIP(public_features.shape[-1])
    elif mode == 'cosine':
        index = faiss.IndexFlatIP(public_features.shape[-1])
        faiss.normalize_L2(public_features)
        faiss.normalize_L2(private_features)
    else:
        raise Exception(f'Unknown mode {mode}')
    if torch.cuda.is_available():
        index = faiss.index_cpu_to_gpu(faiss_res, 0, index)
    logging.debug(f"public_features:\n{public_features}")
    logging.info("Counting votes")
    if direct_variate:
        counts = []
        for i in range(public_features.shape[0]):
            index.add(public_features[i].squeeze())
            _, id = index.search(private_features, k=num_nearest_neighbor)
            count = get_count(id, public_features.shape[1])
            counts.append(count)
            index.reset()
        counts = np.stack(counts)

    else:
        index.add(public_features)
        logging.info(f'Number of samples in index: {index.ntotal}')

        _, ids = index.search(private_features, k=num_nearest_neighbor)
        counts = get_count(ids, num_true_public_features)

    clean_count = counts.copy()
    counts = add_noise(counts, epsilon, delta, num_nearest_neighbor, noise_multiplier, counts.shape[1])
    logging.info(f'Noisy count sum: {np.sum(counts)}')
    logging.info(f'Noisy count num>0: {np.sum(counts > 0)}')
    logging.info(f'Largest noisy counters: {np.flip(np.sort(counts.flatten()))[:50]}')
    counts = np.clip(counts, a_min=threshold, a_max=None)
    counts = counts - threshold
    logging.info(f'Clipped noisy count sum: {np.sum(counts)}')
    logging.info(f'Clipped noisy count num>0: {np.sum(counts > 0)}')
    logging.info(f'Clipped largest noisy counters: {np.flip(np.sort(counts.flatten()))[:50]}')
    counts = sanity_check(counts)
    return counts, clean_count


    

