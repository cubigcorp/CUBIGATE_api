import faiss
import logging
import numpy as np
from collections import Counter
from typing import Dict, Optional
import torch
from dpsda.agm import get_sigma

def revival(counts: np.ndarray, counts_1st_idx: np.ndarray, loser_filter: np.ndarray, synthetic_features: np.ndarray, num_candidate: int, index: faiss.Index):
    logging.info("Losers' revival started")
    shares = counts[~loser_filter]
    winner_filter = np.concatenate((~loser_filter, np.full((counts.shape[0], num_candidate -1), False)), axis=1)

    logging.info(f"Winners indices : {np.where(winner_filter)[0]}")
    logging.info(f"Winners' shares: {shares}")
    logging.info(f"Total vote: {sum(shares)}")

    winners = synthetic_features[winner_filter]
    synthetic_features = synthetic_features.reshape((-1, num_candidate) + synthetic_features.shape[2:])

    logging.info(f"Counting votes for losers")
    for idx in range(counts.shape[0]):
        if loser_filter[idx]:
            index.add(synthetic_features[idx])
            _, ids = index.search(winners, k=1)
            weights = get_weights(ids.flatten(), shares)
            count = get_count_flat(ids, num_candidate, verbose=0, weights=weights)
            count_1st_idx = np.flip(np.argsort(count))[0]
            counts[idx] = count[count_1st_idx]
            counts_1st_idx[idx] = count_1st_idx
        index.reset()

    return counts, counts_1st_idx


def get_weights(ids: np.ndarray, share: np.ndarray) -> Dict:
    weights = dict.fromkeys(np.unique(ids), 0)

    for idx in range(ids.shape[0]):
        weights[ids[idx]] += share[idx]
    return weights
        


def get_count_flat(ids: np.ndarray, dim: int, verbose: int, weights: Optional[np.ndarray]=None) -> np.ndarray:
    counter = Counter(list(ids.flatten()))
    count = np.zeros(shape=dim)
    for k in counter:
        vote = counter[k] if weights is None else weights[k]
        count[k % dim] += vote
        if verbose == 1:
            logging.debug(f"count[{k}]: {count[k]}")
    if verbose == 1:
        logging.info(f'Clean count sum: {np.sum(count)}')
        logging.info(f'Clean count num>0: {np.sum(count > 0)}')
        logging.info(f'Largest clean counters: {sorted(count)[::-1][:50]}')
    count = np.asarray(count)
    return count


def get_count_stack(private_features: np.ndarray, synthetic_features: np.ndarray, num_candidate: int, index: faiss.Index, k: int) -> np.ndarray:
    indices = np.arange(synthetic_features.shape[0]).reshape((-1, num_candidate))
    counts = []
    for i, idx in enumerate(indices) :
        index.add(synthetic_features[idx])
        _, ids = index.search(private_features, k=k)
        counter = Counter(list(ids.flatten()))
        count = np.zeros(shape=num_candidate)
        for c in counter:
            count[c % num_candidate] = counter[c]
        count = np.asarray(count)
        counts.append(count)
        index.reset()
    counts = np.stack(counts)
    logging.info(f'count sum: {np.sum(counts)}')
    logging.info(f'count num>0: {np.sum(counts > 0)}')
    logging.info(f'largest counters: {np.flip(np.sort(counts.flatten()))[:50]}')
    return counts


def add_noise(counts: np.ndarray, epsilon: float, delta: float, num_nearest_neighbor: int, noise_multiplier: float, rng: np.random.Generator, num_candidate: int = 0) -> np.ndarray:
    if epsilon is not None:
        sigma = get_sigma(epsilon=epsilon, delta=delta, GS=1)
        logging.info(f'calculated sigma: {sigma}')
        counts += (rng.normal(scale=sigma, size=len(counts))) * np.sqrt(num_nearest_neighbor)
    else:
        counts += (rng.normal(size=len(counts)) * np.sqrt(num_nearest_neighbor)
                * noise_multiplier)

    if num_candidate > 0 :
        counts = counts.reshape((-1, num_candidate))

    return counts


def sanity_check(counts: np.ndarray) -> bool:
    return len(counts) - np.count_nonzero(counts) == 0


def get_losers(counts: np.ndarray, loser_lower_bound: float, max_vote: int) -> np.ndarray:
    logging.info("Counting losers")
    losers = counts <= loser_lower_bound * max_vote
    logging.info(f"Total losers: {losers.sum()}")
    return losers


def diversity_check(losers: np.ndarray, diversity: float, num_samples: int, diversity_lower_bound: float) -> bool:
    logging.info("Checking diversity")
    updated_div = diversity - losers.sum() / num_samples
    return updated_div > diversity_lower_bound


def dp_nn_histogram(synthetic_features: np.ndarray, private_features: np.ndarray, epsilon: float, delta: float,
                    noise_multiplier: float, rng: np.random.Generator, num_nearest_neighbor: int, mode: str,
                    threshold: float, num_candidate: int, diversity: float, diversity_lower_bound: float = 0.0,
                    loser_lower_bound: float = 0.0, first_vote_only: bool = True, num_packing=1, device: int = 0):
    # public_features shape: (Nsyn * candidate, embedding) if direct_variate
    #                        (Nsyn, embedding) otherwise
    np.set_printoptions(precision=3)
    assert synthetic_features.shape[0] % num_packing == 0

    faiss_res = faiss.StandardGpuResources()
    if mode == 'L2':
        index = faiss.IndexFlatL2(synthetic_features.shape[-1])
    elif mode == 'IP':
        index = faiss.IndexFlatIP(synthetic_features.shape[-1])
    elif mode == 'cosine':
        index = faiss.IndexFlatIP(synthetic_features.shape[-1])
        faiss.normalize_L2(synthetic_features)
        faiss.normalize_L2(private_features)
    else:
        raise Exception(f'Unknown mode {mode}')
    if torch.cuda.is_available():
        index = faiss.index_cpu_to_gpu(faiss_res, device, index)

    logging.debug(f"public_features:\n{synthetic_features}")
    logging.info("Counting votes from private samples")

    index.add(synthetic_features)
    logging.info(f'Number of samples in index: {index.ntotal}')

    _, ids = index.search(private_features, k=num_nearest_neighbor)
    counts = get_count_flat(ids, synthetic_features.shape[0], verbose=1)
    clean_count = counts.copy()
    if epsilon > 0:
        counts = add_noise(counts, epsilon, delta, num_nearest_neighbor, noise_multiplier, rng, num_candidate)
    logging.info(f'Noisy count sum: {np.sum(counts)}')
    logging.info(f'Noisy count num>0: {np.sum(counts > 0)}')
    logging.info(f'Largest noisy counters: {np.flip(np.sort(counts.flatten()))[:50]}')
    counts = np.clip(counts, a_min=threshold, a_max=None)
    counts = counts - threshold
    logging.info(f'Clipped noisy count sum: {np.sum(counts)}')
    logging.info(f'Clipped noisy count num>0: {np.sum(counts > 0)}')
    logging.info(f'Clipped largest noisy counters: {np.flip(np.sort(counts.flatten()))[:50]}')

    if num_candidate > 0:
        index.reset()
        counts_1st_idx = np.flip(np.argsort(counts, axis=1), axis=1)[:, 0]
        counts = counts[np.arange(counts.shape[0]), counts_1st_idx].reshape((-1, 1))  # (Nsyn, 1)
        losers = get_losers(counts, loser_lower_bound, private_features.shape[0]).reshape((-1, 1))  # (Nsyn, 1)
        if losers.sum() == 0:
            return counts.flatten(), clean_count, losers, counts_1st_idx
        # first_vote_only = diversity_check(losers, diversity, counts.shape[0], diversity_lower_bound) if first_vote_only else first_vote_only
        # if first_vote_only:
        #     return counts.flatten(), clean_count, losers, counts_1st_idx
        synthetic_features = synthetic_features.reshape((counts.shape[0], num_candidate) + synthetic_features.shape[1:])  # (Nsyn, num_candidate, ~)
        counts, counts_1st_idx = revival(
            counts=counts,
            counts_1st_idx=counts_1st_idx,
            synthetic_features=synthetic_features,
            loser_filter=losers,
            num_candidate=num_candidate,
            index=index)
        assert sanity_check(counts)
    else:
        losers = counts > 0  # (Nsyn)
        losers = np.expand_dims(losers, axis=1)  # (Nsyn, 1)
        counts_1st_idx = np.zeros_like(counts)

    return counts.flatten(), clean_count, losers, counts_1st_idx


def nn_histogram(synthetic_features, private_features, num_candidate: int, num_packing=1, num_nearest_neighbor=1, mode='L2', device: int = 0):
    # public_features shape: (Nsyn * candidate, embedding) if direct_variate
    #                        (Nsyn, embedding) otherwise
    np.set_printoptions(precision=3)
    assert synthetic_features.shape[0] % num_packing == 0

    faiss_res = faiss.StandardGpuResources()
    if mode == 'L2':
        index = faiss.IndexFlatL2(synthetic_features.shape[-1])
    elif mode == 'IP':
        index = faiss.IndexFlatIP(synthetic_features.shape[-1])
    elif mode == 'cosine':
        index = faiss.IndexFlatIP(synthetic_features.shape[-1])
        faiss.normalize_L2(synthetic_features)
        faiss.normalize_L2(private_features)
    else:
        raise Exception(f'Unknown mode {mode}')
    if torch.cuda.is_available():
        index = faiss.index_cpu_to_gpu(faiss_res, device, index)

    logging.debug(f"public_features:\n{synthetic_features}")
    logging.info("Counting votes from private samples")

    counts = get_count_stack(private_features=private_features, synthetic_features=synthetic_features, num_candidate=num_candidate, index=index, k=num_nearest_neighbor)
    counts_1st_idx = np.flip(np.argsort(counts, axis=1), axis=1)[:, 0]
    counts = counts[np.arange(counts.shape[0]), counts_1st_idx].reshape((-1, 1))  # (Nsyn, 1)
    losers = np.full(counts.shape, False)

    return counts.flatten(), losers, counts_1st_idx
