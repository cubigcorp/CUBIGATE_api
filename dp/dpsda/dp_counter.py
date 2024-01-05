import faiss
import logging
import numpy as np
from collections import Counter
from typing import Dict, Optional
import torch
from dpsda.agm import get_sigma

def revival(counts: np.ndarray, loser_filter: np.ndarray, synthetic_features: np.ndarray, dim: int, index: faiss.Index):
    logging.info("Losers' revival started")
    counts = counts.flatten()
    sorted_idx = np.flip(np.argsort(counts))
    winner_idx = [idx for idx in sorted_idx if counts[idx] > 0]
    logging.info(f"Winners indices : {winner_idx}")

    shares = counts[winner_idx]
    logging.info(f"Winners' shares: {shares}")
    logging.info(f"Total vote: {sum(shares)}")
    loser_idx = loser_filter.flatten().nonzero()[0]
    losers = synthetic_features[loser_idx].reshape((-1, dim, synthetic_features.shape[-1]))
    winners = synthetic_features[winner_idx]

    logging.info(f"Counting votes for {len(losers)} losers")
    loser_counts = []
    for loser in losers:
        index.add(loser)
        _, ids = index.search(winners, k=1)
        weights = get_weights(ids.flatten(), shares)
        count = get_count_flat(ids, dim, verbose=0, weights=weights)
        loser_counts.append(count)
        index.reset()
    loser_counts = np.stack(loser_counts)
    logging.info(f"Counts for losers: {loser_counts}")

    loser_idx = loser_idx.reshape((-1, dim))
    loser_idx = [idx[0] // dim for idx in loser_idx]
    counts = counts.reshape((-1, dim))
    counts[loser_idx] = loser_counts
    return counts


def get_weights(ids: np.ndarray, share: np.ndarray) -> Dict:
    weights = dict.fromkeys(np.unique(ids), 0)

    for idx in range(ids.shape[0]):
        weights[ids[idx]] += share[idx]
    return weights
        


def get_count_flat(ids: np.ndarray, num_candidate: int, verbose: int, weights: Optional[np.ndarray]=None) -> np.ndarray:
    counter = Counter(list(ids.flatten()))
    count = np.zeros(shape=num_candidate)
    for k in counter:
        vote = counter[k] if weights is None else weights[k]
        count[k % num_candidate] += vote
        if verbose == 1:
            logging.debug(f"count[{k}]: {count[k]}")
    if verbose == 1:
        logging.info(f'Clean count sum: {np.sum(count)}')
        logging.info(f'Clean count num>0: {np.sum(count > 0)}')
        logging.info(f'Largest clean counters: {sorted(count)[::-1][:50]}')
    count = np.asarray(count)
    return count


def get_count_stack(private_features: np.ndarray, synthetic_features: np.ndarray, dim: int, index: faiss.Index, k: int) -> np.ndarray:
    indices = np.arange(synthetic_features.shape[0]).reshape((-1, dim))
    counts = []
    for idx in indices :
        index.add(synthetic_features[idx])
        _, ids = index.search(private_features, k=k)
        counter = Counter(list(ids.flatten()))
        count = np.zeros(shape=dim)
        for c in counter:
            count[c % dim] = counter[c]
        count = np.asarray(count)
        counts.append(count)
        index.reset()
    counts = np.stack(counts)
    logging.info(f'count sum: {np.sum(counts)}')
    logging.info(f'count num>0: {np.sum(counts > 0)}')
    logging.info(f'largest counters: {np.flip(np.sort(counts.flatten()))[:50]}')
    return counts


def add_noise(counts: np.ndarray, epsilon: float, delta: float, num_nearest_neighbor: int, noise_multiplier: float, rng: np.random.Generator, dim: int = 0) -> np.ndarray:
    if epsilon is not None:
        sigma = get_sigma(epsilon=epsilon, delta=delta, GS=1)
        logging.info(f'calculated sigma: {sigma}')
        counts += (rng.normal(scale=sigma, size=len(counts))) * np.sqrt(num_nearest_neighbor)
    else:
        counts += (rng.normal(size=len(counts)) * np.sqrt(num_nearest_neighbor)
                * noise_multiplier)

    if dim > 0 :
        counts = counts.reshape((-1, dim))

    return counts


def sanity_check(counts: np.ndarray) -> bool:
    for idx in range(counts.shape[0]):
        if np.all(counts[idx] == 0):
            return False
    return True


def get_losers(counts: np.ndarray, loser_lower_bound: float, dim: int) -> np.ndarray:
    logging.info("Counting losers")
    loser_idx = [[*range(idx * dim, (idx + 1) * dim)] for idx in range(counts.shape[0]) if np.sum(counts[idx]) < loser_lower_bound]
    logging.info(f"Total losers: {len(loser_idx)}")
    loser_idx = np.array(loser_idx).flatten()
    counts = counts.flatten()
    losers = np.array([idx in loser_idx for idx in range(counts.shape[0])]).reshape((-1, dim)) 
    return losers


def diversity_check(losers: np.ndarray, diversity: float, num_samples: int, diversity_lower_bound: float) -> bool:
    logging.info("Checking diversity")
    if diversity < diversity_lower_bound: return False
    updated_div = diversity - losers.sum() / num_samples
    return updated_div > diversity_lower_bound


def dp_nn_histogram(synthetic_features: np.ndarray, private_features: np.ndarray, epsilon: float, delta: float,
                    noise_multiplier: float, rng: np.random.Generator, num_nearest_neighbor: int, mode: str,
                    threshold: float, dim: int, diversity: float, diversity_lower_bound: float = 0.0,
                    loser_lower_bound: float = 0.0, first_vote_only: bool = True, num_packing=1):
    # public_features shape: (Nsyn * lookahead, embedding) if direct_variate
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
        index = faiss.index_cpu_to_gpu(faiss_res, 0, index)

    logging.debug(f"public_features:\n{synthetic_features}")
    logging.info("Counting votes from private samples")

    index.add(synthetic_features)
    logging.info(f'Number of samples in index: {index.ntotal}')

    _, ids = index.search(private_features, k=num_nearest_neighbor)
    counts = get_count_flat(ids, synthetic_features.shape[0], verbose=1)
    clean_count = counts.copy()
    counts = add_noise(counts, epsilon, delta, num_nearest_neighbor, noise_multiplier, rng, dim)
    logging.info(f'Noisy count sum: {np.sum(counts)}')
    logging.info(f'Noisy count num>0: {np.sum(counts > 0)}')
    logging.info(f'Largest noisy counters: {np.flip(np.sort(counts.flatten()))[:50]}')
    counts = np.clip(counts, a_min=threshold, a_max=None)
    counts = counts - threshold
    logging.info(f'Clipped noisy count sum: {np.sum(counts)}')
    logging.info(f'Clipped noisy count num>0: {np.sum(counts > 0)}')
    logging.info(f'Clipped largest noisy counters: {np.flip(np.sort(counts.flatten()))[:50]}')

    if dim > 0:
        index.reset()
        losers = get_losers(counts, loser_lower_bound, dim)
        if losers.sum() == 0:
            return counts, clean_count, losers
        first_vote_only = diversity_check(losers, diversity, synthetic_features.shape[0], diversity_lower_bound) if first_vote_only else first_vote_only
        if first_vote_only:
            counts[losers] = 0
            return counts, clean_count, losers
        counts = revival(
            counts=counts,
            synthetic_features=synthetic_features,
            loser_filter=losers,
            dim=dim,
            index=index)
        assert sanity_check(counts)
    else:
        losers = np.array([False for _ in counts])  # (Nsyn)
        losers = np.expand_dims(losers, axis=1)  # (Nsyn, 1)

    return counts, clean_count, losers


def nn_histogram(synthetic_features, private_features, dim: int, num_packing=1, num_nearest_neighbor=1, mode='L2'):
    # public_features shape: (Nsyn * lookahead, embedding) if direct_variate
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
        index = faiss.index_cpu_to_gpu(faiss_res, 0, index)

    logging.debug(f"public_features:\n{synthetic_features}")
    logging.info("Counting votes from private samples")

    counts = get_count_stack(private_features=private_features, synthetic_features=synthetic_features, dim=dim, index=index, k=num_nearest_neighbor)
    losers = np.full(counts.shape, False)

    return counts, losers
