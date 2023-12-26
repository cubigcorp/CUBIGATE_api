import faiss
import logging
import numpy as np
from collections import Counter
import torch
from dpsda.agm import get_sigma

def revival(counts: np.ndarray, synthetic_features: np.ndarray, dim: int, total_vote: int, top_winner_ratio: float, index: faiss.Index) -> np.ndarray:
    logging.info("Losers' revival started")
    loser_idx = [[*range(idx * dim, (idx + 1) * dim)] for idx in range(counts.shape[0]) if np.all(counts[idx] == 0)]
    counts = counts.flatten()
    sorted_idx = np.flip(np.argsort(counts))
    winner_idx = [idx for idx in sorted_idx if counts[idx] > 0]
    top_winner = np.round(top_winner_ratio * total_vote).astype(int)
    logging.info(f"Selected {top_winner} winners out of {len(winner_idx)} winners")
    winner_idx = winner_idx[:top_winner]
    logging.info(f"Selected winners indices : {winner_idx}")

    shares = np.round(total_vote * counts[winner_idx] / sum(counts[winner_idx]))
    winner_idx = np.concatenate([np.repeat(winner_idx[idx], shares[idx]) for idx in range(len(winner_idx))])
    logging.info(f"Winners' share:{shares}")
    logging.info(f"Total vote: {sum(shares)}")
    losers = synthetic_features[loser_idx].reshape((-1, dim, synthetic_features.shape[-1]))
    winners = synthetic_features[winner_idx]

    logging.info(f"Counting votes for {len(losers)} losers")

    loser_counts = []
    for loser in losers:
        index.add(loser)
        _, ids = index.search(winners, k=1)
        count = get_count(ids, dim, verbose=0)
        loser_counts.append(count)
        index.reset()
    loser_counts = np.stack(loser_counts)
    logging.info(f"Counts for losers: {loser_counts}")

    loser_idx = [idx[0] // 3 for idx in loser_idx]
    counts = counts.reshape((-1, dim))
    counts[loser_idx] = loser_counts
    return counts



def get_count(ids: np.ndarray, num_candidate: int, verbose: int = 1) -> np.ndarray:
    counter = Counter(list(ids.flatten()))
    count = np.zeros(shape=num_candidate)
    for k in counter:
        count[k % num_candidate] += counter[k]
        if verbose == 1:
            logging.debug(f"count[{k}]: {count[k]}")
    if verbose == 1:
        logging.info(f'Clean count sum: {np.sum(count)}')
        logging.info(f'Clean count num>0: {np.sum(count > 0)}')
        logging.info(f'Largest clean counters: {sorted(count)[::-1][:50]}')
    count = np.asarray(count)
    return count


def add_noise(counts: np.ndarray, epsilon: float, delta: float, num_nearest_neighbor: int, noise_multiplier: float, dim: int = 0) -> np.ndarray:
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


def sanity_check(counts: np.ndarray) -> bool:
    for idx in range(counts.shape[0]):
        if np.all(counts[idx] == 0):
            return False
    return True


def dp_nn_histogram(synthetic_features, private_features, epsilon: float, delta: float, 
                    noise_multiplier, num_packing=1, num_nearest_neighbor=1, mode='L2',
                    threshold=0.0, t=None, result_folder: str=None, dim: int = 0, top_winner_ratio: float = 0.1):
    # public_features shape: (Nsyn * lookahead, embedding) if direct_variate
    #                        (Nsyn, embedding) otherwise
    np.set_printoptions(100)
    assert synthetic_features.shape[0] % num_packing == 0
    num_true_public_features = synthetic_features.shape[0] // num_packing
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
    counts = get_count(ids, synthetic_features.shape[0])
    clean_count = counts.copy()
    counts = add_noise(counts, epsilon, delta, num_nearest_neighbor, noise_multiplier, dim)
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
        counts = revival(
            counts=counts,
            synthetic_features=synthetic_features,
            dim=dim,
            total_vote=private_features.shape[0],
            top_winner_ratio=top_winner_ratio,
            index=index)
        assert sanity_check(counts)

    return counts, clean_count


    

