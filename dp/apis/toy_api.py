import numpy as np
from tqdm import tqdm
from .api import API
from dpsda.experiment import get_samples

from typing import Optional, Union


class ToyAPI(API):
    def __init__(self,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(2024)

    @staticmethod
    def command_line_parser():
        parser = super(
            ToyAPI, ToyAPI).command_line_parser()
        return parser

    def random_sampling(self, num_samples, size: int, prompts: str):

        samples = get_samples(prompts[0], self.rng, num_samples, size=size)
        return_prompts = np.repeat(prompts[0], num_samples)

        return samples, np.array(return_prompts)

    def variation(self, samples, additional_info,
                        num_variations_per_sample, size, variation_degree, t=None, candidate=True, demo_samples: Optional[np.ndarray]=None, sample_weight: float = 1.0):
        variations = []
        for _ in tqdm(range(num_variations_per_sample)):
            sub_variations = self._variation(
                samples=samples,
                size=size,
                additional_info=additional_info,
                variation_degree=variation_degree,
                demo_samples=demo_samples,
                sample_weight=sample_weight)
            variations.append(sub_variations)
        return np.stack(variations, axis=1)

    def _variation(self, samples: np.ndarray, size: str, additional_info: np.ndarray, variation_degree: Union[np.ndarray, float], demo_samples: Optional[np.ndarray] = None, sample_weight: float = 1.0):
        x_dim, y_dim = list(map(int, size.split('x')))
        coordinates, colors = np.array_split(samples, 2, axis=1)
        sample_variate = np.zeros_like(coordinates)
        demo_variate = np.zeros_like(coordinates)
        if sample_weight > 0:
            # sample-based
            degrees = variation_degree if isinstance(variation_degree, np.ndarray) else np.repeat(variation_degree, samples.shape[0])
            noises = np.stack([self.rng.normal(0, (1 + deg), 2) for deg in degrees])
            sample_variate = coordinates + noises
        if sample_weight < 1:
            # demonstration-based
            demo_samples = np.delete(demo_samples, 2, axis=1)
            demo_variate = np.mean(demo_samples, axis=1)

        variations = np.clip(sample_weight * sample_variate + (1 - sample_weight) * demo_variate, a_min=0, a_max=x_dim)
        variations = np.concatenate((variations, colors), axis=1)
        return variations
