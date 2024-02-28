import numpy as np
from tqdm.auto import tqdm
from .api import API
from dpsda.experiment import get_samples_out_bounding, get_bounding, get_samples_in_bounding

from typing import Optional, Union


class ToyAPI(API):
    def __init__(self,
                 private_ratio,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ratio = private_ratio
        self.rng = np.random.default_rng(2024)

    @staticmethod
    def command_line_parser():
        parser = super(
            ToyAPI, ToyAPI).command_line_parser()
        parser.add_argument(
            '--private_ratio',
            type=float,
            default=0.4
        )
        return parser

    def random_sampling(self, num_samples, size: int, prompts, generator):
        shape, y_position, x_position = prompts[0].split('_')
        bounding = get_bounding(y_position, x_position, size, self.ratio)
        # samples = get_samples_out_bounding(shape, self.rng, num_samples, bounding, size)
        samples = get_samples_in_bounding(shape, 2024, num_samples, bounding, size)
        return_prompts = np.repeat(prompts, num_samples)

        return samples, np.array(return_prompts)

    def variation(self, samples, additional_info,
                        num_variations_per_sample, size, variation_degree, t=None, candidate=True, demo_samples: Optional[np.ndarray] = None, demo_weights: Optional[np.ndarray] = None, sample_weight: float = 1.0):
        variations = []
        for _ in tqdm(range(num_variations_per_sample)):
            sub_variations = self._variation(
                samples=samples,
                additional_info=additional_info,
                variation_degree=variation_degree,
                demo_samples=demo_samples,
                demo_weights=demo_weights,
                sample_weight=sample_weight)
            variations.append(sub_variations)
        return np.stack(variations, axis=1)

    def _variation(self, samples: np.ndarray, additional_info: np.ndarray, variation_degree: Union[np.ndarray, float], demo_samples: Optional[np.ndarray] = None, demo_weights: Optional[np.ndarray] = None, sample_weight: float = 1.0):
        coordinates, colors = np.array_split(samples, 2, axis=1)
        sample_variate = np.zeros_like(coordinates)
        demo_variate = np.zeros_like(coordinates)
        if sample_weight > 0:
            # sample-based
            degrees = variation_degree if isinstance(variation_degree, np.ndarray) else np.repeat(variation_degree, samples.shape[0])
            noises = np.stack([self.rng.normal(0, deg, 2) for deg in degrees])
            sample_variate = coordinates + noises
        if sample_weight < 1:
            if demo_samples is None:
                sample_weight = 1.0
            else:
                # demonstration-based
                demo_variate = np.array([
                    np.zeros_like(demo_samples[0, 0, :2]) if np.all(demo_samples[i, :, :2] == 0)
                    else np.average(demo_samples[i, :, :2], axis=0, weights=demo_weights[i])
                    for i in range(demo_samples.shape[0])])

        variations = np.clip(sample_weight * sample_variate + (1 - sample_weight) * demo_variate, a_min=-1, a_max=1)
        variations = np.concatenate((variations, colors), axis=1)
        return variations
