import logging
import os
import numpy as np
import torch
from typing import Optional, List
from dp.utils.logging import setup_logging
from dp.data_loader import load_data, load_samples
from dp.extractors.feature_extractor import extract_features
from dp.dp_counter import dp_nn_histogram
from dp.apis import get_api_class_from_name
from dp.data_logger import log_samples, log_count, visualize
from dp.agm import get_epsilon

class CubigDPGenerator():
    def __init__(
        self, 
        api: str,
        model_checkpoint: str,
        feature_extractor: str,
        result_folder: str,
        tmp_folder: str,
        data_folder: str,
        data_loading_batch_size: int,
        feature_extractor_batch_size: int,
        org_img_size: int,
        conditional: bool,
        num_org_data: int,
        ) -> None:
        """
        DP synthetic data generator

        Parameters
        ----------
        api:
            Which foundation model API to use
        result_folder:
            Folder for storing results
        tmp_folder:
            Temporary folder for storing intermediate results
        data_folder:
            Folder for original data
        data_loading_batch_size:
            Batch size for loading the original data
        feature_extractor_batch_size:
            Batch size for feature extractor
        org_img_size:
            Size of original images
        conditional:
            Whether to generate class labels
        num_org_img:
            Number of original data
        feature_extractor:
            Name of feature extractor to use
        """
        # 0-a. Make result directory
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        # 0-b. Set up logging
        setup_logging(os.path.join(result_folder, 'log.log'))

        # 0-c. Declare class variables
        self.api_class = get_api_class_from_name(api)  # Name of the foundation model API
        self.result_folder = result_folder
        self.data_folder = data_folder
        self.data_loading_batch_size = data_loading_batch_size
        self.org_img_size = org_img_size
        self.conditional = conditional
        self.num_org_data = num_org_data
        self.feature_extractor = feature_extractor
        self.feature_extractor_batch_size = feature_extractor_batch_size
        self.tmp_folder = tmp_folder
        

    def train(
        self,
        data_checkpoint_path: Optional[str],
        data_checkpoint_step: Optional[int],
        initial_prompt: Optional[str],
        num_samples_schedule: List[int],
        variation_degree_schedule: List[int],
        lookahead_degree: int,
        img_size: str,
        epsilon: float,
        delta: float,
        count_threshold: int,
        plot_images: bool=False,
        nn_mode: str='L2',
        **api_args):
        """
        Learn the distribution

        Parameters
        ----------
        data_checkpoint_path:
            Path to the data checkpoint
        data_checkpoint_step:
            Iteration of the data checkpoint
        initial_prompt:
            Initial prompt for image generation. It can be specified
            multiple times to provide a list of prompts. If the API accepts
            prompts, the initial samples will be generated with these prompts
        num_samples_schedule:
            Number of samples to generate at each iteration
        variation_degree_schedule:
            Variation degree at each iteration
        lookahead_degree:
            Lookahead degree for computing distances between private and generated images
        img_size:
            Target size of image to generate
        epsilon:
            Privacy parameter, for each step of variations
        delta:
            Privacy parameter
        count_threshold:
            Threshold for DP NN histogram
        plot_images:
            Whether to save generated images in PNG files
        nn_mode:
            Which distance metric to use in DP NN histogram
            
        """
        # 1. Set up API instance
        self.api = self.api_class.from_command_line_args(**api_args)

        # 2. Load original data
        all_private_samples, all_private_labels = load_data(
            data_dir=self.data_folder,
            batch_size=self.data_loading_batch_size,
            image_size=self.org_img_size,
            class_cond=self.conditional,
            num_private_samples=self.num_org_data,
            model=self.feature_extractor)

        private_classes = list(sorted(set(list(all_private_labels))))
        private_num_classes = len(private_classes)
        logging.info(f'Private_num_classes: {private_num_classes}')

        # 3. Extract features of the original data
        logging.info('Extracting features')
        all_private_features = extract_features(
            data=all_private_samples,
            tmp_folder=self.tmp_folder,
            model_name=self.feature_extractor,
            res=self.org_img_size,
            batch_size=self.feature_extractor_batch_size)
        logging.info(f'all_private_features.shape: {all_private_features.shape}')

        # 4-a. Load data checkpoint if any
        if data_checkpoint_path != '':
            logging.info(
                f'Loading data checkpoint from {data_checkpoint_path}')
            samples, additional_info = load_samples(data_checkpoint_path)
            if data_checkpoint_step < 0:
                raise ValueError('data_checkpoint_step should be >= 0')
            start_t = data_checkpoint_step + 1
        # 4-b. Generate initial population
        else:
            logging.info('Generating initial samples')
            samples, additional_info = self.api.random_sampling(
                prompts=initial_prompt,
                num_samples=num_samples_schedule[0],
                size=img_size)
            logging.info(f"Generated initial samples: {len(samples)}")
            log_samples(
                samples=samples,
                additional_info=additional_info,
                folder=f'{self.result_folder}/{0}',
                plot_samples=plot_images)
            if data_checkpoint_step >= 0:
                logging.info('Ignoring data_checkpoint_step')
            start_t = 1

        # 5. Calculate privacy parameters
        T = len(num_samples_schedule)
        if epsilon is not None:
            total_epsilon = get_epsilon(epsilon, T)
            logging.info(f"Expected total epsilon: {total_epsilon:.2f}")
            logging.info(f"Expected privacy cost per t: {epsilon:.2f}")

        # 6. Start learning
        for t in range(start_t, T):
            logging.info(f't={t}')
            assert samples.shape[0] % private_num_classes == 0  #Check if equal number of samples per class
            num_samples_per_class = samples.shape[0] // private_num_classes

            # 7. Prepare to compute histogram
            if lookahead_degree == 0:  # 7-a. Only current population is needed if lookahead == 0 
                packed_samples = np.expand_dims(samples, axis=1)
            else:  # 7-b. Sample is needed As many as lookahead degree
                logging.info('Running image variation')
                packed_samples = self.api.variation(
                    samples=samples,
                    additional_info=additional_info,
                    num_variations_per_sample=lookahead_degree,
                    size=img_size,
                    variation_degree=variation_degree_schedule[t])
            # 7-c. Get features
            packed_features = []
            logging.info('Running feature extraction')
            for i in range(packed_samples.shape[1]):
                sub_packed_features = extract_features(
                    data=packed_samples[:, i],
                    tmp_folder=self.tmp_folder,
                    model_name=self.feature_extractor,
                    res=self.org_img_size,
                    batch_size=self.feature_extractor_batch_size)
                logging.info(
                    f'sub_packed_features.shape: {sub_packed_features.shape}')
                packed_features.append(sub_packed_features)
            packed_features = np.mean(packed_features, axis=0)

            # 8. Compute histogram
            logging.info('Computing histogram')
            count = []
            for class_i, class_ in enumerate(private_classes):
                sub_count, sub_clean_count = dp_nn_histogram(
                    public_features=packed_features[
                        num_samples_per_class * class_i:
                        num_samples_per_class * (class_i + 1)],
                    private_features=all_private_features[
                        all_private_labels == class_],
                    epsilon=epsilon,
                    delta=delta,
                    mode=nn_mode,
                    threshold=count_threshold,
                    result_folder=self.result_folder)
                log_count(
                    sub_count,
                    sub_clean_count,
                    f'{self.result_folder}/{t}/count_class{class_}.npz')
                count.append(sub_count)
            count = np.concatenate(count)
            for class_i, class_ in enumerate(private_classes):
                visualize(
                    samples=samples[
                        num_samples_per_class * class_i:
                        num_samples_per_class * (class_i + 1)],
                    packed_samples=packed_samples[
                        num_samples_per_class * class_i:
                        num_samples_per_class * (class_i + 1)],
                    count=count[
                        num_samples_per_class * class_i:
                        num_samples_per_class * (class_i + 1)],
                    folder=f'{self.result_folder}/{t}',
                    suffix=f'class{class_}')

            # 9. Select parents
            logging.info('Generating new indices')
            assert num_samples_schedule[t] % private_num_classes == 0
            new_num_samples_per_class = (
                num_samples_schedule[t] // private_num_classes)
            new_indices = []
            for class_i in private_classes:
                sub_count = count[
                    num_samples_per_class * class_i:
                    num_samples_per_class * (class_i + 1)]
                sub_new_indices = np.random.choice(
                    np.arange(num_samples_per_class * class_i,
                            num_samples_per_class * (class_i + 1)),
                    size=new_num_samples_per_class,
                    p=sub_count / np.sum(sub_count))
                new_indices.append(sub_new_indices)
            new_indices = np.concatenate(new_indices)
            new_samples = samples[new_indices]
            new_additional_info = additional_info[new_indices]
            logging.debug(f"new_indices: {new_indices}")

            # 10. Generate next generation
            logging.info('Generating new samples')
            new_new_samples = self.api.variation(
                samples=new_samples,
                additional_info=new_additional_info,
                num_variations_per_sample=1,
                size=img_size,
                variation_degree=variation_degree_schedule[t],
                t=t)
            new_new_samples = np.squeeze(new_new_samples, axis=1)
            new_new_additional_info = new_additional_info

            samples = new_new_samples
            additional_info = new_new_additional_info

            log_samples(
                samples=samples,
                additional_info=additional_info,
                folder=f'{self.result_folder}/{t}',
                plot_samples=plot_images)
            logging.info(f"Privacy cost so far: {get_epsilon(epsilon, t):.2f}")


    def generate(
        self,
        base_data: str,
        img_size: str,
        num_samples: int,
        variation_degree: float,
        plot_images: bool,
        **api_args
    ):
        """
        Generate images based on the distribution learned

        Parameters
        ----------
        base_data:
            Set of generated data as the learned distribution
        img_size:
            Target size of image to generate
        num_samples:
            Number of samples to generate
        variation_degree:
            Variation degree
        plot_images:
            Whether to save generated images in PNG files
        """
        # 1. Make sure it has API instance
        if not hasattr(self, 'api'):
            self.api = self.api_class.from_command_line_args(**api_args)

        # 2. Load base data
        samples, additional_info = load_samples(base_data)

        # Generate samples as variations of base data
        assert num_samples % len(samples) == 0
        variation_per_image = num_samples // len(samples)
        width = int(img_size.split('x')[0])
        height = int(img_size.split('x')[1])
        for i, sample in enumerate(samples):
            for v in range(variation_per_image):
                sample = sample.reshape(1, width, height, -1)
                prompt = additional_info[i].reshape(1)
                sample = self.api.variation(
                    samples=sample,
                    additional_info=prompt,
                    num_variations_per_sample=1,
                    size=img_size,
                    variation_degree=variation_degree)
                log_samples(
                    samples=samples,
                    additional_info=additional_info,
                    folder=f'{self.result_folder}/gen',
                    plot_samples=plot_images)
