import argparse
from dp.apis import get_api_class_from_name
from dp.dpsda.logging import setup_logging
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--api',
        type=str,
        required=True,
        choices=['DALLE', 'stable_diffusion', 'improved_diffusion'],
        help='Which foundation model API to use')
    parser.add_argument(
        '--data_checkpoint_path',
        type=str,
        default="",
        help='Path to the data checkpoint')
    parser.add_argument(
    	'--device',
        type=int,
        required=True)
    parser.add_argument(
        '--result_folder',
        type=str,
        default='result',
        help='Folder for storing results')
    parser.add_argument(
        '--image_size',
        type=str,
        default='1024x1024',
        help='Size of generated images in the format of HxW')
    parser.add_argument(
        '--variation_degree',
        type=float,
        default=0.92,
        help='Variation degree at each iteration')
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10)
    parser.add_argument(
        '--variation_per_image',
        required=False,
        default=1,
        type=int
    )
    parser.add_argument(
        '--data_dir',
        required=True,
        type=str
    )
    parser.add_argument(
        '--suffix',
        required=False,
        default='',
        type=str
    )
    parser.add_argument(
        '--prefix',
        required=False,
        default='',
        type=str
    )
    parser.add_argument(
        '--save_single_org',
        action='store_true',
        required=False
    )
    args, api_args = parser.parse_known_args()
    api_class = get_api_class_from_name(args.api)
    api = api_class.from_command_line_args(api_args)

    return args, api

def load_samples(path):
    data = np.load(path)
    samples = data['samples']
    additional_info = data['additional_info']
    return samples, additional_info

def save_original(data_dir: str, num: int, result_dir: str, suffix: str, save_single: bool, width: int, height: int, rows: int=2, cols: int=5) -> None:
    samples = []
    for img in os.listdir(data_dir):
        full = os.path.join(data_dir, img)
        sample = cv2.imread(full)
        sample = cv2.resize(sample, dsize=(width, height))
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        samples.append(sample)
        if save_single:
            plt.imsave(os.path.join(result_dir, f'original_{len(samples)}_{suffix}.jpg'), sample)
        if len(samples) >= num:
            break
    samples = np.array(samples)

    samples = samples.reshape(rows, cols, width, height, -1).swapaxes(1, 2).reshape(height*rows, width*cols, -1)
    plt.imsave(os.path.join(result_dir, f"original_{suffix}.jpg"), samples)

def save_single_sample(sample: np.ndarray, dir: str, idx: int, suffix: str=None, prefix: str=None) -> None:
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.imsave(os.path.join(dir, f"{prefix}_{idx}_{suffix}.jpg"), sample)

def save_samples(packed_samples: np.ndarray, dir: str, suffix: str, prefix: str, rows: int=2, cols: int=5) -> None:
    if not os.path.exists(dir):
        os.makedirs(dir)

    width = packed_samples[0].shape[0]
    height= packed_samples[0].shape[1]
    samples = packed_samples.reshape(rows, cols, width, height, -1).swapaxes(1, 2).reshape(height*rows, width*cols, -1)
    plt.imsave(os.path.join(dir, f"{prefix}_grid_{suffix}.jpg"), samples)

if __name__ == "__main__":
    args, api = argument()
    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)
    setup_logging(os.path.join(args.result_folder, 'log.log'))
    logging.info(f'config: {args}')
    logging.info(f'API config: {api.args}')
    samples, additional_info = load_samples(args.data_checkpoint_path)
    rng = np.random.default_rng(2023)
    indices = rng.choice(len(samples)-1, args.num_samples)
    samples = samples[indices]
    additional_info = additional_info[indices]
    width = int(args.image_size.split('x')[0])
    height = int(args.image_size.split('x')[1])
    total_sample = args.num_samples * args.variation_per_image
    cols = 10 if total_sample > 10 else 5
    rows = total_sample // cols + (total_sample % cols > 0)
    logging.info('Running image variation')
    packed_samples = []
    for i, sample in enumerate(samples):
        for v in range(args.variation_per_image):
            sample = sample.reshape(1, width, height, -1)
            prompt = additional_info[i].reshape(1)
            sample = api.variation(
                images=sample,
                additional_info=prompt,
                num_variations_per_image=1,
                size=args.image_size,
                variation_degree=args.variation_degree)
            sample = sample.reshape(width, height, -1)
            save_single_sample(sample, args.result_folder, i * args.variation_per_image + v, args.suffix, args.prefix)
            packed_samples.append(sample)
    packed_samples = np.array(packed_samples).reshape(total_sample, width, height, -1)
    save_samples(packed_samples, args.result_folder, args.suffix, args.prefix, rows, cols)
    save_original(args.data_dir, total_sample, args.result_folder, args.suffix, args.save_single_org, width, height, rows, cols)