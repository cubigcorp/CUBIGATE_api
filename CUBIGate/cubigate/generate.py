import logging
import os
import numpy as np
import imageio
from torchvision.utils import make_grid
import torch
from dpsda.logging import setup_logging
from dpsda.data_loader import load_data, load_samples
from dpsda.feature_extractor import extract_features
from dpsda.metrics import make_fid_stats
from dpsda.metrics import compute_fid
from dpsda.dp_counter import dp_nn_histogram
from dpsda.arg_utils import str2bool
from apis import get_api_class_from_name
from dpsda.data_logger import log_samples
from dpsda.tokenizer import tokenize
from dpsda.agm import get_epsilon

class CubigDPGenerator():
    def __init__(
        self, 
        api: str,
        model_checkpoint: str,
        feature_extractor: str,
        result_folder: str,
        tmp_folder: str,
        modality: str,
        data_folder: str,
        ) -> None:
        pass

    def train(
        self,
        condition_guidance_scale: float,
        inference_steps: int,
        batch_size: int,
        variation_strength: float,
        variation_degree_schedule: str,
        count_threshold: float,
        image_size: str,
        
        num_initial_samples: int,
        initial_prompt: str,
        ):
        pass

    def generate(
        self,
        base_data: str,
        image_size: str,
        num_samples: int,
        suffix: str
    ):
        pass