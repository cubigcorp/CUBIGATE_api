import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from typing import Optional, Union
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

from .api import API
from dpsda.pytorch_utils import dev

def _round_to_uint8(image):
    return np.around(np.clip(image * 255, a_min=0, a_max=255)).astype(np.uint8)

class StableDiffusionAPI(API):
    def __init__(self, random_sampling_checkpoint,
                 random_sampling_guidance_scale,
                 random_sampling_num_inference_steps,
                 random_sampling_batch_size,
                 variation_checkpoint,
                 variation_guidance_scale,
                 variation_num_inference_steps,
                 variation_batch_size,
                 api_device,
                 lora,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch.cuda.empty_cache()
        self._random_sampling_checkpoint = random_sampling_checkpoint
        self._random_sampling_guidance_scale = random_sampling_guidance_scale
        self._random_sampling_num_inference_steps = \
            random_sampling_num_inference_steps
        self._random_sampling_batch_size = random_sampling_batch_size
        self._random_sampling_pipe =  AutoPipelineForText2Image.from_pretrained(
            self._random_sampling_checkpoint, torch_dtype=torch.float16)
        self._random_sampling_pipe.set_progress_bar_config(disable=True)
        if lora is not None:
            self._random_sampling_pipe.unet.load_attn_procs(lora)
        self._random_sampling_pipe.safety_checker = None
        self.device = f"cuda:{api_device}"
        self._random_sampling_pipe = self._random_sampling_pipe.to(self.device)

        self._variation_checkpoint = variation_checkpoint
        self._variation_guidance_scale = variation_guidance_scale
        self._variation_num_inference_steps = variation_num_inference_steps
        self._variation_batch_size = variation_batch_size

        if self._variation_checkpoint == self._random_sampling_checkpoint:
            # 동일한 checkpoint일 경우 재활용으로 메모리 절약
            self._variation_pipe = AutoPipelineForImage2Image.from_pipe(self._random_sampling_pipe)
        else:
            self._variation_pipe = \
                    AutoPipelineForImage2Image.from_pretrained(
                        self._variation_checkpoint,
                        torch_dtype=torch.float16, )
        self._variation_pipe.safety_checker = None
        self._variation_pipe.set_progress_bar_config(disable=True)
        if lora is not None:
            self._variation_pipe.unet.load_attn_procs(lora)
        self._variation_pipe = self._variation_pipe.to(self.device)

    @staticmethod
    def command_line_parser():
        parser = super(
            StableDiffusionAPI, StableDiffusionAPI).command_line_parser()
        parser.add_argument(
            '--lora',
            type=str,
            required=False,
            help="LoRA"
        )
        parser.add_argument(
            '--api_device',
            type=int,
            required=True
        )
        parser.add_argument(
            '--random_sampling_checkpoint',
            type=str,
            required=True,
            help='The path to the checkpoint for random sampling API')
        parser.add_argument(
            '--random_sampling_guidance_scale',
            type=float,
            default=0,
            help='The guidance scale for random sampling API')
        parser.add_argument(
            '--random_sampling_num_inference_steps',
            type=int,
            default=50,
            help='The number of diffusion steps for random sampling API')
        parser.add_argument(
            '--random_sampling_batch_size',
            type=int,
            default=10,
            help='The batch size for random sampling API')

        parser.add_argument(
            '--variation_checkpoint',
            type=str,
            required=True,
            help='The path to the checkpoint for variation API')
        parser.add_argument(
            '--variation_guidance_scale',
            type=float,
            default=7.5,
            help='The guidance scale for variation API')
        parser.add_argument(
            '--variation_num_inference_steps',
            type=int,
            default=50,
            help='The number of diffusion steps for variation API')
        parser.add_argument(
            '--variation_batch_size',
            type=int,
            default=10,
            help='The batch size for variation API')
        return parser

    def random_sampling(self, num_samples, size, prompts):
        """
        Generates a specified number of random image samples based on a given
        prompt and size using OpenAI's Image API.

        Args:
            num_samples (int):
                The number of image samples to generate.
            size (str, optional):
                The size of the generated images in the format
                "widthxheight". Options include "256x256", "512x512", and
                "1024x1024".
            prompts (List[str]):
                The text prompts to generate images from. Each promot will be
                used to generate num_samples/len(prompts) number of samples.

        Returns:
            numpy.ndarray:
                A numpy array of shape [num_samples x width x height x
                channels] with type np.uint8 containing the generated image
                samples as numpy arrays.
            numpy.ndarray:
                A numpy array with length num_samples containing prompts for
                each image.
        """
        max_batch_size = self._random_sampling_batch_size
        images = []
        return_prompts = []
        width, height = list(map(int, size.split('x')))
        for prompt_i, prompt in enumerate(prompts):
            num_samples_for_prompt = (num_samples + prompt_i) // len(prompts)
            num_iterations = int(np.ceil(
                float(num_samples_for_prompt) / max_batch_size))
            for iteration in tqdm(range(num_iterations), desc="Generating initial samples", unit="batch"):
                batch_size = min(
                    max_batch_size,
                    num_samples_for_prompt - iteration * max_batch_size)
                images.append(_round_to_uint8(self._random_sampling_pipe(
                    prompt=prompt,
                    #negative_prompts=negative_prompts,
                    width=width,
                    height=height,
                    num_inference_steps=(
                        self._random_sampling_num_inference_steps),
                    guidance_scale=self._random_sampling_guidance_scale,
                    num_images_per_prompt=batch_size,
                    output_type='np').images))
            return_prompts.extend([prompt] * num_samples_for_prompt)
        return np.concatenate(images, axis=0), np.array(return_prompts)

    def variation(self, samples, additional_info,
                        num_variations_per_sample, size, variation_degree: Union[np.ndarray, float], t: int = None, candidate: bool = False, demo_samples: Optional[np.ndarray] = None, demo_weights: Optional[np.ndarray] = None, sample_weight=None):
        """
        Generates a specified number of variations for each image in the input
        array using OpenAI's Image Variation API.

        Args:
            images (numpy.ndarray):
                A numpy array of shape [num_samples x width x height
                x channels] containing the input images as numpy arrays of type
                uint8.
            additional_info (numpy.ndarray):
                A numpy array with the first dimension equaling to
                num_samples containing prompts provided by
                image_random_sampling.
            num_variations_per_image (int):
                The number of variations to generate for each input image.
            size (str):
                The size of the generated image variations in the
                format "widthxheight". Options include "256x256", "512x512",
                and "1024x1024".
            variation_degree (float):
                The image variation degree, between 0~1. A larger value means
                more variation.

        Returns:
            numpy.ndarray:
                A numpy array of shape [num_samples x num_variations_per_image
                x width x height x channels] containing the generated image
                variations as numpy arrays of type uint8.
        """
        if np.any(0 > variation_degree) or np.any(variation_degree > 1):
            raise ValueError('variation_degree should be between 0 and 1')
        variations = []
        for _ in tqdm(range(num_variations_per_sample), desc="Counting candidates", unit='candidate'):
            sub_variations = self._image_variation(
                samples=samples,
                prompts=list(additional_info),
                size=size,
                variation_degree=variation_degree)
            variations.append(sub_variations)
        return np.stack(variations, axis=1)

    def _image_variation(self, samples, prompts, size, variation_degree: Union[np.ndarray, float]):
        width, height = list(map(int, size.split('x')))
        variation_transform = T.Compose([
            T.Resize(
                (width, height),
                interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5])])
        samples = [variation_transform(Image.fromarray(im))
                  for im in samples]
        samples = torch.stack(samples).to(self.device)
        max_batch_size = 1 if isinstance(variation_degree, np.ndarray) else self._variation_batch_size
        variations = []
        num_iterations = int(np.ceil(
            float(samples.shape[0]) / max_batch_size))
        for iteration in tqdm(range(num_iterations), leave=False, desc="Generating candidates", unit='batch'):
            degree = variation_degree[iteration] if isinstance(variation_degree, np.ndarray) else variation_degree
            inference_steps = int(np.ceil(self._variation_num_inference_steps / degree)) if 'turbo' in self._variation_checkpoint else self._variation_num_inference_steps
                
            variations.append(self._variation_pipe(
                prompt=prompts[iteration * max_batch_size:
                               (iteration + 1) * max_batch_size],
                image=samples[iteration * max_batch_size:
                             (iteration + 1) * max_batch_size],
                num_inference_steps=inference_steps,
                strength=degree,
                guidance_scale=self._variation_guidance_scale,
                num_images_per_prompt=1,
                output_type='np').images)
        variations = _round_to_uint8(np.concatenate(variations, axis=0))
        return variations
