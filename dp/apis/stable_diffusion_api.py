import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from apis.adapter.ip_adapter import IPAdapter
from typing import Optional, Union, List
from dpsda.prompt_generator import PromptGenerator
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

from .api import API
import os

def _round_to_uint8(image):
    return np.around(np.clip(image * 255, a_min=0, a_max=255)).astype(np.uint8)


def tuple_into_tensor(in_tuple):
    out = [torch.cat(item) for item in in_tuple]
    return out


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
            self.lora = lora
            self._random_sampling_pipe.unet.load_attn_procs(self.lora)
        self._random_sampling_pipe.safety_checker = None
        self.device = f"cuda:{api_device}"
        self._random_sampling_pipe = self._random_sampling_pipe.to(self.device)

        self._variation_checkpoint = variation_checkpoint
        self._variation_guidance_scale = variation_guidance_scale
        self._variation_num_inference_steps = variation_num_inference_steps
        self._variation_batch_size = variation_batch_size


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

    def random_sampling(self, num_samples, size, prompts: Optional[List] = None, generator: Optional[PromptGenerator] = None):
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
        assert (prompts is None) ^ (generator is None)
        max_batch_size = self._random_sampling_batch_size
        num_iterations = int(np.ceil(num_samples / max_batch_size))
        images = []
        return_prompts = []
        width, height = list(map(int, size.split('x')))
        if isinstance(prompts, List):
            prompts = prompts * int(np.ceil(num_samples // len(prompts)))
            prompts = prompts[:num_samples]
        else:
            prompts = generator.generate(num_samples)

        for iteration in tqdm(range(num_iterations), desc="Generating initial samples", unit="batch"):
            target_prompts = prompts[iteration * max_batch_size:
                                    (iteration + 1) * max_batch_size]
            image = _round_to_uint8(self._random_sampling_pipe(
                prompt=target_prompts,
                #negative_prompts=negative_prompts,
                width=width,
                height=height,
                num_inference_steps=(
                    self._random_sampling_num_inference_steps),
                guidance_scale=self._random_sampling_guidance_scale,
                num_images_per_prompt=1,
                output_type='np').images)
            images.append(image)
            return_prompts.extend(target_prompts)
        self._initial_variate()
        return np.concatenate(images, axis=0), np.array(return_prompts)

    def variation(self, samples, additional_info,
                num_variations_per_sample, size, variation_degree: Union[np.ndarray, float], 
                t: int = None, candidate: bool = False, demo_samples: Optional[np.ndarray] = None, 
                demo_weights: Optional[np.ndarray] = None, sample_weight: float = 1.0):
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
        
        if not hasattr(self, '_variation_API'):
            self._initial_variate()
        
        if np.any(0 > variation_degree) or np.any(variation_degree > 1):
            raise ValueError('variation_degree should be between 0 and 1')
        width, height = list(map(int, size.split('x')))
        variations = []
        max_batch_size = 1 if ('turbo' in self._variation_checkpoint) or (isinstance(variation_degree, np.ndarray)) else self._variation_batch_size
        prompts = list(additional_info)
        if demo_samples is None:
            num_less_demo = 0
        else:
            negative = "worst quality, low quality, illustration, 3d, 2d, painting"
            out = self._get_weights_images(prompts=prompts, negative_prompt=negative, demo_samples=demo_samples, demo_weights=demo_weights)
            num_less_demo = demo_samples.shape[1]
            # max_batch_size = self._variation_batch_size // (demo_samples.shape[1] + 1)
        num_iterations = int(np.ceil(
            float(samples.shape[0] - num_less_demo) / max_batch_size) + num_less_demo)

        variation_transform = T.Compose([
            T.Resize(
                (width, height),
                interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5])])
        samples = [variation_transform(Image.fromarray(sample))
                for sample in samples]
        samples = torch.stack(samples).to(self.device)

        for iteration in tqdm(range(num_iterations), desc="Generating candidates", unit='sample'):
            batch_size = 1 if iteration < num_less_demo else max_batch_size
            start_idx = iteration if iteration <= num_less_demo else num_less_demo + (iteration - num_less_demo) * batch_size
            end_idx = start_idx + batch_size
            degree = variation_degree[start_idx:end_idx].squeeze() if isinstance(variation_degree, np.ndarray) else variation_degree
            if 'turbo' in self._variation_checkpoint.lower():
                inference_steps = int(np.ceil(self._variation_num_inference_steps / degree))
            if demo_samples is not None:
                pos_embeds, neg_embeds, pooled_pos_embeds, pooled_neg_embeds = zip(*out[start_idx:end_idx])
                pos_embeds, neg_embeds, pooled_pos_embeds, pooled_neg_embeds = tuple_into_tensor((pos_embeds, neg_embeds, pooled_pos_embeds, pooled_neg_embeds))
                image = self._variation_pipe(
                    image=samples[start_idx:end_idx],
                    prompt_embeds=pos_embeds,
                    negative_prompt_embeds=neg_embeds,
                    pooled_prompt_embeds=pooled_pos_embeds,
                    negative_pooled_prompt_embeds=pooled_neg_embeds,
                    width=width,
                    height=height,
                    strength=degree,
                    guidance_scale=self._variation_guidance_scale,
                    num_inference_steps=inference_steps,
                    num_images_per_prompt=num_variations_per_sample,
                    output_type='np'
                ).images
            else:
                image = self._variation_pipe(
                    prompt=prompts[start_idx:end_idx],
                    image=samples[start_idx:end_idx],
                    width=width,
                    height=height,
                    strength=degree,
                    guidance_scale=self._variation_guidance_scale,
                    num_inference_steps=self._variation_num_inference_steps,
                    num_images_per_prompt=num_variations_per_sample,
                    output_type='np'
                ).images
            batch_image = np.stack(image, axis=0)
            batch_image = batch_image.reshape((-1, num_variations_per_sample) + batch_image.shape[1:])
            variations.append(batch_image)
        variations = _round_to_uint8(np.concatenate(variations, axis=0))
        return variations


    def _initial_variate(self):
        del self._random_sampling_pipe
        self._variation_pipe = \
                    AutoPipelineForImage2Image.from_pretrained(
                        self._variation_checkpoint,
                        torch_dtype=torch.float16, )
        dir = os.path.dirname(os.path.abspath(__file__))

        self._variation_pipe.safety_checker = None
        self._variation_pipe.set_progress_bar_config(disable=True)

        if hasattr(self, 'lora'):
            self._variation_pipe.unet.load_attn_procs(self.lora)
        self._variation_pipe = self._variation_pipe.to(self.device)
        self._variation_API = IPAdapter(
            self._variation_pipe,
            image_encoder_path=f'{dir}/adapter/sdxl_models/image_encoder',
            ip_ckpt=f'{dir}/adapter/sdxl_models/ip-adapter_sdxl.bin',
            device=self.device
        )
        self._variation_API.set_scale(0.5)


    def _get_weights_images(self,
                             prompts: np.ndarray,
                             demo_samples: np.ndarray,
                             demo_weights: np.ndarray,
                             negative_prompt: Optional[str] = None):
        out = []
        for idx in range(len(demo_samples)):
            if idx < demo_samples.shape[1]:
                target_images = demo_samples[idx][:1 + idx]
                target_weights = demo_weights[idx][:1 + idx]
            else:
                target_images = demo_samples[idx]
                target_weights = demo_weights[idx]
            out.append(self._variation_API.get_prompt_embeds(images=target_images, prompt=prompts[idx], negative_prompt=negative_prompt, weight=target_weights))
        
        return out

