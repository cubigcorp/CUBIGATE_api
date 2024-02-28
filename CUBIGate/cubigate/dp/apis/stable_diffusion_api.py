import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from typing import Optional, Union
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
from cubigate.dp.utils.pytorch_utils import dev

from .api import API


def _round_to_uint8(image):
    return np.around(np.clip(image * 255, a_min=0, a_max=255)).astype(np.uint8)

class StableDiffusionAPI(API):
    def __init__(self, 
                 API_checkpoint,
                 guidance_scale,
                 inference_steps,
                 API_batch_size,
                 prompt,
                 lora, gpu_num,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt = prompt
        self.lora=lora
        self._random_sampling_checkpoint = API_checkpoint
        self._random_sampling_guidance_scale = guidance_scale
        self._random_sampling_num_inference_steps = \
            inference_steps
        self._random_sampling_batch_size = API_batch_size

        self._variation_checkpoint = API_checkpoint
        self._variation_guidance_scale = guidance_scale
        self._variation_num_inference_steps = inference_steps
        self._variation_batch_size = API_batch_size
        self.gpu_num=gpu_num
        self._random_sampling_pipe =  None
    

    @staticmethod
    def command_line_parser():
        parser = super(
            StableDiffusionAPI, StableDiffusionAPI).command_line_parser()
        parser.add_argument(
            '--prompt',
            type=list,
            help="If the API accepts a prompt, the initial samples will be generated with the prompt"
        )
        parser.add_argument(
            '--API_checkpoint',
            type=str,
            required=False,
            default='/root/Cubigate_ai_engine/CUBIGate/models/stable_diffusion/sdxl',
            help='The path to the checkpoint for API')
        parser.add_argument(
            '--guidance_scale',
            type=float,
            default=7.5,
            help='The guidance scale for API')
        parser.add_argument(
            '--inference_steps',
            type=int,
            default=50,
            help='The number of diffusion steps for API')
        parser.add_argument(
            '--API_batch_size',
            type=int,
            default=4,
            help='The batch size for API')
        parser.add_argument(
            '--lora',
            type=str,
            required=False,
            default="None",
            help='If you use lora it is true')
        parser.add_argument(
            '--gpu_num',
            type=str,
            default="0",
            help='gpu number')

        return parser

    def random_sampling(self, num_samples, size):
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
        self._random_sampling_pipe =  AutoPipelineForText2Image.from_pretrained(
            self._random_sampling_checkpoint, torch_dtype=torch.float16)
        self._random_sampling_pipe.set_progress_bar_config(disable=True)
        self._random_sampling_pipe.safety_checker = None
        print(self.lora)
        
        if self.lora!= "None":
            print("Lora")
            self._random_sampling_pipe.load_lora_weights(self.lora)
            self._random_sampling_pipe.fuse_lora(lora_scale=1.0)
        self._random_sampling_pipe.to(f"cuda:{self.gpu_num}")
        max_batch_size = self._random_sampling_batch_size
        images = []
        width, height = list(map(int, size.split('x')))
        iteration = int(np.ceil(
            float(num_samples) / max_batch_size))
        ##TODO: Batch size개수마다 생성하도록 하기
        print(self.prompt)
        for i in range(iteration):
            batch_size = min(max_batch_size, (num_samples - max_batch_size * i))
            print(f"batch_size:{batch_size}")
            images.append(_round_to_uint8(self._random_sampling_pipe(
                prompt=self.prompt[batch_size*i: batch_size*(i+1)],
                width=width,
                height=height,
                num_inference_steps=(
                    self._random_sampling_num_inference_steps),
                guidance_scale=self._random_sampling_guidance_scale,
                num_images_per_prompt=1,
                output_type='np').images))
        print("variate")
        self._init_variate()
        return np.concatenate(images, axis=0)

    def variation(self, samples,
                        num_variations_per_sample, size, variation_degree):
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

        if not (0 <= variation_degree <= 1):
            raise ValueError('variation_degree should be between 0 and 1')
        variations = []
        for _ in tqdm(range(num_variations_per_sample)):
            sub_variations = self._variation(
                samples=samples,
                size=size,
                variation_degree=variation_degree)
            variations.append(sub_variations)
        return np.stack(variations, axis=1)

    def _variation(self, samples, size, variation_degree):

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
        samples = torch.stack(samples).to(f"cuda:{self.gpu_num}")
        max_batch_size = self._variation_batch_size
        variations = []
        num_iterations = int(np.ceil(
            float(samples.shape[0]) / max_batch_size))
        for iteration in tqdm(range(num_iterations), leave=False):
            batch_size = min(max_batch_size, (samples.shape[0] - max_batch_size * iteration))
            variations.append(self._variation_pipe(
                prompt=self.prompt[iteration*batch_size:(iteration+1)*batch_size],
                image=samples[iteration * max_batch_size:
                             (iteration + 1) * max_batch_size],
                num_inference_steps=self._variation_num_inference_steps,
                strength=variation_degree,
                guidance_scale=self._variation_guidance_scale,
                num_images_per_prompt=1,
                output_type='np').images)  
        variations = _round_to_uint8(np.concatenate(variations, axis=0))
        return variations


    def _init_variate(self):
        print(1)
        del self._random_sampling_pipe
        print(self._variation_checkpoint)
        print(self.lora)
        self._variation_pipe = \
                    AutoPipelineForImage2Image.from_pretrained(
                        self._variation_checkpoint,
                        torch_dtype=torch.float16, )
        if self.lora!= "None":
            #TODO: lorar값 바꾸기
            print("lora")
            self._variation_pipe.load_lora_weights(self.lora)
            self._variation_pipe.fuse_lora(lora_scale=1.0)
        self._variation_pipe.safety_checker = None
        self._variation_pipe.set_progress_bar_config(disable=True)
        print(f"gpu:{self.gpu_num}")
        self._variation_pipe.to(f"cuda:{self.gpu_num}")