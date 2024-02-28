import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from .api import API
# from dpsda.pytorch_utils import dev


class GPT2API(API):
    def __init__(self, random_sampling_checkpoint,
                 random_sampling_batch_size,
                 variation_checkpoint,
                 variation_guidance_scale,
                 variation_batch_size,
                 api_device,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._random_sampling_checkpoint = random_sampling_checkpoint
        self._random_sampling_batch_size = random_sampling_batch_size
        self.device = f'cuda:{api_device}'
        self.tokenizer = GPT2Tokenizer.from_pretrained(random_sampling_checkpoint)
        self._random_sampling_api = GPT2LMHeadModel.from_pretrained(
            self._random_sampling_checkpoint, torch_dtype=torch.float16)
       # self._random_sampling_pipe = self._random_sampling_pipe.to(dev())
        self._random_sampling_api = self._random_sampling_api.to(self.device)

        self._variation_checkpoint = variation_checkpoint
        self._variation_guidance_scale = variation_guidance_scale
        self._variation_batch_size = variation_batch_size

        self._variation_api = \
            GPT2LMHeadModel.from_pretrained(
                self._variation_checkpoint,
                torch_dtype=torch.float16)
        self._variation_api.safety_checker = None
        self._variation_api = self._variation_api.to(self.device)
        #self._variation_pipe = self._variation_pipe.to(dev())

    @staticmethod
    def command_line_parser():
        parser = super(
            GPT2API, GPT2API).command_line_parser()
        parser.add_argument(
            '--api_device',
            type=int,
            required=True,
            help='The path to the checkpoint for random sampling API')
        parser.add_argument(
            '--random_sampling_checkpoint',
            type=str,
            required=True,
            help='The path to the checkpoint for random sampling API')
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
            '--variation_batch_size',
            type=int,
            default=10,
            help='The batch size for variation API')
        return parser

    def random_sampling(self, num_samples, prompts, size=None):
        """
        Generates a specified number of random samples based on a given
        prompt and size using OpenAI's API.

        Args:
            num_samples (int):
                The number of  samples to generate.
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
        texts = []
        return_prompts = []

        for prompt_i, prompt in enumerate(prompts):
            num_samples_for_prompt = (num_samples + prompt_i) // len(prompts)
            num_iterations = int(np.ceil(
                float(num_samples_for_prompt) / max_batch_size))
            for iteration in tqdm(range(num_iterations)):
                batch_size = min(
                    max_batch_size,
                    num_samples_for_prompt - iteration * max_batch_size)
                input_ids = self.tokenizer.encode(prompt, return_tensors='pt', skip_special_tokens=True).to(self.device)
                generated_txt = self._random_sampling_api.generate(input_ids, pad_token_id=self.tokenizer.eos_token_id)
                decoded_txt = self.tokenizer.decode(generated_txt[0], skip_special_tokens=True)
                texts.append(decoded_txt)
                
            return_prompts.extend([prompt] * num_samples_for_prompt)
        return np.concatenate(texts, axis=0), np.array(return_prompts)

    def variation(self, samples, additional_info,
                        num_variations_per_image, size, variation_degree):
        """
        Generates a specified number of variations for each image in the input
        array using OpenAI's Image Variation API.

        Args:
            images (numpy.ndarray):
                A numpy array of shape [num_samples x width x height
                x channels] containing the input images as numpy arrays of type
                uint8.
                07
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
        for _ in tqdm(range(num_variations_per_image)):
            sub_variations = self._variation(
                samples=samples,
                prompts=list(additional_info),
                size=size,
                variation_degree=variation_degree)
            variations.append(sub_variations)
        return np.stack(variations, axis=1)

    def _variation(self, samples, prompts, size, variation_degree):
        samples = torch.stack(samples).to(self.device)
        max_batch_size = self._variation_batch_size
        variations = []
        num_iterations = int(np.ceil(
            float(samples.shape[0]) / max_batch_size))
        for iteration in tqdm(range(num_iterations), leave=False):
            input_ids = self.tokenizer.encode(samples, return_tensors='pt', skip_special_tokens=True).to(self.device)
            generated_txt = self._random_sampling_api.generate(input_ids, pad_token_id=self.tokenizer.eos_token_id, do_sample=True, temperature=variation_degree, top_k=0)
            decoded_txt = self.tokenizer.decode(generated_txt[0], skip_special_tokens=True)
            variations.append(decoded_txt)
        variations = np.concatenate(variations, axis=0)
        return variations
