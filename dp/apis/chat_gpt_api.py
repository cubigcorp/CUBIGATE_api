import torch
import numpy as np
from tqdm import tqdm
import openai
from .api import API
from wrapt_timeout_decorator import timeout
from typing import Dict
# from dpsda.pytorch_utils import dev


class ChatGPTAPI(API):
    def __init__(self, random_sampling_checkpoint,
                 random_sampling_batch_size,
                 variation_checkpoint,
                 variation_batch_size,
                 api_key,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._random_sampling_checkpoint = random_sampling_checkpoint
        self._random_sampling_batch_size = random_sampling_batch_size
        openai.api_key = api_key

        self._variation_checkpoint = variation_checkpoint
        self._variation_batch_size = variation_batch_size

        self._variation_api = variation_checkpoint
        #self._variation_pipe = self._variation_pipe.to(dev())

    @staticmethod
    def command_line_parser():
        parser = super(
            ChatGPTAPI, ChatGPTAPI).command_line_parser()
        parser.add_argument(
            '--api_key',
            type=str,
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
        system_prompt = {
        "role": "system",
        "content": f"1. Append 'END' at the end of each answer\n2.  Repeat the prompts {max_batch_size} times"
        }

        for prompt_i, prompt in enumerate(prompts):
            num_samples_for_prompt = (num_samples + prompt_i) // len(prompts)
            num_iterations = int(np.ceil(
                float(num_samples_for_prompt) / max_batch_size))
            for iteration in tqdm(range(num_iterations)):
                batch_size = min(
                    max_batch_size,
                    num_samples_for_prompt - iteration * max_batch_size)
                messages = [
                    system_prompt,
                    {"role": "user", "content": prompt}
                ]

                text = self._generate(model=self._random_sampling_checkpoint, messages=messages).split('END')
                text = [t.strip('\n') for t in text]
                texts.append(text)
                
            return_prompts.extend([prompt] * num_samples_for_prompt)
        return np.concatenate(texts, axis=0), np.array(return_prompts)

    def variation(self, samples, additional_info,
                        num_variations_per_sample, size, variation_degree):
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
        for _ in tqdm(range(num_variations_per_sample)):
            sub_variations = self._variation(
                samples=samples,
                prompts=list(additional_info),
                size=size,
                variation_degree=variation_degree)
            variations.append(sub_variations)
        return np.stack(variations, axis=1)

    def _variation(self, samples, prompts, size, variation_degree):
        max_batch_size = self._variation_batch_size
        variations = []
        num_iterations = int(np.ceil(
            float(samples.shape[0]) / max_batch_size))
        repeat = int(variation_degree + 1)
        for iteration in tqdm(range(num_iterations), leave=False):
            start_idx = iteration * max_batch_size
            end_idx = (iteration + 1) * max_batch_size
            target_samples = samples[start_idx:end_idx]

            prompts = "\nEND\n".join(target_samples)
            prompts = prompts + "\n Above is a document. Paraphrase it. Leave 'END' unchanged."
            messages = [
                    {"role": "user", "content": prompts}
                ]
            target_samples = self._generate(model=self._variation_checkpoint, messages=messages, temperature=variation_degree).split('END')
            target_samples = [t.strip('\n') for t in target_samples]
            variation = target_samples
            variations.append(variation)
        variations = np.concatenate(variations, axis=0)
        return variations
    
    @timeout(100)
    def _generate(self, model: str, messages: Dict, max_tokens: int=2048, n: int=1, stop: str=None, temperature: float=1):
        response = openai.ChatCompletion.create(
                  model=model, 
                  messages=messages,
                  max_tokens=max_tokens,
                  request_timeout = 100,
                  n=n,
                  stop=stop,
                  temperature=temperature)

        return response.choices[0].message.content
