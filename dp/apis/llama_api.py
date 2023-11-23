import torch
import numpy as np
from tqdm import tqdm
from .api import API
from transformers import AutoTokenizer
import transformers
import gc
# from dpsda.pytorch_utils import dev


class Llama2API(API):
    def __init__(self, random_sampling_checkpoint,
                 random_sampling_batch_size,
                 max_seq_len,
                 top_k,
                 variation_checkpoint,
                 variation_batch_size,
                 api_device,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(random_sampling_checkpoint)
        self._random_sampling_api = transformers.pipeline(
            "text-generation",
            model = random_sampling_checkpoint,
            device=api_device,
            do_sample=True,
            top_k=top_k,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            tokenizer=self.tokenizer
        )
        self.flag = '\n'
        self.tokenizer.pad_token_id = self._random_sampling_api.model.config.eos_token_id
        
        self._random_sampling_batch_size = random_sampling_batch_size

        self._variation_api = self._random_sampling_api
        self._variation_batch_size = variation_batch_size

        #self._variation_pipe = self._variation_pipe.to(dev())

    @staticmethod
    def command_line_parser():
        parser = super(
            Llama2API, Llama2API).command_line_parser()
        parser.add_argument(
            '--max_seq_len',
            type=int,
            required=True,
            help='The path to the checkpoint for random sampling API')
        parser.add_argument(
            '--top_k',
            type=int,
            required=True,
            help='The path to the checkpoint for random sampling API')
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
            '--variation_batch_size',
            type=int,
            default=10,
            help='The batch size for variation API')
        return parser

    def random_sampling(self, num_samples, prompts, size=None):

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
                response = self._random_sampling_api([prompt] * batch_size, batch_size=batch_size)
                text = [r[0]['generated_text'] for r in response]
                indices = [t.find(self.flag) for t in text]
                text = [t[idx+len(self.flag):].strip('\n') for t, idx in zip(text, indices)]

                texts.append(text)
                torch.cuda.empty_cache()
                gc.collect()
            return_prompts.extend([prompt] * num_samples_for_prompt)
        return np.concatenate(texts, axis=0), np.array(return_prompts)

    def variation(self, samples, additional_info,
                        num_variations_per_sample, size, variation_degree):
        variations = []
        for _ in tqdm(range(num_variations_per_sample)):
            sub_variations = self._variation(
                samples=samples,
                variation_degree=variation_degree)
            variations.append(sub_variations)
        return np.stack(variations, axis=1)

    def _variation(self, samples, variation_degree):
        max_batch_size = self._variation_batch_size
        variations = []
        num_iterations = int(np.ceil(
            float(samples.shape[0]) / max_batch_size))
        repeat = int(variation_degree + 1)
        for iteration in tqdm(range(num_iterations), leave=False):
            start_idx = iteration * max_batch_size
            end_idx = (iteration + 1) * max_batch_size
            target_samples = samples[start_idx:end_idx]
            r = 0
            while r < repeat:
                prompts = [sample + "\n Above is a document. Paraphrase it while keeping its basic structure." for sample in target_samples]
                response = self._variation_api(prompts, batch_size=len(prompts))
                text = [r[0]['generated_text'] for r in response]
                indices = [t.find(self.flag) for t in text]
                text = [t[idx + len(self.flag)].strip('\n') for t, idx in zip(text, indices)]
                target_samples = text
                r += 1
            variation = target_samples
            variations.append(variation)
        variations = np.concatenate(variations, axis=0)
        return variations
    