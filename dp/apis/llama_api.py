import torch
import numpy as np
from tqdm.auto import tqdm
from .api import API
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import gc, os
from typing import List
from dpsda.data_logger import log_samples
from dpsda.data_loader import load_samples
# from dpsda.pytorch_utils import dev


class Llama2API(API):
    def __init__(self, random_sampling_checkpoint,
                 random_sampling_batch_size,
                 variation_checkpoint,
                 variation_batch_size,
                 api_device,
                 goal,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = f'cuda:{api_device}'
        self.tokenizer = AutoTokenizer.from_pretrained(random_sampling_checkpoint)
        self._random_sampling_api = AutoModelForCausalLM.from_pretrained(random_sampling_checkpoint, max_length=40960).to(self.device)

        self.tokenizer.pad_token_id = self._random_sampling_api.model.config.eos_token_id
        self.goal = goal
        self._random_sampling_batch_size = random_sampling_batch_size

        self._variation_api = self._random_sampling_api
        self._variation_batch_size = variation_batch_size


    @staticmethod
    def command_line_parser():
        parser = super(
            Llama2API, Llama2API).command_line_parser()
        parser.add_argument(
            '--goal',
            type=str
        )
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
                text = self._generate([f'[INST]\n{prompt}\n[\INST]\n\n'] * batch_size, batch_size=batch_size, variation=False)
                texts.append(text)
                torch.cuda.empty_cache()
                gc.collect()
            return_prompts.extend([prompt] * num_samples_for_prompt)
        return np.concatenate(texts, axis=0), np.array(return_prompts)

    def variation(self, samples, additional_info,
                        num_variations_per_sample, size, variation_degree, t=None, candidate=None, demo=None):
        variations = []
        for _ in tqdm(range(num_variations_per_sample)):
            sub_variations = self._variation(
                samples=samples,
                variation_degree=variation_degree,
                additional_info=additional_info)
            variations.append(sub_variations)
        return np.stack(variations, axis=1)

    def _variation(self, samples, variation_degree, additional_info):
        max_batch_size = self._variation_batch_size
        variations = []
        num_iterations = int(np.ceil(
            float(samples.shape[0]) / max_batch_size))
        for iteration in tqdm(range(num_iterations), leave=False):
            start_idx = iteration * max_batch_size
            end_idx = (iteration + 1) * max_batch_size
            target_samples = samples[start_idx:end_idx]
            prompts = [f'[INST]\nParaphrase: {sample} \n[\INST]\n\n' for sample, initial in zip(target_samples, additional_info)]
            variation = self._generate(prompts, batch_size=len(prompts), variation=True, variation_degree=variation_degree)
            variations.append(variation)
            torch.cuda.empty_cache()
            gc.collect()
        variations = np.concatenate(variations, axis=0)
        return variations


    def _sanity_check(self, generated: List[str], goal: str, variation: bool, max_length=4096) -> List[bool]:
        prompts = [f'[INST]\nAnswer if {gen} is {goal} only with "Yes" or "No" without any further explanation.\n[\INST]\n\n' for gen in generated]
        input_ids = self.tokenizer(
            prompts,
            return_tensors="pt", padding="max_length",
            max_length=max_length,
            truncation=True,
        ).input_ids
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            if variation: 
                response = self._variation_api.generate(input_ids)
            else:
                response = self._random_sampling_api.generate(input_ids)
    
        responses = ['Yes' in str(r) for r in response]
        return responses


    def _generate(self, prompts: str, batch_size: int, variation: bool, variation_degree: float=None, max_length: int=4096):
        input_ids = self.tokenizer(
            prompts,
            return_tensors="pt", padding="max_length",
            max_length=max_length,
            truncation=True,
        ).input_ids
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            if variation:
                generated = self._variation_api.generate(input_ids, temperature=variation_degree, do_sample=True)
                response = self.tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            else:
                generated = self._random_sampling_api.generate(input_ids)
                response = self.tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        if not isinstance(response, list):
            response = [response]
        # prompt가 대답에 그대로 나타날 경우 제거
        indices = [text.find(prompt) for text, prompt in zip(response, prompts)]
        texts = [response[i][indices[i]+len(prompts[i]):].strip('\n') for i in range(batch_size) if indices[i] >= 0]
        # Filtering
        for idx in range(len(texts)):
            if texts[idx].startswith('Sure, here'):
                idx = texts[idx].find(':')
                texts[idx] = texts[idx][idx+1:].strip('\n')
                
        filter = self._sanity_check(texts, self.goal, variation)
        texts = [texts[idx] for idx in range(len(texts)) if filter[idx]]
        # 정해진 개수만큼 만들어지지 않은 경우
        remain = batch_size - len(texts)
        while remain > 0 :
            sub_texts = self._generate(prompts=prompts[:remain], batch_size=remain, variation=False)[:remain]
            texts.extend(sub_texts)
            remain = batch_size - len(texts)
        return texts

    def _live_save(self, samples: List, additional_info, prefix: str):
        log_samples(
            samples=samples,
            additional_info=additional_info,
            folder=self._result_folder,
            save_each_sample=False,
            save_npz=True,
            prefix=prefix)


    def _live_load(self, path: str):
        if path is None:
            return None, 0
        samples, _ = load_samples(path)
        iteration = int(path.split('_')[-2])
        if iteration:
            sub_samples = samples
            samples = []
            samples.append(sub_samples)
            sub_iteration = iteration
            while sub_iteration:
                sub_iteration -= 1
                dirname = os.path.dirname(path)
                basename = os.path.basename(path).split("_")[:-2]
                prev = os.path.join(dirname, f"{basename}_{iteration}_samples.npz")
                if os.path.exists(prev):
                    sub_samples, _ = load_samples(prev)
                    samples.append(sub_samples)
                else:
                    return None, 0
        iteration += 1
                
        return samples, iteration
