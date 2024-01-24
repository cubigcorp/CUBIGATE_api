import torch
import numpy as np
from tqdm.auto import tqdm
from .api import API
from transformers import AutoTokenizer
import transformers
import gc
import os
from typing import Optional
from typing import List
from dpsda.data_logger import log_samples
from dpsda.data_loader import load_samples
# from dpsda.pytorch_utils import dev


class ChatLlama2API(API):
    def __init__(self, random_sampling_checkpoint,
                 random_sampling_batch_size,
                 top_k,
                 variation_checkpoint,
                 variation_prompt_path,
                 variation_batch_size,
                 api_device,
                 control_prompt,
                 goal,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(random_sampling_checkpoint)
        self._random_sampling_api = transformers.pipeline(
            "conversational",
            model = random_sampling_checkpoint,
            device=api_device,
            do_sample=True,
            top_k=top_k,
            num_return_sequences=1,
            eos_token_id=self._tokenizer.eos_token_id,
            tokenizer=self._tokenizer,
            max_length=4096
        )
        self._goal = goal
        self._control_prompt = control_prompt

        self._tokenizer.pad_token_id = self._random_sampling_api.model.config.eos_token_id
        
        self._random_sampling_batch_size = random_sampling_batch_size

        if variation_checkpoint == random_sampling_checkpoint:
            self._variation_api = self._random_sampling_api
        else:
            self._variation_api = transformers.pipeline(
            "coversational",
            model = variation_checkpoint,
            device=api_device,
            do_sample=True,
            top_k=top_k,
            num_return_sequences=1,
            eos_token_id=self._tokenizer.eos_token_id,
            tokenizer=self._tokenizer
        )
        self._variation_batch_size = variation_batch_size
        with open(variation_prompt_path, 'r') as f:
            self._variation_prompt = f.read()

    @staticmethod
    def command_line_parser():
        parser = super(
            ChatLlama2API, ChatLlama2API).command_line_parser()
        parser.add_argument(
            '--variation_prompt_path',
            type=str,
            required=True
        )
        parser.add_argument(
            '--control_prompt',
            type=str
        )
        parser.add_argument(
            '--goal',
            type=str,
            help="The description of what to generate"
        )
        parser.add_argument(
            '--top_k',
            type=int,
            required=False,
            default=1,
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
                text = self._generate([f'{prompt} {self._control_prompt}'] * batch_size, batch_size=batch_size, variation=False)
                texts.append(text)
                torch.cuda.empty_cache()
                gc.collect()
            return_prompts.extend([prompt] * num_samples_for_prompt)
        return np.concatenate(texts, axis=0), np.array(return_prompts)

    def variation(self, samples, additional_info,
                        num_variations_per_sample, size, variation_degree, t=None, candidate=True, demo_samples: Optional[np.ndarray]=None, demo_weights: Optional[np.ndarray] = None, sample_weight: float = 1.0):
        variations = []
        for _ in tqdm(range(num_variations_per_sample)):
            sub_variations = self._variation(
                samples=samples,
                additional_info=additional_info,
                variation_degree=variation_degree,
                demo_samples=demo_samples,
                sample_weight=sample_weight)
            variations.append(sub_variations)
        return np.stack(variations, axis=1)

    def _variation(self, samples, additional_info, variation_degree, demo_samples: Optional[np.ndarray] = None, sample_weight: float = 1.0):
        max_batch_size = self._variation_batch_size
        variations = []
        num_iterations = int(np.ceil(
            float(samples.shape[0]) / max_batch_size))
        for iteration in tqdm(range(num_iterations), leave=False):
            start_idx = iteration * max_batch_size
            end_idx = (iteration + 1) * max_batch_size
            target_samples = samples[start_idx:end_idx]

            base_prompt = self._variation_prompt.replace('PROMPT', additional_info[0])
            prompts = [base_prompt.replace('SAMPLE', sample) for sample in target_samples]
            variation = self._generate(prompts, batch_size=len(prompts), variation=True, variation_degree=variation_degree)
            variations.append(variation)
            torch.cuda.empty_cache()
            gc.collect()
        variations = np.concatenate(variations, axis=0)
        return variations


    def _sanity_check(self, generated: str, goal: str, variation: bool) -> List[bool]:
        prompts = [[{"role":"user", "content":f"Is {gen} {goal}'? Answer only with 'Yes' or 'No' without any further explanation"}] for gen in generated]
        with torch.no_grad():
            if variation: 
                response = self._variation_api(prompts, batch_size=len(prompts))
            else:
                response = self._random_sampling_api(prompts, batch_size=len(prompts))
    
        responses = ['Yes' in str(r) for r in response]
        return responses


    def _generate(self, prompts: str, batch_size: int, variation: bool, variation_degree: float=None):
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        with torch.no_grad():
            if variation:
                response = self._variation_api(messages, batch_size=batch_size, temperature=variation_degree)
            else:
                response = self._random_sampling_api(messages, batch_size=batch_size)
        
        if not isinstance(response, list):
            response = [response]
        responses = [str(r) for r in response]
        # prompt가 대답에 그대로 나타날 경우 제거
        flag = 'assistant: '
        indices = [text.find(flag) for text in responses]
        texts = [text[idx+len(flag):].strip(' ') for text, idx in zip(responses, indices) if idx >= 0]
        
        for idx in range(len(texts)):
            if texts[idx].startswith('Sure, here'):
                anchor = texts[idx].find(':')
                texts[idx] = texts[idx][anchor+1:].strip('\n')

        # Sanity Check
        checks = self._sanity_check(texts, self._goal, variation)
        texts = [texts[idx] for idx in range(len(texts)) if checks[idx]]
        # 정해진 개수만큼 만들어지지 않은 경우
        remain = batch_size - len(texts)
        while remain:
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
