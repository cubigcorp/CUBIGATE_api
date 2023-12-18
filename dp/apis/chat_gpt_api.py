import numpy as np
from tqdm import tqdm
import openai
from .api import API
from wrapt_timeout_decorator import timeout
from typing import Dict, List
from dpsda.data_logger import log_samples
from dpsda.data_loader import load_samples
import os
import logging
import time
# from dpsda.pytorch_utils import dev


class ChatGPTAPI(API):
    def __init__(self, random_sampling_checkpoint,
                 random_sampling_batch_size,
                 variation_checkpoint,
                 variation_batch_size,
                 api_key,
                 variation_prompt_path,
                 control_prompt,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._random_sampling_checkpoint = random_sampling_checkpoint
        self._random_sampling_batch_size = random_sampling_batch_size
        with open(api_key, 'r') as f:
            openai.api_key = f.read()

        self._variation_checkpoint = variation_checkpoint
        self._variation_batch_size = variation_batch_size

        self._variation_api = variation_checkpoint
        with open(variation_prompt_path, 'r') as f:
            self.variation_prompt = f.read()
        self.control_prompt = control_prompt

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
            '--control_prompt',
            type=str,
            required=False,
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
        parser.add_argument(
            '--variation_prompt_path',
            required=True,
            type=str
        )
        return parser

    def random_sampling(self, num_samples, prompts, size=None):
        max_batch_size = self._random_sampling_batch_size
        texts = []
        return_prompts = []

        for prompt_i, prompt in enumerate(prompts):
            num_samples_for_prompt = (num_samples + prompt_i) // len(prompts)
            num_iterations = int(np.ceil(
                float(num_samples_for_prompt) / max_batch_size))

            # Load the target file if any
            if self._live == 1:
                samples, pre_iter = self._live_load(self._live_loading_target)
                if samples is not None:
                    texts.append(samples)
                    num_iterations -= pre_iter
                self._live = 0
                logging.debug(f"Loaded {self._live_loading_target}")
                logging.debug(f"Start iteration from {iteration}")
                logging.debug(f"Remaining {num_iterations} iteration")

            for iteration in tqdm(range(num_iterations)):
                batch_size = min(
                    max_batch_size,
                    num_samples_for_prompt - iteration * max_batch_size)
                                # For batch computing
                if 'BATCH' in prompt:
                    prompt = prompt.replace('BATCH', f'{batch_size}')
                if self._modality == 'text':
                    messages = [
                        {"role": "user", "content": prompt + self.control_prompt }
                    ]

                    response = self._generate(model=self._random_sampling_checkpoint, messages=messages).strip('END').split('END')
                    text = [t.strip('\n') for t in response]
                    text = [t for t in text if t][:batch_size]
                    remain = batch_size - len(text)
                    while remain > 0 :
                        prompt = prompt.replace(f'{batch_size}', f'{remain}')
                        messages = [
                        {"role": "user", "content": prompt + self.control_prompt }
                    ]
                        response = self._generate(model=self._random_sampling_checkpoint, messages=messages).strip('END').split('END')
                        temp = [t.strip('\n') for t in response]
                        text = (text + [t for t in temp if t])[:batch_size]
                        remain = batch_size - len(text)
                    texts.append(text)
                # 중간 저장을 할 경우
                _save = (self._save_freq < np.inf) and (iteration % self._save_freq == 0)
                if self._live == 0 and _save:
                    self._live_save(
                        samples=text,
                        additional_info=[f'{iteration} iteration for random sampling'] * len(text),
                        prefix=f'initial_{iteration}'
                    )
            return_prompts.extend([prompt] * num_samples_for_prompt)
        return np.concatenate(texts, axis=0), np.array(return_prompts)

    def variation(self, samples, additional_info,
                        num_variations_per_sample, size, variation_degree, t=None):
        if not (0 <= variation_degree <= 1):
            raise ValueError('variation_degree should be between 0 and 1')
        variations = []

        if (self._live == 1) and ('sub' not in self._live_loading_target):
            sub_variations, iteration = self._live_load(self._live_loading_target)
            variations.append(sub_variations)
            num_variations_per_sample -= iteration
            self._live = 0
            logging.debug(f"Loaded {self._live_loading_target}")
            logging.debug(f"Start iteration from {iteration}")
            logging.debug(f"Remaining {num_variations_per_sample} iteration")
        for iteration in tqdm(range(num_variations_per_sample)):
            sub_variations = self._variation(
                samples=samples,
                additional_info=list(additional_info),
                size=size,
                variation_degree=variation_degree,
                t=t)

            variations.append(sub_variations)
            _save = (self._save_freq < np.inf) and (iteration % self._save_freq == 0)
            if self._live == 0 and _save:
                self._live_save(
                    samples=sub_variations,
                    additional_info=[f'{iteration} iteration for {t} variation'] * len(sub_variations),
                    prefix=f'variation_{t}_{iteration}'
                )
        return np.stack(variations, axis=1)

    def _variation(self, samples, additional_info, size, variation_degree, t):
        max_batch_size = self._variation_batch_size
        variations = []
        num_iterations = int(np.ceil(
            float(samples.shape[0]) / max_batch_size))
        if (self._live == 1) and ('sub' in self._live_loading_target):
            variation, iteration = self._live_load(self._live_loading_target)
            variations.append(variation)
            num_iterations -= iteration
            self._live = 0
            logging.debug(f"Loaded {self._live_loading_target}")
            logging.debug(f"Start iteration from {iteration}")
            logging.debug(f"Remaining {num_iterations} iteration")

        for iteration in tqdm(range(num_iterations), leave=False):
            start_idx = iteration * max_batch_size
            end_idx = (iteration + 1) * max_batch_size
            target_samples = samples[start_idx:end_idx]
            logging.debug(f"Number of samples: {len(target_samples)}")
            if self._modality == 'text':
                prompts = "\nEND\n".join(target_samples)
                prompts = f"{prompts}\n{self.variation_prompt.replace('PROMPT', additional_info[0])}"
                messages = [
                        {"role": "user", "content": prompts}
                    ]
                response = self._generate(model=self._variation_checkpoint, messages=messages, temperature=variation_degree)
                response = response.strip('END').split('END')
                logging.debug(f"{iteration}_response length: {len(response)}")
                variation = [r.strip('\n') for r in response]
                logging.debug(f"{iteration}_variation length: {len(variation)}")
            variations.append(variation)
            _save = (self._save_freq < np.inf) and (iteration % self._save_freq == 0)
            if self._live == 0 and _save:
                self._live_save(
                    samples=variation,
                    additional_info=[f'{iteration} iteration for sub-variation'] * len(variation),
                    prefix=f'sub_variation_{t}_{iteration}'
                )
        variations = np.concatenate(variations, axis=0)

        logging.debug(f"{iteration}_final shape: {variations.shape}")
        return variations
    
    @timeout(1000)
    def _generate(self, model: str, messages: Dict, batch_size=1, stop: str=None, temperature: float=1, sleep=10):
        response = openai.ChatCompletion.create(
                  model=model, 
                  messages=messages,
                  request_timeout = 1000,
                  stop=stop,
                  temperature=temperature)
        time.sleep(sleep)
        # generated = response.choices[0].message.content.strip('END').split('END')
        # text = []
        # for temp in generated:
        #     temp = temp.strip('\n')
        #     if temp is not None:
        #         text.append(temp)

        
        return response.choices[0].message.content
    
    def _live_save(self, samples: List, additional_info, prefix: str):
        log_samples(
            samples=samples,
            additional_info=additional_info,
            folder=self._result_folder,
            plot_samples=False,
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
        
