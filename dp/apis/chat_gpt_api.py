import numpy as np
from typing import Optional
import openai
from .api import API
from wrapt_timeout_decorator import timeout
from typing import Dict, List
from dpsda.data_logger import log_samples
from dpsda.data_loader import load_samples
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import time
# from dpsda.pytorch_utils import dev


class ChatGPTAPI(API):
    def __init__(self, random_sampling_checkpoint,
                 random_sampling_batch_size,
                 variation_checkpoint,
                 variation_batch_size,
                 api_key,
                 api_device,
                 variation_prompt_path,
                 control_prompt,
                 use_auxiliary_model,
                 auxiliary_model_checkpoint,
                 verbose,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._random_sampling_checkpoint = random_sampling_checkpoint
        self._random_sampling_batch_size = random_sampling_batch_size
        self.verbose = verbose
        with open(api_key, 'r') as f:
            openai.api_key = f.read()

        self._variation_checkpoint = variation_checkpoint
        self._variation_batch_size = variation_batch_size

        self._variation_api = variation_checkpoint
        with open(variation_prompt_path, 'r') as f:
            self.variation_prompt = f.read()
        self.control_prompt = control_prompt
        if use_auxiliary_model:
            self.use_auxiliary_model = use_auxiliary_model
            self.device = f"cuda:{api_device}"
            self.tokenizer = AutoTokenizer.from_pretrained(auxiliary_model_checkpoint)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.auxiliary_model = AutoModelForCausalLM.from_pretrained(auxiliary_model_checkpoint).to(self.device)
        else:
            self.use_auxiliary_model = False

    @staticmethod
    def command_line_parser():
        parser = super(
            ChatGPTAPI, ChatGPTAPI).command_line_parser()
        parser.add_argument(
            '--verbose',
            action='store_true'
        )
        parser.add_argument(
            '--use_auxiliary_model',
            action='store_true'
        )
        parser.add_argument(
            '--auxiliary_model_checkpoint',
            type=str,
            required=False
        )
        parser.add_argument(
            '--api_device',
            type=int,
            required=False
        )
        parser.add_argument(
            '--api_key',
            type=str,
            required=True,
            help='The path to the key for chatGPT API')
        parser.add_argument(
            '--control_prompt',
            type=str,
            required=False,
            help='Control prompt for random sampling API')
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
            start_iter = 0
            # Load the target file if any
            if self._live == 1:
                samples, start_iter = self._live_load(self._live_loading_target)
                if samples is not None:
                    texts.append(samples)
                self._live = 0
                logging.info(f"Loaded {self._live_loading_target}")
                logging.info(f"Start iteration from {start_iter}")
                logging.info(f"Remaining {num_iterations} iteration")
            idx = start_iter

            while idx < num_iterations :
                batch_size = min(
                    max_batch_size,
                    num_samples_for_prompt - idx * max_batch_size)
                                # For batch computing
                if 'BATCH' in prompt:
                    prompt = prompt.replace('BATCH', f'{batch_size}')
                if self._modality == 'text':
                    messages = [
                        {"role": "user", "content": prompt + self.control_prompt }
                    ]

                    response = self._generate(model=self._random_sampling_checkpoint, messages=messages)
                    if batch_size > 1:
                        response = response.strip('--').split('--')
                    else:
                        response = [response]
                    text = [t.strip('\n') for t in response]
                    text = [t for t in text if t][:batch_size]
                    remain = batch_size - len(text)
                    while remain > 0 :
                        prompt = prompt.replace(f'{batch_size}', f'{remain}')
                        messages = [
                        {"role": "user", "content": prompt + self.control_prompt }
                    ]
                        response = self._generate(model=self._random_sampling_checkpoint, messages=messages)
                        if batch_size > 1 :
                            response = response.strip('--').split('--')
                        temp = [t.strip('\n') for t in response]
                        text = (text + [t for t in temp if t])[:batch_size]
                        remain = batch_size - len(text)
                    texts.append(text)
                # 중간 저장을 할 경우
                _save = (self._save_freq < np.inf) and (idx % self._save_freq == 0)
                if self._live == 0 and _save:
                    self._live_save(
                        samples=text,
                        additional_info=[f'{idx} iteration for random sampling'] * len(text),
                        prefix=f'initial_{idx}'
                    )
                idx += 1
            return_prompts.extend([prompt] * num_samples_for_prompt)
        return np.concatenate(texts, axis=0), np.array(return_prompts)

    def variation(self, samples: np.ndarray, additional_info: np.ndarray,
                        num_variations_per_sample: int, size: int, variation_degree: float, t=None, lookahead: bool = True, demo_samples: Optional[np.ndarray] = None, sample_weight: float = 1.0):
        if not (0 <= variation_degree <= 1):
            raise ValueError('variation_degree should be between 0 and 1')
        variations = []

        start_iter = 0
        if (self._live == 1) and ('sub' not in self._live_loading_target) and lookahead:
            sub_variations, start_iter = self._live_load(self._live_loading_target)
            variations.extend(sub_variations)
            self._live = 0
            logging.info(f"Loaded {self._live_loading_target}")
            logging.info(f"Start iteration from {start_iter}")
            logging.info(f"Remaining {num_variations_per_sample} iteration")
        idx = start_iter
        while idx < num_variations_per_sample:
            sub_variations = self._variation(
                samples=samples,
                additional_info=list(additional_info),
                size=size,
                variation_degree=variation_degree,
                t=t,
                l=idx,
                lookahead=lookahead,
                demo_samples=demo_samples,
                sample_weight=sample_weight)

            variations.append(sub_variations)

            if self._live == 0 and lookahead:
                self._live_save(
                    samples=variations,
                    additional_info=[f'{idx} iteration for {t} variation'] * len(sub_variations),
                    prefix=f'variation_{t}_{idx}'
                )
            idx += 1
        return np.stack(variations, axis=1)

    def _variation(self, samples: np.ndarray, additional_info: np.ndarray, size, variation_degree: float, t: int, l: int, lookahead: bool, demo_samples: Optional[np.ndarray] = None, sample_weight: float = 1.0):
        """
        samples : (Nsyn, ~) 변형해야 할 실제 샘플
        additional_info: (Nsyn,) 초기 샘플을 생성할 때 사용한 프롬프트
        size: 이미지에서만 필요, 사용x
        t, l: 중간 저장 시 이름을 구분하기 위한 변수.중요x
        lookahead: lookahead으로 만들어지는 샘플들만 저장하기 위해서 필요한 변수. 중요x
        demo_samples: (Nsyn, num_demo, ~) 데모로 사용할 샘플
        sample_weight: w
        """
        num_demo = demo_samples.shape[1] if demo_samples is not None else 0
        max_batch_size = self._variation_batch_size
        variations = []
        num_iterations = int(np.ceil(
            float(samples.shape[0]) / max_batch_size))
        start_iter = 0
        if (self._live == 1) and ('sub' in self._live_loading_target) and lookahead:
            variation, start_iter = self._live_load(self._live_loading_target)
            variations.extend(variation)
            self._live = 0
            logging.info(f"Loaded {self._live_loading_target}")
            logging.info(f"Start iteration from {start_iter}")
            logging.info(f"Remaining {num_iterations - start_iter} iteration")
        logging.info(f"Number of demonstrations: {num_demo}")
        logging.info(f"Number of samples in a batch: {max_batch_size}")
        idx = start_iter
    
        while idx < num_iterations:
            start_idx = idx * max_batch_size
            end_idx = (idx + 1) * max_batch_size
            target_samples = samples[start_idx:end_idx]
            target_demo = demo_samples[start_idx:end_idx].flatten() if num_demo > 0 else None

            if self._modality == 'text':
                if self.use_auxiliary_model:
                    prompts = [f'<s>[INST] Paraphrase: {sample} [/INST]' for sample in target_samples]
                    response = self._paraphrase(prompts=prompts, temperature=variation_degree)
                else:
                    if num_demo > 0 :
                        # Prepare emostrations
                        demo_prompts = "\n--\n".join(target_demo)
                        prompts = f"{demo_prompts}\n{self.variation_prompt}"
                    else:
                        prompts = "\n--\n".join(target_samples)
                        prompts = f"{prompts}\n{self.variation_prompt.replace('PROMPT', additional_info[0])}"

                    messages = [
                            {"role": "user", "content": prompts}
                        ]
                    response = self._generate(model=self._variation_checkpoint, messages=messages, temperature=variation_degree)
                    if max_batch_size > 1 :
                        response = response.strip('--').split('--')
                    else:
                        response = [response]
                variation = [r.strip('\n') for r in response]
                if self.verbose:
                    logging.info(f"{idx}_response length: {len(response)}")
                    logging.info(f"{idx}_variation length: {len(variation)}")
            variations.append(variation)
            _save = (self._save_freq < np.inf) and (idx % self._save_freq == 0)
            if self._live == 0 and _save and lookahead:
                self._live_save(
                    samples=variations,
                    additional_info=[f'{idx} iteration for sub-variation'] * len(variation),
                    prefix=f'sub_variation_{t}_{l}_{idx}'
                )
            idx += 1
        variations = np.concatenate(variations, axis=0)

        logging.info(f"{idx}_final shape: {variations.shape}")
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
        iteration += 1
        return samples, iteration
        

    def _paraphrase(
        self,
        prompts,
        num_return_sequences=1,
        temperature=0.7,
        max_length=2048
    ):
        input_ids = self.tokenizer(
            prompts,
            return_tensors="pt", padding="max_length",
            max_length=max_length,
            truncation=True,
        ).input_ids
        input_ids = input_ids.to(self.device)
        
        outputs = self.auxiliary_model.generate(
            input_ids, temperature=temperature, num_return_sequences=num_return_sequences, pad_token_id=self.tokenizer.eos_token_id, do_sample=True, max_new_tokens=2048)

        res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return res