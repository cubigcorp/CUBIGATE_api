import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Union
from .api import API
from wrapt_timeout_decorator import timeout
from typing import Dict, List
from dpsda.data_logger import log_samples
from dpsda.data_loader import load_samples
import io
import os
import logging
import time
import random
# from dpsda.pytorch_utils import dev


class NoAPI(API):
    def __init__(self, random_sampling_batch_size,
                 variation_batch_size,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._random_sampling_batch_size = random_sampling_batch_size
        self._variation_batch_size = variation_batch_size


    @staticmethod
    def command_line_parser():
        parser = super(
            NoAPI, NoAPI).command_line_parser()
        parser.add_argument(
            '--random_sampling_batch_size',
            type=int,
            default=10,
            help='The batch size for random sampling API')
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
                    
                if self._modality == 'time-series':
                    text = []
                    row,column = list(map(int, size.split('x')))
                    for _ in range(batch_size):
                        # (self.row, self.column) 크기의 행렬 생성
                        matrix = np.random.uniform(-1, 1, (row, column)) 
                        # 행렬을 CSV 형태의 텍스트로 변환
                        matrix_text = '\n'.join(','.join(str(cell) for cell in row) for row in matrix)
                        text.append(matrix_text)
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
                        num_variations_per_sample: int, size: int, variation_degree: Union[float, np.ndarray], t=None, candidate: bool = True, demo_samples: Optional[np.ndarray] = None, sample_weight: float = 1.0):
        if isinstance(variation_degree, np.ndarray):
            if np.any(0 > variation_degree) or np.any(variation_degree > 1):
                raise ValueError('variation_degree should be between 0 and 1')
        elif not (0 <= variation_degree <= 1):
            raise ValueError('variation_degree should be between 0 and 1')
        variations = []

        start_iter = 0
        if (self._live == 1) and ('sub' not in self._live_loading_target) and candidate:
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
                candidate=candidate,
                demo_samples=demo_samples,
                sample_weight=sample_weight)

            variations.append(sub_variations)

            if self._live == 0 and candidate:
                self._live_save(
                    samples=variations,
                    additional_info=[f'{idx} iteration for {t} variation'] * len(sub_variations),
                    prefix=f'variation_{t}_{idx}'
                )
            idx += 1
        return np.stack(variations, axis=1)

    def _variation(self, samples: np.ndarray, additional_info: np.ndarray, size, variation_degree: Union[float, np.ndarray], t: int, l: int, candidate: bool, demo_samples: Optional[np.ndarray] = None, sample_weight: float = 1.0):
        """
        samples : (Nsyn, ~) 변형해야 할 실제 샘플
        additional_info: (Nsyn,) 초기 샘플을 생성할 때 사용한 프롬프트
        size: 이미지에서만 필요, 사용x
        t, l: 중간 저장 시 이름을 구분하기 위한 변수.중요x
        candidate: candidate으로 만들어지는 샘플들만 저장하기 위해서 필요한 변수. 중요x
        demo_samples: (Nsyn, num_demo, ~) 데모로 사용할 샘플
        sample_weight: w
        """
        num_demo = demo_samples.shape[1] if demo_samples is not None else 0
        max_batch_size = self._variation_batch_size
        variations = []
        num_iterations = int(np.ceil(
            float(samples.shape[0]) / max_batch_size))
        start_iter = 0
        if (self._live == 1) and ('sub' in self._live_loading_target) and candidate:
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
            target_degree = variation_degree[start_idx:end_idx] if isinstance(variation_degree, np.ndarray) else variation_degree
            if self._modality == 'time-series':
                variation = []
                skipped_samples = 0  # 걸러진 샘플들의 수를 추적하기 위한 변수
                row,column = list(map(int, size.split('x')))
                
                if num_demo > 0:
                    combined_variations = []
                    for demo_sample, sample in zip(target_demo, target_samples):
                        try:
                            # 데모 샘플을 DataFrame으로 변환
                            demo_sample_df = pd.read_csv(io.StringIO(demo_sample.replace(" ", "")), header=None)

                            # 실제 샘플을 DataFrame으로 변환
                            sample_df = pd.read_csv(io.StringIO(sample.replace(" ", "")), header=None)

                            # (실제 샘플에서만) 각 데이터 포인트에 가우시안 노이즈 추가
                            noisy_sample_df = sample_df.apply(lambda col: col + np.random.normal(0, target_degree, size=col.shape))

                            # 가중치 적용
                            weighted_noisy_sample = noisy_sample_df * sample_weight
                            weighted_demo_sample = demo_sample_df * (1 - sample_weight)

                            # 가중치가 적용된 두 DataFrame을 합산
                            combined_df = weighted_noisy_sample + weighted_demo_sample

                            # DataFrame을 다시 문자열로 변환하여 variation 리스트에 추가
                            combined_sample = combined_df.to_csv(header=False, index=False).strip('\n')
                            variation.append(combined_sample)

                        except Exception as e:
                            # 변환 과정에서 에러 발생 시 건너뛰기
                            continue

                else:
                    for sample in target_samples:
                        try:
                            sample_no_space = sample.replace(" ", "")  # 공백 제거
                            # 문자열 데이터를 DataFrame으로 변환
                            sample_df = pd.read_csv(io.StringIO(sample_no_space), header=None)
                            
                            # DataFrame의 크기가 기존 크기가 아닌 경우 건너뛰기
                            if sample_df.shape != (row, column):
                                skipped_samples += 1
                                continue

                            # 각 데이터 포인트에 가우시안 노이즈 추가
                            noisy_df = sample_df.apply(lambda col: col + np.random.normal(0, target_degree, size=col.shape))
                            # DataFrame을 다시 문자열로 변환하여 variation 리스트에 추가
                            noisy_sample = noisy_df.to_csv(header=False, index=False).strip('\n')
                            variation.append(noisy_sample)

                        except Exception as e:
                            # 변환 과정에서 에러 발생 시 건너뛰기
                            skipped_samples += 1
                            continue
                    
            variations.append(variation)
            _save = (self._save_freq < np.inf) and (idx % self._save_freq == 0)
            if self._live == 0 and _save and candidate:
                self._live_save(
                    samples=variations,
                    additional_info=[f'{idx} iteration for sub-variation'] * len(variation),
                    prefix=f'sub_variation_{t}_{l}_{idx}'
                )
            idx += 1
        variations = np.concatenate(variations, axis=0)

        if self._modality == 'time-series':
            # len(variations)가 len(samples)와 같아질 때까지 샘플 추가
            while len(variations) < len(samples):
                if len(variations) > 0:
                    # variations 배열에서 무작위로 샘플을 선택하여 추가
                    additional_sample = random.choice(variations)
                    variations = np.append(variations, [additional_sample], axis=0)
                else:
                    # variations가 비어있다면, 루프를 중단하고 빈 배열 반환
                    break

        logging.info(f"{idx}_final shape: {variations.shape}")
        return variations
   
    
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
        

