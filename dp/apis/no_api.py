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
    
    def GM_initial_sampling(self, num_samples,  public_dir):
        data_list=os.listdir(public_dir)
        random.shuffle(data_list)
        data_path=[os.path.join(public_dir, data) for data in data_list]
        max_batch_size = self._random_sampling_batch_size
        
        samples=[]
        return_prompts = []
        num_iterations=int(num_samples/max_batch_size)
        for i in range(num_iterations):
            texts = ""
            for j in range(max_batch_size):
                data_num=int((i*num_iterations+j)%len(data_list))
                
                with open(data_path[data_num], mode="r") as _file:
                    text=_file.readlines()[0]
            #        print(text)
                texts+=text
                texts+="\n"
                samples.append(text)
                
            #samples.append(texts)
            if self._live == 0:
                        self._live_save(
                            samples=texts,
                            additional_info=[f'{i} iteration for random sampling'] * len(text),
                            prefix=f'initial_{i}'
                    )
        #return_prompts.extend(["GM"] * num_samples)
        return_prompts=["GM"] * num_samples
        return np.array(samples), np.array(return_prompts)
    
    def text_to_tabular(self, target_sample, column):
        #print(target_sample)
        # print(type(target_sample[0]))
        
        for i in range(len(target_sample)):
            if type(target_sample[0])==np.ndarray:
                target=target_sample[i][0]
                # print(target)
            else:
                target=target_sample[i]

            target=target.replace("\'", "")
            target=target.replace("\"", "")
            #target=target.strip("\"")
            rows=target.replace("\n\n", "\n")
            #print(rows)
            df_list=rows.split("\n")
    
        
            df=[]
            for j in df_list:
            
                rows=j.split(",")

                if "" in rows:
                    rows=rows.remove("")
                if rows==None:
                    continue
                if len(rows)==1:
                    rows=j.split(" ")
                    # print(rows)
                    # print(len(rows))
                    if "" in rows:
                        rows=rows.remove("")
                    
                    #print(rows)
                if rows==None:
                    continue
                if len(rows)==16:
                    rows=rows[1:]
                #print(rows)
                
                if len(rows)==15:
                    df.append(rows)
                
                
            if i==0:
                total_df=pd.DataFrame(df, columns=column)

            else:
                df=pd.DataFrame(df, columns=column)
                total_df=pd.concat([total_df,df])
        
        # print(total_df)
        # print(total_df.shape)
        return total_df

    def tabular_to_text(self, tabular_data):
    #  text_list=[]
        text=[]
        count=0
        for idx, row in tabular_data.iterrows():
            #pre=str(idx)+". "
            row[row.isnull()] = '?'
            # if row.empty:
            #     continue
            row=" ".join(row)+"\n"
            # if "." in row:
            #     row=row.split(".")[1]
            
            row.strip("\"")
    #      print(row)
            #text+=row
            
            text.append(row)
            count+=1
        return text
    
    def _tabular_variation(self, target_sample, column, variation_degree=0.001, cat_var=0.1, public_info={}):

    
        #if t==1:
        tabular=self.text_to_tabular(target_sample, column)

        #TODO: 최빈값으로 바꾸기
        
        categorical_col=public_info[1].keys()
        num_col=public_info[0].keys()
    
        cat_public_info=public_info[1]
       
            
        
        for col in num_col:   
            tabular[col]=tabular[col].apply(pd.to_numeric, errors='coerce')
            # print(pd.DataFrame(tabular[col]).mean(numeric_only=True))
            tabular[col]=tabular[col].fillna(pd.DataFrame(tabular[col]).mean(numeric_only=True)[col])
            tabular[col]=tabular[col].astype(float)
        
        for i, col in enumerate(num_col):
            print(variation_degree)
            try:
                gaussian_noise_col = np.random.normal(0, float(variation_degree), tabular.shape[0])
            except:
                print(len(variation_degree))
                gaussian_noise_col=np.array(variation_degree)
            # print(f"guassain: {gaussian_noise_col}")
            tabular[col]+=gaussian_noise_col
            # print(tabular[col])
            # tabular[col]=norm_col*tabular[col].std()+tabular[col].mean()
            # tabular[col][tabular[col]<0]=0
            tabular[col]=tabular[col].apply(pd.to_numeric, errors='coerce')
            tabular[col]=tabular[col].fillna(tabular[col].mean(numeric_only=True))
            tabular[col]=tabular[col].astype(float)
            tabular[col]=tabular[col].round(decimals=5)
        
        for j in categorical_col:
            print(cat_var)
            num=int(len(tabular[j])*cat_var)
            # print(list(range(0, len(tabular[j]))))
            # print(num)
            num_list=[random.choice(list(range(0, len(tabular[j])))) for i in range(num)]
            
            for k in num_list:
                tabular[j].iloc[k]=random.choice(cat_public_info[j])

        for col in  column:
            tabular[col]=tabular[col].astype(str)
    
        text=self.tabular_to_text(tabular)

        text=np.array(text)
        
        # print(f"text: {text}")
        return text
    def random_initial_sampling(self, num_samples,  public_info, columns):

        # data_list=os.listdir(public_dir)
        # random.shuffle(data_list)
        # # data_path=[os.path.join(public_dir, data) for data in data_list]
        # print(f"columns:{columns}")
        max_batch_size = self._random_sampling_batch_size
        num_public_info=public_info[0]
        cat_public_info=public_info[1]
        # print(f"num:{num_public_info}")
        # print(f"cat:{cat_public_info}")
        samples=[]
        return_prompts = []
        num_iterations=int(num_samples/max_batch_size)
        for i in range(num_iterations):
            texts = ""
            str_text=""
            for j in range(max_batch_size):
                #time.sleep(0.5)
                str_text=""
                #data_num=int((i*num_iterations+j)%len(data_list))
                
                #with open(data_path[data_num], mode="r") as _file:
                text=dict.fromkeys(columns)
                for key, value in num_public_info.items():
                    text[key]=np.random.normal(0, 1)
                    #text[key]=random.randrange(value[0], value[1])
                for key, value in cat_public_info.items():
                    text[key]=random.choice(value)
            
                for idx, key in enumerate(columns):
                    # print(key)
                    # print(text[key])
                    str_text+=str(text[key])
                    if idx==14:
                        break
                    str_text+=" "
      
                texts+=str_text
                texts+="\n"
                samples.append(str_text)
                
            #samples.append(texts)
            if self._live == 0:
                        self._live_save(
                            samples=texts,
                            additional_info=[f'{i} iteration for random sampling'] * len(text),
                            prefix=f'initial_{i}'
                    )
        #return_prompts.extend(["GM"] * num_samples)
        return_prompts=["GM"] * num_samples
      #  print(samples)
       # print(return_prompts)
       # print(samples)
        return np.array(samples), np.array(return_prompts)    
    def random_sampling(self, public_dir, num_samples, prompts, size=None, modality="tabular", seed_population="GM", public_info={}, columns=[]):
        max_batch_size = self._random_sampling_batch_size
        texts = []
        return_prompts = []
        if modality=="tabular" and seed_population=="GM":
            samples, return_prompt=self.GM_initial_sampling(num_samples, public_dir)
        elif modality=="tabular" and seed_population=="random":
            samples, return_prompt=self.random_initial_sampling(num_samples, public_info, columns)
            #random_initial_sampling(self, num_samples,  public_info, columns
        else:
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
            texts=np.concatenate(texts, axis=0)
            return_prompt=np.array(return_prompts)
        return samples, return_prompt

    def variation(self, samples: np.ndarray, additional_info: np.ndarray,
                        num_variations_per_sample: int, size: int, variation_degree: Union[float, np.ndarray], t=None, candidate: bool = True, demo_samples: Optional[np.ndarray] = None, sample_weight: float = 1.0,
                        modality="tabular", columns=[], 
                        cat_var=0.1, public_info={}):
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
                sample_weight=sample_weight, column=columns, categorical_variation=cat_var, public_info=public_info)

            variations.append(sub_variations)

            if self._live == 0 and candidate:
                self._live_save(
                    samples=variations,
                    additional_info=[f'{idx} iteration for {t} variation'] * len(sub_variations),
                    prefix=f'variation_{t}_{idx}'
                )
            idx += 1
        return np.stack(variations, axis=1)

    def _variation(self, samples: np.ndarray, additional_info: np.ndarray, size, variation_degree: Union[float, np.ndarray], t: int, l: int, candidate: bool, demo_samples: Optional[np.ndarray] = None, sample_weight: float = 1.0,
                   column=[], categorical_variation=0.01, public_info={}):
        """
        samples : (Nsyn, ~) 변형해야 할 실제 샘플
        additional_info: (Nsyn,) 초기 샘플을 생성할 때 사용한 프롬프트
        size: 이미지에서만 필요, 사용x
        t, l: 중간 저장 시 이름을 구분하기 위한 변수.중요x
        candidate: candidate으로 만들어지는 샘플들만 저장하기 위해서 필요한 변수. 중요x
        demo_samples: (Nsyn, num_demo, ~) 데모로 사용할 샘플
        sample_weight: w
        """
        print(f"1:{variation_degree}")
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
            elif self._modality == 'tabular':
          
                print(public_info)
                variation=self._tabular_variation(target_samples, column, target_degree , categorical_variation, public_info)
                    
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
        

