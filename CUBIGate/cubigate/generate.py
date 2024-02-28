import logging
import os
import numpy as np
import uuid
from typing import List
from cubigate.dp.utils.logging import setup_logging
from cubigate.dp.data_loader import load_data, load_samples, load_count
from cubigate.dp.extractors.feature_extractor import extract_features
from cubigate.dp.dp_counter import dp_nn_histogram
from cubigate.dp.apis import get_api_class_from_name
from cubigate.dp.data_logger import log_samples, log_count
from cubigate.dp.agm import get_epsilon
from PIL import Image
import shutil
import zipfile
import torch

from transformers import BlipProcessor, BlipForConditionalGeneration

class CubigDPGenerator():
    def __init__(
        self, 
        api: str = "stable_diffusion",
        feature_extractor: str = "inception_v3",
        result_folder: str = "/var/dp_msv/checkpoint/",
        tmp_folder: str = "./tmp/cookie",
        data_loading_batch_size: int = 1,
        feature_extractor_batch_size: int = 500,
        prv_img_size: int = 512,
        conditional: bool = False,
        num_prv_data: int = 10,
        prompt: str = """white or gray ragdoll cat, intricate, 
                        high detail, dramatic, extremely realistic eyes, 
                        extremely realistic hair, 
                        hair color can be white,gray,black or other else""",
        seed: int = 2024,
        gpu_num: int=0
        ) -> None:
        print(2)
        
        
        """
        DP synthetic data generator

        Parameters
        ----------
        api:
            Which foundation model API to use
        result_folder:
            Folder for storing results
        tmp_folder:
            Temporary folder for storing intermediate results
        data_loading_batch_size:
            Batch size for loading the original data
        feature_extractor_batch_size:
            Batch size for feature extractor
        prv_img_size:
            Size of original images
        conditional:
            Whether to generate class labels
        num_prv_img:
            Number of original data
        feature_extractor:
            Name of feature extractor to use
        prompt:
            Initial prompt for image generation
        seed:
            Random seed for reproducibility
        """
        
        # 0-a. Make result directory
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)


        # 0-b. Declare class variables
        print(1)
        self.api_class = get_api_class_from_name(api)  # Name of the foundation model API
        print(3)
        self.result_folder = result_folder
        self.data_loading_batch_size = data_loading_batch_size
        self.prv_img_size = prv_img_size
        self.conditional = conditional
        self.num_prv_data = num_prv_data
        self.feature_extractor = feature_extractor
        self.feature_extractor_batch_size = feature_extractor_batch_size
        self.tmp_folder = tmp_folder
        self.prompt = prompt
        self.rng = np.random.default_rng(seed)
        self.name = uuid.uuid4().hex
        self.gpu_num=gpu_num
        
        # 0-x. Set up logging
        setup_logging(os.path.join(result_folder, f'{self.name}.log'))

    def initialize(
        self,
        api_args: List = [],
        data_folder: str = "./input_data/cookie",
        num_samples: int = 10,
        img_size: str = '512x512',
        plot_images: bool = False,
        checkpoint_path: str = "./result/cookie/1/_samples.npz",
        checkpoint_step: int = 1) -> str:
        """
        Prepare everything needed for learning such
        as loading the private data, extracting 
        private features, and generaing initial samples

        Parameters
        ----------
        api_args:
            Arguments for API
        num_samples:
            Number of samples to generate
        img_size:
            Target size of image to generate
        plot_images:
            Whether to save generated images in PNG files
        checkpoint_path:
            Path to the data checkpoint
        checkpoint_step:
            Iteration of the data checkpoint

        Returns
        ----------
        str:
            Path for the initial samples
        """
        print(f"checkpoint_path:{checkpoint_path}")
        print(f"data_folder:{data_folder}")
        dataname=data_folder.split("/")[-1]
        print(f"dataname: {dataname}")
        
        if dataname=="cookie":
            self.prompt=["white or gray ragdoll cat, intricate, \
                        high detail, dramatic, extremely realistic eyes, \
                        extremely realistic hair, \
                        hair color can be white,gray,black or other else "]*self.num_prv_data
            if len(api_args) == 0:
                api_args =   [
                            '--API_checkpoint', '/root/Cubigate_ai_engine/CUBIGate/models/stable_diffusion/sdxl',
                            '--guidance_scale', '7',
                            '--inference_steps', '40',
                            '--API_batch_size', '4',
                            '--lora', "/root/Cubigate_ai_engine/CUBIGate/models/Lora/cookie_sd_xl.safetensors"
                            ]
       
        elif dataname=="lfw":
            self.prompt=["Smart Sharpe, A bust shot of "]*self.num_prv_data
            if len(api_args) == 0:
                api_args = [
                            '--API_checkpoint', '/root/Cubigate_ai_engine/CUBIGate/models/stable_diffusion/sdxl',
                            '--guidance_scale', '7',
                            '--inference_steps', '40',
                            '--API_batch_size', '4'
                            ,
                             '--lora', "/root/Cubigate_ai_engine/CUBIGate/models/Lora/naver_lora.safetensors"
                             #TODO: lora값 바꾸기
                            ]
        self.result_folder=os.path.join(self.result_folder,dataname)
        
        #TODO: auto_prompt 선텍여부
        self.auto_prompt(data_folder, dataname)    
        
        
        api_args.extend(['--prompt', self.prompt])
        # 1. Set up API instance
        self.api = self.api_class.from_command_line_args(api_args)
        # 2. Load private data
        self.all_private_samples, self.all_private_labels = load_data(
            data_dir=data_folder,
            batch_size=self.data_loading_batch_size,
            image_size=self.prv_img_size,
            class_cond=self.conditional,
            num_private_samples=self.num_prv_data)

        self.private_classes = list(sorted(set(list(self.all_private_labels))))
        self.private_num_classes = len(self.private_classes)
        assert num_samples % self.private_num_classes == 0  # Check if equal number of samples per class

        # 3. Extract features of the original data
        self.all_private_features = extract_features(
            data=self.all_private_samples,
            tmp_folder=self.tmp_folder,
            model_name=self.feature_extractor,
            res=self.prv_img_size,
            batch_size=self.feature_extractor_batch_size)

        # 4-a. Load data checkpoint if any
        if checkpoint_path != '':
            logging.info(
                f'Loading data checkpoint from {checkpoint_path}')
            self.samples = load_samples(checkpoint_path)
            if checkpoint_step < 0:
                raise ValueError('data_checkpoint_step should be >= 0')
            self.start_t = checkpoint_step + 1
            self.folder = f'{self.result_folder}/train/{self.start_t}'
        # 4-b. Generate initial population
        else:
            samples = self.api.random_sampling(
                num_samples=num_samples,
                size=img_size)
            self.folder = f'{self.result_folder}/{1}'
            log_samples(
                samples=samples,
                folder=self.folder,
                plot_samples=plot_images,
                prefix=self.name)
            if checkpoint_step >= 0:
                logging.info('Ignoring data_checkpoint_step')
            self.start_t = 1

        return f'{self.folder}/{self.name}_samples.npz'


    def train(
        self,
        iteration: int,
        epsilon: float,
        delta: float,
        data_folder: str,
        checkpoint_path: str = "",
        checkpoint_step: int = 1,
        num_samples: int = 10,
        variation_degree_schedule: List[float] = [],
        num_candidate: int = 4,
        threshold: float = 0.0,
        plot_images: bool = False,
        img_size: str = '512x512',
        mode: str = 'L2',
        api_args: List = []
        ) -> str:

        """
        Learn the distribution

        Parameters
        ----------
        api_args:
            Arguments for API
        data_folder:
            Folder of the original data
        checkpoint_path:
            Path to the data checkpoint
        checkpoint_step:
            Iteration of the data checkpoint
        num_samples:
            Number of samples to generate at each iteration
        variation_degree_schedule:
            Variation degree at each iteration
        num_candidate:
            Number of candidates to be selected for each sample
        img_size:
            Target size of image to generate
        epsilon:
            Privacy parameter, for each iteration
        delta:
            Privacy parameter
        threshold:
            Threshold to estimate the distribution
        plot_images:
            Whether to save generated images in PNG files
        mode:
            Which distance metric to use in measure

        Returns
        ----------
        str:
            Path for the final checkpoint
        """
        
        
        if len(variation_degree_schedule) == 0:
            variation_degree_schedule = [1.0-i*0.02 for i in range(iteration)]
            
        dataname=data_folder.split("/")[-1]
        if dataname=="cookie":
            self.prompt=["white or gray ragdoll cat, intricate, \
                        high detail, dramatic, extremely realistic eyes, \
                        extremely realistic hair, \
                        hair color can be white,gray,black or other else "]*self.num_prv_data
            if len(api_args) == 0:
                api_args =   [
                            '--API_checkpoint', '/root/Cubigate_ai_engine/CUBIGate/models/stable_diffusion/sdxl',
                            '--guidance_scale', '7',
                            '--inference_steps', '40',
                            '--API_batch_size', '4',
                            '--lora', "/root/Cubigate_ai_engine/CUBIGate/models/Lora/cookie_sd_xl.safetensors"
                            ]
       
       
        elif dataname=="lfw":
            self.prompt=["Smart Sharpe, A bust shot of "]*self.num_prv_data
            if len(api_args) == 0:
                api_args = [
                            '--API_checkpoint', '/root/Cubigate_ai_engine/CUBIGate/models/stable_diffusion/sdxl',
                            '--guidance_scale', '7',
                            '--inference_steps', '40',
                            '--API_batch_size', '4'
                            ,
                             '--lora', "/root/Cubigate_ai_engine/CUBIGate/models/Lora/naver_lora.safetensors"
                             #TODO: lora값 바꾸기
                            ]
       
  
        api_args.extend(['--prompt', self.prompt])
        api_args.extend(['--gpu_num', str(self.gpu_num)])
        
        # 1. Initialize

        samples_path = self.initialize(
            api_args=api_args,
            data_folder=data_folder,
            num_samples=num_samples,
            img_size=img_size,
            plot_images=plot_images,
            checkpoint_path=checkpoint_path,
            checkpoint_step=checkpoint_step
        )
        self.folder = f'{self.result_folder}/train/{self.start_t + 1}' if self.start_t == 1 else f'{self.result_folder}/train/{self.start_t}'
        # Start learning
        for t in range(self.start_t, iteration):
            logging.info(f"t={t}")
            

            # 2. Variate current samples to produce candidates
            packed_samples_path = self.variate(
                samples_path=samples_path,
                num_packing=num_candidate,
                img_size=img_size,
                variation_degree=variation_degree_schedule[t]
            )
            # 3. Measure how well the candidates of each sample fit in the distribution
            count_path = self.measure(
                samples_path=packed_samples_path,
                epsilon=epsilon,
                delta=delta,
                num_candidate=num_candidate,
                threshold=threshold,
                mode=mode
            )
            self.folder = f'{self.result_folder}/{t + 1}'
            # if not os.path.exists(f'{self.result_folder}/{dataname}'):
            #     os.mkdir(f'{self.result_folder}/{dataname}')
            # 4. Select the fittest candidate of each sample
            self.select(
                dist_path=count_path,
                samples_path=packed_samples_path,
                num_candidate=num_candidate
            )
            logging.info(f"Privacy cost so far: {get_epsilon(epsilon, t):.2f}")
        shutil.rmtree(self.tmp_folder, ignore_errors=True)
        print(self.folder)
            
        return f'{self.folder}/{self.name}_samples.npz'

    ## Generate단계에서눈 auto prompt 적용하지 않았음. 
    ## TODO: 필요시 사용
    def generate(
        self,
        base_data: str,
        img_size: str = '512x512',
        num_samples: int = 10,
        variation_degree: float = 0.5,
        api_args: List = []
    ) -> zipfile.ZipFile:
        """
        Generate images based on the distribution learned

        Parameters
        ----------
        base_data:
            Set of generated data as the learned distribution
        img_size:
            Target size of image to generate
        num_samples:
            Number of samples to generate
        variation_degree:
            Variation degree

        Returns
        ----------
        ZipFile:
            Zip file of the generated images
        """
        # /var/dp_msv/
        print(f"base_data:{base_data}")
        print(f"result_folder:{self.result_folder}")
        dataname=base_data.split("/")[4]
        # .split("/")[0]
        print(dataname)
    
    
        if dataname=="cookie":
            self.prompt=["white or gray ragdoll cat, intricate, \
                        high detail, dramatic, extremely realistic eyes, \
                        extremely realistic hair, \
                        hair color can be white,gray,black or other else"]*self.num_prv_data
            if len(api_args) == 0:
                api_args =  [
                            '--API_checkpoint', '/root/Cubigate_ai_engine/CUBIGate/models/stable_diffusion/sdxl',
                            '--guidance_scale', '7',
                            '--inference_steps', '40',
                            '--API_batch_size', '1',
                            '--lora', "/root/Cubigate_ai_engine/CUBIGate/models/Lora/cookie_sd_xl.safetensors"
                            ]
       
       
        elif dataname=="lfw":
            self.prompt=["Smart Sharpe, A bust shot of one person"]*self.num_prv_data
            if len(api_args) == 0:
                api_args = [
                            '--API_checkpoint', '/root/Cubigate_ai_engine/CUBIGate/models/stable_diffusion/sdxl',
                            '--guidance_scale', '7',
                            '--inference_steps', '40',
                            '--API_batch_size', '1',
                             '--lora', '/root/Cubigate_ai_engine/CUBIGate/models/Lora/naver_lora.safetensors'
                             #TODO: lora값 바꾸기
                            ]
        api_args.extend(['--prompt', self.prompt])
        api_args.extend(['--gpu_num', str(self.gpu_num)])
        print(f"gpu_num", f'{str(self.gpu_num)}')
        print(api_args)
        print(self.prompt)
        print(base_data)

        # 1. Make sure it has API instance
        if not hasattr(self, 'api'):
            print("attr")
            self.api = self.api_class.from_command_line_args(api_args)
            self.api._init_variate()

        # 2. Load base data
        samples = load_samples(base_data)

        # 3. Generate samples as variations of base data
        if num_samples != len(samples):
            target_idx = np.random.choice(len(samples), num_samples, replace=True)
            target_samples = samples[target_idx]
        else:
            target_samples=samples
        width, height = list(map(int, img_size.split('x')))
        gen_samples = self.api.variation(
            samples=target_samples,
            num_variations_per_sample=1,
            size=img_size,
            variation_degree=variation_degree).reshape((num_samples, width, height, -1))
        log_samples(
            samples=gen_samples,
            folder=f'{self.result_folder}/generate/{self.name}',
            plot_samples=True,
            save_npz=False)
        shutil.make_archive(
            f'{self.result_folder}/generate/{self.name}',
            'zip',
            root_dir=self.result_folder,
            base_dir=f'generate/{self.name}')
        shutil.rmtree(f'{self.result_folder}/generate/{self.name}', ignore_errors=True)
        
        #TODO: zip파일을 서버에 저장안하도록 바꾸면 더 좋을 것 같다.
        synthetic_img_zip=zipfile.ZipFile(f'{self.result_folder}/generate/{self.name}.zip')
        
        return synthetic_img_zip


    def variate(
        self,
        samples_path: str,
        num_packing: int = 4,
        img_size: str = '512x512',
        variation_degree: float = 0.5
    ) -> str:
        """
        Variate current samples to produce candidates

        Parameters
        ----------
        num_packing:
            Number of variations for each sample
        img_size:
            Target size of image to generate
        variation_degree:
            Strength of variation

        Returns:
        ----------
        str:
            Path for the result samples
        """
        samples = load_samples(samples_path)
        packed_samples = self.api.variation(
            samples=samples,
            num_variations_per_sample=num_packing,
            size=img_size,
            variation_degree=variation_degree)
        log_samples(
            samples=packed_samples,
            folder=self.folder,
            plot_samples=False,
            save_npz=True,
            prefix=f'{self.name}_packed')
        
        
        print(f"packed: {packed_samples.shape}")
        try:
            image=Image.fromarray(packed_samples[0][0])
            image.save("./lfw_sample.png")
        except:
            print("fail to save")
            pass
        return f'{self.folder}/{self.name}_packed_samples.npz'


    def measure(
        self,
        samples_path: str,
        epsilon: float,
        delta: float,
        num_candidate: int = 4,
        threshold: float = 1.0,
        mode: str = 'L2',
    ) -> str:
        """
        Measure how well the candidates of each sample fit in the distribution

        Parameters
        ----------
        samples_path:
            Path for the candidates
        epsilon:
            Privacy parameter to securely estimate the distribution
        delta:
            Privacy parameter
        num_candidate:
            Number of candidate for each sample
        threshold:
            Threshold to estimate the distribution
        mode:
            Which distance metric to use

        Returns
        ----------
        str:
            Path for the estimated distribution
        """
        samples = load_samples(samples_path)
        packed_features = []
        for sample in samples:
            sub_packed_features = extract_features(
                data=sample,
                tmp_folder=self.tmp_folder,
                model_name=self.feature_extractor,
                res=self.prv_img_size,
                batch_size=self.feature_extractor_batch_size)

            packed_features.append(sub_packed_features)
        # packed_features shape: (N_syn * num_candidate, embedding)
        packed_features = np.concatenate(packed_features, axis=0)



        count = []
        self.num_samples_per_class = samples.shape[0] // len(self.private_classes)
        num_samples_per_class_w_candidates = self.num_samples_per_class * num_candidate
        for class_i, class_ in enumerate(self.private_classes):
            sub_count, _, _ = dp_nn_histogram(
                synthetic_features=packed_features[
                    num_samples_per_class_w_candidates * class_i:
                    num_samples_per_class_w_candidates * (class_i + 1)],
                private_features=self.all_private_features[
                    self.all_private_labels == class_],
                epsilon=epsilon,
                delta=delta,
                rng=self.rng,
                mode=mode,
                threshold=threshold,
                num_candidate=num_candidate)
            count.append(sub_count)
        count = np.concatenate(count)
        log_count(count, f'{self.folder}/{self.name}_count.npz')
        return f'{self.folder}/{self.name}_count.npz'


    def select(
        self,
        dist_path: str,
        samples_path: str,
        num_candidate: int = 4
    ) -> str:
        """
        Select the fittest candidate of each sample

        Parameters
        ----------
        dist_path:
            Path for the estimated distribution
        samples_path:
            Path for the samples
        num_candidate:
            Number of candidates for each sample

        Returns
        ----------
        str:
            Path for the selected samples
        """
        samples = load_samples(samples_path)
        dist = load_count(dist_path)
        assert samples.shape[0] % self.private_num_classes == 0
        num_samples_per_class = samples.shape[0] // self.private_num_classes
        selected = []
        for class_i in range(self.private_num_classes):
            sub_count = dist[
                num_samples_per_class * class_i:
                num_samples_per_class * (class_i + 1)]
            for i in range(sub_count.shape[0]):
                indices = self.rng.choice(
                    np.arange(num_candidate),
                    size=1,
                    p = sub_count[i] / np.sum(sub_count[i])
                )
                selected.append(indices)
        selected = np.concatenate(selected)
        samples = samples[np.arange(samples.shape[0]), selected]
        print(self.folder)
        log_samples(
            samples=samples,
            folder=self.folder,
            plot_samples=False,
            save_npz=True,
            prefix=self.name)
        return f'{self.folder}/{self.name}_samples.npz'
    def auto_prompt(self, data_folder, dataname):
        
        model="Salesforce/blip-image-captioning-base"
        prompt_save_file=os.path.join(self.result_folder, dataname)
        pipe=BlipForConditionalGeneration.from_pretrained(model)
        processor=BlipProcessor.from_pretrained(model)
        # Load and preprocess the image
        # background_pipe=Interrogator(Config())
        # background_table=LabelTable(["indoor", "outdoor", "unspecific", "crowded", "simple"], "terms", background_pipe)

        filelist=np.sort(os.listdir(data_folder))

        for i, _file in enumerate(filelist):
            if i==len(self.prompt):
                break
            path=os.path.join(data_folder, _file)
            image=Image.open(path)
            inputs=processor(image, return_tensors="pt")
            out=pipe.generate(**inputs)
            # print(gen_prompts[i])
            # print(pipe(image))
            prompt=processor.decode(out[0], skip_special_token=True)
            prompt=prompt.replace("[SEP]", "")
            # prompt=prompt.replace("a man", "man")
            # prompt=prompt.replace("a woman", "woman")
            prompt=prompt.replace("tuxed", "")
            prompt=prompt.replace("fr", "")
            
            prompt=prompt.split(" and ")[0]
            # prompt=prompt+" on the photo with RGB-scale, "
            # prompt=prompt+f"background is {background_table.rank(background_pipe.image_to_features(image), top_count=1)[0]}"
            self.prompt[i]+=prompt
            
            if i ==0:
                with open( prompt_save_file, "w") as _prompt:
                    _prompt.write(self.prompt[i]+"\n")
            else:
                with open(prompt_save_file, "a") as _prompt:
                    _prompt.write(self.prompt[i]+"\n")
                    
        print(f"Auto_prompt: {self.prompt}")
        del pipe
        del processor
        torch.cuda.empty_cache()
        print(torch.cuda.memory_allocated()/1024**2)
        print(torch.cuda.memory_cached()/1024**2)
   

