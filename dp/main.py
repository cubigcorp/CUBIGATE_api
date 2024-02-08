import argparse
import logging
import os
import numpy as np
import wandb
from tqdm.auto import tqdm
from multiprocessing import Process
from dpsda.logging import setup_logging
from dpsda.data_loader import load_private_data, load_samples, load_public_data, load_count
from dpsda.feature_extractor import extract_features
from dpsda.metrics import make_fid_stats, compute_metric
from dpsda.dp_counter import dp_nn_histogram, nn_histogram
from dpsda.arg_utils import str2bool, split_args, split_schedulers_args, slice_scheduler_args
from apis import get_api_class_from_name
from dpsda.data_logger import log_samples, log_count, log_fid, visualize, log_plot, t_sne
from dpsda.tokenizer import tokenize
from dpsda.agm import get_epsilon
from dpsda.experiment import get_toy_data
from dpsda.schedulers import get_scheduler_class_from_name
from dpsda.prompt_generator import PromptGenerator
from dpsda.data_splitter import hard_split, soft_split



def parse_args():
    parser = argparse.ArgumentParser()
    toy_group = parser.add_argument_group("Toy experiment")
    toy_group.add_argument(
        '--experimental',
        type=str2bool,
        default=False,
        help="Whether it is just experimental with toy data."
    )
    toy_group.add_argument(
        '--toy_data_type',
        type=str,
        default="square_upper_right",
        help="[SHAPE]_[Y_POSITION]_[X_POSITION]"
    )
    toy_group.add_argument(
        '--toy_private_bounding_ratio',
        type=float,
        default=0.0
    )
    wandb_group = parser.add_argument_group("Wandb")
    wandb_group.add_argument(
        '--wandb_log_notes',
        type=str,
        default="",
        help="Notes to describe the experiment on wandb"
    )
    wandb_group.add_argument(
        '--wandb_log_tags',
        type=str,
        default='',
        help="Tags to classify the experiement on wandb"
    )
    wandb_group.add_argument(
        '--wandb_log_dir',
        type=str,
        default='/mnt/cubigate/',
        help="An absolute path to a directory where metadata will be stored"
    )
    wandb_group.add_argument(
        '--wandb_resume_id',
        type=str,
        default=None,
        help="Wandb run ID to resume"
    )
    privacy = parser.add_argument_group('Privacy parameters')
    privacy.add_argument(
        '--dp',
        type=str2bool,
        default=True,
        help="Whether to apply differential privacy"
    )
    privacy.add_argument(
        '--epsilon_delta_dp',
        type=str2bool,
        default=True,
        help="Whther to apply (eps, del)-DP"
    )
    privacy.add_argument(
        '--epsilon',
        type=float,
        default=0.0,
        required=False)
    privacy.add_argument(
        '--delta',
        default=0.0,
        type=float,
        required=False)

    ours_group = parser.add_argument_group("Ours")
    ours_group.add_argument(
        '--num_sub_labels_per_class',
        type=str
    )
    ours_group.add_argument(
        '--use_weight_scheduler',
        type=str2bool,
        default=False
    )
    ours_group.add_argument(
        '--weight_scheduler',
        type=str,
        default="constant",
        choices=['step', 'exponential', 'linear', 'constant', 'wlinear']
    )
    ours_group.add_argument(
        '--sample_weight',
        type=float,
        default=1.0,
        help="Weights for sample variation compared to demonstration"
    )
    ours_group.add_argument(
        '--direct_variate',
        type=str2bool,
        required=False,
        help="Whether to use candidate variations"
    )
    ours_group.add_argument(
        '--use_public_data',
        type=str2bool,
        default=False,
        help="Whether to use public data")
    ours_group.add_argument(
        '--public_data_folder',
        type=str,
        required=False,
        default='',
        help="Folder for public data if any"
    )
    ours_group.add_argument(
        '--demonstration',
        type=int,
        required=False,
        default=0)
    
    ours_group.add_argument(
        '--use_sample_specific_prompt',
        type=str2bool,
        default=False,
        help="Whether to use sample-specific prompt"
    )
    
    general = parser.add_argument_group("Generaal")
    general.add_argument(
        '--random_seed',
        type=int,
        default=2024
    )
    general.add_argument(
        '--device',
        type=int,
        default=0)
    general.add_argument(
        '--save_samples_live',
        action='store_true')
    general.add_argument(
        '--save_samples_live_freq',
        type=int,
        required=False,
        default=np.inf,
        help="Live saving Frequency")
    general.add_argument(
        '--live_loading_target',
        type=str,
        required=False)
    
    general.add_argument(
        '--modality',
        type=str,
        choices=['image', 'text', 'time-series', "tabular"], 
        default='toy')
    general.add_argument(
        '--api',
        type=str,
        required=True,
        choices=['DALLE', 'stable_diffusion', 'improved_diffusion', 'chatgpt', 'llama2', 'chat_llama2', 'toy', 'noapi'], 
        help='Which foundation model API to use')

    general.add_argument(
        '--save_each_sample',
        type=str2bool,
        default=True,
        help='Whether to save generated images in PNG files')  #False
    general.add_argument(
        '--checkpoint_sub_label',
        type=int,
        default=-1
    )
    general.add_argument(
        '--data_checkpoint_path',
        type=str,
        default="",
        help='Path to the data checkpoint')
    general.add_argument(
        '--count_checkpoint_path',
        type=str,
        default="",
        help="Path to the count checkpoint"
    )
    general.add_argument(
        '--data_checkpoint_step',
        type=int,
        default=-1,
        help='Iteration of the data checkpoint')
    general.add_argument(
        '--num_samples_schedule',
        type=str,
        default='50000,'*9 + '50000',
        help='Number of samples to generate at each iteration')
    general.add_argument(
        '--num_samples',
        type=int,
        default=0
    )
    general.add_argument(
        '--T',
        type=int,
        default=0
    )
    general.add_argument(
        '--variation_degree_schedule',
        type=str,
        default='0,'*9 + '0',
        help='Variation degree at each iteration')
    general.add_argument(
        '--use_degree_scheduler',
        type=str2bool,
        default=True
    )
    general.add_argument(
        '--degree_scheduler',
        type=str,
        default='constant',
        choices=['step', 'exponential', 'linear', 'constant', 'wlinear'],
        help='Variation degree scheduler')
    parser.add_argument(
        '--adaptive_variation_degree',
        type=str2bool,
        default=False
    )
    parser.add_argument(
        '--num_fid_samples',
        type=int,
        default=50000,
        help='Number of generated samples to compute FID')
    parser.add_argument(
        '--num_private_samples',
        type=int,
        default=50000,
        help='Number of private samples to load')
    parser.add_argument(
        '--noise_multiplier',
        type=float,
        default=0.0,
        required=False,
        help='Noise multiplier for DP NN histogram')    #noise_multiplier => how??
    parser.add_argument(
        '--num_candidate',
        type=int,
        default=0,
        help=('candidate degree for computing distances between private and '  #
              'generated images'))
    parser.add_argument(
        '--feature_extractor',
        type=str,
        default='',
        choices=['bert_base_nli_mean_tokens', 'all_mpnet_base_v2', 'inception_v3', 'clip_vit_b_32', 'original'], 
        help='Which image feature extractor to use')
    parser.add_argument(
        '--num_nearest_neighbor',
        type=int,
        default=1,
        help='Number of nearest neighbors to find in DP NN histogram')   ###몇개가 적절할지??
    parser.add_argument(
        '--nn_mode',
        type=str,
        default='L2',
        choices=['L2', 'IP', 'cosine'],
        help='Which distance metric to use in DP NN histogram')   ##Bert랑 같이 similarty로 갈지 논의
    parser.add_argument(
        '--private_sample_size',
        type=int,
        default=1024,
        help='Size of private images')
    parser.add_argument(
        '--tmp_folder',
        type=str,
        default='result/tmp',
        help='Temporary folder for storing intermediate results')
    parser.add_argument(
        '--result_folder',
        type=str,
        default='result',
        help='Folder for storing results')
    parser.add_argument(
        '--data_folder',
        type=str,
        required=False,
        default='',
        help='Folder that contains the private images')
    parser.add_argument(
        '--count_threshold',
        type=float,
        default=0.0,
        help='Threshold for DP NN histogram')  #얼마로??
    parser.add_argument(
        '--compute_fid',
        type=str2bool,
        default=False,
        help='Whether to compute FID')
    parser.add_argument(
        '--fid_dataset_name',
        type=str,
        default='customized_dataset',
        help=('Name of the dataset for computing FID against. If '
              'fid_dataset_name and fid_dataset_split in combination are one '
              'of the precomputed datasets in '
              'https://github.com/GaParmar/clean-fid and make_fid_stats=False,'
              ' then the precomputed statistics will be used. Otherwise, the '
              'statistics will be computed using the private samples and saved'
              ' with fid_dataset_name and fid_dataset_split for future use.'))
    parser.add_argument(
        '--fid_dataset_split',
        type=str,
        default='train',
        help=('Split of the dataset for computing FID against. If '
              'fid_dataset_name and fid_dataset_split in combination are one '
              'of the precomputed datasets in '
              'https://github.com/GaParmar/clean-fid and make_fid_stats=False,'
              ' then the precomputed statistics will be used. Otherwise, the '
              'statistics will be computed using the private samples and saved'
              ' with fid_dataset_name and fid_dataset_split for future use.'))
    parser.add_argument(
        '--fid_model_name',
        type=str,
        default='inception_v3',
        choices=['inception_v3', 'clip_vit_b_32'],
        help='Which embedding network to use for computing FID')
    parser.add_argument(
        '--make_fid_stats',
        type=str2bool,
        default=True,
        help='Whether to compute FID stats for the private samples')
    parser.add_argument(
        '--data_loading_batch_size',
        type=int,
        default=100,
        help='Batch size for loading private samples')
    parser.add_argument(
        '--feature_extractor_batch_size',   #BERT생각하면서 고민
        type=int,
        default=500,
        help='Batch size for feature extraction')
    parser.add_argument(
        '--fid_batch_size',
        type=int,
        default=500,
        help='Batch size for computing FID')
    parser.add_argument(
        '--gen_class_cond',
        type=str2bool,
        default=False,
        help='Whether to generate class labels')
    parser.add_argument(
        '--initial_prompt',
        action='append',
        type=str,
        help='Initial prompt for image generation. It can be specified '
             'multiple times to provide a list of prompts. If the API accepts '
             'prompts, the initial samples will be generated with these '
             'prompts')
    parser.add_argument(
        '--sample_size',
        type=str,
        default='1024x1024',
        help='Size of generated images in the format of HxW')
    args, other_args = parser.parse_known_args()
    
    args.num_samples_schedule = list(map(
        int, args.num_samples_schedule.split(',')))
    if args.direct_variate:
        assert len(set(args.num_samples_schedule)) == 1, "Number of samples should remain same during the variations"
    variation_degree_type = (float if '.' in args.variation_degree_schedule
                             else int)
    args.variation_degree_schedule = list(map(
        variation_degree_type, args.variation_degree_schedule.split(',')))
    args.wandb_log_tags = args.wandb_log_tags.split(',') if args.wandb_log_tags != '' else None
    args.toy_data_type = args.toy_data_type.split('_')

    if args.num_samples > 0:
        assert args.T > 0, "Specify how many variations to run."

    if ((not args.use_degree_scheduler) and args.T == 0 ):
        if (len(args.num_samples_schedule) != len(args.variation_degree_schedule)):
            raise ValueError('The length of num_samples_schedule and '
                            'variation_degree_schedule should be the same')
    if args.sample_weight < 1:
        assert args.demonstration > 0

    args.num_sub_labels_per_class = list(map(int, args.num_sub_labels_per_class.split(',')))
    
    return args, other_args


def function(args, other_args, all_private_samples, all_private_labels, all_private_features, num_samples, sub_label) -> str:
    # Schedulers, prompt generator
    if args.use_degree_scheduler or args.use_weight_scheduler or args.use_sample_specific_prompt:
        api_args, scheduler_args, prompt_args = split_args(other_args)
        if not args.use_degree_scheduler:  # weight scheduler only
            weight_args = slice_scheduler_args(scheduler_args)
        elif not args.use_weight_scheduler:  # degree scheduler only 
            degree_args = slice_scheduler_args(scheduler_args)
        else:  # both
            weight_args, degree_args = split_schedulers_args(scheduler_args)
    else:
        api_args = other_args
    # API
    api_class = get_api_class_from_name(args.api)
    live_save_folder = args.result_folder if args.save_samples_live else None
    api = api_class.from_command_line_args(api_args, live_save_folder, args.live_loading_target, args.save_samples_live_freq, args.modality)
    # Schedulers
    T = args.T if args.T > 0 else len(args.num_samples_schedule)
    if args.use_degree_scheduler:
        degree_scheduler_class = get_scheduler_class_from_name(args.degree_scheduler)
        degree_scheduler = degree_scheduler_class.from_command_line_args(args=degree_args, T=T)
    else:
        degree_scheduler =None
    if args.use_weight_scheduler:
        assert args.demonstration > 0
        weight_scheduler_class = get_scheduler_class_from_name(args.weight_scheduler)
        weight_scheduler = weight_scheduler_class.from_command_line_args(args=weight_args, T=T)
    else:
        weight_scheduler =None

    if args.epsilon_delta_dp:
        args.delta = 1 / num_samples

    private_classes = list(map(int, sorted(set(list(all_private_labels)))))
    private_num_classes = len(private_classes)
    logging.info(f'Private_num_classes: {private_num_classes}')

    metric = "FID" if args.num_private_samples > 2048 else "KID"
    rng = np.random.default_rng(args.random_seed)
    
    config = dict(vars(args), **vars(api.args))
    if args.use_degree_scheduler:
        config.update(**vars(degree_scheduler.args))
    if args.use_weight_scheduler:
        config.update(**vars(weight_scheduler.args))
    wandb.init(
        entity='cubig_ai',
        project="AZOO",
        config=config,
        notes=args.wandb_log_notes,
        tags=args.wandb_log_tags,
        dir=args.wandb_log_dir,
        id=args.wandb_resume_id
    )
    run_folder = f'{args.result_folder}/{sub_label}_{wandb.run.name}'
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
    if (not args.dp) and ((args.epsilon > 0) or (args.noise_multiplier > 0) or (args.direct_variate)) :
        logging.info("You set it non-dp. Privacy parameters are ignored.")
        args.direct_variate = True

    if args.experimental:
        logging.info("This is an experimental run. All the unnecessary parameters are ignored.")
        args.make_fid_stats = False
        args.data_checkpoint_path = args.public_data_folder = args.data_folder = ''
        args.use_public_data = False
        args.modality = 'toy'
        args.feature_extractor = ''
        args.save_each_sample = False
        
    logging.info(f'config: {args}')
    logging.info(f'API config: {api.args}')

    if args.make_fid_stats:
        logging.info(f'Computing {metric} stats')
        make_fid_stats(
            samples=all_private_samples,
            dataset=args.fid_dataset_name,
            dataset_res=args.private_sample_size,
            dataset_split=args.fid_dataset_split,
            tmp_folder=args.tmp_folder,
            model_name=args.fid_model_name,
            batch_size=args.fid_batch_size,
            modality=args.modality,
            metric=metric)

    
    num_samples_per_class = num_samples // private_num_classes
    synthetic_labels = np.repeat(private_classes, num_samples_per_class)

    # Generating initial samples.
    if args.data_checkpoint_path != '':
        logging.info(
            f'Loading data checkpoint from {args.data_checkpoint_path}')
        samples, additional_info = load_samples(args.data_checkpoint_path)
        if args.direct_variate and args.data_checkpoint_step >= 1:
            assert args.count_checkpoint_path != '', "Count information must be provided with data checkpoint."
            (count, losers) = load_count(args.count_checkpoint_path)
            assert samples.shape[0] == count.shape[0], "The number of count should be equal to the number of synthetic samples"
        if args.data_checkpoint_step < 0:
            raise ValueError('data_checkpoint_step should be >= 0')
        if args.use_weight_scheduler:
            weight_scheduler.set_from_t(args.data_checkpoint_step)
        if args.use_degree_scheduler:
            degree_scheduler.set_from_t(args.data_checkpoint_step)
        start_t = args.data_checkpoint_step + 1
    else:
        if args.use_public_data:
            logging.info(f'Using public data in {args.public_data_folder} as initial samples')
            samples, additional_info = load_public_data(
                data_folder=args.public_data_folder,
                modality=args.modality,
                num_public_samples=num_samples,
                prompt=args.initial_prompt)
            start_t = 1
        else:
            if args.use_sample_specific_prompt:
                generator = PromptGenerator(args.initial_prompt[0], prompt_args, rng)
                generator.generate(num_samples)
                wandb.config.update(generator.tag_prompts)
                prompts = None
            else:
                prompts = args.initial_prompt
                generator = None
            logging.info('Generating initial samples')
            samples, additional_info = api.random_sampling(
                prompts=prompts,
                generator=generator,
                num_samples=num_samples,
                size=args.sample_size)
            logging.info(f"Generated initial samples: {len(samples)}")
            if args.modality == 'text':
                tokens = [tokenize(args.feature_extractor, sample) for sample in samples]
                tokens = np.array(tokens)
            elif args.modality == 'toy':
                tokens = samples[:, :2]
            else:
                tokens = samples
            tsne_p = Process(target=t_sne, kwargs={
                'private_samples': all_private_samples,
                'synthetic_samples': tokens,
                'private_labels': all_private_labels,
                'synthetic_labels': synthetic_labels,
                't': 0,
                'dir': run_folder
            })
            tsne_p.start()
            if args.modality == 'image':
                visualize(
                    samples=samples[:100],
                    count=np.arange(len(samples)),
                    folder=run_folder,
                    suffix='first_100',
                    t=0)
            log_samples(
                samples=samples,
                additional_info=additional_info,
                folder=f'{run_folder}/{0}',
                save_each_sample=args.save_each_sample,
                modality=args.modality)
            if args.data_checkpoint_step >= 0:
                logging.info('Ignoring data_checkpoint_step'),
            start_t = 1

        if args.experimental:
            log_plot(private_samples=all_private_samples,
                     synthetic_samples=samples, 
                     step=0,
                     dir=run_folder,)
    
    
    if args.compute_fid:
        logging.info(f'Computing {metric}')
        if args.modality == 'text' or args.modality == 'time-series' or args.modality=="tabular":
                tokens = [tokenize(args.fid_model_name, sample) for sample in samples]
                tokens = np.array(tokens)
        else:
            tokens = samples

        fid = compute_metric(
            samples=tokens,
            tmp_folder=args.tmp_folder,
            num_fid_samples=args.num_fid_samples,
            dataset_res=args.private_sample_size,
            dataset=args.fid_dataset_name,
            dataset_split=args.fid_dataset_split,
            model_name=args.fid_model_name,
            batch_size=args.fid_batch_size,
            modality=args.modality,
            metric=metric)
        logging.info(f'{metric}={fid}')
        log_fid(run_folder, fid, 0)
        wandb.log({f'{metric}': fid})

    T = len(args.num_samples_schedule) if args.T == 0 else args.T + 1
    if args.epsilon > 0 and args.dp:
        total_epsilon = get_epsilon(args.epsilon, T)
        logging.info(f"Expected total epsilon: {total_epsilon:.2f}")
        logging.info(f"Expected privacy cost per t: {args.epsilon:.2f}")

    for t in range(start_t, T):
        logging.info(f't={t}')
        assert samples.shape[0] % private_num_classes == 0
        num_samples_per_class = samples.shape[0] // private_num_classes
        variation_degree_t = degree_scheduler.step() if args.use_degree_scheduler else args.variation_degree_schedule[t]
        wandb.log({"variation_degree":variation_degree_t, "t": t})
        if args.num_candidate == 0:
            packed_samples = np.expand_dims(samples, axis=1)
        else:
            # adaptive variation degree - count 정보가 있을 때만 적용
            if (args.adaptive_variation_degree) and ('count' in vars()):
                logging.info("Calculating adaptive variation degree")
                logging.info(f"Maximum degree for t={t}: {variation_degree_t:.2f}")
                variation_degree = []
                for class_i, class_ in enumerate(private_classes):
                    # count: (Nsyn)
                    # samples: (Nsyn, ~)
                    sub_count = count[
                            num_samples_per_class * class_i:
                            num_samples_per_class * (class_i + 1)]
                    sub_num_vote = all_private_features[
                        all_private_labels == class_].shape[0] - args.count_threshold
                    sub_ratio = np.divide(sub_count, sub_num_vote)
                    share = 1 - np.clip(sub_ratio, 0, 0.9)
                    sub_degree = np.multiply(share, variation_degree_t)
                    variation_degree.append(sub_degree)
                variation_degree = np.concatenate(variation_degree)
                logging.info(f'Largest variation degrees: {np.flip(np.sort(variation_degree))[:50]}')
            else:
                variation_degree = variation_degree_t
                    
            # demonstration
            if ((args.sample_weight < 1) or args.use_weight_scheduler) and ('count' in vars()):
                demo_indices = []  # (Nsyn * demonstrations)
                logging.info('Getting demonstrations')
                for class_i in private_classes:
                    sub_count = count[
                        num_samples_per_class * class_i:
                        num_samples_per_class * (class_i + 1)]
                    sub_losers = losers[
                        num_samples_per_class * class_i:
                        num_samples_per_class * (class_i + 1)]
                    sub_count[sub_losers] = 0
                    # Sort counts
                    sub_counts_sorted_idx = np.flip(np.argsort(sub_count))
                    # Only superior samples as demonstration
                    sub_counts_idx = np.tile(sub_counts_sorted_idx, (len(sub_counts_sorted_idx), 1))
                    sub_row_idx, sub_col_idx = np.indices(sub_counts_idx.shape)
                    sub_counts_idx[sub_col_idx >= sub_row_idx] = -1
                    sub_p = np.array([sub_count[idx] if idx >= 0 else 0 for idx in sub_counts_idx.flat]).reshape((num_samples, num_samples))
                    with np.errstate(divide='ignore', invalid='ignore'):
                        sub_p = np.nan_to_num(sub_p / np.sum(sub_p, axis=1).reshape((-1, 1)))
                    # Sampling demonstrations' index
                    num_demo = [args.demonstration if idx >= args.demonstration else idx for idx in range(len(sub_count))]
                    sub_indices = [
                        np.repeat(-1, args.demonstration) if np.all(sub_p[idx]==0)
                        else np.concatenate((rng.choice(sub_counts_idx[idx], num_demo[idx], replace=False, p=sub_p[idx]), np.full(abs(args.demonstration - num_demo[idx]), -1)))
                        for idx in range(len(sub_count))]
                    sub_indices = np.stack(sub_indices)
                    demo_indices.append(sub_indices)
                demo_indices = np.concatenate(demo_indices)
                no_demo_filter = demo_indices == -1
                demo_indices[no_demo_filter] = 0
                demo_samples = samples[demo_indices]
                demo_samples[no_demo_filter] = np.zeros_like(samples[0])  # (Nsyn, demo, ~)
                logging.info(f'Demonstration samples shape: {demo_samples.shape}')
                
                # Assigning weights to demonstrations
                demo_counts = count[demo_indices].reshape((-1, args.demonstration))
                demo_counts[no_demo_filter] = 0.
                with np.errstate(divide='ignore', invalid='ignore'):
                    demo_weights = np.nan_to_num(demo_counts / np.sum(demo_counts, axis=1).reshape((-1, 1)))
            else:
                demo_samples  = demo_weights = None
            logging.info('Running sample variation')
            sample_weight_t = weight_scheduler.step() if args.use_weight_scheduler else args.sample_weight
            wandb.log({"sample_weight": sample_weight_t, "t": t})
            packed_samples =api.variation(
                samples=samples,
                additional_info=additional_info,
                num_variations_per_sample=args.num_candidate,
                size=args.sample_size,
                variation_degree=variation_degree,
                t=t,
                candidate=True,
                demo_samples=demo_samples,
                demo_weights = demo_weights,
                sample_weight=sample_weight_t)
            if args.direct_variate:
                # 현재 샘플도 후보로 넣음
                packed_samples = np.concatenate((np.expand_dims(samples, axis=1), packed_samples), axis=1)

        if args.modality == 'text' or args.modality == 'time-series' or args.modality=="tabular":
            packed_tokens = []
            for packed_sample in packed_samples:
                tokens = [tokenize(args.feature_extractor, t) for t in packed_sample]
                sub_tokens = np.array(tokens)
                packed_tokens.append(sub_tokens)
            packed_tokens = np.array(packed_tokens)
        
        else:
            packed_tokens = packed_samples

        packed_features = []
        logging.info('Running feature extraction')

        for packed in tqdm(packed_tokens, "Extracting features from synthetic samples", unit="sample"):
            if args.experimental:
                sub_packed_features = packed[:, :2]
            else:
                sub_packed_features = extract_features(
                    data=packed,
                    tmp_folder=args.tmp_folder,
                    model_name=args.feature_extractor,
                    res=args.private_sample_size,
                    batch_size=args.feature_extractor_batch_size,
                    device=f'cuda:{args.device}',
                    use_dataparallel=False,
                    modality=args.modality,
                    verbose=False)

            packed_features.append(sub_packed_features)
        if args.direct_variate:  # candidate로 생성한 variation을 사용할 경우
            # packed_features shape: (N_syn * num_candidate, embedding)
            packed_features = np.concatenate(packed_features, axis=0)
        else:  # 기존 DPSDA
            # packed_features shape: (N_syn, embedding)
            packed_features = np.mean(packed_features, axis=1)

        logging.info('Computing histogram')
        losers = []
        count = []
        count_1st_idx = []
        if args.direct_variate:
            num_samples_per_class_w_candidate = num_samples_per_class * (args.num_candidate + 1)
            num_candidate = packed_samples.shape[1]
        else:
            num_candidate = 0
            num_samples_per_class_w_candidate = num_samples_per_class
        for class_i, class_ in enumerate(private_classes):
            if args.dp:
                sub_count, sub_losers, sub_1st_idx = dp_nn_histogram(
                    synthetic_features=packed_features[
                        num_samples_per_class_w_candidate * class_i:
                        num_samples_per_class_w_candidate * (class_i + 1)],
                    private_features=all_private_features[
                        all_private_labels == class_],
                    epsilon=args.epsilon,
                    delta=args.delta,
                    noise_multiplier=args.noise_multiplier,
                    num_nearest_neighbor=args.num_nearest_neighbor,
                    mode=args.nn_mode,
                    threshold=args.count_threshold,
                    num_candidate=num_candidate,
                    rng=rng,
                    device=args.device,
                    dir=run_folder,
                    step = t)
            else:
                sub_count, sub_losers, sub_1st_idx = nn_histogram(
                    synthetic_features=packed_features[
                        num_samples_per_class_w_candidate * class_i:
                        num_samples_per_class_w_candidate * (class_i + 1)],
                    private_features=all_private_features[
                        all_private_labels == class_],
                    mode=args.nn_mode,
                    num_candidate=num_candidate,
                    device=args.device
                )
            losers.append(sub_losers)
            count.append(sub_count)
            count_1st_idx.append(sub_1st_idx)
        losers = np.concatenate(losers)
        count = np.concatenate(count)
        count_1st_idx = np.concatenate(count_1st_idx)
        
        logging.info('Generating new indices')
        if num_samples == 0:
            assert args.num_samples_schedule[t] % private_num_classes == 0
            new_num_samples_per_class = (
                args.num_samples_schedule[t] // private_num_classes)
        else:
            new_num_samples_per_class = num_samples_per_class

        if args.direct_variate:
            logging.info(f"Selected candidates: {count_1st_idx}")
            new_new_samples = packed_samples[np.arange(num_samples), count_1st_idx]
            new_new_additional_info = additional_info
            log_count(
                count,
                None,
                losers,
                f'{run_folder}/{t}/count.npz')
        else:
            new_indices = []
            for class_i in private_classes:
                sub_count = count[
                    num_samples_per_class * class_i:
                    num_samples_per_class * (class_i + 1)]
                sub_indices = rng.choice(
                    np.arange(num_samples_per_class * class_i,
                            num_samples_per_class * (class_i + 1)),
                    size=new_num_samples_per_class,
                    p=sub_count / np.sum(sub_count))
                    
                new_indices.append(sub_indices)
            new_indices = np.concatenate(new_indices)
            new_samples = samples[new_indices]
            new_additional_info = additional_info[new_indices]
            logging.info(f"new_indices: {new_indices}")
            logging.info('Generating new samples')
            new_new_samples = api.variation(
                samples=new_samples,
                additional_info=new_additional_info,
                num_variations_per_sample=1,
                size=args.sample_size,
                variation_degree=variation_degree,
                t=t,
                candidate=False)
            new_new_samples = np.squeeze(new_new_samples, axis=1)
            new_new_additional_info = new_additional_info
        

        if args.compute_fid:
            logging.info(f'Computing {metric}')
            if args.modality == 'text' or args.modality == 'time-series' or args.modality=="tabular":
                new_new_tokens = [tokenize(args.fid_model_name, sample) for sample in new_new_samples]
                new_new_tokens = np.array(new_new_tokens)
            else:
                new_new_tokens = new_new_samples
            new_new_fid = compute_metric(
                new_new_tokens,
                tmp_folder=args.tmp_folder,
                num_fid_samples=args.num_fid_samples,
                dataset_res=args.private_sample_size,
                dataset=args.fid_dataset_name,
                dataset_split=args.fid_dataset_split,
                model_name=args.fid_model_name,
                batch_size=args.fid_batch_size,
                modality=args.modality,
                device=f'cuda:{args.device}',
                metric=metric)
            logging.info(f'fid={new_new_fid}')
            log_fid(run_folder, new_new_fid, t)

        tsne_p.join()
        wandb.log({'t-SNE': wandb.Image(f'{run_folder}/{t-1}_t-SNE.png'), 't': t-1})
        samples = new_new_samples
        additional_info = new_new_additional_info
        if args.modality == 'text':
                tokens = [tokenize(args.feature_extractor, sample) for sample in samples]
                tokens = np.array(packed_tokens)
        elif args.modality == 'toy':
                tokens = samples[:, :2]
        else:
            tokens = samples
        tsne_p = Process(target=t_sne, kwargs={
                'private_samples': all_private_samples,
                'synthetic_samples': tokens,
                'private_labels': all_private_labels,
                'synthetic_labels': synthetic_labels,
                't': t,
                'dir': run_folder
            })
        tsne_p.start()
        if args.modality == 'image':
            for class_i, class_ in enumerate(private_classes):
                visualize(
                    samples=samples[
                        num_samples_per_class_w_candidate * class_i:
                        num_samples_per_class_w_candidate * (class_i + 1)],
                    packed_samples=packed_samples[
                        num_samples_per_class_w_candidate * class_i:
                        num_samples_per_class_w_candidate * (class_i + 1)],
                    count=count[
                        num_samples_per_class_w_candidate * class_i:
                        num_samples_per_class_w_candidate * (class_i + 1)],
                    folder=run_folder,
                    t=t,
                    suffix=f'class{class_}')
                visualize(
                    samples=samples[
                        num_samples_per_class_w_candidate * class_i:
                        num_samples_per_class_w_candidate * (class_i + 1)],
                    count=count[
                        num_samples_per_class_w_candidate * class_i:
                        num_samples_per_class_w_candidate * (class_i + 1)],
                    folder=run_folder,
                    t=t,
                    suffix=f'class{class_}')
        if args.experimental:
            log_plot(private_samples=all_private_samples,
                     synthetic_samples=samples, 
                     step=t,
                     dir=run_folder)
    
        log_samples(
            samples=samples,
            additional_info=additional_info,
            folder=f'{run_folder}/{t}',
            save_each_sample=args.save_each_sample,
            modality=args.modality)
        if args.dp:
            eps = get_epsilon(args.epsilon, t)
            logging.info(f"Privacy cost so far: {eps:.2f}")
            wandb.log({"epsilon": eps, "t": t})

    
    tsne_p.join()
    wandb.log({'t-SNE': wandb.Image(f'{run_folder}/{args.T}_t-SNE.png'), 't': args.T})
    wandb.finish()
    return f'{run_folder}/{args.T}/_samples.npz'


def main(super_label: int):    
    args, other_args = parse_args()
    os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()
    args.result_folder = f'{args.result_folder}/{os.environ["WANDB_RUN_GROUP"]}'
    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)
    log_file = os.path.join(args.result_folder, 'log.log')
    setup_logging(log_file)
    num_sub_labels = args.num_sub_labels_per_class[super_label]
    logging.info(f'{os.environ["WANDB_RUN_GROUP"]} started')

    if args.experimental:
        all_private_samples, all_private_labels = get_toy_data(
            shape=args.toy_data_type[0],
            y_position=args.toy_data_type[1],
            x_position=args.toy_data_type[2],
            num_data=args.num_private_samples,
            ratio=args.toy_private_bounding_ratio,
            num_labels=1,
            size=args.sample_size,
            seed=args.random_seed
        )
        all_private_features = all_private_samples[:, :2]

    else:
        all_private_samples, all_private_labels = load_private_data(
            data_dir=args.data_folder,
            batch_size=args.data_loading_batch_size,
            sample_size=args.private_sample_size,
            class_cond=args.gen_class_cond,
            num_private_samples=args.num_private_samples,
            modality=args.modality,
            model=args.feature_extractor)
        logging.info('Extracting features')
        all_private_features = extract_features(
            data=all_private_samples,
            tmp_folder=args.tmp_folder,
            model_name=args.feature_extractor,
            res=args.private_sample_size,
            batch_size=args.feature_extractor_batch_size,
            device=f'cuda:{args.device}',
            use_dataparallel=False,
            modality=args.modality)
    logging.info(f'all_private_features.shape: {all_private_features.shape}')

    # Clustering
    all_private_sub_labels = soft_split(X=all_private_features, num_sub_label=num_sub_labels, folder=args.result_folder)
    private_sub_labels = list(map(int, sorted(set(list(all_private_sub_labels)))))
    logging.info(f'Total {num_sub_labels} sub-labels for super-label {super_label}')

    all_synthetic_samples = []
    additional_info = []
    total_samples = args.num_samples_schedule[0] if args.num_samples == 0 else args.num_samples
    accum_samples = 0
    for i, sub_label in enumerate(private_sub_labels):
        if sub_label < args.checkpoint_sub_label:
            continue
        indices = all_private_sub_labels == sub_label
        # 딱 떨어지지 않는 경우 마지막에 더 만들어줌
        if i == len(private_sub_labels) - 1:
            num_samples = total_samples - accum_samples
        else:
            portion = indices.sum() / args.num_private_samples
            num_samples = int(np.ceil(total_samples * portion))
            accum_samples += num_samples
        logging.info(f"{i+1}th sub-label's private_samples: {indices.sum()}")
        logging.info(f'Num synthetic samples: {num_samples}')
        path = function(args=args,
                 other_args=other_args,
                 all_private_samples=all_private_samples[indices],
                 all_private_labels=all_private_labels[indices],
                 all_private_features=all_private_features[indices],
                 num_samples=num_samples,
                 sub_label=sub_label)
        sub_synthetic_samples, sub_additional_info = load_samples(path)
        logging.info(f'{i+1}th sub-labels samples: {sub_synthetic_samples.shape}')
        all_synthetic_samples.append(sub_synthetic_samples)
        additional_info.extend(sub_additional_info)
    all_synthetic_samples = np.concatenate(all_synthetic_samples)
    additional_info = np.array(additional_info)
    log_samples(
        samples=all_synthetic_samples,
        folder=args.result_folder,
        save_each_sample=args.save_each_sample,
        modality=args.modality,
        additional_info=additional_info,
        save_npz=True
    )
    if args.experimental:
        log_plot(private_samples=all_private_samples, synthetic_samples=all_synthetic_samples, dir=args.result_folder)
    
    # artifact = wandb.Artifact(name="log", type="dataset")
    # artifact.add_file(local_path=log_file, name=os.environ["WANDB_RUN_GROUP"])
    # wandb.log_artifact(artifact)
    # os.remove(log_file)

    


if __name__ == '__main__':
    main(0)
