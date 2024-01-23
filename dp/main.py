import argparse
import logging
import os
import numpy as np
import imageio
import wandb
from torchvision.utils import make_grid
import torch
from typing import List
from dpsda.logging import setup_logging
from dpsda.data_loader import load_private_data, load_samples, load_public_data, load_count
from dpsda.feature_extractor import extract_features
from dpsda.metrics import make_fid_stats
from dpsda.metrics import compute_metric
from dpsda.dp_counter import dp_nn_histogram, nn_histogram
from dpsda.arg_utils import str2bool, split_args, split_schedulers_args
from apis import get_api_class_from_name
from dpsda.data_logger import log_samples, log_count
from dpsda.tokenizer import tokenize
from dpsda.agm import get_epsilon
from dpsda.experiment import get_toy_data, log_plot
from dpsda.schedulers.scheduler import get_scheduler_class_from_name



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
    parser.add_argument(
        '--diversity_lower_bound',
        type=float,
        default=0.5,
        help="Lower bound for diversity as the ratio of samples who are winners"
    )
    parser.add_argument(
        '--loser_lower_bound',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--dp',
        type=str2bool,
        default=True
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=2024
    )
    ours_group = parser.add_argument_group("Ours")
    ours_group.add_argument(
        '--use_weight_scheduler',
        type=str2bool,
        default=False
    )
    ours_group.add_argument(
        '--weight_scheduler',
        type=str,
        default=""
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
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.0,
        required=False)
    parser.add_argument(
        '--delta',
        default=0.0,
        type=float,
        required=False)
    parser.add_argument(
        '--device',
        type=int,
        default=0)
    parser.add_argument(
        '--save_samples_live',
        action='store_true')
    parser.add_argument(
        '--save_samples_live_freq',
        type=int,
        required=False,
        default=np.inf,
        help="Live saving Frequency")
    parser.add_argument(
        '--live_loading_target',
        type=str,
        required=False)
    ours_group.add_argument(
        '--demonstration',
        type=int,
        required=False,
        default=0)
    parser.add_argument(
        '--modality',
        type=str,
        choices=['image', 'text', 'time-series', "tabular"], 
        default='toy')
    parser.add_argument(
        '--api',
        type=str,
        required=True,
        choices=['DALLE', 'stable_diffusion', 'improved_diffusion', 'chatgpt', 'llama2', 'chat_llama2', 'toy', 'noapi'], 
        help='Which foundation model API to use')
    parser.add_argument(
        '--plot_images',
        type=str2bool,
        default=True,
        help='Whether to save generated images in PNG files')  #False
    parser.add_argument(
        '--data_checkpoint_path',
        type=str,
        default="",
        help='Path to the data checkpoint')
    parser.add_argument(
        '--count_checkpoint_path',
        type=str,
        default="",
        help="Path to the count checkpoint"
    )
    parser.add_argument(
        '--data_checkpoint_step',
        type=int,
        default=-1,
        help='Iteration of the data checkpoint')
    parser.add_argument(
        '--num_samples_schedule',
        type=str,
        default='50000,'*9 + '50000',
        help='Number of samples to generate at each iteration')
    parser.add_argument(
        '--num_samples',
        type=int,
        default=0
    )
    parser.add_argument(
        '--T',
        type=int,
        default=0
    )
    parser.add_argument(
        '--variation_degree_schedule',
        type=str,
        default='0,'*9 + '0',
        help='Variation degree at each iteration')
    parser.add_argument(
        '--use_degree_scheduler',
        type=str2bool
    )
    parser.add_argument(
        '--variation_degree_scheduler',
        type=str,
        default='linear',
        choices=['step', 'exponential', 'linear', 'constant'],
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
    if args.use_degree_scheduler or args.use_weight_scheduler:
        api_args, scheduler_args = split_args(other_args)
        if not args.use_degree_scheduler:  # weight scheduler only
            weight_args = scheduler_args
        elif not args.use_weight_scheduler:  # degree scheduler only 
            degree_args = scheduler_args
        else:  # both
            weight_args, degree_args = split_schedulers_args(scheduler_args)
    else:
        api_args = other_args
    live_save_folder = args.result_folder if args.save_samples_live else None
    args.num_samples_schedule = list(map(
        int, args.num_samples_schedule.split(',')))
    if args.direct_variate:
        assert len(set(args.num_samples_schedule)) == 1, "Number of samples should remain same during the variations"
    # if args.loser_lower_bound == 0:
    #     args.loser_lower_bound = 1 / args.num_candidate
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
    T = args.T if args.T > 0 else len(args.num_samples_schedule)
    if args.sample_weight < 1:
        assert args.demonstration > 0
    api_class = get_api_class_from_name(args.api)
    api = api_class.from_command_line_args(api_args, live_save_folder, args.live_loading_target, args.save_samples_live_freq, args.modality)
    if args.use_degree_scheduler:
        degree_scheduler_class = get_scheduler_class_from_name(args.variation_degree_scheduler, 'degree')
        degree_scheduler = degree_scheduler_class.from_command_line_args(args=degree_args, T=T)
    else:
        degree_scheduler =None
    if args.use_weight_scheduler:
        assert args.demonstration > 0
        weight_scheduler_class = get_scheduler_class_from_name(args.weight_scheduler, 'weight')
        weight_scheduler = weight_scheduler_class.from_command_line_args(args=weight_args, T=T)
    else:
        weight_scheduler =None
    return args, api, degree_scheduler, weight_scheduler


def round_to_uint8(image):
    return np.around(np.clip(image, a_min=0, a_max=255)).astype(np.uint8)


def visualize(samples, packed_samples, count, folder, suffix=''):
    if not os.path.exists(folder):
        os.makedirs(folder)
    samples = samples.transpose((0, 3, 1, 2))
    packed_samples = packed_samples.transpose((0, 1, 4, 2, 3))

    ids = np.argsort(count)[::-1][:5]
    print(count[ids])
    vis_samples = []
    for i in range(len(ids)):
        vis_samples.append(samples[ids[i]])
        for j in range(packed_samples.shape[1]):
            vis_samples.append(packed_samples[ids[i]][j])
    vis_samples = np.stack(vis_samples)
    vis_samples = make_grid(
        torch.Tensor(vis_samples),
        nrow=packed_samples.shape[1] + 1).numpy().transpose((1, 2, 0))
    vis_samples = round_to_uint8(vis_samples)
    imageio.imsave(
        os.path.join(folder, f'visualize_top_{suffix}.png'), vis_samples)

    ids = np.argsort(count)[:5]
    print(count[ids])
    vis_samples = []
    for i in range(len(ids)):
        vis_samples.append(samples[ids[i]])
        for j in range(packed_samples.shape[1]):
            vis_samples.append(packed_samples[ids[i]][j])
    vis_samples = np.stack(vis_samples)
    vis_samples = make_grid(
        torch.Tensor(vis_samples),
        nrow=packed_samples.shape[1] + 1).numpy().transpose((1, 2, 0))
    vis_samples = round_to_uint8(vis_samples)
    imageio.imsave(
        os.path.join(folder, f'visualize_bottom_{suffix}.png'), vis_samples)


def log_fid(folder, fid, t):
    with open(os.path.join(folder, 'fid.csv'), 'a') as f:
        f.write(f'{t} {fid}\n')


def wandb_logging(private_labels: List, diversity: List, first_vote_only: List, t: int, losers: List=None):
    if len(private_labels) > 1:
        # diversity logging
        wandb.Table.MAX_ROWS = len(diversity)
        div_table = wandb.Table(columns=private_labels, data=diversity)
        vote_table = wandb.Table(columns=private_labels, data=first_vote_only)
        wandb.log({"diversity": div_table, "fist_vote_only": vote_table, "t": t})
    else:
        wandb.log({"diversity": diversity[0][0], "first_vote_only": first_vote_only[0][0], "t":t})

    if losers is not None:
        loser_table = wandb.Table(columns=private_labels, data=losers)
        wandb.log({"has_lost": loser_table, "t": t})


def main():
    args, api, degree_scheduler, weight_scheduler = parse_args()
    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)
    log_file = os.path.join(args.result_folder, 'log.log')
    setup_logging(log_file)
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
        args.initial_prompt = args.toy_data_type
        
    logging.info(f'config: {args}')
    logging.info(f'API config: {api.args}')
    config = dict(vars(args), **vars(api.args))
    if args.use_degree_scheduler:
        config.update(**vars(degree_scheduler.args))
    wandb.init(
        entity='cubig_ai',
        project="AZOO",
        config=config,
        notes=args.wandb_log_notes,
        tags=args.wandb_log_tags,
        dir=args.wandb_log_dir,
        id=args.wandb_resume_id
    )

    metric = "FID" if args.num_private_samples > 2048 else "KID"
    rng = np.random.default_rng(args.random_seed)

    if args.experimental:
        all_private_samples, all_private_labels = get_toy_data(
            shape=args.toy_data_type[0],
            y_position=args.toy_data_type[1],
            x_position=args.toy_data_type[2],
            num_data=args.num_private_samples,
            ratio=args.toy_private_bounding_ratio,
            num_labels=1,
            size=args.sample_size,
            rng=rng
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

    private_classes = list(map(int, sorted(set(list(all_private_labels)))))
    private_num_classes = len(private_classes)
    logging.info(f'Private_num_classes: {private_num_classes}')

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
            device=f'cuda:{args.device}',
            metric=metric)

    num_samples = args.num_samples_schedule[0] if args.num_samples == 0 else args.num_samples
    num_samples_per_class = num_samples // private_num_classes

    # Generating initial samples.
    if args.data_checkpoint_path != '':
        logging.info(
            f'Loading data checkpoint from {args.data_checkpoint_path}')
        samples, additional_info = load_samples(args.data_checkpoint_path)
        if args.direct_variate:
            assert args.count_checkpoint_path != '', "Count information must be provided with data checkpoint."
            (count, accum_loser) = load_count(args.count_checkpoint_path)
            assert samples.shape[0] == count.shape[0], "The number of count should be equal to the number of synthetic samples"
            diversity = 1 - np.sum(accum_loser, axis=1) / num_samples_per_class
            first_vote_only = diversity > args.diversity_lower_bound
        if args.data_checkpoint_step < 0:
            raise ValueError('data_checkpoint_step should be >= 0')
        start_t = args.data_checkpoint_step + 1
    else:
        accum_loser = np.full((private_num_classes, num_samples), False)
        diversity = np.full((private_num_classes), 1.0)
        first_vote_only = np.full((private_num_classes), True)
        if args.use_public_data:
            logging.info(f'Using public data in {args.public_data_folder} as initial samples')
            samples, additional_info = load_public_data(
                data_folder=args.public_data_folder,
                modality=args.modality,
                num_public_samples=num_samples,
                prompt=args.initial_prompt)
            start_t = 1
        else:
            
            logging.info('Generating initial samples')
            samples, additional_info = api.random_sampling(
                prompts=args.initial_prompt,
                num_samples=num_samples,
                size=args.sample_size)
            logging.info(f"Generated initial samples: {len(samples)}")
            if not args.experimental:
                log_samples(
                    samples=samples,
                    additional_info=additional_info,
                    folder=f'{args.result_folder}/{0}',
                    plot_samples=args.plot_images,
                    modality=args.modality)
            if args.data_checkpoint_step >= 0:
                logging.info('Ignoring data_checkpoint_step'),
            start_t = 1
        if args.direct_variate:
            wandb_logging(private_classes, [diversity.tolist()], [first_vote_only.tolist()], 0, accum_loser.T.tolist())

        if args.experimental:
            log_plot(private_samples=all_private_samples,
                     synthetic_samples=samples, 
                     size=args.sample_size,
                     step=0,
                     dir=args.result_folder,)
    
    
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
            device=f'cuda:{args.device}',
            metric=metric)
        logging.info(f'{metric}={fid}')
        log_fid(args.result_folder, fid, 0)
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
                for class_i in private_classes:
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
                    sub_losers = accum_loser[class_i, 
                        num_samples_per_class * class_i:
                        num_samples_per_class * (class_i + 1)]
                    sub_count[sub_losers] = 0
                    # Sort counts
                    sub_counts_sorted_idx = np.flip(np.argsort(sub_count))
                    # Only superior samples as demonstration
                    sub_counts_idx = np.tile(sub_counts_sorted_idx, (len(sub_counts_sorted_idx), 1))
                    sub_row_idx, sub_col_idx = np.indices(sub_counts_idx.shape)
                    sub_counts_idx[sub_col_idx >= sub_row_idx] = -1
                    sub_p = np.array([sub_count[idx] if idx >= 0 else 0 for idx in sub_counts_idx.flat]).reshape((args.num_samples, args.num_samples))
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

        for packed in packed_tokens:
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
                    modality=args.modality)

            packed_features.append(sub_packed_features)
        if args.direct_variate:  # candidate로 생성한 variation을 사용할 경우
            # packed_features shape: (N_syn * num_candidate, embedding)
            packed_features = np.concatenate(packed_features, axis=0)
        else:  # 기존 DPSDA
            # packed_features shape: (N_syn, embedding)
            packed_features = np.mean(packed_features, axis=1)

        logging.info('Computing histogram')
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
                logging.info(f"Current diversity: {diversity[class_i]}")
                sub_count, sub_clean_count, sub_losers, sub_1st_idx = dp_nn_histogram(
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
                    diversity=diversity[class_i],
                    diversity_lower_bound=args.diversity_lower_bound,
                    loser_lower_bound=args.loser_lower_bound,
                    first_vote_only=first_vote_only[class_i],
                    device=args.device)
                if first_vote_only[class_i]:
                    accum_loser[class_i] = np.logical_or(accum_loser[class_i], np.any(sub_losers, axis=1, keepdims=True).flatten())
                    updated_div = 1 - accum_loser[class_i].sum() / num_samples_per_class
                    logging.info(f"Diversity loss: {diversity[class_i] - updated_div}")
                    diversity[class_i] = updated_div
                    first_vote_only[class_i] = diversity[class_i] > args.diversity_lower_bound
                    wandb_logging(private_classes, [diversity.tolist()], [first_vote_only.tolist()], t, accum_loser.T.tolist())
                # if t < 8:
                #     first_vote_only[class_i] = True
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
                sub_clean_count = sub_count.copy()
            log_count(
                sub_count,
                sub_clean_count,
                None,
                f'{args.result_folder}/{t}/count_class{class_}.npz')
            count.append(sub_count)
            count_1st_idx.append(sub_1st_idx)
        count = np.concatenate(count)
        count_1st_idx = np.concatenate(count_1st_idx)
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
                    folder=f'{args.result_folder}/{t}',
                    suffix=f'class{class_}')
        logging.info('Generating new indices')
        if args.num_samples == 0:
            assert args.num_samples_schedule[t] % private_num_classes == 0
            new_num_samples_per_class = (
                args.num_samples_schedule[t] // private_num_classes)
        else:
            new_num_samples_per_class = num_samples_per_class

        if args.direct_variate:
            selected = []
            for class_i in private_classes:
                class_indices = np.arange(num_samples_per_class * class_i, num_samples_per_class * (class_i + 1))
                sub_count = count[class_indices]
                for i in range(sub_count.shape[0]):
                    # loser
                    # 패자부할전을 할 경우에 이 단계에서 loser로 분류되는 샘플은 없으므로 첫 번째 투표만 하는 경우에만 해당
                    if sub_count[i] == 0:
                        idx = rng.choice(
                            np.arange(num_samples_per_class),
                            size=1,
                            p=sub_count / np.sum(sub_count)
                        )[0]
                        logging.info(f"Winner selected in place of loser at {i}: {idx}")
                        idx = [class_indices[idx], count_1st_idx[class_indices[idx]]]
                    # winner
                    else:
                        idx = [class_indices[i], count_1st_idx[class_indices[i]]]
                    selected.append(idx)
            selected = np.stack(selected)
            new_new_samples = packed_samples[selected[:, 0], selected[:, 1]]
            new_new_additional_info = additional_info
            new_new_count = count[selected[:,0]]
            count = new_new_count
            log_count(
                count,
                None,
                accum_loser,
                f'{args.result_folder}/{t}/count.npz')
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
            log_fid(args.result_folder, new_new_fid, t)

        samples = new_new_samples
        additional_info = new_new_additional_info

        if args.experimental:
            log_plot(private_samples=all_private_samples,
                     synthetic_samples=samples, 
                     size=args.sample_size,
                     step=t,
                     dir=args.result_folder)
    
        else:
            log_samples(
                samples=samples,
                additional_info=additional_info,
                folder=f'{args.result_folder}/{t}',
                plot_samples=args.plot_images,
                modality=args.modality)
        if args.dp:
            eps = get_epsilon(args.epsilon, t)
            logging.info(f"Privacy cost so far: {eps:.2f}")
            wandb.log({"epsilon": eps, "t": t})

    artifact = wandb.Artifact(name="log", type="dataset")
    artifact.add_file(local_path=log_file, name=wandb.run.name)
    wandb.log_artifact(artifact)
    os.remove(log_file)

if __name__ == '__main__':
    main()
