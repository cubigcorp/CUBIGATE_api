from dpsda.metrics import compute_metric, make_fid_stats
from dpsda.data_loader import load_private_data, load_samples
from dpsda.tokenizer import tokenize
import argparse
import numpy as np

def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device',
        type=int,
        required=True)
    parser.add_argument(
        '--modality',
        type=str,
        choices=['image', 'text'], #Tabular: text
        required=True)
    parser.add_argument(
        '--data_checkpoint_path',
        type=str,
        default="",
        help='Path to the data checkpoint')
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
        '--tmp_folder',
        type=str,
        default='result/tmp',
        help='Temporary folder for storing intermediate results')

    parser.add_argument(
        '--data_folder',
        type=str,
        required=True,
        help='Folder that contains the private images')

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
        '--feature_extractor',
        type=str,
        default='clip_vit_b_32',
        choices=['bert_base_nli_mean_tokens', 'inception_v3', 'clip_vit_b_32', 'original'], 
        help='Which image feature extractor to use')
    parser.add_argument(
        '--fid_batch_size',
        type=int,
        default=500,
        help='Batch size for computing FID')
    parser.add_argument(
        '--gen_class_cond',
        action='store_true',
        help='Whether to generate class labels')
    parser.add_argument(
        '--private_image_size',
        type=int,
        default=1024,
        help='Size of private images')
    parser.add_argument(
        '--data_loading_batch_size',
        type=int,
        default=100,
        help='Batch size for loading private samples')

    args, _ = parser.parse_known_args()
    return args

args = argument()
all_private_samples, all_private_labels = load_private_data(
        data_dir=args.data_folder,
        batch_size=args.data_loading_batch_size,
        image_size=args.private_image_size,
        class_cond=args.gen_class_cond,
        num_private_samples=args.num_private_samples,
        modality=args.modality,
        model=args.feature_extractor)
metric = 'FID' if len(all_private_samples) > 2048 else 'KID'
make_fid_stats(
            samples=all_private_samples,
            dataset=args.fid_dataset_name,
            dataset_res=args.private_image_size,
            dataset_split=args.fid_dataset_split,
            tmp_folder=args.tmp_folder,
            model_name=args.fid_model_name,
            batch_size=args.fid_batch_size,
            modality=args.modality,
            device=f'cuda:{args.device}',
            metric=metric)
samples, additional_info = load_samples(args.data_checkpoint_path)
if args.modality == 'text':
    tokens = [tokenize(args.fid_model_name, sample) for sample in samples]
    tokens = np.array(tokens)
else:
    tokens = samples
score = compute_metric(
    samples=tokens,
    modality=args.modality,
    tmp_folder=args.tmp_folder,
    dataset=args.fid_dataset_name,
    dataset_res=args.private_image_size,
    dataset_split=args.fid_dataset_split,
    batch_size=args.fid_batch_size,
    num_fid_samples=args.num_fid_samples,
    model_name=args.fid_model_name,
    device=f'cuda:{args.device}',
    metric=metric
)
print(score)


