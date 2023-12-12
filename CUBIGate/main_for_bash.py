from cubigate.generate import CubigDPGenerator
from cubigate.dp.utils.arg_utils import str2bool
import argparse
import os

def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train',
        action='store_true',
        help="Whether to train generator to learn the distribution before generating samples."
    )
    parser.add_argument(
        '--generate',
        action='store_true',
        help="Whether to generate new samples after train the generator"
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        required=False)
    parser.add_argument(
        '--delta',
        default=0.0,
        type=float,
        required=False)
    parser.add_argument(
        '--api',
        type=str,
        required=True,
        choices=['DALLE', 'stable_diffusion', 'improved_diffusion'],
        help='Which foundation model API to use')
    parser.add_argument(
        '--plot_images',
        type=str2bool,
        default=True,
        help='Whether to save generated images in PNG files') 
    parser.add_argument(
        '--data_checkpoint_path',
        type=str,
        default="",
        help='Path to the data checkpoint')
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
        '--variation_degree_schedule',
        type=str,
        default='0,'*9 + '0',
        help='Variation degree at each iteration')
    parser.add_argument(
        '--num_org_data',
        type=int,
        default=50000,
        help='Number of original data to load')
    parser.add_argument(
        '--num_samples',
        type=int,
        default=50000,
        help='Number of samples to generate')
    parser.add_argument(
        '--variation_degree',
        type=float,
        default=0.5,
        help='Variation degree for final generaton.')
    parser.add_argument(
        '--lookahead_degree',
        type=int,
        default=0,
        help=('Lookahead degree for computing distances between private and '  #
              'generated images'))
    parser.add_argument(
        '--feature_extractor',
        type=str,
        default='clip_vit_b_32',
        choices=['bert_base_nli_mean_tokens', 'inception_v3', 'clip_vit_b_32', 'original'], 
        help='Which image feature extractor to use')
    parser.add_argument(
        '--nn_mode',
        type=str,
        default='L2',
        choices=['L2', 'IP'],
        help='Which distance metric to use in DP NN histogram')   ##Bert랑 같이 similarty로 갈지 논의
    parser.add_argument(
        '--org_img_size',
        type=int,
        default=1024,
        help='Size of original images')
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
        required=True,
        help='Folder that contains the private images')
    parser.add_argument(
        '--count_threshold',
        type=float,
        default=0.0,
        help='Threshold for DP NN histogram')
    parser.add_argument(
        '--data_loading_batch_size',
        type=int,
        default=100,
        help='Batch size for loading private samples')
    parser.add_argument(
        '--feature_extractor_batch_size', 
        type=int,
        default=500,
        help='Batch size for feature extraction')
    parser.add_argument(
        '--conditional',
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
        '--img_size',
        type=str,
        default='1024x1024',
        help='Size of generated images in the format of HxW')
    args, api_args = parser.parse_known_args()
    args.num_samples_schedule = list(map(
        int, args.num_samples_schedule.split(',')))
    variation_degree_type = (float if '.' in args.variation_degree_schedule
                             else int)
    args.variation_degree_schedule = list(map(
        variation_degree_type, args.variation_degree_schedule.split(',')))
    assert len(args.num_samples_schedule) == len(args.variation_degree_schedule)
    return args, api_args

def main():
    args, api_args = argument()
    print(api_args)
    generator = CubigDPGenerator(
        api = args.api,
        feature_extractor=args.feature_extractor,
        result_folder=args.result_folder,
        tmp_folder=args.tmp_folder,
        data_loading_batch_size=args.data_loading_batch_size,
        feature_extractor_batch_size=args.feature_extractor_batch_size,
        org_img_size=args.org_img_size,
        conditional=args.conditional,
        num_org_data=args.num_org_data
    )
    if args.train:
        generator.train(
            data_folder=args.data_folder,
            data_checkpoint_path=args.data_checkpoint_path,
            data_checkpoint_step=args.data_checkpoint_step,
            initial_prompt=args.initial_prompt,
            num_samples_schedule=args.num_samples_schedule,
            variation_degree_schedule=args.variation_degree_schedule,
            lookahead_degree=args.lookahead_degree,
            img_size=args.img_size,
            epsilon=args.epsilon,
            delta=args.delta,
            count_threshold=args.count_threshold,
            plot_images=args.plot_images,
            nn_mode=args.nn_mode,
            api_args=api_args
        )
    args.data_checkpoint_path = os.path.join(args.result_folder, str(len(args.variation_degree_schedule) - 1), '_samples.npz')
    if args.generate:
        generator.generate(
            base_data=args.data_checkpoint_path,
            img_size=args.img_size,
            num_samples=args.num_samples,
            variation_degree=args.variation_degree,
            plot_images=args.plot_images,
            api_args=api_args
        )

if __name__ == '__main__':
    main()