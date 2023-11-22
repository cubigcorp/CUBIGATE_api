from cubigate.generate import CubigDPGenerator

generator = CubigDPGenerator(
    api=args.api,
    model_checkpoint=args.model_checkpoint,
    feature_extractor=args.feature_extractor,
    result_folder=args.result_folder,
    tmp_folder=args.tmp_folder,
    modality=args.modality,
    data_folder=args.data_folder)

generator.train(
    condition_guidance_scale=args.condition_guidance_scale,
    inference_steps=args.inference_step,
    batch_size=args.batch_size,
    variation_strength=args.variation_strength,
    variation_degree_schedule=args.variation_degree_schedule,
    count_threshold=args.count_threshold,
    image_size=args.image_size,
    num_initial_samples=args.num_initial_samples,
    initial_prompt=args.initial_prompts)
generator.generate(
    base_data=args.base_data,
    image_size=args.image_size,
    num_samples=args.num_samples,
    suffix=args.suffix
)