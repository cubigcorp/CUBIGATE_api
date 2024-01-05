python main.py \
--device 1 \
--api_device 1 \
--feature_extractor clip_vit_b_32 \
--fid_model_name inception_v3 \
--fid_dataset_name customized_dataset \
--count_threshold 2.0 \
--noise_multiplier 2.0 \
--num_candidate 8 \
--image_size 512x512 \
--private_image_size 512 \
--data_folder /home/minsy/dpsda/DPSDA/data/chest/train/NORMAL \
--num_samples_schedule 100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100 \
--variation_degree_schedule 1.0,0.98,0.96,0.94,0.92,0.88,0.84,0.8,0.76,0.72,0.68,0.64,0.6,0.56,0.52,0.48,0.44,0.4 \
--num_fid_samples 100 \
--num_private_samples 100 \
--initial_prompt "A photo of normal chest xray" \
--make_fid_stats True \
--result_folder /home/minsy/dpsda/DPSDA/result/chest/normal/lora \
--tmp_folder /tmp/chest/normal/lora \
--api stable_diffusion \
--random_sampling_checkpoint 'runwayml/stable-diffusion-v1-5' \
--random_sampling_guidance_scale 7.5 \
--random_sampling_num_inference_steps 50 \
--random_sampling_batch_size 10 \
--variation_checkpoint 'runwayml/stable-diffusion-v1-5' \
--variation_guidance_scale 7.5 \
--variation_num_inference_steps 50 \
--variation_batch_size 10 \
--lora /home/minsy/dpsda/lora/chest/
