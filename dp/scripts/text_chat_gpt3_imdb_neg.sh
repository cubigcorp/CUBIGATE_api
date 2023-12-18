python main.py \
--device 2 \
--modality text \
--feature_extractor clip_vit_b_32 \
--count_threshold 2.0 \
--noise_multiplier 1.0 \
--lookahead_degree 1 \
--data_folder /mnt/cubigate/minsy/dp_data/IMDB/private/negative \
--num_samples_schedule 100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100 \
--variation_degree_schedule 1.0,0.98,0.96,0.94,0.92,0.88,0.84,0.8,0.76,0.72,0.68,0.64,0.6,0.56,0.52,0.48,0.44,0.4 \
--num_private_samples 100 \
--initial_prompt "Generate BATCH negative movie reviews as if they were posted on IMDB." \
--control_prompt "The review does not have a title, a number, or anything other than itself." \
--make_fid_stats True \
--compute_fid True \
--num_fid_samples 100 \
--fid_model_name clip_vit_b_32 \
--fid_dataset_name imdb_neg \
--result_folder /mnt/cubigate/result/IMDB/gpt3/negative \
--tmp_folder /tmp/IMDB/neg/chatgpt3 \
--api chatgpt \
--random_sampling_checkpoint gpt-3.5-turbo-1106 \
--random_sampling_batch_size 1 \
--variation_checkpoint gpt-3.5-turbo-1106 \
--variation_batch_size 1 \
--api_key keys/minsy.key \
--variation_prompt_path prompts/text_variation_chatgpt.txt \
--use_public_data true \
--public_data_folder /mnt/cubigate/minsy/dp_data/IMDB/public/negative \
--save_samples_live \
--epsilon 1.0 \
--delta 0.0 \