python main.py \
--device 2 \
--modality text \
--feature_extractor clip_vit_b_32 \
--count_threshold 2.0 \
--noise_multiplier 1.0 \
--lookahead_degree 8 \
--data_folder /home/minsy/CUBIG/dp/data/clinical/train \
--num_samples_schedule 100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100 \
--variation_degree_schedule 1.0,0.98,0.96,0.94,0.92,0.88,0.84,0.8,0.76,0.72,0.68,0.64,0.6,0.56,0.52,0.48,0.44,0.4 \
--num_private_samples 100 \
--initial_prompt "Generate a dialog between a doctor and a patient" \
--make_fid_stats False \
--result_folder /home/minsy/CUBIG/dp/result/clinical/chatgpt \
--tmp_folder /tmp/clinical/chatgpt \
--api chatgpt \
--random_sampling_checkpoint gpt-4 \
--random_sampling_batch_size 50 \
--variation_checkpoint gpt-4 \
--variation_batch_size 50 \
--api_key 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 