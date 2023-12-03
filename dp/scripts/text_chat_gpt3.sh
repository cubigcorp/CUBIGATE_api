python main.py \
--device 2 \
--modality text \
--feature_extractor clip_vit_b_32 \
--count_threshold 2.0 \
--noise_multiplier 1.0 \
--lookahead_degree 1 \
--data_folder /home/minsy/CUBIG/dp/data/clinical/train \
--num_samples_schedule 100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100 \
--variation_degree_schedule 1.0,0.98,0.96,0.94,0.92,0.88,0.84,0.8,0.76,0.72,0.68,0.64,0.6,0.56,0.52,0.48,0.44,0.4 \
--num_private_samples 100 \
--initial_prompt "Generate BATCH dialogs between a doctor and a patient at a clinic. Each item does not have any titles or numbers but only a tail written as 'END' without any additional dividers at all." \
--make_fid_stats False \
--result_folder /home/minsy/CUBIG/dp/result/clinical/chatgpt3 \
--tmp_folder /tmp/clinical/chatgpt3 \
--api chatgpt \
--random_sampling_checkpoint gpt-3.5-turbo-1106 \
--random_sampling_batch_size 2 \
--variation_checkpoint gpt-3.5-turbo-1106 \
--variation_batch_size 2 \
--save_samples_live \
--api_key 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 \
# --data_checkpoint_path /home/minsy/CUBIG/dp/result/clinical/chatgpt/1/_samples.npz \
# --data_checkpoint_step 1 \
# --live_loading_target /home/minsy/CUBIG/dp/result/clinical/chatgpt/variation_2_0_samples.npz

