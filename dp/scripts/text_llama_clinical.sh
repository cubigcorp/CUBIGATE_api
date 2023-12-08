python main.py \
--device 2 \
--modality text \
--feature_extractor clip_vit_b_32 \
--count_threshold 2.0 \
--noise_multiplier 1.0 \
--lookahead_degree 1 \
--data_folder /home/minsy/CUBIG/dp/data/clinical/train \
--num_samples_schedule 10,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100 \
--variation_degree_schedule 1.0,0.98,0.96,0.94,0.92,0.88,0.84,0.8,0.76,0.72,0.68,0.64,0.6,0.56,0.52,0.48,0.44,0.4 \
--num_private_samples 10 \
--initial_prompt "Generate a dialog between a doctor and a patient" \
--make_fid_stats False \
--result_folder /home/minsy/CUBIG/dp/result/clinical/test \
--tmp_folder /tmp/clinical/test \
--api llama2 \
--random_sampling_checkpoint meta-llama/Llama-2-7b-chat-hf \
--random_sampling_batch_size 4 \
--variation_checkpoint meta-llama/Llama-2-7b-chat-hf \
--variation_batch_size 1 \
--max_seq_len 512 \
--top_k 10 \
--api_device 2 \
--epsilon 1.0 \
--delta 0.0 \
# --data_checkpoint_path /home/minsy/CUBIG/dp/result/clinical/17/_samples.npz \
# --data_checkpoint_step 17
