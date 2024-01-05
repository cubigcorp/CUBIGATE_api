python main.py \
--device 2 \
--modality text \
--feature_extractor clip_vit_b_32 \
--count_threshold 2.0 \
--noise_multiplier 1.0 \
--num_candidate 1 \
--data_folder /home/minsy/CUBIG/dp/data/IMDB/private/negative \
--num_samples_schedule 100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100 \
--variation_degree_schedule 1.0,0.98,0.96,0.94,0.92,0.88,0.84,0.8,0.76,0.72,0.68,0.64,0.6,0.56,0.52,0.48,0.44,0.4 \
--num_private_samples 100 \
--initial_prompt "Generate BATCH negative movie reviews as if they were negted on IMDB. The review does not have a title, a number, or anything other than itself." \
--make_fid_stats True \
--compute_fid True \
--num_fid_samples 100 \
--fid_model_name clip_vit_b_32 \
--fid_dataset_name imdb_neg_llama \
--result_folder /home/minsy/CUBIG/dp/result/IMDB/llama/negative \
--tmp_folder /tmp/IMDB/neg/llama \
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
--use_public_data true \
--public_data_folder /home/minsy/CUBIG/dp/data/IMDB/public/negative \
