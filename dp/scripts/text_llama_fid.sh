python FID.py \
--device 2 \
--modality text \
--feature_extractor clip_vit_b_32 \
--fid_model_name clip_vit_b_32 \
--data_folder /home/minsy/CUBIG/dp/data/clinical/train \
--num_private_samples 100 \
--num_fid_samples 100 \
--initial_prompt "Generate a dialog between a doctor and a patient" \
--result_folder /home/minsy/CUBIG/dp/result/clinical/test \
--tmp_folder /tmp/clinical/test \
--variation_batch_size 1 \
--max_seq_len 512 \
--top_k 10 \
--api_device 2 \
--epsilon 1.0 \
--delta 0.0 \
--data_checkpoint_path /home/minsy/CUBIG/dp/result/clinical/0/_samples.npz \
--data_checkpoint_step 17
