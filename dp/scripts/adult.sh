python main.py \
--device 2 \
--modality text \
--feature_extractor bert_base_nli_mean_tokens \
--count_threshold 2.0 \
--noise_multiplier 1.0 \
--num_candidate 3 \
--data_folder /home/yerinyoon/Cubigate_ai_engine/dp/data/adult_files/train \
--num_samples_schedule 100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100 \
--variation_degree_schedule 1.0,0.98,0.96,0.94,0.92,0.88,0.84,0.8,0.76,0.72,0.68,0.64,0.6,0.56,0.52,0.48,0.44,0.4 \
--num_private_samples 100 \
--initial_prompt "Generate BATCH data with same format of the given data but you have to keep below constraint.\
    You need to make tabular data with value {income:<-50k, workclass: Private, education:11th}
    " \
--make_fid_stats False \
--result_folder /home/yerinyoon/Cubigate_ai_engine/dp/result/adult/chatgpt \
--tmp_folder /home/yerinyoon/Cubigate_ai_engine/tmp/adult/chatcpt \
--api chatgpt \
--random_sampling_checkpoint gpt-4 \
--random_sampling_batch_size 50 \
--variation_checkpoint gpt-4 \
--variation_batch_size 10 \
--save_samples_live \
--api_key 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 \
--data_checkpoint_path /home/yerinyoon/Cubigate_ai_engine/dp/result/adult/chatgpt/_samples.npz \
--data_checkpoint_step 0 \
--live_loading_target /home/yerinyoon/Cubigate_ai_engine/dp/result/adult/chatgpt/vari_samples.npz 

