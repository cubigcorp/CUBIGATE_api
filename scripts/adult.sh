python main.py \
--device 2  \
--modality "text" \
--api "chatgpt" \ 
--data_checkpoint_path "/home/yerinyoon/Cubigate_ai_engine/dp/result/adult/samples.npz" \
--lookahead_degree 8 \
--num_samples 50 \
--feature_extractor "bert_base_nli_mean_tokens" \
--num_nearest_neighbor 3 \
\