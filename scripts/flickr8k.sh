#!/bin/bash 

export PYTHONPATH=$PYTHONPATH:/data/sls/scratch/clai24/word-seg/codebook-learning

data_dir=/data/sls/scratch/clai24/word-seg/flicker8k/
expdir=exp/debug/ 

stage=0

if [ $stage -eq 0 ]; then 
    # train 
    CUDA_LAUNCH_BLOCKING=1 python main_v2_1.py \
               --train_token_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_train.txt \
               --dev_token_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_dev.txt \
               --train_embed_file_paths ${data_dir}/preprocess/speechrepresentations/flickr_8k_wav2vec2_large_lv60_layer10,11,12,13,14_pca256_embeddings_train.npz \
               --dev_embed_file_paths ${data_dir}/preprocess/speechrepresentations/flickr_8k_wav2vec2_large_lv60_layer10,11,12,13,14_pca256_embeddings_dev.npz \
               --word_seg_file_path ${data_dir}/word_seg_features/flicker_speech_features.mat \
               --save_dir ${expdir} \
               --batch_size 16 \
               --repre_dim 256 \
               --num_layer_repre 5 \
               --vocab_size 1027 --d_model 512 --nhead 8 --num_encoder_layers 6 --num_decoder_layers 6 --dim_feedforward 2048 --model_activation "gelu" \
               --dropout 0.1 --max_seq_length 100 --epochs 10 --log_interval 20 --learning_rate 1e-4 --gradient_acc_steps 16 --optimizer_type "adam" --label_smoothing 0.1
fi 

if [ $stage -eq 1 ]; then 
    # inference 
    CUDA_LAUNCH_BLOCKING=1 python flickr8k_inference.py \
               --train_token_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_train.txt \
               --dev_token_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_dev.txt \
               --test_token_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_test.txt \
               --train_embed_file_path ${data_dir}/preprocess/speechrepresentations/flickr_8k_wav2vec2_large_lv60_layer14_embeddings_train.npz \
               --dev_embed_file_path ${data_dir}/preprocess/speechrepresentations/flickr_8k_wav2vec2_large_lv60_layer14_embeddings_dev.npz \
               --test_embed_file_path ${data_dir}/preprocess/speechrepresentations/flickr_8k_wav2vec2_large_lv60_layer14_embeddings_test.npz \
               --word_seg_file_path ${data_dir}/word_seg_features/flicker_speech_features.mat \
               --save_dir ${expdir} \
               --load_model_path exp/debug/best_loss_model.pth \
               --batch_size 1 \
               --repre_dim 1024 \
               --vocab_size 1027 --d_model 256 --nhead 8 --num_encoder_layers 5 --num_decoder_layers 3 --dim_feedforward 1024 \
               --dropout 0.1 --max_seq_length 100 --epochs 10 --log_interval 20 --learning_rate 1e-4 --optimizer_type "adam" --label_smoothing 0.1
fi 
