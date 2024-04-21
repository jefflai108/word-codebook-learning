#!/bin/bash 

export PYTHONPATH=$PYTHONPATH:/data/sls/scratch/clai24/word-seg/codebook-learning

data_dir=/data/sls/scratch/clai24/word-seg/mix-corpus
expdir=exp/debug2/ 
mkdir -p $expdir 

stage=0

if [ $stage -eq 0 ]; then 
    # train 
    NUM_PCA=512
    CUDA_LAUNCH_BLOCKING=1 python main_v2_1_flickr8k_spokencoco.py \
               --train_token_file_path ${data_dir}/speechtokens/rvq1/flickr_8k_spokencoco/flickr_8k_spokencoco_rvq1_tokens_train_split_SPLIT.txt \
               --dev_token_file_path ${data_dir}/speechtokens/rvq1/flickr_8k_spokencoco/flickr_8k_spokencoco_rvq1_tokens_dev.txt \
               --train_embed_file_paths ${data_dir}/speechrepresentations/flickr_8k_spokencoco/flickr_8k_spokencoco_wav2vec2_large_lv60_layer10,11,12,13,14_pca${NUM_PCA}_embeddings_train_split_SPLIT.npz \
               --dev_embed_file_paths ${data_dir}/speechrepresentations/flickr_8k_spokencoco/flickr_8k_spokencoco_wav2vec2_large_lv60_layer10,11,12,13,14_pca${NUM_PCA}_embeddings_dev.npz \
               --total_train_split_num 20 \
               --word_seg_file_path ${data_dir}/word_seg_features/flickr_8k_spokencoco_speech_features.mat \
               --save_dir ${expdir} \
               --batch_size 20 --gradient_acc_steps 16 \
               --repre_dim ${NUM_PCA} \
               --num_layer_repre 5 \
               --word_pooling "lde32" --norm_type "batchnorm" \
               --vocab_size 1027 --d_model 768 --nhead 8 --num_encoder_layers 3 --num_decoder_layers 3 --dim_feedforward 3072 --model_activation "gelu" \
               --dropout 0.1 --max_seq_length 100 --epochs 16 --log_interval 20 --learning_rate 1e-4 --optimizer_type "adamw" --label_smoothing 0.1
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
