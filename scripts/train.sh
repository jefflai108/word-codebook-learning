#!/bin/bash 

export PYTHONPATH=$PYTHONPATH:/data/sls/scratch/clai24/word-seg/codebook-learning

data_dir=/data/sls/scratch/clai24/word-seg/flicker8k/
expdir=exp/debug/ 

stage=0

if [ $stage -eq 0 ]; then 
    # train 
    CUDA_LAUNCH_BLOCKING=1 python main.py \
               --train_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_train.txt \
               --dev_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_dev.txt \
               --word_seg_file_path ${data_dir}/word_seg_features/flicker_speech_features.mat \
               --save_dir ${expdir} \
               --batch_size 32 \
               --segment_context_size 7 \
               --vocab_size 1027 --d_model 512 --nhead 8 --num_encoder_layers 3 --num_decoder_layers 3 --dim_feedforward 2048 \
               --dropout 0.1 --max_seq_length 130 --epochs 10 --log_interval 20 --learning_rate 1e-4 --optimizer_type "adam"
fi 

if [ $stage -eq 1 ]; then 
    # inference 
    CUDA_LAUNCH_BLOCKING=1 python inference.py \
               --train_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_train.txt \
               --test_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_test.txt \
               --dev_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_dev.txt \
               --word_seg_file_path ${data_dir}/word_seg_features/flicker_speech_features.mat \
               --save_dir ${expdir} \
               --load_model_path exp/debug_collection/contextsize7_dmodel512_enclayer5_declayer1_ffn_dim2048_dp0.1_lr1e-3/best_loss_model.pth \
               --batch_size 1 \
               --segment_context_size 5 \
               --vocab_size 1027 --d_model 512 --nhead 8 --num_encoder_layers 5 --num_decoder_layers 1 --dim_feedforward 2048 \
               --dropout 0.1 --max_seq_length 130 --epochs 10 --log_interval 20 --learning_rate 1e-3 --optimizer_type "adam"
fi 
