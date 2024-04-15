#!/bin/bash 

export PYTHONPATH=$PYTHONPATH:/data/sls/scratch/clai24/word-seg/codebook-learning

data_dir=/data/sls/scratch/clai24/word-seg/flicker8k/
expdir=exp/debug/ 

stage=1

if [ $stage -eq 1 ]; then 
    # inference 
    w2v2_large_layerid=14
    CUDA_LAUNCH_BLOCKING=1 python slm21_inference.py \
               --test_synthetic_path ${data_dir}/preprocess/speechtokens/rvq1/slm21_semantic_test_synthetic.txt \
               --test_librispeech_path ${data_dir}/preprocess/speechtokens/rvq1/slm21_semantic_test_librispeech.txt \
               --dev_synthetic_path ${data_dir}/preprocess/speechtokens/rvq1/slm21_semantic_dev_synthetic.txt \
               --dev_librispeech_path ${data_dir}/preprocess/speechtokens/rvq1/slm21_semantic_dev_librispeech.txt \
               --test_synthetic_embed_paths ${data_dir}/preprocess/speechrepresentations/slm21_semantic_test_synthetic_wav2vec2_large_lv60_layer10,11,12,13,14_pca256_embeddings.npz \
               --test_librispeech_embed_paths ${data_dir}/preprocess/speechrepresentations/slm21_semantic_test_librispeech_wav2vec2_large_lv60_layer10,11,12,13,14_pca256_embeddings.npz \
               --dev_synthetic_embed_paths ${data_dir}/preprocess/speechrepresentations/slm21_semantic_dev_synthetic_wav2vec2_large_lv60_layer10,11,12,13,14_pca256_embeddings.npz \
               --dev_librispeech_embed_paths ${data_dir}/preprocess/speechrepresentations/slm21_semantic_dev_librispeech_wav2vec2_large_lv60_layer10,11,12,13,14_pca256_embeddings.npz \
               --save_dir ${expdir} \
               --load_model_path exp/debug/best_loss_model.pth \
               --batch_size 1 \
               --repre_dim 256 \
               --num_layer_repre 5 \
               --vocab_size 1027 --d_model 512 --nhead 8 --num_encoder_layers 6 --num_decoder_layers 6 --dim_feedforward 2048 --model_activation "gelu" \
               --dropout 0.1 --max_seq_length 100 --epochs 10 --log_interval 20 --learning_rate 1e-4 --gradient_acc_steps 16 --optimizer_type "adam" --label_smoothing 0.1
fi 

