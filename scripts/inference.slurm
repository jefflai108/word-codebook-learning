#!/bin/bash 
#SBATCH -J cb-inference
#SBATCH -o /data/sls/scratch/clai24/slurm_dumps/codebook-debug-inference_%j.out   
#SBATCH -e /data/sls/scratch/clai24/slurm_dumps/codebook-debug-inference_%j.err   
#SBATCH --qos=regular 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 
#SBATCH --partition=a5,a6,2080
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00 
#SBATCH --mem=80G
#SBATCH --requeue

## Set the python environment you want to use for your code 
CONDA_ROOT=/data/sls/scratch/clai24/anaconda3
source ${CONDA_ROOT}/etc/profile.d/conda.sh 
conda activate /data/sls/scratch/clai24/anaconda3/envs/prompt-gpt4

export PYTHONPATH=$PYTHONPATH:/data/sls/scratch/clai24/word-seg/codebook-learning

data_dir=/data/sls/scratch/clai24/word-seg/flicker8k/

d_model=$1
num_encoder_layers=$2
num_decoder_layers=$3
ffn_dim=$4
dp=$5
lr=$6
context_size=$7
labelsmooth=$8
w2v2_large_layerid=$9 
pca_dim=${10}

stage="v2.1" # v0 / v1 / v2 

#for w2v2_large_layerid in 10 11 12 13 14; do 
#for context_size in 1 3 5 7; do 
#for w2v2_large_layerid in 10 14; do 
for lr in 1e-3 5e-4 1e-4; do 
for dp in 0.2 0.3; do 

    if [ "$stage" == "v0" ]; then
        expdir=/data/sls/scratch/clai24/word-seg/codebook-learning/exp/debug_collection/contextsize${context_size}_dmodel${d_model}_enclayer${num_encoder_layers}_declayer${num_decoder_layers}_ffn_dim${ffn_dim}_dp${dp}_lr${lr}_labelsmooth${labelsmooth}
        expdir=/data/sls/scratch/clai24/word-seg/codebook-learning/exp/debug_collection/contextsize${context_size}_dmodel${d_model}_enclayer${num_encoder_layers}_declayer${num_decoder_layers}_ffn_dim${ffn_dim}_dp${dp}_lr${lr}

        CUDA_LAUNCH_BLOCKING=1 python inference.py \
                       --train_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_train.txt \
                       --test_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_test.txt \
                       --dev_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_dev.txt \
                       --word_seg_file_path ${data_dir}/word_seg_features/flicker_speech_features.mat \
                       --load_model_path ${expdir}/best_loss_model.pth \
                       --save_dir ${expdir} \
                       --batch_size 1 \
                       --segment_context_size ${context_size} \
                       --vocab_size 1027 --d_model ${d_model} --nhead 8 --num_encoder_layers ${num_encoder_layers} --num_decoder_layers ${num_decoder_layers} --dim_feedforward ${ffn_dim} \
                       --dropout ${dp} --max_seq_length 130 --epochs 30 --log_interval 20 --learning_rate ${lr} --optimizer_type "adam" --label_smoothing $labelsmooth \
                       >> ${expdir}/inference.log 2>&1

        CUDA_LAUNCH_BLOCKING=1 python slm21_inference.py \
                       --test_synthetic_path ${data_dir}/preprocess/speechtokens/rvq1/slm21_semantic_test_synthetic.txt \
                       --test_librispeech_path ${data_dir}/preprocess/speechtokens/rvq1/slm21_semantic_test_librispeech.txt \
                       --dev_synthetic_path ${data_dir}/preprocess/speechtokens/rvq1/slm21_semantic_dev_synthetic.txt \
                       --dev_librispeech_path ${data_dir}/preprocess/speechtokens/rvq1/slm21_semantic_dev_librispeech.txt \
                       --load_model_path ${expdir}/best_loss_model.pth \
                       --save_dir ${expdir} \
                       --batch_size 1 \
                       --segment_context_size ${context_size} \
                       --vocab_size 1027 --d_model ${d_model} --nhead 8 --num_encoder_layers ${num_encoder_layers} --num_decoder_layers ${num_decoder_layers} --dim_feedforward ${ffn_dim} \
                       --dropout ${dp} --max_seq_length 130 --epochs 30 --log_interval 20 --learning_rate ${lr} --optimizer_type "adam" --label_smoothing $labelsmooth \
                       >> ${expdir}/slm21_inference.log 2>&1
    fi 

    if [ "$stage" == "v1" ]; then
        expdir=/data/sls/scratch/clai24/word-seg/codebook-learning/exp/debug_collection_v1/wav2vec2_large_lv60_layer${w2v2_large_layerid}_contextsize${context_size}_dmodel${d_model}_enclayer${num_encoder_layers}_declayer${num_decoder_layers}_ffn_dim${ffn_dim}_dp${dp}_lr${lr}_labelsmooth${labelsmooth}

        #CUDA_LAUNCH_BLOCKING=1 python inference.py \
        #       --train_token_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_train.txt \
        #       --dev_token_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_dev.txt \
        #       --test_token_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_test.txt \
        #       --train_embed_file_path ${data_dir}/preprocess/speechrepresentations/flickr_8k_wav2vec2_large_lv60_layer${w2v2_large_layerid}_embeddings_train.npz \
        #       --dev_embed_file_path ${data_dir}/preprocess/speechrepresentations/flickr_8k_wav2vec2_large_lv60_layer${w2v2_large_layerid}_embeddings_dev.npz \
        #       --test_embed_file_path ${data_dir}/preprocess/speechrepresentations/flickr_8k_wav2vec2_large_lv60_layer${w2v2_large_layerid}_embeddings_test.npz \
        #       --word_seg_file_path ${data_dir}/word_seg_features/flicker_speech_features.mat \
        #       --load_model_path ${expdir}/best_loss_model.pth \
        #       --save_dir ${expdir} \
        #       --batch_size 1 \
        #       --segment_context_size ${context_size} \
        #       --repre_dim 1024 \
        #       --vocab_size 1027 --d_model ${d_model} --nhead 8 --num_encoder_layers ${num_encoder_layers} --num_decoder_layers ${num_decoder_layers} --dim_feedforward ${ffn_dim} \
        #       --dropout ${dp} --max_seq_length 130 --epochs 30 --log_interval 20 --learning_rate ${lr} --optimizer_type "adam" --label_smoothing $labelsmooth \
        #       >> ${expdir}/inference.log 2>&1

        CUDA_LAUNCH_BLOCKING=1 python slm21_inference.py \
                       --test_synthetic_path ${data_dir}/preprocess/speechtokens/rvq1/slm21_semantic_test_synthetic.txt \
                       --test_librispeech_path ${data_dir}/preprocess/speechtokens/rvq1/slm21_semantic_test_librispeech.txt \
                       --dev_synthetic_path ${data_dir}/preprocess/speechtokens/rvq1/slm21_semantic_dev_synthetic.txt \
                       --dev_librispeech_path ${data_dir}/preprocess/speechtokens/rvq1/slm21_semantic_dev_librispeech.txt \
                       --test_synthetic_embed_path ${data_dir}/preprocess/speechrepresentations/slm21_semantic_test_synthetic_wav2vec2_large_lv60_layer${w2v2_large_layerid}_embeddings.npz \
                       --test_librispeech_embed_path ${data_dir}/preprocess/speechrepresentations/slm21_semantic_test_librispeech_wav2vec2_large_lv60_layer${w2v2_large_layerid}_embeddings.npz \
                       --dev_synthetic_embed_path ${data_dir}/preprocess/speechrepresentations/slm21_semantic_dev_synthetic_wav2vec2_large_lv60_layer${w2v2_large_layerid}_embeddings.npz \
                       --dev_librispeech_embed_path ${data_dir}/preprocess/speechrepresentations/slm21_semantic_dev_librispeech_wav2vec2_large_lv60_layer${w2v2_large_layerid}_embeddings.npz \
                       --load_model_path ${expdir}/best_loss_model.pth \
                       --save_dir ${expdir} \
                       --batch_size 1 \
                       --segment_context_size ${context_size} \
                       --repre_dim 1024 \
                       --vocab_size 1027 --d_model ${d_model} --nhead 8 --num_encoder_layers ${num_encoder_layers} --num_decoder_layers ${num_decoder_layers} --dim_feedforward ${ffn_dim} \
                       --dropout ${dp} --max_seq_length 130 --epochs 30 --log_interval 20 --learning_rate ${lr} --optimizer_type "adam" --label_smoothing $labelsmooth \
                       >> ${expdir}/slm21_inference.log 2>&1
    fi 

    if [ "$stage" == "v2" ]; then
        expdir=/data/sls/scratch/clai24/word-seg/codebook-learning/exp/debug_collection_v2/mean_pooled_wav2vec2_large_lv60_layer${w2v2_large_layerid}_dmodel${d_model}_enclayer${num_encoder_layers}_declayer${num_decoder_layers}_ffn_dim${ffn_dim}_dp${dp}_lr${lr}_labelsmooth${labelsmooth}
        expdir=/data/sls/scratch/clai24/word-seg/codebook-learning/exp/debug_collection_v2/mean_pooled_wav2vec2_large_lv60_layer${w2v2_large_layerid}_pca${pca_dim}_dmodel${d_model}_enclayer${num_encoder_layers}_declayer${num_decoder_layers}_ffn_dim${ffn_dim}_dp${dp}_lr${lr}_labelsmooth${labelsmooth}

        #CUDA_LAUNCH_BLOCKING=1 python slm21_inference.py \
        #               --test_synthetic_path ${data_dir}/preprocess/speechtokens/rvq1/slm21_semantic_test_synthetic.txt \
        #               --test_librispeech_path ${data_dir}/preprocess/speechtokens/rvq1/slm21_semantic_test_librispeech.txt \
        #               --dev_synthetic_path ${data_dir}/preprocess/speechtokens/rvq1/slm21_semantic_dev_synthetic.txt \
        #               --dev_librispeech_path ${data_dir}/preprocess/speechtokens/rvq1/slm21_semantic_dev_librispeech.txt \
        #               --test_synthetic_embed_path ${data_dir}/preprocess/speechrepresentations/slm21_semantic_test_synthetic_wav2vec2_large_lv60_layer${w2v2_large_layerid}_embeddings.npz \
        #               --test_librispeech_embed_path ${data_dir}/preprocess/speechrepresentations/slm21_semantic_test_librispeech_wav2vec2_large_lv60_layer${w2v2_large_layerid}_embeddings.npz \
        #               --dev_synthetic_embed_path ${data_dir}/preprocess/speechrepresentations/slm21_semantic_dev_synthetic_wav2vec2_large_lv60_layer${w2v2_large_layerid}_embeddings.npz \
        #               --dev_librispeech_embed_path ${data_dir}/preprocess/speechrepresentations/slm21_semantic_dev_librispeech_wav2vec2_large_lv60_layer${w2v2_large_layerid}_embeddings.npz \
        #               --load_model_path ${expdir}/best_loss_model.pth \
        #               --save_dir ${expdir} \
        #               --batch_size 1 \
        #               --repre_dim 1024 \
        #               --vocab_size 1027 --d_model ${d_model} --nhead 8 --num_encoder_layers ${num_encoder_layers} --num_decoder_layers ${num_decoder_layers} --dim_feedforward ${ffn_dim} \
        #               --dropout ${dp} --max_seq_length 100 --epochs 40 --log_interval 20 --learning_rate ${lr} --optimizer_type "adam" --label_smoothing $labelsmooth \
        #               >> ${expdir}/slm21_inference.log 2>&1

        CUDA_LAUNCH_BLOCKING=1 python slm21_inference.py \
                       --test_synthetic_path ${data_dir}/preprocess/speechtokens/rvq1/slm21_semantic_test_synthetic.txt \
                       --test_librispeech_path ${data_dir}/preprocess/speechtokens/rvq1/slm21_semantic_test_librispeech.txt \
                       --dev_synthetic_path ${data_dir}/preprocess/speechtokens/rvq1/slm21_semantic_dev_synthetic.txt \
                       --dev_librispeech_path ${data_dir}/preprocess/speechtokens/rvq1/slm21_semantic_dev_librispeech.txt \
                       --test_synthetic_embed_path ${data_dir}/preprocess/speechrepresentations/slm21_semantic_test_synthetic_wav2vec2_large_lv60_layer${w2v2_large_layerid}_pca${pca_dim}_embeddings.npz \
                       --test_librispeech_embed_path ${data_dir}/preprocess/speechrepresentations/slm21_semantic_test_librispeech_wav2vec2_large_lv60_layer${w2v2_large_layerid}_pca${pca_dim}_embeddings.npz \
                       --dev_synthetic_embed_path ${data_dir}/preprocess/speechrepresentations/slm21_semantic_dev_synthetic_wav2vec2_large_lv60_layer${w2v2_large_layerid}_pca${pca_dim}_embeddings.npz \
                       --dev_librispeech_embed_path ${data_dir}/preprocess/speechrepresentations/slm21_semantic_dev_librispeech_wav2vec2_large_lv60_layer${w2v2_large_layerid}_pca${pca_dim}_embeddings.npz \
                       --load_model_path ${expdir}/best_loss_model.pth \
                       --save_dir ${expdir} \
                       --batch_size 1 \
                       --repre_dim ${pca_dim} \
                       --vocab_size 1027 --d_model ${d_model} --nhead 8 --num_encoder_layers ${num_encoder_layers} --num_decoder_layers ${num_decoder_layers} --dim_feedforward ${ffn_dim} \
                       --dropout ${dp} --max_seq_length 100 --epochs 40 --log_interval 20 --learning_rate ${lr} --optimizer_type "adam" --label_smoothing $labelsmooth \
                       >> ${expdir}/slm21_inference.log 2>&1
    fi 


    if [ "$stage" == "v2.1" ]; then
        expdir=/data/sls/scratch/clai24/word-seg/codebook-learning/exp/debug_collection_v2.1/mean_pooled_wav2vec2_large_lv60_layer10,11,12,13,14_pca${pca_dim}_dmodel${d_model}_enclayer${num_encoder_layers}_declayer${num_decoder_layers}_ffn_dim${ffn_dim}_dp${dp}_lr${lr}_labelsmooth${labelsmooth}

        CUDA_LAUNCH_BLOCKING=1 python slm21_inference.py \
                       --test_synthetic_path ${data_dir}/preprocess/speechtokens/rvq1/slm21_semantic_test_synthetic.txt \
                       --test_librispeech_path ${data_dir}/preprocess/speechtokens/rvq1/slm21_semantic_test_librispeech.txt \
                       --dev_synthetic_path ${data_dir}/preprocess/speechtokens/rvq1/slm21_semantic_dev_synthetic.txt \
                       --dev_librispeech_path ${data_dir}/preprocess/speechtokens/rvq1/slm21_semantic_dev_librispeech.txt \
                       --test_synthetic_embed_path ${data_dir}/preprocess/speechrepresentations/slm21_semantic_test_synthetic_wav2vec2_large_lv60_layer10,11,12,13,14_pca${pca_dim}_embeddings.npz \
                       --test_librispeech_embed_path ${data_dir}/preprocess/speechrepresentations/slm21_semantic_test_librispeech_wav2vec2_large_lv60_layer10,11,12,13,14_pca${pca_dim}_embeddings.npz \
                       --dev_synthetic_embed_path ${data_dir}/preprocess/speechrepresentations/slm21_semantic_dev_synthetic_wav2vec2_large_lv60_layer10,11,12,13,14_pca${pca_dim}_embeddings.npz \
                       --dev_librispeech_embed_path ${data_dir}/preprocess/speechrepresentations/slm21_semantic_dev_librispeech_wav2vec2_large_lv60_layer10,11,12,13,14_pca${pca_dim}_embeddings.npz \
                       --load_model_path ${expdir}/best_loss_model.pth \
                       --save_dir ${expdir} \
                       --batch_size 1 \
                       --repre_dim ${pca_dim} \
                       --num_layer_repre 5 \
                       --vocab_size 1027 --d_model ${d_model} --nhead 8 --num_encoder_layers ${num_encoder_layers} --num_decoder_layers ${num_decoder_layers} --dim_feedforward ${ffn_dim} \
                       --dropout ${dp} --max_seq_length 100 --epochs 40 --log_interval 20 --learning_rate ${lr} --gradient_acc_steps 16 --optimizer_type "adam" --label_smoothing $labelsmooth \
                       >> ${expdir}/slm21_inference.log 2>&1

    fi 

done ; done ; 
