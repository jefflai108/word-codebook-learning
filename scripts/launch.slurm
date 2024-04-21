#!/bin/bash 
#SBATCH -J cb-hyper-search
#SBATCH -o /data/sls/scratch/clai24/slurm_dumps/codebook-debug-hyperparam-search_%j.out   
#SBATCH -e /data/sls/scratch/clai24/slurm_dumps/codebook-debug-hyperparam-search_%j.err   
#SBATCH --qos=regular 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 
#SBATCH --partition=a5,a6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00 
#SBATCH --requeue
#SBATCH --mem=80G

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
word_pool=${11}
word_norm=${12}

stage="v2.1_flickr8k+spokencoco" # v0 / v1 / v2 / v2.1 / v2.1_flickr8k+spokencoco

if [ "$stage" == "v0" ]; then
    expdir=/data/sls/scratch/clai24/word-seg/codebook-learning/exp/debug_collection_v0/contextsize${context_size}_dmodel${d_model}_enclayer${num_encoder_layers}_declayer${num_decoder_layers}_ffn_dim${ffn_dim}_dp${dp}_lr${lr}
    mkdir -p ${expdir}

    CUDA_LAUNCH_BLOCKING=1 python main.py \
                   --train_token_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_train.txt \
                   --dev_token_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_dev.txt \
                   --train_embed_file_path ${data_dir}/preprocess/speechrepresentations/flickr_8k_wav2vec2_large_lv60_layer${w2v2_large_layerid}_embeddings_train.npz \
                   --dev_embed_file_path ${data_dir}/preprocess/speechrepresentations/flickr_8k_wav2vec2_large_lv60_layer${w2v2_large_layerid}_embeddings_dev.npz \
                   --word_seg_file_path ${data_dir}/word_seg_features/flicker_speech_features.mat \
                   --save_dir ${expdir} \
                   --batch_size 64 \
                   --segment_context_size ${context_size} \
                   --repre_dim 1024 \
                   --vocab_size 1027 --d_model ${d_model} --nhead 8 --num_encoder_layers ${num_encoder_layers} --num_decoder_layers ${num_decoder_layers} --dim_feedforward ${ffn_dim} \
                   --dropout ${dp} --max_seq_length 130 --epochs 30 --log_interval 20 --learning_rate ${lr} --optimizer_type "adam" --label_smoothing $labelsmooth \
                   >> ${expdir}/training.log 2>&1
fi 

if [ "$stage" == "v1" ]; then
    expdir=/data/sls/scratch/clai24/word-seg/codebook-learning/exp/debug_collection_v1/wav2vec2_large_lv60_layer${w2v2_large_layerid}_contextsize${context_size}_dmodel${d_model}_enclayer${num_encoder_layers}_declayer${num_decoder_layers}_ffn_dim${ffn_dim}_dp${dp}_lr${lr}_labelsmooth${labelsmooth}
    mkdir -p ${expdir}

    CUDA_LAUNCH_BLOCKING=1 python main_v1.py \
                   --train_token_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_train.txt \
                   --dev_token_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_dev.txt \
                   --train_embed_file_path ${data_dir}/preprocess/speechrepresentations/flickr_8k_wav2vec2_large_lv60_layer${w2v2_large_layerid}_embeddings_train.npz \
                   --dev_embed_file_path ${data_dir}/preprocess/speechrepresentations/flickr_8k_wav2vec2_large_lv60_layer${w2v2_large_layerid}_embeddings_dev.npz \
                   --word_seg_file_path ${data_dir}/word_seg_features/flicker_speech_features.mat \
                   --save_dir ${expdir} \
                   --batch_size 64 \
                   --segment_context_size ${context_size} \
                   --repre_dim 1024 \
                   --vocab_size 1027 --d_model ${d_model} --nhead 8 --num_encoder_layers ${num_encoder_layers} --num_decoder_layers ${num_decoder_layers} --dim_feedforward ${ffn_dim} \
                   --dropout ${dp} --max_seq_length 130 --epochs 30 --log_interval 20 --learning_rate ${lr} --optimizer_type "adam" --label_smoothing $labelsmooth \
                   >> ${expdir}/training.log 2>&1
fi 

if [ "$stage" == "v2" ]; then
    expdir=/data/sls/scratch/clai24/word-seg/codebook-learning/exp/debug_collection_v2/mean_pooled_wav2vec2_large_lv60_layer${w2v2_large_layerid}_dmodel${d_model}_enclayer${num_encoder_layers}_declayer${num_decoder_layers}_ffn_dim${ffn_dim}_dp${dp}_lr${lr}_labelsmooth${labelsmooth}
    expdir=/data/sls/scratch/clai24/word-seg/codebook-learning/exp/debug_collection_v2/mean_pooled_wav2vec2_large_lv60_layer${w2v2_large_layerid}_pca${pca_dim}_dmodel${d_model}_enclayer${num_encoder_layers}_declayer${num_decoder_layers}_ffn_dim${ffn_dim}_dp${dp}_lr${lr}_labelsmooth${labelsmooth}
    mkdir -p ${expdir}

    #CUDA_LAUNCH_BLOCKING=1 python main_v2.py \
    #           --train_token_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_train.txt \
    #           --dev_token_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_dev.txt \
    #           --train_embed_file_path ${data_dir}/preprocess/speechrepresentations/flickr_8k_wav2vec2_large_lv60_layer${w2v2_large_layerid}_embeddings_train.npz \
    #           --dev_embed_file_path ${data_dir}/preprocess/speechrepresentations/flickr_8k_wav2vec2_large_lv60_layer${w2v2_large_layerid}_embeddings_dev.npz \
    #           --word_seg_file_path ${data_dir}/word_seg_features/flicker_speech_features.mat \
    #           --save_dir ${expdir} \
    #           --batch_size 64 \
    #           --repre_dim 1024 \
    #           --vocab_size 1027 --d_model ${d_model} --nhead 8 --num_encoder_layers ${num_encoder_layers} --num_decoder_layers ${num_decoder_layers} --dim_feedforward ${ffn_dim} \
    #           --dropout ${dp} --max_seq_length 100 --epochs 40 --log_interval 20 --learning_rate ${lr} --gradient_acc_steps 4 --optimizer_type "adam" --label_smoothing $labelsmooth \
    #           >> ${expdir}/training.log 2>&1

    CUDA_LAUNCH_BLOCKING=1 python main_v2.py \
               --train_token_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_train.txt \
               --dev_token_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_dev.txt \
               --train_embed_file_path ${data_dir}/preprocess/speechrepresentations/flickr_8k_wav2vec2_large_lv60_layer${w2v2_large_layerid}_pca${pca_dim}_embeddings_train.npz \
               --dev_embed_file_path ${data_dir}/preprocess/speechrepresentations/flickr_8k_wav2vec2_large_lv60_layer${w2v2_large_layerid}_pca${pca_dim}_embeddings_dev.npz \
               --word_seg_file_path ${data_dir}/word_seg_features/flicker_speech_features.mat \
               --save_dir ${expdir} \
               --batch_size 64 \
               --repre_dim ${pca_dim} \
               --vocab_size 1027 --d_model ${d_model} --nhead 8 --num_encoder_layers ${num_encoder_layers} --num_decoder_layers ${num_decoder_layers} --dim_feedforward ${ffn_dim} \
               --dropout ${dp} --max_seq_length 100 --epochs 40 --log_interval 20 --learning_rate ${lr} --gradient_acc_steps 4 --optimizer_type "adam" --label_smoothing $labelsmooth \
               >> ${expdir}/training.log 2>&1
fi 

if [ "$stage" == "v2.1" ]; then
    expdir=/data/sls/scratch/clai24/word-seg/codebook-learning/exp/debug_collection_v2.1/mean_pooled_wav2vec2_large_lv60_layer10,11,12,13,14_pca${pca_dim}_dmodel${d_model}_enclayer${num_encoder_layers}_declayer${num_decoder_layers}_ffn_dim${ffn_dim}_dp${dp}_lr${lr}_labelsmooth${labelsmooth}
    mkdir -p ${expdir}

    CUDA_LAUNCH_BLOCKING=1 python main_v2_1.py \
               --train_token_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_train.txt \
               --dev_token_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_dev.txt \
               --train_embed_file_paths ${data_dir}/preprocess/speechrepresentations/flickr_8k_wav2vec2_large_lv60_layer10,11,12,13,14_pca${pca_dim}_embeddings_train.npz \
               --dev_embed_file_paths ${data_dir}/preprocess/speechrepresentations/flickr_8k_wav2vec2_large_lv60_layer10,11,12,13,14_pca${pca_dim}_embeddings_dev.npz \
               --word_seg_file_path ${data_dir}/word_seg_features/flicker_speech_features.mat \
               --save_dir ${expdir} \
               --batch_size 16 \
               --repre_dim ${pca_dim} \
               --num_layer_repre 5 \
               --vocab_size 1027 --d_model ${d_model} --nhead 8 --num_encoder_layers ${num_encoder_layers} --num_decoder_layers ${num_decoder_layers} --dim_feedforward ${ffn_dim} \
               --dropout ${dp} --max_seq_length 100 --epochs 40 --log_interval 20 --learning_rate ${lr} --gradient_acc_steps 16 --optimizer_type "adam" --label_smoothing $labelsmooth \
               >> ${expdir}/training.log 2>&1
fi 

if [ "$stage" == "v2.1_flickr8k+spokencoco" ]; then 
    data_dir=/data/sls/scratch/clai24/word-seg/mix-corpus/
    expdir=/data/sls/scratch/clai24/word-seg/codebook-learning/exp/debug_collection_v2.1_flickr8k+spokencoco/${word_pool}_pooled_${word_norm}_normed_wav2vec2_large_lv60_layer10,11,12,13,14_pca${pca_dim}_dmodel${d_model}_enclayer${num_encoder_layers}_declayer${num_decoder_layers}_ffn_dim${ffn_dim}_dp${dp}_lr${lr}_labelsmooth${labelsmooth}
    mkdir -p ${expdir}

    CUDA_LAUNCH_BLOCKING=1 python main_v2_1_flickr8k_spokencoco.py \
               --train_token_file_path ${data_dir}/speechtokens/rvq1/flickr_8k_spokencoco/flickr_8k_spokencoco_rvq1_tokens_train_split_SPLIT.txt \
               --dev_token_file_path ${data_dir}/speechtokens/rvq1/flickr_8k_spokencoco/flickr_8k_spokencoco_rvq1_tokens_dev.txt \
               --train_embed_file_paths ${data_dir}/speechrepresentations/flickr_8k_spokencoco/flickr_8k_spokencoco_wav2vec2_large_lv60_layer10,11,12,13,14_pca${pca_dim}_embeddings_train_split_SPLIT.npz \
               --dev_embed_file_paths ${data_dir}/speechrepresentations/flickr_8k_spokencoco/flickr_8k_spokencoco_wav2vec2_large_lv60_layer10,11,12,13,14_pca${pca_dim}_embeddings_dev.npz \
               --total_train_split_num 20 \
               --word_seg_file_path ${data_dir}/word_seg_features/flickr_8k_spokencoco_speech_features.mat \
               --save_dir ${expdir} \
               --batch_size 20 --gradient_acc_steps 16 \
               --repre_dim ${pca_dim} \
               --num_layer_repre 5 \
               --word_pooling ${word_pool} --norm_type ${word_norm} \
               --vocab_size 1027 --d_model ${d_model} --nhead 8 --num_encoder_layers ${num_encoder_layers} --num_decoder_layers ${num_decoder_layers} --dim_feedforward ${ffn_dim} --model_activation "gelu" \
               --dropout ${dp} --max_seq_length 100 --epochs 16 --log_interval 20 --learning_rate ${lr} --optimizer_type "adamw" --label_smoothing $labelsmooth \
               >> ${expdir}/training.log 2>&1
fi

