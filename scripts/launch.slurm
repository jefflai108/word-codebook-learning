#!/bin/bash 
#SBATCH -J cb-hyper-search
#SBATCH -o /data/sls/scratch/clai24/slurm_dumps/codebook-debug-hyperparam-search_%j.out   
#SBATCH -e /data/sls/scratch/clai24/slurm_dumps/codebook-debug-hyperparam-search_%j.err   
#SBATCH --qos=regular 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 
#SBATCH --partition=a5,a6
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00 

## Set the python environment you want to use for your code 
CONDA_ROOT=/data/sls/scratch/clai24/anaconda3
source ${CONDA_ROOT}/etc/profile.d/conda.sh 
conda activate /data/sls/scratch/clai24/anaconda3/envs/prompt-gpt4

data_dir=/data/sls/scratch/clai24/word-seg/flicker8k/

d_model=$1
num_encoder_layers=$2
num_decoder_layers=$3
ffn_dim=$4
dp=$5
lr=$6
context_size=$7
expdir=/data/sls/scratch/clai24/word-seg/codebook-learning/exp/debug_collection/contextsize${context_size}_dmodel${d_model}_enclayer${num_encoder_layers}_declayer${num_decoder_layers}_ffn_dim${ffn_dim}_dp${dp}_lr${lr}
mkdir -p ${expdir}

CUDA_LAUNCH_BLOCKING=1 python main.py \
               --train_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_train.txt \
               --dev_file_path ${data_dir}/preprocess/speechtokens/rvq1/flickr_8k_rvq1_tokens_dev.txt \
               --word_seg_file_path ${data_dir}/word_seg_features/flicker_speech_features.mat \
               --save_dir ${expdir} \
               --batch_size 48 \
               --segment_context_size ${context_size} \
               --vocab_size 1025 --d_model ${d_model} --nhead 8 --num_encoder_layers ${num_encoder_layers} --num_decoder_layers ${num_decoder_layers} --dim_feedforward ${ffn_dim} \
               --dropout ${dp} --max_seq_length 128 --epochs 20 --log_interval 20 --learning_rate ${lr} --optimizer_type "adam" \
               >> ${expdir}/training.log 2>&1

