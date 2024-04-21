#!/bin/bash 

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

#for pca_dim in 64 128 256 512; do 
for pca_dim in 128; do # save (256, 512) for satori 
for label_smooth_alpha in 0.0 0.1 0.2; do 
for lr in 1e-3 5e-4 1e-4; do 
for dp in 0.2 0.3; do 
    sbatch scripts/launch.slurm 256 3 3 1024 ${dp} ${lr} -100 ${label_smooth_alpha} -100 ${pca_dim} "mean" "batchnorm"

    #sbatch scripts/launch.slurm 512 3 3 2048 ${dp} ${lr} -100 ${label_smooth_alpha} -100 ${pca_dim} "mean" "batchnorm"
    #sbatch scripts/launch.slurm 768 3 3 3072 ${dp} ${lr} -100 ${label_smooth_alpha} -100 ${pca_dim} "mean" "batchnorm"

    #sbatch scripts/launch.slurm 256 3 3 1024 ${dp} ${lr} -100 ${label_smooth_alpha} -100 ${pca_dim} "weighted_mean"
    #sbatch scripts/launch.slurm 512 3 3 2048 ${dp} ${lr} -100 ${label_smooth_alpha} -100 ${pca_dim} "weighted_mean"
    #sbatch scripts/launch.slurm 768 3 3 3072 ${dp} ${lr} -100 ${label_smooth_alpha} -100 ${pca_dim} "weighted_mean"

    #sbatch scripts/launch.slurm 256 3 3 1024 ${dp} ${lr} -100 ${label_smooth_alpha} -100 ${pca_dim} "conv_mean"
    #sbatch scripts/launch.slurm 512 3 3 2048 ${dp} ${lr} -100 ${label_smooth_alpha} -100 ${pca_dim} "conv_mean"
    #sbatch scripts/launch.slurm 768 3 3 3072 ${dp} ${lr} -100 ${label_smooth_alpha} -100 ${pca_dim} "conv_mean"

    #sbatch scripts/launch.slurm 256 3 3 1024 ${dp} ${lr} -100 ${label_smooth_alpha} -100 ${pca_dim} "lde8"
    #sbatch scripts/launch.slurm 512 3 3 2048 ${dp} ${lr} -100 ${label_smooth_alpha} -100 ${pca_dim} "lde8"
    #sbatch scripts/launch.slurm 768 3 3 3072 ${dp} ${lr} -100 ${label_smooth_alpha} -100 ${pca_dim} "lde8"

    #sbatch scripts/launch.slurm 256 3 3 1024 ${dp} ${lr} -100 ${label_smooth_alpha} -100 ${pca_dim} "lde32"
    #sbatch scripts/launch.slurm 512 3 3 2048 ${dp} ${lr} -100 ${label_smooth_alpha} -100 ${pca_dim} "lde32"
    #sbatch scripts/launch.slurm 768 3 3 3072 ${dp} ${lr} -100 ${label_smooth_alpha} -100 ${pca_dim} "lde32"
done; done; done; done;

exit 0 

for w2v2_large_layerid in 10 11 12 13 14; do 
for context_size in 1 3 5 7; do 
    ## launched
    #sbatch scripts/launch.slurm 256 3 3 1024 0.3 1e-3 ${context_size} 0.0 ${w2v2_large_layerid}
    #sbatch scripts/launch.slurm 256 3 3 1024 0.3 5e-4 ${context_size} 0.0 ${w2v2_large_layerid}
    #sbatch scripts/launch.slurm 256 3 3 1024 0.3 1e-4 ${context_size} 0.0 ${w2v2_large_layerid}
    #sbatch scripts/launch.slurm 256 2 3 1024 0.3 1e-3 ${context_size} 0.0 ${w2v2_large_layerid}
    #sbatch scripts/launch.slurm 256 2 3 1024 0.3 5e-4 ${context_size} 0.0 ${w2v2_large_layerid}
    #sbatch scripts/launch.slurm 256 2 3 1024 0.3 1e-4 ${context_size} 0.0 ${w2v2_large_layerid}
    #sbatch scripts/launch.slurm 256 1 3 1024 0.3 1e-3 ${context_size} 0.0 ${w2v2_large_layerid}
    #sbatch scripts/launch.slurm 256 1 3 1024 0.3 5e-4 ${context_size} 0.0 ${w2v2_large_layerid}
    #sbatch scripts/launch.slurm 256 1 3 1024 0.3 1e-4 ${context_size} 0.0 ${w2v2_large_layerid}

    #sbatch scripts/launch.slurm 256 3 3 1024 0.1 1e-4 ${context_size} 0.0
    #sbatch scripts/launch.slurm 256 3 3 1024 0.1 5e-4 ${context_size} 0.0
    #sbatch scripts/launch.slurm 256 3 3 1024 0.1 1e-3 ${context_size} 0.0
    #sbatch scripts/launch.slurm 256 5 1 1024 0.1 1e-4 ${context_size} 0.0
    #sbatch scripts/launch.slurm 256 5 1 1024 0.1 5e-4 ${context_size} 0.0
    #sbatch scripts/launch.slurm 256 5 1 1024 0.1 1e-3 ${context_size} 0.0

    #sbatch scripts/launch.slurm 256 3 3 1024 0.1 1e-4 ${context_size} 0.1
    #sbatch scripts/launch.slurm 256 3 3 1024 0.1 5e-4 ${context_size} 0.1
    #sbatch scripts/launch.slurm 256 3 3 1024 0.1 1e-3 ${context_size} 0.1
    #sbatch scripts/launch.slurm 256 5 1 1024 0.1 1e-4 ${context_size} 0.1
    #sbatch scripts/launch.slurm 256 5 1 1024 0.1 5e-4 ${context_size} 0.1
    #sbatch scripts/launch.slurm 256 5 1 1024 0.1 1e-3 ${context_size} 0.1

    #sbatch scripts/launch.slurm 256 3 3 1024 0.1 1e-4 ${context_size} 0.2
    #sbatch scripts/launch.slurm 256 3 3 1024 0.1 5e-4 ${context_size} 0.2
    #sbatch scripts/launch.slurm 256 3 3 1024 0.1 1e-3 ${context_size} 0.2
    #sbatch scripts/launch.slurm 256 5 1 1024 0.1 1e-4 ${context_size} 0.2
    #sbatch scripts/launch.slurm 256 5 1 1024 0.1 5e-4 ${context_size} 0.2
    #sbatch scripts/launch.slurm 256 5 1 1024 0.1 1e-3 ${context_size} 0.2

    #sbatch scripts/launch.slurm 256 3 3 1024 0.2 1e-4 ${context_size} 0.0
    #sbatch scripts/launch.slurm 256 3 3 1024 0.2 5e-4 ${context_size} 0.0
    #sbatch scripts/launch.slurm 256 3 3 1024 0.2 1e-3 ${context_size} 0.0
    #sbatch scripts/launch.slurm 256 5 1 1024 0.2 1e-4 ${context_size} 0.0
    #sbatch scripts/launch.slurm 256 5 1 1024 0.2 5e-4 ${context_size} 0.0
    #sbatch scripts/launch.slurm 256 5 1 1024 0.2 1e-3 ${context_size} 0.0

    #sbatch scripts/launch.slurm 256 3 3 1024 0.3 1e-4 ${context_size} 0.0
    #sbatch scripts/launch.slurm 256 3 3 1024 0.3 5e-4 ${context_size} 0.0
    #sbatch scripts/launch.slurm 256 3 3 1024 0.3 1e-3 ${context_size} 0.0
    #sbatch scripts/launch.slurm 256 5 1 1024 0.3 1e-4 ${context_size} 0.0
    #sbatch scripts/launch.slurm 256 5 1 1024 0.3 5e-4 ${context_size} 0.0
    #sbatch scripts/launch.slurm 256 5 1 1024 0.3 1e-3 ${context_size} 0.0
done     
done 
