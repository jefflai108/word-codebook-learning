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

for pca_dim in 256 512; do 
for label_smooth_alpha in 0.0 0.1 0.2; do 
    sbatch scripts/inference.slurm 256 3 3 1024 -1 -1 -100 ${label_smooth_alpha} -100 ${pca_dim}
    sbatch scripts/inference.slurm 256 5 3 1024 -1 -1 -100 ${label_smooth_alpha} -100 ${pca_dim}
    sbatch scripts/inference.slurm 512 3 3 2048 -1 -1 -100 ${label_smooth_alpha} -100 ${pca_dim}
    sbatch scripts/inference.slurm 512 6 6 2048 -1 -1 -100 ${label_smooth_alpha} -100 ${pca_dim}
done; done; 

exit 0

#for w2v2_large_layerid in 10 11 12 13 14; do --> iterate over these in scripts/inference.slurm
#for context_size in 1 3 5 7; do --> iterate over these in scripts/inference.slurm
for w2v2_large_layerid in 10; do 
for context_size in 1; do 
    ## launched
    #sbatch scripts/inference.slurm 256 3 3 1024 0.3 1e-3 ${context_size} 0.0 ${w2v2_large_layerid}
    #sbatch scripts/inference.slurm 256 3 3 1024 0.3 5e-4 ${context_size} 0.0 ${w2v2_large_layerid}
    #sbatch scripts/inference.slurm 256 3 3 1024 0.3 1e-4 ${context_size} 0.0 ${w2v2_large_layerid}
    #sbatch scripts/inference.slurm 256 2 3 1024 0.3 1e-3 ${context_size} 0.0 ${w2v2_large_layerid}
    #sbatch scripts/inference.slurm 256 2 3 1024 0.3 5e-4 ${context_size} 0.0 ${w2v2_large_layerid}
    #sbatch scripts/inference.slurm 256 2 3 1024 0.3 1e-4 ${context_size} 0.0 ${w2v2_large_layerid}
    #sbatch scripts/inference.slurm 256 1 3 1024 0.3 1e-3 ${context_size} 0.0 ${w2v2_large_layerid}
    #sbatch scripts/inference.slurm 256 1 3 1024 0.3 5e-4 ${context_size} 0.0 ${w2v2_large_layerid}
    #sbatch scripts/inference.slurm 256 1 3 1024 0.3 1e-4 ${context_size} 0.0 ${w2v2_large_layerid}
   
    #sbatch scripts/inference.slurm 256 3 3 1024 0.1 1e-4 ${context_size} 0.0
    #sbatch scripts/inference.slurm 256 3 3 1024 0.1 5e-4 ${context_size} 0.0
    #sbatch scripts/inference.slurm 256 3 3 1024 0.1 1e-3 ${context_size} 0.0
    #sbatch scripts/inference.slurm 256 5 1 1024 0.1 1e-4 ${context_size} 0.0
    #sbatch scripts/inference.slurm 256 5 1 1024 0.1 5e-4 ${context_size} 0.0
    #sbatch scripts/inference.slurm 256 5 1 1024 0.1 1e-3 ${context_size} 0.0

    #sbatch scripts/inference.slurm 256 3 3 1024 0.1 1e-4 ${context_size} 0.1
    #sbatch scripts/inference.slurm 256 3 3 1024 0.1 5e-4 ${context_size} 0.1
    #sbatch scripts/inference.slurm 256 3 3 1024 0.1 1e-3 ${context_size} 0.1
    #sbatch scripts/inference.slurm 256 5 1 1024 0.1 1e-4 ${context_size} 0.1
    #sbatch scripts/inference.slurm 256 5 1 1024 0.1 5e-4 ${context_size} 0.1
    #sbatch scripts/inference.slurm 256 5 1 1024 0.1 1e-3 ${context_size} 0.1

    #sbatch scripts/inference.slurm 256 3 3 1024 0.1 1e-4 ${context_size} 0.2
    #sbatch scripts/inference.slurm 256 3 3 1024 0.1 5e-4 ${context_size} 0.2
    #sbatch scripts/inference.slurm 256 3 3 1024 0.1 1e-3 ${context_size} 0.2
    #sbatch scripts/inference.slurm 256 5 1 1024 0.1 1e-4 ${context_size} 0.2
    #sbatch scripts/inference.slurm 256 5 1 1024 0.1 5e-4 ${context_size} 0.2
    #sbatch scripts/inference.slurm 256 5 1 1024 0.1 1e-3 ${context_size} 0.2

    #sbatch scripts/inference.slurm 256 3 3 1024 0.2 1e-4 ${context_size} 0.0
    #sbatch scripts/inference.slurm 256 3 3 1024 0.2 5e-4 ${context_size} 0.0
    #sbatch scripts/inference.slurm 256 3 3 1024 0.2 1e-3 ${context_size} 0.0
    #sbatch scripts/inference.slurm 256 5 1 1024 0.2 1e-4 ${context_size} 0.0
    #sbatch scripts/inference.slurm 256 5 1 1024 0.2 5e-4 ${context_size} 0.0
    #sbatch scripts/inference.slurm 256 5 1 1024 0.2 1e-3 ${context_size} 0.0

    #sbatch scripts/inference.slurm 256 3 3 1024 0.3 1e-4 ${context_size} 0.0
    #sbatch scripts/inference.slurm 256 3 3 1024 0.3 5e-4 ${context_size} 0.0
    #sbatch scripts/inference.slurm 256 3 3 1024 0.3 1e-3 ${context_size} 0.0
    #sbatch scripts/inference.slurm 256 5 1 1024 0.3 1e-4 ${context_size} 0.0
    #sbatch scripts/inference.slurm 256 5 1 1024 0.3 5e-4 ${context_size} 0.0
    #sbatch scripts/inference.slurm 256 5 1 1024 0.3 1e-3 ${context_size} 0.0
done 
done 
