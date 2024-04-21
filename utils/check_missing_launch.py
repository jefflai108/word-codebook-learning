import os

# Define the base directory path
base_path = "/data/sls/scratch/clai24/word-seg/codebook-learning/exp/debug_collection_v2.1_flickr8k+spokencoco"

# Read existing directories from the base path
existing_directories = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

# Define the hyper-parameters and their possible values
pca_dim = [64, 128]
dp_values = [0.2, 0.3]
lr_values = ['1e-3', '5e-4', '1e-4']
labelsmooth_values = [0.0, 0.1, 0.2]

# All possible combinations
all_combinations = [(pca, dp, lr, ls) for dp in dp_values for lr in lr_values for ls in labelsmooth_values for pca in pca_dim]

# Template for directory names
template = "mean_pooled_batchnorm_normed_wav2vec2_large_lv60_layer10,11,12,13,14_pca{pca_dim}_dmodel256_enclayer3_declayer3_ffn_dim1024_dp{dp}_lr{lr}_labelsmooth{ls}"

# Identify missing combinations
missing_commands = []
for pca, dp, lr, ls in all_combinations:
    dir_name = template.format(pca_dim=pca, dp=dp, lr=lr, ls=ls)
    full_dir_path = os.path.join(base_path, dir_name)
    if full_dir_path not in existing_directories:
        # Generate the sbatch command for the missing configuration
        command = f"sbatch scripts/launch.slurm 256 3 3 1024 {dp} {lr} -100 {ls} -100 {pca} \"mean\" \"batchnorm\""
        missing_commands.append(command)

# Output the sbatch commands for missing directories
print("Missing Launch Configurations:")
for cmd in missing_commands:
    print(cmd)

