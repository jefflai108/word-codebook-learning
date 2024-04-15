import os

# Path to the root directory
root_dir = '/data/sls/scratch/clai24/word-seg/codebook-learning/exp/debug_collection_v2'

top_n = 15

# Structure to hold the UED scores for each system
systems_scores = []

# Walk through each subdirectory and file
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file == 'unit_edit_distance.log':
            file_path = os.path.join(subdir, file)
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    # Parsing the UED values
                    dev_synthetic_ued = float(lines[1].split()[-1])
                    dev_librispeech_ued = float(lines[3].split()[-1])
                    test_synthetic_ued = float(lines[0].split()[-1])
                    test_librispeech_ued = float(lines[2].split()[-1])
                    # Calculating average Dev UED
                    avg_dev_ued = (dev_synthetic_ued + dev_librispeech_ued) / 2
                    # Storing all relevant scores along with the system path
                    systems_scores.append((subdir, avg_dev_ued, dev_synthetic_ued, dev_librispeech_ued, test_synthetic_ued, test_librispeech_ued))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# Sorting systems by their average Dev UED (ascending order)
systems_scores_sorted = sorted(systems_scores, key=lambda x: x[1])

# Selecting the top N systems based on average Dev UED
top_n_systems = systems_scores_sorted[:top_n]

# Printing the systems with their respective scores
for system in top_n_systems:
    print(f"System: {system[0]}, "
          f"Average Dev UED: {system[1]}, "
          f"Dev Synthetic UED: {system[2]}, "
          f"Dev Librispeech UED: {system[3]}, "
          f"Test Synthetic UED: {system[4]}, "
          f"Test Librispeech UED: {system[5]}")

