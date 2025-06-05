#!/bin/bash
sample_sizes=(100 500 1000 1500 3000 6000 10000)

# Run for each sample size
for n in "${sample_sizes[@]}"; do
    echo "Running evaluation with n_select = $n"
    python experiments/sampling/test_sampling_gmm.py \
        --strategy full \
        --n_select $n \
        --random-seed 42 \
        --reset
done

echo "All evaluations completed!"