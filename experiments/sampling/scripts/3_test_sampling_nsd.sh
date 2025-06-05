#!/bin/bash
subject_ids=(1 2 3 4 5 6 7 8)

# Run for each sample size

    echo "Running evaluation with subject_id = $subject_id"
    python experiments/sampling/test_sampling_nsd.py \
        --strategy full \
        --n_selects 100 500 1000 1500 2000 3000 4000 5000 \
        --subjects 1
done

echo "All evaluations completed!"