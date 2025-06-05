#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np

from how_to_sample.coreset import ClusteringHelper
from how_to_sample.ed import estimate_ED
from how_to_sample.sampling_full import (
    sample_greedy_ed,
    sample_kcenter,
    sample_kmeans,
    sample_margin,
    sample_random,
    sample_stratified,
)
from how_to_sample.simulations import (
    cluster_identification_accuracy,
    generate_gaussian_mixture_samples,
    ridge_regression_closed_form,
)


def eval_dataset(
    X_tr,
    Y_tr,
    mu_tr,
    teacher_weights,
    sigma2_intra,
    sigma2_inter,
    sigma2_noise,
    lam,
):
    """
    Given a training subset (X_tr, Y_tr), compute the ridge regression solution W,
    then evaluate both the in-distribution and out-of-distribution cluster identification accuracy,
    and also compute the effective dimensionality (ED) of X_tr.
    """
    W = ridge_regression_closed_form(X_tr, Y_tr, lam)
    in_dist_accuracy = cluster_identification_accuracy(
        W=W,
        teacher_weights=teacher_weights,
        cluster_centers=mu_tr,
        sigma2_intra=sigma2_intra,
        sigma2_noise=sigma2_noise,
        n=100,
        t=32,
        ood=False,
    )

    out_of_dist_accuracy = cluster_identification_accuracy(
        W=W,
        teacher_weights=teacher_weights,
        cluster_centers=mu_tr,
        sigma2_intra=sigma2_intra,
        sigma2_noise=sigma2_noise,
        n=100,
        t=32,
        ood=True,
        sigma2_inter=sigma2_inter,
    )

    ed = estimate_ED(X_tr)
    return W, in_dist_accuracy, out_of_dist_accuracy, ed


def main(strategy, n_select, n_repeats, output_dir, random_seed, reset):
    # Determine output directory based on script location and name.
    script_path = Path(__file__).resolve()
    script_name = script_path.stem
    if output_dir is None:
        output_dir = script_path.parent / "outputs" / script_name
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if results already exist
    output_file = (
        output_dir
        / f"results_method={strategy}_n_images={n_select}_random_seed={random_seed}.json"
    )
    if output_file.exists() and not reset:
        print(f"Results already exist at {output_file}, skipping...")
        return

    np.random.seed(random_seed)

    # Dictionary to hold overall results.
    results = {
        "strategy": strategy,
        "n_select": n_select,
        "n_repeats": n_repeats,
        "repeats": [],
    }

    # Common parameters for simulation
    lam = 1.0

    D = 512
    num_in_clusters = 100
    sigma2_intra = 10.0 / D
    sigma2_inter = 100.0 / D
    sigma2_noise = 0.25
    N_total = 500000  # Full simulation dataset size
    # Create teacher weights (used to generate responses)
    teacher_weights = np.random.normal(loc=0.0, scale=(1.0 / np.sqrt(D)), size=(D, D))
    print("Generating GMM dataset...")
    X_full, Y_full, mu = generate_gaussian_mixture_samples(
        N_total,
        num_in_clusters,
        D,
        sigma2_intra,
        sigma2_inter,
        teacher_weights,
        sigma2_noise,
        random_state=42,
    )
    cluster_labels = np.zeros(X_full.shape[0])
    n_per_cluster = N_total // num_in_clusters
    for i in range(num_in_clusters):
        cluster_labels[i * n_per_cluster : (i + 1) * n_per_cluster] = i
    eval_params = {
        "mu": mu,
        "sigma2_intra": sigma2_intra,
        "sigma2_inter": sigma2_inter,
        "sigma2_noise": sigma2_noise,
        "lam": lam,
        "teacher_weights": teacher_weights,
    }

    if strategy == "random" or strategy == "stratified":
        for repeat in range(n_repeats):
            print(f"Repeat {repeat + 1}/{n_repeats}")
            seed = 1000 + repeat
            np.random.seed(seed)
            if strategy == "random":
                selected_indices = sample_random(X_full, n_select, random_state=seed)
            elif strategy == "stratified":
                selected_indices = sample_stratified(
                    X_full, cluster_labels, n_select, random_state=seed
                )

            X_selected = X_full[selected_indices]
            Y_selected = Y_full[selected_indices]

            W, in_acc, ood_acc, ed = eval_dataset(
                X_selected,
                Y_selected,
                eval_params["mu"],
                eval_params["teacher_weights"],
                eval_params["sigma2_intra"],
                eval_params["sigma2_inter"],
                eval_params["sigma2_noise"],
                lam,
            )
            result = {
                "repeat": repeat,
                "in_distribution_accuracy": in_acc,
                "ood_accuracy": ood_acc,
                "effective_dimensionality": ed,
            }
            results["repeats"].append(result)
    else:
        clustering_helper = ClusteringHelper(X_full, mu, cluster_labels)
        if strategy == "kmeans":
            selected_indices = sample_kmeans(X_full, n_select)
        elif strategy == "kcenter":
            selected_indices = sample_kcenter(X_full, n_select)
        elif strategy == "greedy_ed":
            selected_indices = sample_greedy_ed(X_full, n_select, clustering_helper)
        elif strategy == "greedy_ed_kmeans":
            selected_indices = sample_greedy_ed(
                X_full, n_select, clustering_helper, init_kmeans=True
            )
        elif strategy == "margin":
            selected_indices = sample_margin(
                X_full,
                Y_full,
                n_select,
                clustering_helper,
                initial_seed=100,
            )
        X_selected = X_full[selected_indices]
        Y_selected = Y_full[selected_indices]

        W, in_acc, ood_acc, ed = eval_dataset(
            X_selected,
            Y_selected,
            eval_params["mu"],
            eval_params["teacher_weights"],
            eval_params["sigma2_intra"],
            eval_params["sigma2_inter"],
            eval_params["sigma2_noise"],
            lam,
        )
        result = {
            "in_distribution_accuracy": in_acc,
            "ood_accuracy": ood_acc,
            "effective_dimensionality": ed,
        }
        results["repeats"].append(result)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sampling strategy evaluation for generalization."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=[
            "random",
            "stratified",
            "kmeans",
            "kcenter",
            "greedy_ed",
            "greedy_ed_kmeans",
            "margin",
            "full",
        ],
        required=True,
        help="Sampling strategy to evaluate",
    )
    parser.add_argument(
        "--n_select", type=int, default=6000, help="Number of samples to select"
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=100,
        help="Number of repeats for random operations",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory for results JSON"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Run strategies even if results already exist",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    if args.strategy == "full":
        for strategy in [
            "random",
            "stratified",
            "kmeans",
            "kcenter",
            "greedy_ed",
            "greedy_ed_kmeans",
            "margin",
        ]:
            print(f"\nProcessing strategy: {strategy}")
            main(
                strategy,
                args.n_select,
                args.n_repeats,
                args.output_dir,
                args.random_seed,
                args.reset,
            )
    else:
        main(
            args.strategy,
            args.n_select,
            args.n_repeats,
            args.output_dir,
            args.random_seed,
            args.reset,
        )
