#!/usr/bin/env python
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from how_to_sample.coreset import ClusteringHelper
from how_to_sample.ed import estimate_ED
from how_to_sample.sampling_full import (
    sample_greedy_ed,
    sample_kcenter,
    sample_kmeans,
    sample_margin,
    sample_oracle,
    sample_random,
    sample_stratified,
)
from how_to_sample.simulations import (
    cluster_identification_accuracy,
    ridge_regression_closed_form,
)


def find_cluster_groups_and_outliers(cluster_centers, n_groups=6, min_group_size=3):
    """
    Find groups of similar clusters and the most isolated clusters.

    Args:
        cluster_centers: Array of cluster centers
        n_groups: Number of groups to identify
        min_group_size: Minimum size for a group to be considered

    Returns:
        groups: List of lists containing indices of clusters in each group
        outliers: Array of indices for the most isolated clusters
    """
    # Calculate pairwise distances
    pairwise_distances = pdist(cluster_centers, metric="euclidean")
    distance_matrix = squareform(pairwise_distances)

    # Find outliers (most isolated clusters)
    mean_distances = np.mean(distance_matrix, axis=1)
    outlier_indices = np.argsort(mean_distances)[-n_groups:]  # Top N most isolated

    # Perform hierarchical clustering
    linkage_matrix = hierarchy.linkage(pairwise_distances, method="ward")

    # Cut the dendrogram to get n_groups
    labels = hierarchy.fcluster(linkage_matrix, n_groups, criterion="maxclust")

    # Organize clusters into groups
    groups = []
    for i in range(1, n_groups + 1):
        group = np.where(labels == i)[0]
        if len(group) >= min_group_size:
            groups.append(group)

    return groups, outlier_indices


def cluster_full_dataset(features, n_clusters, batch_size=10000):
    # Scale features
    scaler = StandardScaler()

    # Fit scaler on a sample if memory is a concern
    sample_idx = np.random.choice(len(features), 100000, replace=False)
    scaler.fit(features[sample_idx])

    # Initialize MiniBatchKMeans
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, batch_size=batch_size, random_state=42, n_init="auto"
    )

    # Process in batches
    for i in range(0, len(features), batch_size):
        batch = features[i : i + batch_size]
        batch_scaled = scaler.transform(batch)
        if i == 0:
            kmeans.partial_fit(batch_scaled)
        else:
            kmeans.partial_fit(batch_scaled)

        if i % 1000000 == 0:
            print(f"Processed {i} samples")

    return kmeans, scaler


def assign_to_clusters(new_features, kmeans_model, scaler, batch_size=10000):
    labels = []
    for i in tqdm(range(0, len(new_features), batch_size)):
        batch = new_features[i : i + batch_size]
        scaled_features = scaler.transform(batch)
        labels.append(kmeans_model.predict(scaled_features))

    labels = np.concatenate(labels)

    return labels


def get_cluster_accuracy(cluster_centers, target_idx, predicted_samples, n_jobs=32):
    """Calculate cluster prediction accuracy with optimized performance.

    Args:
        cluster_centers: Array of cluster centers (n_clusters, n_features)
        target_idx: Index of target cluster
        predicted_samples: Array of predicted samples (n_samples, n_features)
        n_jobs: Number of jobs for parallel processing. None for no parallelization.

    Returns:
        float: Accuracy score
    """
    # For small datasets, use vectorized numpy operations
    if len(predicted_samples) < 10000 or n_jobs is None:
        # Normalize vectors for faster correlation
        norm_pred = (
            predicted_samples / np.linalg.norm(predicted_samples, axis=1)[:, np.newaxis]
        )
        norm_centers = (
            cluster_centers / np.linalg.norm(cluster_centers, axis=1)[:, np.newaxis]
        )

        # Calculate correlations with all clusters at once
        correlations = norm_pred @ norm_centers.T

        # Get correlation with target and max correlation with others
        target_corrs = correlations[:, target_idx]
        other_centers = np.ones(len(cluster_centers), dtype=bool)
        other_centers[target_idx] = False
        other_corrs = np.max(correlations[:, other_centers], axis=1)

        # Count where target correlation exceeds others
        n_correct = np.sum(target_corrs > other_corrs)
        return n_correct / len(predicted_samples)

    # For large datasets, use parallel processing
    else:
        from concurrent.futures import ProcessPoolExecutor
        from math import ceil

        chunk_size = ceil(len(predicted_samples) / n_jobs)
        chunks = [
            predicted_samples[i : i + chunk_size]
            for i in range(0, len(predicted_samples), chunk_size)
        ]

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Process each chunk in parallel
            futures = [
                executor.submit(
                    get_cluster_accuracy,
                    cluster_centers,
                    target_idx,
                    chunk,
                    None,  # Prevent recursive parallelization
                )
                for chunk in chunks
            ]

            # Weight results by chunk size
            weighted_acc = 0
            total_samples = 0
            for future, chunk in zip(futures, chunks):
                chunk_acc = future.result()
                chunk_size = len(chunk)
                weighted_acc += chunk_acc * chunk_size
                total_samples += chunk_size

            return weighted_acc / total_samples


def eval_group_ood_performance(
    X_train, Y_train, X_test, test_cluster_labels, cluster_centers, scaler
):
    ridge = RidgeCV()
    ridge.fit(X_train, Y_train)
    Y_pred = ridge.predict(X_test)
    Y_pred = scaler.transform(Y_pred)
    test_accuracies = []
    for cluster in np.unique(test_cluster_labels):
        cluster = int(cluster)
        cluster_indices = np.where(test_cluster_labels == cluster)[0]
        test_accuracies.append(
            get_cluster_accuracy(cluster_centers, cluster, Y_pred[cluster_indices])
        )

    return test_accuracies


def main(
    strategy, n_select, n_repeats, output_dir, random_seed, laion_features_path, reset
):
    # Determine output directory based on script location and name.
    script_path = Path(__file__).resolve()
    script_name = script_path.stem
    if output_dir is None:
        output_dir = script_path.parent / "outputs" / script_name
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define all possible strategies
    all_strategies = [
        "random",
        "stratified",
        "kmeans",
        "kcenter",
        "greedy_ed",
        "greedy_ed_kmeans",
        "oracle",
        "margin",
    ]

    # Check which strategies need to be run
    strategies_to_run = []
    if strategy == "full":
        for s in all_strategies:
            output_file = (
                output_dir
                / f"results_method={s}_n_images={n_select}_random_seed={random_seed}.json"
            )
            if not output_file.exists() or reset:
                strategies_to_run.append(s)
        if not strategies_to_run:
            print("All strategies have already been run. Nothing to do.")
            return
    else:
        output_file = (
            output_dir
            / f"results_method={strategy}_n_images={n_select}_random_seed={random_seed}.json"
        )
        if output_file.exists() and not reset:
            print(f"Results for strategy {strategy} already exist. Nothing to do.")
            return
        strategies_to_run = [strategy]

    print(f"Will run the following strategies: {strategies_to_run}")

    np.random.seed(random_seed)

    # Only load data if there are strategies to run
    print("Loading LAION features...")
    laion_features = np.load(laion_features_path)
    D = laion_features.shape[1]
    sigma2_noise = 1
    teacher_weights = np.random.normal(loc=0.0, scale=(1.0 / np.sqrt(D)), size=(D, D))

    # Cluster the LAION features (here we use 1000 clusters)
    optimal_k = 1000

    # Load or compute clustering
    laion_features_dir = Path(laion_features_path).parent
    clustering_cache_path = laion_features_dir / f"laion_clustering_k{optimal_k}.pkl"

    if clustering_cache_path.exists():
        print(f"Loading cached clustering results from {clustering_cache_path}")
        with open(clustering_cache_path, "rb") as f:
            clustering_data = pickle.load(f)
            kmeans = clustering_data["kmeans"]
            scaler = clustering_data["scaler"]
            laion_labels = clustering_data["labels"]
    else:
        print("Clustering LAION features with MiniBatchKMeans...")
        kmeans, scaler = cluster_full_dataset(laion_features, optimal_k)
        laion_labels = assign_to_clusters(laion_features, kmeans, scaler)

        # Cache the results
        print(f"Saving clustering results to {clustering_cache_path}")
        clustering_data = {"kmeans": kmeans, "scaler": scaler, "labels": laion_labels}
        with open(clustering_cache_path, "wb") as f:
            pickle.dump(clustering_data, f)

    mu = kmeans.cluster_centers_

    # Draw a subset (500 samples per cluster)
    print("Creating LAION subset...")
    unique_labels, cts = np.unique(laion_labels, return_counts=True)
    laion_subset = []
    laion_subset_labels = []
    for i in range(optimal_k):
        laion_subset.append(
            laion_features[np.random.choice(np.where(laion_labels == i)[0], 500)]
        )
        laion_subset_labels.append(i * np.ones(500))
    laion_subset = np.concatenate(laion_subset)
    laion_subset_labels = np.concatenate(laion_subset_labels)

    del laion_features  # Free up memory

    laion_subset_fMRI = (laion_subset @ teacher_weights) + np.random.normal(
        loc=0.0, scale=np.sqrt(sigma2_noise), size=(laion_subset.shape)
    )

    groups, outliers = find_cluster_groups_and_outliers(mu)

    smaller_kmeans = MiniBatchKMeans(
        n_clusters=100, batch_size=10000, random_state=42, n_init="auto"
    )
    smaller_labels = smaller_kmeans.fit_predict(laion_subset)
    smaller_mu = smaller_kmeans.cluster_centers_

    def run_single_strategy(current_strategy):
        output_file = (
            output_dir
            / f"results_method={current_strategy}_n_images={n_select}_random_seed={random_seed}.json"
        )
        if output_file.exists() and not reset:
            print(
                f"Results for strategy {current_strategy} already exist at {output_file}, skipping..."
            )
            return

        results = {
            "strategy": current_strategy,
            "n_select": n_select,
            "n_repeats": n_repeats,
            "repeats": [],
        }

        for cluster_group_idx, cluster_group in tqdm(
            enumerate(groups), total=len(groups), desc="Cluster groups"
        ):
            N_test_samples = 1000  # Number of samples to test on

            # Get test samples from the target group
            group_clusters = groups[cluster_group_idx]
            group_indices = []
            for cluster in group_clusters:
                group_indices.extend(np.where(laion_subset_labels == cluster)[0])
            group_indices = np.array(group_indices)

            # Get random subset of group samples for testing
            # Sample test indices in a stratified manner across clusters in the group
            test_indices = np.concatenate(
                [
                    np.random.choice(
                        np.where(laion_subset_labels == cluster)[0],
                        N_test_samples
                        // len(group_clusters),  # Divide evenly across group clusters
                        replace=False,
                    )
                    for cluster in group_clusters
                ]
            )

            test_cluster_labels = laion_subset_labels[test_indices]
            X_test = laion_subset_fMRI[test_indices]
            Y_test = laion_subset[test_indices]

            available_indices = np.setdiff1d(
                np.arange(laion_subset.shape[0]), test_indices
            )
            X_available = laion_subset_fMRI[available_indices]
            available_labels = laion_subset_labels[available_indices]
            Y_available = laion_subset[available_indices]

            if current_strategy == "random" or current_strategy == "stratified":
                for repeat in range(n_repeats):
                    print(f"Repeat {repeat + 1}/{n_repeats}")
                    if current_strategy == "random":
                        selected_indices = sample_random(
                            X_available, n_select, random_state=None
                        )
                    elif current_strategy == "stratified":
                        selected_indices = sample_stratified(
                            X_available, available_labels, n_select, random_state=repeat
                        )

                    X_selected = X_available[selected_indices]
                    Y_selected = Y_available[selected_indices]

                    ood_acc = eval_group_ood_performance(
                        X_selected, Y_selected, X_test, test_cluster_labels, mu, scaler
                    )
                    ed = estimate_ED(Y_selected)
                    result = {
                        "repeat": repeat,
                        "ood_accuracy": ood_acc,
                        "effective_dimensionality": ed,
                    }
                    results["repeats"].append(result)
            else:
                if current_strategy == "kmeans":
                    selected_indices = sample_kmeans(X_available, n_select)
                elif current_strategy == "kcenter":
                    selected_indices = sample_kcenter(X_available, n_select)
                elif current_strategy == "greedy_ed":
                    clustering_helper = ClusteringHelper(
                        X_available, smaller_mu, smaller_labels[available_indices]
                    )
                    selected_indices = sample_greedy_ed(
                        X_available, n_select, clustering_helper
                    )
                elif current_strategy == "greedy_ed_kmeans":
                    clustering_helper = ClusteringHelper(
                        X_available, smaller_mu, smaller_labels[available_indices]
                    )
                    selected_indices = sample_greedy_ed(
                        X_available, n_select, clustering_helper, init_kmeans=True
                    )
                elif current_strategy == "oracle":
                    clustering_helper = ClusteringHelper(
                        X_available, mu, available_labels
                    )
                    selected_indices = sample_oracle(
                        X_available,
                        Y_available,
                        n_select,
                        clustering_helper,
                        initial_seed=100,
                    )
                elif current_strategy == "margin":
                    clustering_helper = ClusteringHelper(
                        X_available, mu, available_labels
                    )
                    selected_indices = sample_margin(
                        X_available,
                        Y_available,
                        n_select,
                        clustering_helper,
                        initial_seed=100,
                    )

                X_selected = X_available[selected_indices]
                Y_selected = Y_available[selected_indices]

                ood_acc = eval_group_ood_performance(
                    X_selected,
                    Y_selected,
                    X_test,
                    test_cluster_labels,
                    mu,
                    scaler,
                )
                ed = estimate_ED(Y_selected)
                result = {
                    "ood_accuracy": ood_acc,
                    "effective_dimensionality": ed,
                }
                results["repeats"].append(result)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_file}")

    if strategy == "full":
        for s in strategies_to_run:
            print(f"\nRunning strategy: {s}")
            run_single_strategy(s)
    else:
        run_single_strategy(strategy)


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
            "oracle",
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
        "--random-seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Run strategies even if results already exist",
    )
    parser.add_argument(
        "--laion_features_path",
        type=str,
        default="./data/laion_features.npy",
        help="Path to LAION features file",
    )
    args = parser.parse_args()

    main(
        args.strategy,
        args.n_select,
        args.n_repeats,
        args.output_dir,
        args.random_seed,
        args.laion_features_path,
        args.reset,
    )
