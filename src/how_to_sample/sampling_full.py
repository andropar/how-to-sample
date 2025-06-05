import concurrent.futures
from typing import List, Optional, Set, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import RidgeCV
from tqdm import tqdm

from .coreset import kCenterGreedy
from .ed import estimate_ED, estimate_ED_by_update, update_cov


def sample_random(
    X: np.ndarray, n_select: int, random_state: Optional[int] = None
) -> np.ndarray:
    """Uniformly random sampling from all data.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        n_select: Number of samples to select
        random_state: Random seed for reproducibility

    Returns:
        Array of selected sample indices
    """
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.random.choice(X.shape[0], n_select, replace=False)
    return indices


def sample_stratified(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    n_select: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Stratified sampling: select approximately equal number of samples per cluster.

    If not enough samples are present in a cluster, take all available samples.
    If n_select is less than number of clusters, randomly select n_select clusters
    and take one sample from each.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        cluster_labels: Cluster assignment for each sample
        n_select: Number of samples to select
        random_state: Random seed for reproducibility

    Returns:
        Array of selected sample indices
    """
    if random_state is not None:
        np.random.seed(random_state)
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    # Handle case where n_select < n_clusters
    if n_select < n_clusters:
        selected_clusters = np.random.choice(unique_clusters, n_select, replace=False)
        selected_indices = []
        for cluster in selected_clusters:
            cluster_idx = np.where(cluster_labels == cluster)[0]
            selected = np.random.choice(cluster_idx, 1, replace=False)
            selected_indices.extend(selected)
        return np.array(selected_indices)

    # Distribute samples approximately equally across clusters
    n_per_cluster = n_select // n_clusters
    selected_indices = []
    for cluster in unique_clusters:
        cluster_idx = np.where(cluster_labels == cluster)[0]
        if len(cluster_idx) < n_per_cluster:
            # Take all samples if cluster is too small
            selected = cluster_idx
        else:
            selected = np.random.choice(cluster_idx, n_per_cluster, replace=False)
        selected_indices.extend(selected)

    selected_indices = np.array(selected_indices)

    # Fill remaining slots if needed
    if len(selected_indices) < n_select:
        remaining = np.setdiff1d(np.arange(X.shape[0]), selected_indices)
        extra = np.random.choice(
            remaining, n_select - len(selected_indices), replace=False
        )
        selected_indices = np.concatenate([selected_indices, extra])
    return selected_indices


def sample_kmeans(
    X: np.ndarray, n_select: int, random_state: Optional[int] = None
) -> np.ndarray:
    """K-means based sampling: cluster data and select nearest sample to each centroid.

    Runs MiniBatchKMeans with k = n_select clusters, then selects the sample
    closest to each centroid. Uses Annoy for efficient approximate nearest
    neighbor search.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        n_select: Number of samples to select (equals number of clusters)
        random_state: Random seed for reproducibility

    Returns:
        Array of selected sample indices
    """
    from annoy import AnnoyIndex
    from sklearn.cluster import MiniBatchKMeans

    if random_state is not None:
        np.random.seed(random_state)

    # Run KMeans clustering
    kmeans = MiniBatchKMeans(
        n_clusters=n_select, batch_size=1000, random_state=random_state
    )
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_

    # Build Annoy index for fast nearest neighbor search
    n_features = X.shape[1]
    annoy_index = AnnoyIndex(n_features, "euclidean")
    for i in range(X.shape[0]):
        annoy_index.add_item(i, X[i])
    annoy_index.build(n_trees=50)  # More trees = better accuracy but slower build

    # Find nearest sample for each centroid
    selected_indices = []
    for centroid in centroids:
        idx, _ = annoy_index.get_nns_by_vector(centroid, 1, include_distances=True)
        selected_indices.extend(idx)

    return np.array(selected_indices)


def sample_kcenter(
    X: np.ndarray, n_select: int, random_state: Optional[int] = None
) -> np.ndarray:
    """K-center greedy sampling for diverse subset selection.

    Uses the k-center greedy algorithm to iteratively select points that
    maximize the minimum distance to previously selected points, promoting
    diversity in the selected subset.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        n_select: Number of samples to select
        random_state: Random seed for reproducibility

    Returns:
        Array of selected sample indices
    """
    if random_state is not None:
        np.random.seed(random_state)
    kcg = kCenterGreedy(X)
    selected_indices = kcg.select_batch_([], n_select)
    return np.array(selected_indices)


def compute_ed_of_set(i: int, selected_set: np.ndarray) -> Tuple[float, int]:
    """Compute effective dimensionality of a set of samples.

    Args:
        i: Index identifier for parallel processing
        selected_set: Set of feature vectors

    Returns:
        Tuple of (effective_dimensionality, index)
    """
    return estimate_ED(selected_set), i


def compute_ed_of_set_incremental(
    i: int,
    current_mean: np.ndarray,
    current_cov: np.ndarray,
    n_samples: int,
    new_feature: np.ndarray,
) -> Tuple[float, int]:
    """Compute effective dimensionality incrementally when adding a new sample.

    Args:
        i: Index identifier for parallel processing
        current_mean: Current mean of selected samples
        current_cov: Current covariance matrix of selected samples
        n_samples: Number of currently selected samples
        new_feature: New feature vector to add

    Returns:
        Tuple of (effective_dimensionality, index)
    """
    return estimate_ED_by_update(current_mean, current_cov, n_samples, new_feature), i


def sample_greedy_ed(
    X: np.ndarray,
    n_select: int,
    clustering_helper,
    init_kmeans: bool = False,
    n_jobs: int = 32,
) -> np.ndarray:
    """Greedily select samples that maximize effective dimensionality (ED).

    This method iteratively selects samples that maximize the effective
    dimensionality of the selected subset. Uses incremental ED calculation
    with parallelization for efficiency. Can optionally initialize with
    k-means centroids for better starting points.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        n_select: Number of samples to select
        clustering_helper: ClusteringHelper instance for candidate sampling
        init_kmeans: If True, initialize with k-means centroids
        n_jobs: Number of parallel workers for ED computation

    Returns:
        Array of selected sample indices
    """
    selected = []

    # Initialize selection strategy
    if init_kmeans:
        if n_select < len(clustering_helper.centroids):
            # Select from random clusters if we need fewer samples than centroids
            random_clusters = np.random.choice(
                np.arange(len(clustering_helper.centroids)),
                n_select // 5,
                replace=False,
            )
            for cluster_idx in random_clusters:
                random_sample_indices, _ = clustering_helper.get_closest_samples(
                    cluster_idx
                )
                selected.extend(random_sample_indices)
        else:
            # Use all cluster centroids as initial selection
            initial, _ = clustering_helper.get_closest_samples()
            selected.extend(initial)
    else:
        # Start with 2 randomly chosen samples
        selected = np.random.choice(np.arange(X.shape[0]), 2, replace=False).tolist()

    remaining = set(range(X.shape[0])) - set(selected)

    # Initialize statistics for incremental ED computation
    selected_samples = X[selected]
    current_mean = np.mean(selected_samples, axis=0)
    current_cov = np.cov(selected_samples.T)

    # Greedy selection loop
    while len(selected) < n_select:
        if len(remaining) == 0:
            break

        # Sample candidate pool from clusters
        random_sample_indices, _ = clustering_helper.get_random_samples(10)
        candidate_pool = X[random_sample_indices]

        # Filter candidates that are too close to existing samples
        distances = cdist(candidate_pool, X[selected], metric="euclidean")
        min_distances = np.min(distances, axis=1)
        candidate_pool = candidate_pool[min_distances > 0.1]
        random_sample_indices = np.array(random_sample_indices)[min_distances > 0.1]

        # Calculate ED values in parallel using incremental update
        ed_values = np.zeros(len(candidate_pool))
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    compute_ed_of_set_incremental,
                    i,
                    current_mean,
                    current_cov,
                    len(selected),
                    candidate,
                )
                for i, candidate in enumerate(candidate_pool)
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(candidate_pool),
                desc="Computing ED values",
            ):
                result, i = future.result()
                ed_values[i] = result

        # Select candidate with highest ED
        best_idx = np.argmax(ed_values)
        best_candidate = random_sample_indices[best_idx]
        selected.append(best_candidate)
        remaining.remove(best_candidate)

        # Update statistics incrementally
        new_feature = X[best_candidate]
        current_mean = (current_mean * len(selected) + new_feature) / (
            len(selected) + 1
        )
        current_cov = update_cov(current_cov, current_mean, new_feature, len(selected))

        # Progress reporting
        if len(selected) % 100 == 0:
            print(
                f"Greedy ED sampling: selected {len(selected)} samples, ED: {ed_values[best_idx]}"
            )

    return np.array(selected)


def sample_oracle(
    X: np.ndarray,
    Y: np.ndarray,
    n_select: int,
    clustering_helper,
    random_state: Optional[int] = None,
    initial_seed: int = 100,
    batch_size: int = 100,
) -> np.ndarray:
    """Model-based active learning using prediction error as acquisition function.

    Starts with an initial random seed set, trains a ridge regression model,
    then iteratively selects batches of points with highest prediction error.
    The model is retrained after each batch until n_select samples are collected.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        Y: Target matrix of shape (n_samples, n_targets)
        n_select: Number of samples to select
        clustering_helper: ClusteringHelper instance for candidate sampling
        random_state: Random seed for reproducibility
        initial_seed: Size of initial random seed set
        batch_size: Number of samples to add in each iteration

    Returns:
        Array of selected sample indices
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Initialize with random seed set
    selected = list(
        np.random.choice(np.arange(X.shape[0]), initial_seed, replace=False)
    )
    remaining = set(range(X.shape[0])) - set(selected)

    # Train initial model
    model = RidgeCV()
    model.fit(X[selected], Y[selected])

    # Iterative batch selection
    while len(selected) < n_select:
        # Sample candidate pool from clusters
        candidate_pool = np.array(clustering_helper.get_random_samples(10)[0])

        # Compute prediction errors
        preds = model.predict(X[candidate_pool])
        errors = np.mean((preds - Y[candidate_pool]) ** 2, axis=1)

        # Select samples with highest errors
        top_indices = np.argsort(errors)[-batch_size:]
        selected_candidates = np.array(candidate_pool)[top_indices]
        selected.extend(selected_candidates.tolist())
        remaining = remaining - set(selected_candidates.tolist())

        # Retrain model with new samples
        model.fit(X[selected], Y[selected])
        print(f"Model active sampling: selected {len(selected)} samples")

        if len(remaining) == 0:
            break

    # Trim to exact size if needed
    if len(selected) > n_select:
        selected = selected[:n_select]
    return np.array(selected)


def sample_diversity(
    X: np.ndarray,
    n_select: int,
    clustering_helper,
    random_state: Optional[int] = None,
    initial_seed: int = 100,
    batch_size: int = 100,
) -> np.ndarray:
    """Diversity-based active learning using maximum minimum distance criterion.

    Selects samples that are most different from the currently selected set
    by maximizing the minimum distance to existing samples. This approach
    is much faster than uncertainty-based methods as it doesn't require
    model training.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        n_select: Number of samples to select
        clustering_helper: ClusteringHelper instance for candidate sampling
        random_state: Random seed for reproducibility
        initial_seed: Size of initial random seed set
        batch_size: Number of samples to add in each iteration

    Returns:
        Array of selected sample indices
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Initial random selection
    selected = list(
        np.random.choice(np.arange(X.shape[0]), initial_seed, replace=False)
    )
    remaining = set(range(X.shape[0])) - set(selected)

    # Iterative diversity-based selection
    while len(selected) < n_select:
        if len(remaining) == 0:
            break

        # Sample candidate pool from clusters
        candidate_pool = np.array(clustering_helper.get_random_samples(10)[0])

        # Compute minimum distances to selected set
        distances = cdist(X[candidate_pool], X[selected], metric="euclidean")
        min_distances = np.min(distances, axis=1)

        # Select points with largest minimum distance (most diverse)
        top_indices = np.argsort(min_distances)[-batch_size:]
        selected_candidates = candidate_pool[top_indices]

        selected.extend(selected_candidates.tolist())
        remaining = remaining - set(selected_candidates.tolist())

        print(f"Diversity sampling: selected {len(selected)} samples")

    # Trim to exact size if needed
    if len(selected) > n_select:
        selected = selected[:n_select]
    return np.array(selected)


def sample_margin(
    X: np.ndarray,
    Y: np.ndarray,
    n_select: int,
    clustering_helper,
    random_state: Optional[int] = None,
    initial_seed: int = 100,
    batch_size: int = 100,
) -> np.ndarray:
    """Margin-based active learning using logistic regression uncertainty.

    Selects samples where the model is least confident (closest to decision
    boundary) by measuring the margin between top two class probabilities.
    Converts regression targets to classification by binning. Faster than
    neural network-based uncertainty sampling while maintaining good performance.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        Y: Target matrix of shape (n_samples, n_targets)
        n_select: Number of samples to select
        clustering_helper: ClusteringHelper instance for candidate sampling
        random_state: Random seed for reproducibility
        initial_seed: Size of initial random seed set
        batch_size: Number of samples to add in each iteration

    Returns:
        Array of selected sample indices
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if random_state is not None:
        np.random.seed(random_state)

    # Initial random selection
    selected = list(
        np.random.choice(np.arange(X.shape[0]), initial_seed, replace=False)
    )
    remaining = set(range(X.shape[0])) - set(selected)

    # Standardize features for logistic regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert regression targets to classification by binning
    Y_binned = np.zeros_like(Y)
    for i in range(Y.shape[1]):
        Y_binned[:, i] = np.digitize(Y[:, i], bins=np.quantile(Y[:, i], [0.33, 0.66]))

    # Iterative margin-based selection
    while len(selected) < n_select:
        if len(remaining) == 0:
            break

        # Train separate model for each target dimension
        margins = []
        for i in range(Y.shape[1]):
            model = LogisticRegression(random_state=random_state)
            model.fit(X_scaled[selected], Y_binned[selected, i])

            # Sample candidate pool from clusters
            candidate_pool = np.array(clustering_helper.get_random_samples(10)[0])

            # Get prediction probabilities
            probs = model.predict_proba(X_scaled[candidate_pool])
            # Margin is difference between top two class probabilities
            margin = np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]
            margins.append(margin)

        # Average margins across all target dimensions
        avg_margins = np.mean(margins, axis=0)

        # Select points with smallest margins (highest uncertainty)
        top_indices = np.argsort(avg_margins)[:batch_size]
        selected_candidates = candidate_pool[top_indices]

        selected.extend(selected_candidates.tolist())
        remaining = remaining - set(selected_candidates.tolist())

        print(f"Margin sampling: selected {len(selected)} samples")

    # Trim to exact size if needed
    if len(selected) > n_select:
        selected = selected[:n_select]
    return np.array(selected)
