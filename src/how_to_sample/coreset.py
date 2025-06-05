from typing import Dict, List, Optional, Tuple

import numpy as np
from annoy import AnnoyIndex
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from tqdm.auto import tqdm


class ClusteringHelper:
    """Helper class for clustering-based sampling and analysis of feature vectors.

    This class provides functionality for clustering high-dimensional features,
    finding nearest neighbors, and sampling from clusters using various strategies.
    Supports both exact distance computation and approximate nearest neighbor
    search using Annoy for large datasets.
    """

    def __init__(
        self,
        features: np.ndarray,
        centroids: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        n_trees: int = 10,
    ) -> None:
        """Initialize the clustering helper.

        Args:
            features: Feature vectors of shape (n_samples, n_features)
            centroids: Pre-computed cluster centroids of shape (n_clusters, n_features)
            labels: Pre-computed cluster labels of shape (n_samples,)
            n_trees: Number of trees for Annoy index (used for large datasets)
        """
        self.features = features
        self.labels = labels
        self.n_trees = n_trees

        if centroids is not None:
            self.centroids = centroids
            self.labels = self.get_labels() if labels is None else labels
            self.cluster_index_mapping = self.get_cluster_index_mapping()

        # Cache for storing computed distances to avoid recomputation
        self.distances: Dict[int, np.ndarray] = {}

    def cluster(self, n_clusters: int) -> np.ndarray:
        """Perform K-means clustering on the features.

        Args:
            n_clusters: Number of clusters to create

        Returns:
            Cluster centroids of shape (n_clusters, n_features)
        """
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, batch_size=(256 * 64 + 1), random_state=42
        )
        kmeans.fit(self.features)
        self.centroids = kmeans.cluster_centers_
        self.labels = kmeans.labels_

        self.cluster_index_mapping = self.get_cluster_index_mapping()

        return self.centroids

    def get_labels(self) -> np.ndarray:
        """Assign cluster labels to features based on nearest centroids.

        Uses Annoy index for approximate nearest neighbor search when dealing
        with large datasets (>10k samples), otherwise uses exact computation.

        Returns:
            Cluster labels for each feature vector
        """
        if self.labels is None:
            # Use approximate nearest neighbor search for large datasets
            if len(self.features) > 1e4:
                index = AnnoyIndex(self.centroids.shape[1], "euclidean")
                for i, feature in enumerate(self.centroids):
                    index.add_item(i, feature)

                index.build(self.n_trees)

                labels = []
                for sample in tqdm(self.features):
                    labels.append(index.get_nns_by_vector(sample, 1)[0])
            else:
                # Use exact computation for smaller datasets
                labels = pairwise_distances(self.features, self.centroids).argmin(
                    axis=1
                )
        else:
            labels = self.labels

        return labels

    def get_cluster_index_mapping(self) -> Dict[int, np.ndarray]:
        """Create mapping from cluster IDs to sample indices.

        Returns:
            Dictionary mapping cluster_id -> array of sample indices in that cluster
        """
        mapping = {}
        for cluster_idx in np.unique(self.labels):
            cluster_indices = np.where(self.labels == cluster_idx)[0]
            mapping[cluster_idx] = cluster_indices

        return mapping

    def get_random_samples(
        self, n_samples: int, cluster_idx: Optional[int] = None
    ) -> Tuple[List[int], List[int]]:
        """Sample random points from clusters.

        Args:
            n_samples: Number of samples to draw from each cluster
            cluster_idx: Specific cluster to sample from. If None, samples from all clusters

        Returns:
            Tuple of (sampled_indices, corresponding_cluster_labels)
        """
        clusters_to_sample_from = (
            np.arange(len(self.centroids)) if cluster_idx is None else [cluster_idx]
        )

        sampled_indices = []
        labels = []
        for cluster_idx in clusters_to_sample_from:
            cluster_indices = self.cluster_index_mapping[cluster_idx]
            n_samples_for_cluster = min(n_samples, len(cluster_indices))
            sampled_indices.extend(
                np.random.choice(cluster_indices, n_samples_for_cluster, replace=False)
            )
            labels.extend([cluster_idx] * n_samples_for_cluster)

        return sampled_indices, labels

    def get_closest_samples(
        self, cluster_idx: Optional[int] = None, n_closest_samples: int = 1
    ) -> Tuple[List[int], List[int]]:
        """Sample points closest to cluster centroids.

        Args:
            cluster_idx: Specific cluster to sample from. If None, samples from all clusters
            n_closest_samples: Number of closest samples to return per cluster

        Returns:
            Tuple of (sampled_indices, corresponding_cluster_labels)
        """
        clusters_to_sample_from = (
            np.arange(len(self.centroids)) if cluster_idx is None else [cluster_idx]
        )

        sampled_indices = []
        labels = []
        for cluster_idx in clusters_to_sample_from:
            cluster_indices = self.cluster_index_mapping[cluster_idx]
            cluster_features = self.features[cluster_indices]
            cluster_centroid = self.centroids[cluster_idx]

            # Cache distances to avoid recomputation
            if cluster_idx not in self.distances:
                distances = pairwise_distances(
                    cluster_features, cluster_centroid.reshape(1, -1)
                )
                self.distances[cluster_idx] = distances

            closest_indices = cluster_indices[
                self.distances[cluster_idx]
                .argsort(axis=0)
                .flatten()[:n_closest_samples]
            ]
            sampled_indices.extend(closest_indices)
            labels.extend([cluster_idx] * n_closest_samples)

        return sampled_indices, labels


class kCenterGreedy:
    """Greedy k-center algorithm for diverse subset selection.

    This class implements the k-center greedy algorithm which iteratively selects
    points that are farthest from all previously selected points. This promotes
    diversity in the selected subset and is commonly used in active learning
    and coreset construction.

    Adapted from https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py.
    """

    def __init__(self, features: np.ndarray, metric: str = "euclidean") -> None:
        """Initialize the k-center greedy selector.

        Args:
            features: Feature vectors of shape (n_samples, n_features)
            metric: Distance metric to use ('euclidean', 'cosine', etc.)
        """
        self.features = features
        self.name = "kcenter"
        self.metric = metric
        self.min_distances: Optional[np.ndarray] = None
        self.n_obs = self.features.shape[0]
        self.already_selected: List[int] = []

    def update_distances(
        self,
        cluster_centers: List[int],
        only_new: bool = True,
        reset_dist: bool = False,
    ) -> None:
        """Update minimum distances to cluster centers.

        For each data point, maintains the distance to the nearest cluster center.
        This is used to efficiently find the point that is farthest from all
        existing centers.

        Args:
            cluster_centers: Indices of cluster centers
            only_new: If True, only calculate distances for newly selected points
            reset_dist: If True, reset all minimum distances
        """
        if reset_dist:
            self.min_distances = None

        if only_new:
            # Only process centers that haven't been processed before
            cluster_centers = [
                d for d in cluster_centers if d not in self.already_selected
            ]

        if cluster_centers:
            # Compute distances from all points to new cluster centers
            x = self.features[cluster_centers]
            dist = cdist(self.features, x, metric=self.metric)

            if self.min_distances is None:
                # Initialize with distances to first center(s)
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                # Update minimum distances
                self.min_distances = np.minimum(self.min_distances, dist)

    def select_batch_(self, already_selected: List[int], N: int, **kwargs) -> List[int]:
        """Select a diverse batch of points using greedy k-center algorithm.

        This method greedily selects points that maximize the minimum distance
        to existing cluster centers, promoting diversity in the selected batch.

        Args:
            already_selected: Indices of points already selected
            N: Number of points to select in this batch
            **kwargs: Additional arguments (unused)

        Returns:
            List of indices of selected points
        """
        self.update_distances(already_selected, only_new=True, reset_dist=False)

        new_batch = []

        for _ in tqdm(range(N)):
            if self.already_selected is None:
                # Initialize with a random point if no points selected yet
                ind = np.random.choice(np.arange(self.n_obs))
            else:
                # Select the point that is farthest from all existing centers
                ind = np.argmax(self.min_distances)

            # Ensure we don't select already chosen points
            assert ind not in already_selected

            # Update distances with the newly selected point
            self.update_distances([ind], only_new=True, reset_dist=False)
            new_batch.append(ind)

        self.already_selected = already_selected

        return new_batch
