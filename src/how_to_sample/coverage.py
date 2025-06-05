from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_scaled_features(
    features: List[np.ndarray], dataset_names: List[str]
) -> Dict[str, np.ndarray]:
    """
    Scale features using StandardScaler fitted on all datasets combined.

    This function fits a single StandardScaler on the concatenated features from all
    datasets to ensure consistent scaling across different datasets.

    Args:
        features: List of feature arrays, one per dataset
        dataset_names: List of dataset names corresponding to the features

    Returns:
        Dictionary mapping dataset names to their scaled features

    Raises:
        ValueError: If features and dataset_names have different lengths
    """
    if len(features) != len(dataset_names):
        raise ValueError("features and dataset_names must have the same length")

    scaler = StandardScaler()
    # Fit scaler on all features combined for consistent scaling
    scaler.fit(np.vstack(features))

    return {
        dataset_name: scaler.transform(feature_array)
        for dataset_name, feature_array in zip(dataset_names, features)
    }


def fit_evt_threshold(covering_errors: np.ndarray, q: float = 0.95) -> float:
    """
    Fit Generalized Extreme Value (GEV) distribution to errors and return threshold.

    This function uses Extreme Value Theory (EVT) to model the tail behavior of
    reconstruction errors, providing a principled way to set coverage thresholds.

    Args:
        covering_errors: Array of reconstruction errors from the covering dataset
        q: Quantile for threshold calculation (default 0.95 for 95th percentile)

    Returns:
        Threshold value at the specified quantile of the fitted GEV distribution

    Raises:
        ValueError: If q is not between 0 and 1
    """
    if not 0 < q < 1:
        raise ValueError("Quantile q must be between 0 and 1")

    # Fit GEV distribution to the covering errors
    gev_params = stats.genextreme.fit(covering_errors)

    # Calculate threshold as quantile of fitted distribution
    threshold = stats.genextreme.ppf(q, *gev_params)

    return threshold


def test_pca_coverage(
    covering_features_scaled: np.ndarray,
    covered_features_scaled: np.ndarray,
    n_components: Union[int, float] = 0.95,
    error_threshold: Optional[float] = None,
) -> Tuple[float, PCA]:
    """
    Test coverage of one dataset by another using PCA reconstruction error.

    This function trains a PCA model on the covering dataset and evaluates how well
    it can reconstruct features from the covered dataset. Coverage is measured as
    the percentage of covered samples with reconstruction error below a threshold.

    Args:
        covering_features_scaled: Scaled features of the dataset that should provide coverage
        covered_features_scaled: Scaled features of the dataset being tested for coverage
        n_components: Number of PCA components to use. If float (0-1), represents
                     variance ratio to retain. If int, represents exact number of components
        error_threshold: Optional manual threshold for coverage decision. If None,
                        threshold is automatically calculated using EVT on covering dataset

    Returns:
        Tuple containing:
            - coverage_percentage: Percentage of covered samples within the threshold
            - pca: Fitted PCA model for further analysis

    Raises:
        ValueError: If n_components is invalid or arrays have incompatible shapes
    """
    if isinstance(n_components, float) and not (0 < n_components <= 1):
        raise ValueError("When n_components is float, it must be between 0 and 1")
    if isinstance(n_components, int) and n_components <= 0:
        raise ValueError("When n_components is int, it must be positive")

    # Train PCA on covering dataset to learn the data manifold
    pca = PCA(n_components=n_components)
    pca.fit(covering_features_scaled)

    # Calculate reconstruction error for covering dataset
    # This establishes the baseline error distribution
    covering_pca = pca.transform(covering_features_scaled)
    reconstructed_covering = pca.inverse_transform(covering_pca)
    covering_errors = np.mean(
        np.square(covering_features_scaled - reconstructed_covering), axis=1
    )

    # Calculate reconstruction error for covered dataset
    # Higher errors indicate samples that are poorly covered
    covered_pca = pca.transform(covered_features_scaled)
    reconstructed_covered = pca.inverse_transform(covered_pca)
    covered_errors = np.mean(
        np.square(covered_features_scaled - reconstructed_covered), axis=1
    )

    # Set threshold based on covering dataset if not provided
    # Uses EVT to model tail behavior of reconstruction errors
    if error_threshold is None:
        error_threshold = fit_evt_threshold(covering_errors)

    # Calculate coverage as percentage of samples below threshold
    n_covered = np.sum(covered_errors <= error_threshold)
    coverage_percentage = (n_covered / len(covered_errors)) * 100

    return coverage_percentage, pca
