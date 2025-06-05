import numpy as np


def estimate_ED(features: np.ndarray) -> float:
    """
    Estimate the Effective Dimension (ED) of a feature matrix.

    The effective dimension is computed as the ratio of the squared sum of eigenvalues
    to the sum of squared eigenvalues of the covariance matrix. This provides a measure
    of the intrinsic dimensionality of the data.

    Args:
        features: Feature matrix of shape (n_samples, n_features)

    Returns:
        Effective dimension as a float value

    Note:
        ED = (sum(λ))² / sum(λ²) where λ are the eigenvalues of the covariance matrix
    """
    cov = np.cov(features.T)
    eigenspectrum = np.linalg.eigvalsh(cov)

    return eigenspectrum.sum() ** 2 / (eigenspectrum**2).sum()


def update_cov(
    cov: np.ndarray, mean: np.ndarray, x_new: np.ndarray, n: int
) -> np.ndarray:
    """
    Update covariance matrix incrementally with a new data point.

    This function implements an incremental update of the covariance matrix
    when a new sample is added to the dataset, avoiding recomputation from scratch.

    Args:
        cov: Current covariance matrix of shape (n_features, n_features)
        mean: Current mean vector of shape (n_features,)
        x_new: New data point of shape (n_features,)
        n: Number of existing samples (before adding x_new)

    Returns:
        Updated covariance matrix of shape (n_features, n_features)
    """
    delta = x_new - mean
    new_cov = cov + np.outer(delta, delta) / (n + 1)
    return new_cov * n / (n + 1)


def estimate_ED_by_update(
    old_mean: np.ndarray,
    old_cov: np.ndarray,
    n_old_features: int,
    new_feature: np.ndarray,
) -> float:
    """
    Estimate Effective Dimension after incrementally adding a new feature.

    This function computes the ED by first updating the covariance matrix
    with the new feature, then calculating the effective dimension from
    the updated eigenspectrum.

    Args:
        old_mean: Previous mean vector of shape (n_features,)
        old_cov: Previous covariance matrix of shape (n_features, n_features)
        n_old_features: Number of samples in the previous dataset
        new_feature: New feature vector to add of shape (n_features,)

    Returns:
        Updated effective dimension as a float value
    """
    new_cov_X = update_cov(old_cov, old_mean, new_feature, n_old_features)
    eigenspectrum = np.linalg.eigvalsh(new_cov_X)
    return eigenspectrum.sum() ** 2 / (eigenspectrum**2).sum()
