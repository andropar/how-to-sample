from typing import Optional, Tuple, Union

import numpy as np


def participation_ratio(
    eigenvals: Union[np.ndarray, list], eps: float = 1e-12
) -> float:
    """
    Calculate the participation ratio of eigenvalues.

    The participation ratio measures the effective dimensionality of a system
    by computing (sum of eigenvalues)^2 / (sum of squared eigenvalues).

    Args:
        eigenvals: Array or list of eigenvalues
        eps: Threshold for filtering out small/negative eigenvalues (numerical noise)

    Returns:
        Participation ratio as a float. Returns 0.0 if no valid eigenvalues remain.
    """
    # Filter out extremely small or negative eigenvalues (numerical chatter)
    w = np.array(eigenvals, dtype=float)
    w = w[w > eps]
    if len(w) == 0:
        return 0.0
    return (w.sum() ** 2) / np.sum(w**2)


def generate_gaussian_mixture_samples(
    N: int,
    C: int,
    D: int,
    sigma2_intra: float,
    sigma2_inter: float,
    teacher_weights: np.ndarray,
    sigma2_noise: float,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate training samples from a Gaussian mixture model in latent space.

    This function implements the generative model described in equation (5):
    - Training distribution: p_tr(y) = (1/C) sum_{c=1..C} N(y | mu_tr_c, sigma2_intra * I)
    - Cluster centers: mu_tr_c ~ N(0, sigma2_inter * I)
    - Observations: x = A^T y + xi, where xi ~ N(0, sigma2_noise * I)

    Args:
        N: Total number of training samples to generate
        C: Number of clusters in the mixture model
        D: Dimensionality of both latent space y and observation space x
        sigma2_intra: Within-cluster variance (controls spread around cluster centers)
        sigma2_inter: Between-cluster variance (controls spread of cluster centers)
        teacher_weights: Linear transformation matrix A of shape (D, D)
        sigma2_noise: Observation noise variance
        random_state: Random seed for reproducibility

    Returns:
        Tuple containing:
        - Xtr: Observed training data of shape (N, D)
        - Ytr: Latent training data of shape (N, D)
        - mu_tr: Cluster centers of shape (C, D)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Draw cluster centers mu_tr_c from N(0, sigma2_inter * I)
    # Each mu_tr_c is in R^D
    mu_tr = np.random.normal(loc=0.0, scale=np.sqrt(sigma2_inter), size=(C, D))

    # Distribute samples evenly across clusters (assumes N divisible by C)
    samples_per_cluster = N // C

    all_y = []
    all_x = []

    for c in range(C):
        # Generate latent features y ~ N(mu_tr_c, sigma2_intra * I)
        y_c = mu_tr[c] + np.random.normal(
            loc=0.0, scale=np.sqrt(sigma2_intra), size=(samples_per_cluster, D)
        )
        # Transform to observation space: x = A^T y + noise
        # Note: teacher_weights represents A, so x = y @ A + noise
        x_c = (y_c @ teacher_weights) + np.random.normal(
            loc=0.0, scale=np.sqrt(sigma2_noise), size=(samples_per_cluster, D)
        )
        all_y.append(y_c)
        all_x.append(x_c)

    # Concatenate samples from all clusters
    Ytr = np.vstack(all_y)  # shape: (N, D)
    Xtr = np.vstack(all_x)  # shape: (N, D)

    return Xtr, Ytr, mu_tr


def ridge_regression_closed_form(
    X: np.ndarray, Y: np.ndarray, lam: float
) -> np.ndarray:
    """
    Train a ridge regression model using the closed-form solution.

    Solves the ridge regression problem: W = argmin_W ||XW - Y||_F^2 + lambda ||W||_F^2
    The closed-form solution is: W = (X^T X + lambda I)^{-1} X^T Y

    Args:
        X: Input features of shape (N, D) where N is number of samples
        Y: Target values of shape (N, D)
        lam: Ridge regularization parameter (lambda)

    Returns:
        Weight matrix W of shape (D, D)
    """
    D = X.shape[1]
    # Compute regularized normal equations: (X^T X + lambda * I)
    A = X.T @ X + lam * np.eye(D)
    B = X.T @ Y
    # Use pseudoinverse for numerical stability
    W = np.linalg.pinv(A) @ B
    return W


def generate_in_distribution_samples(
    n: int,
    mu_tr_c: np.ndarray,
    sigma2_intra: float,
    teacher_weights: np.ndarray,
    sigma2_noise: float,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate in-distribution test samples from a known cluster center.

    Samples are drawn from the same distribution as training data:
    - Latent: y ~ N(mu_tr_c, sigma2_intra * I)
    - Observed: x = A^T y + noise, where noise ~ N(0, sigma2_noise * I)

    Args:
        n: Number of test samples to generate
        mu_tr_c: Cluster center from training set, shape (D,)
        sigma2_intra: Within-cluster variance
        teacher_weights: Linear transformation matrix A of shape (D, D)
        sigma2_noise: Observation noise variance
        random_state: Random seed for reproducibility

    Returns:
        Tuple containing:
        - x_test: Observed test data of shape (n, D)
        - y_test: Latent test data of shape (n, D)
    """
    if random_state is not None:
        np.random.seed(random_state)

    D = mu_tr_c.shape[0]
    # Sample latent variables around the given cluster center
    y_test = mu_tr_c + np.random.normal(
        loc=0.0, scale=np.sqrt(sigma2_intra), size=(n, D)
    )
    # Transform to observation space with noise
    x_test = (y_test @ teacher_weights) + np.random.normal(
        loc=0.0, scale=np.sqrt(sigma2_noise), size=(n, D)
    )
    return x_test, y_test


def generate_out_of_distribution_samples(
    n: int,
    D: int,
    sigma2_intra: float,
    sigma2_inter: float,
    teacher_weights: np.ndarray,
    sigma2_noise: float,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate out-of-distribution (OOD) test samples from a novel cluster center.

    OOD samples follow the same generative process but with a new cluster center:
    1. Sample new cluster center: mu_ood ~ N(0, sigma2_inter * I)
    2. Sample latent variables: y ~ N(mu_ood, sigma2_intra * I)
    3. Transform to observations: x = A^T y + noise

    Args:
        n: Number of test samples to generate
        D: Dimensionality of latent and observation spaces
        sigma2_intra: Within-cluster variance
        sigma2_inter: Between-cluster variance (for sampling new center)
        teacher_weights: Linear transformation matrix A of shape (D, D)
        sigma2_noise: Observation noise variance
        random_state: Random seed for reproducibility

    Returns:
        Tuple containing:
        - x_test: Observed test data of shape (n, D)
        - y_test: Latent test data of shape (n, D)
        - mu_ood: Novel cluster center of shape (D,)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Sample a novel cluster center from the prior distribution
    mu_ood = np.random.normal(loc=0.0, scale=np.sqrt(sigma2_inter), size=(D,))
    # Generate latent samples around this new center
    y_test = mu_ood + np.random.normal(
        loc=0.0, scale=np.sqrt(sigma2_intra), size=(n, D)
    )
    # Transform to observation space with noise
    x_test = (y_test @ teacher_weights) + np.random.normal(
        loc=0.0, scale=np.sqrt(sigma2_noise), size=(n, D)
    )
    return x_test, y_test, mu_ood


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient between two 1D vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Correlation coefficient between -1 and 1
    """
    return np.corrcoef(a, b)[0, 1]


def cluster_identification_accuracy(
    W: np.ndarray,
    teacher_weights: np.ndarray,
    cluster_centers: np.ndarray,
    sigma2_intra: float,
    sigma2_noise: float,
    n: int = 100,
    t: int = 32,
    ood: bool = False,
    sigma2_inter: Optional[float] = None,
) -> float:
    """
    Evaluate zero-shot cluster identification accuracy as described in Figure 9.

    This function tests the model's ability to identify which cluster center
    a test sample belongs to by correlating predicted latent representations
    with candidate cluster centers.

    For in-distribution testing:
    - Test samples are drawn from known training cluster centers
    - Model must identify the correct center among all training centers

    For out-of-distribution testing:
    - Test samples are drawn from novel cluster centers not seen during training
    - Model must identify the novel center among all candidates (training + novel)

    Args:
        W: Learned weight matrix from ridge regression, shape (D, D)
        teacher_weights: True transformation matrix A, shape (D, D)
        cluster_centers: Training cluster centers, shape (C, D)
        sigma2_intra: Within-cluster variance for generating test samples
        sigma2_noise: Observation noise variance
        n: Number of test samples per trial
        t: Number of trials to run for statistical reliability
        ood: If True, test on out-of-distribution samples; if False, test in-distribution
        sigma2_inter: Between-cluster variance (required for OOD testing)

    Returns:
        Median accuracy across all trials (between 0 and 1)

    Raises:
        ValueError: If ood=True but sigma2_inter is not provided
    """
    if ood and sigma2_inter is None:
        raise ValueError("sigma2_inter must be provided for OOD testing")

    # Fix random seed for reproducible results across function calls
    np.random.seed(0)

    accuracies = []
    C = cluster_centers.shape[0]  # Number of training clusters
    D = cluster_centers.shape[1]  # Dimensionality

    for trial in range(t):
        if not ood:
            # In-distribution testing: select a random training cluster center
            chosen_indices = np.random.choice(C, size=1, replace=False)
            mu_true = cluster_centers[chosen_indices[0]]
            # Generate test samples from this known center
            x_test, y_test = generate_in_distribution_samples(
                n, mu_true, sigma2_intra, teacher_weights, sigma2_noise
            )
            # Candidate centers are all training centers
            candidate_centers = cluster_centers
            true_center_idx = chosen_indices[0]

        else:
            # Out-of-distribution testing: generate novel cluster center
            x_test, y_test, mu_ood = generate_out_of_distribution_samples(
                n, D, sigma2_intra, sigma2_inter, teacher_weights, sigma2_noise
            )
            # Candidate centers include training centers plus the novel center
            candidate_centers = np.vstack([cluster_centers, mu_ood.reshape(1, -1)])
            # True center is the novel one (last in the candidate list)
            true_center_idx = candidate_centers.shape[0] - 1

        # Generate predictions using learned model
        y_hat = x_test @ W

        # For each test sample, identify the most similar cluster center
        correct_count = 0
        for i in range(n):
            # Compute correlation between prediction and each candidate center
            correlations = [
                correlation(y_hat[i], center) for center in candidate_centers
            ]
            # Predict the center with highest correlation
            predicted_center_idx = np.argmax(correlations)

            # Check if prediction matches the true center
            if predicted_center_idx == true_center_idx:
                correct_count += 1

        # Record accuracy for this trial
        trial_accuracy = correct_count / n
        accuracies.append(trial_accuracy)

    # Return median accuracy across all trials for robustness
    return np.median(accuracies)
