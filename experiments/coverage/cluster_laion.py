import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from how_to_sample.data import get_random_samples_from_LAION

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--n-samples",
        type=int,
        default=int(1e7),
        help="Number of samples to use for clustering",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=5443, help="Number of clusters"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n-jobs", type=int, default=32, help="Number of parallel jobs"
    )
    args = parser.parse_args()

    # Create output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir / "outputs" / "cluster_laion"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        filename=output_dir / "log.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

    logging.info(f"Getting {args.n_samples} random samples from LAION")

    # Get random samples from LAION
    used_feature_fps, features, included_indices = get_random_samples_from_LAION(
        n_samples=args.n_samples,
        seed=args.seed,
        n_jobs=args.n_jobs,
        text_features=False,
        clf_thresh=0.75,
        feature_model="clip",
        clf_op="gt",
    )

    logging.info(f"Loaded features with shape: {features.shape}")
    logging.info(f"Memory usage: {sys.getsizeof(features) / 1024**3:.2f} GB")

    # Fit KMeans model
    logging.info(f"Fitting KMeans with {args.n_clusters} clusters")
    kmeans = MiniBatchKMeans(
        n_clusters=args.n_clusters, batch_size=(256 * 64 + 1), random_state=args.seed
    )
    kmeans.fit(features)

    # Get cluster centers
    centers = kmeans.cluster_centers_
    logging.info(f"Cluster centers shape: {centers.shape}")

    # Save cluster centers
    output_fp = output_dir / "cluster_centers.npy"
    np.save(output_fp, centers)
    logging.info(f"Saved cluster centers to {output_fp}")
