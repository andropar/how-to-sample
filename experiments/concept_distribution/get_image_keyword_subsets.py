import argparse
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
from how_to_sample.coreset import ClusteringHelper
from how_to_sample.sampling_full import (
    sample_greedy_ed,
    sample_kcenter,
    sample_kmeans,
    sample_random,
    sample_stratified,
)
from sklearn.cluster import MiniBatchKMeans


def get_available_methods() -> List[str]:
    """Get list of available sampling methods.

    Returns:
        List of method names including 'all' option.
    """
    return [
        "random",
        "stratified",
        "kmeans",
        "kcenter",
        "greedy_ed",
        "greedy_ed_kmeans",
        "all",
    ]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Sample subset of images using different sampling methods"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=get_available_methods(),
        help='Sampling method to use. Use "all" to run all methods.',
    )
    parser.add_argument(
        "--n-select",
        type=int,
        default=6000,
        help="Number of samples to select",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=100,
        help="Number of clusters for clustering-based methods",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=92,
        help="Number of parallel jobs for greedy_ed methods",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to run sampling methods on image features."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    np.random.seed(args.random_seed)

    # Load features saved from get_LAION_image_keywords.py
    features_fp = (
        Path(__file__).parent / "outputs" / "get_LAION_image_keywords" / "features.npy"
    )
    if not features_fp.exists():
        logger.error(f"Features file not found: {features_fp}")
        return
    features = np.load(features_fp)
    logger.info(f"Loaded features with shape {features.shape} from {features_fp}")

    # Create output directory for sampling results
    output_dir = Path(__file__).parent / "outputs" / "get_image_keyword_subsets"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to store sampling results
    sampling_results: Dict[str, np.ndarray] = {}

    # Define method implementations
    sampling_implementations: Dict[str, Callable[[], np.ndarray]] = {
        "random": lambda: sample_random(
            features, args.n_select, random_state=args.random_seed
        ),
        "stratified": lambda: sample_stratified(
            features, cluster_labels, args.n_select, random_state=args.random_seed
        ),
        "kmeans": lambda: sample_kmeans(
            features, args.n_select, random_state=args.random_seed
        ),
        "kcenter": lambda: sample_kcenter(
            features, args.n_select, random_state=args.random_seed
        ),
        "greedy_ed": lambda: sample_greedy_ed(
            features,
            args.n_select,
            clustering_helper,
            init_kmeans=False,
            n_jobs=args.n_jobs,
        ),
        "greedy_ed_kmeans": lambda: sample_greedy_ed(
            features,
            args.n_select,
            clustering_helper,
            init_kmeans=True,
            n_jobs=args.n_jobs,
        ),
    }

    # Determine which methods to run
    methods_to_run = (
        list(sampling_implementations.keys()) if args.method == "all" else [args.method]
    )

    # Run clustering if needed
    clustering_helper: Any = None
    cluster_labels: np.ndarray = None
    if args.method in ["all", "stratified", "greedy_ed", "greedy_ed_kmeans"]:
        logger.info("Running clustering for required methods")
        clustering_model = MiniBatchKMeans(
            n_clusters=args.n_clusters, random_state=args.random_seed
        )
        cluster_labels = clustering_model.fit_predict(features)
        clustering_helper = ClusteringHelper(
            features, clustering_model.cluster_centers_, cluster_labels
        )

    # Run selected methods
    for method in methods_to_run:
        logger.info(f"Starting {method} sampling")
        selected_indices = sampling_implementations[method]()
        sampling_results[method] = selected_indices
        logger.info(
            f"{method.capitalize()} sampling selected {len(selected_indices)} indices"
        )

        # Save results immediately
        out_fp = output_dir / f"selected_indices_{method}.npy"
        np.save(out_fp, selected_indices)
        logger.info(f"Saved {method} sampling indices to {out_fp}")


if __name__ == "__main__":
    main()
