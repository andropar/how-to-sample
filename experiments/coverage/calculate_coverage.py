import logging
import os
import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from how_to_sample.coverage import (
    get_scaled_features,
    test_pca_coverage,
)
from how_to_sample.data import (
    LAIONSampleHelper,
    get_random_samples_from_LAION,
    load_pickle_file,
)

LAION_CACHE_BASE_DIR = "./laion_embeddings_cache"


def load_laion_data(model, model_out_dir_suffix, n_samples, clf_thresh=0.75):
    """Load or generate LAION features."""
    model_cache_dir = Path(LAION_CACHE_BASE_DIR) / model / model_out_dir_suffix
    os.makedirs(model_cache_dir, exist_ok=True)

    used_feature_fps_fp = model_cache_dir / "used_laion_feature_fps.pkl"
    included_indices_fp = model_cache_dir / "included_indices.pkl"

    if os.path.exists(used_feature_fps_fp) and os.path.exists(included_indices_fp):
        sample_helper = LAIONSampleHelper(
            load_pickle_file(used_feature_fps_fp),
            load_pickle_file(included_indices_fp),
        )
        laion_features = sample_helper.get_sample_features().squeeze()
    else:
        logging.info(
            f"Generating LAION features for model {model}, type {model_out_dir_suffix}, clf_thresh {clf_thresh}"
        )
        used_laion_feature_fps, laion_features, included_indices = (
            get_random_samples_from_LAION(
                n_samples, clf_thresh=clf_thresh, feature_model=model
            )
        )
        laion_features = laion_features.squeeze()

        with open(used_feature_fps_fp, "wb") as f:
            pickle.dump(used_laion_feature_fps, f)
        with open(included_indices_fp, "wb") as f:
            pickle.dump(included_indices, f)
    logging.info(
        f"Loaded/generated LAION features for {model} ({model_out_dir_suffix}): {laion_features.shape}"
    )
    return laion_features


def load_nsd_features(nsd_feature_path):
    """Load NSD features"""
    nsd_features = (
        np.load(nsd_feature_path)["arr_0"]
        if ".npz" in nsd_feature_path
        else np.load(nsd_feature_path)
    ).squeeze()
    logging.info(f"Loaded NSD features from {nsd_feature_path}: {nsd_features.shape}")
    return nsd_features


def load_things_features(things_feature_path):
    things_features = (
        np.load(things_feature_path)["arr_0"]
        if ".npz" in things_feature_path
        else np.load(things_feature_path)
    ).squeeze()
    logging.info(
        f"Loaded THINGS features from {things_feature_path}: {things_features.shape}"
    )
    return things_features


def run_pca_coverage(
    features_to_cover,
    covering_features,
):
    """Run PCA-based coverage analysis."""
    scaled_features = get_scaled_features(
        [
            features_to_cover,
            covering_features,
        ],
        ["covered", "covering"],
    )

    # Use n_components=None for original PCA coverage logic
    coverage, _ = test_pca_coverage(
        scaled_features["covered"], scaled_features["covering"], n_components=None
    )
    logging.info(f"PCA Coverage: {coverage:.2f}%")
    return coverage


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="clip",
        choices=["clip", "alexnet", "barlowtwins", "alexnet_layer3"],
        help="Model for feature extraction.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["datasets_vs_laion_natural", "laion_natural_vs_laion_2b"],
        help="Specific analysis task to perform.",
    )
    parser.add_argument(
        "--nsd-features-path",
        type=str,
        required=True,
        help="Path to NSD features file.",
    )
    parser.add_argument(
        "--things-features-path",
        type=str,
        required=True,
        help="Path to THINGS features file.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=int(1e8),
        help="Number of samples to attempt to load for LAION datasets.",
    )
    parser.add_argument(
        "--covering-subset-size",
        type=int,
        default=6000,
        help="Number of samples for the random LAION-natural subset when it's a covering dataset.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations/seeds for 'laion_natural_vs_laion_2b' task.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Main random seed for reproducibility."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    script_dir = Path(__file__).parent
    base_output_dir = Path("./outputs/calculate_coverage")
    out_dir = base_output_dir / args.model / args.task
    os.makedirs(out_dir, exist_ok=True)

    logging.basicConfig(
        filename=out_dir / "log.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(f"Running with arguments: {args}")

    if args.task == "datasets_vs_laion_natural":
        logging.info("Starting task: datasets_vs_laion_natural")

        # Load LAION-natural (covered)
        laion_natural_features = load_laion_data(
            args.model, "laion_natural", args.n_samples, clf_thresh=0.75
        )

        # NSD vs LAION-natural
        logging.info("Calculating NSD coverage of LAION-natural...")
        nsd_features = load_nsd_features(args.nsd_features_path)
        nsd_coverage = run_pca_coverage(laion_natural_features, nsd_features)

        # THINGS vs LAION-natural
        logging.info("Calculating THINGS coverage of LAION-natural...")
        things_features = load_things_features(args.things_features_path)
        things_coverage = run_pca_coverage(laion_natural_features, things_features)

        # Random LAION-natural subset vs LAION-natural
        logging.info(
            f"Calculating Random LAION-natural subset coverage of LAION-natural over {args.iterations} iterations..."
        )
        laion_subset_coverage_scores = []
        for i in range(args.iterations):
            current_iter_seed = args.seed + i
            np.random.seed(current_iter_seed)
            logging.info(
                f"Iteration {i + 1}/{args.iterations} for LAION-natural subset, using seed {current_iter_seed}"
            )
            subset_indices = np.random.choice(
                np.arange(laion_natural_features.shape[0]),
                size=args.covering_subset_size,
                replace=False,
            )
            laion_natural_subset = laion_natural_features[subset_indices]
            coverage = run_pca_coverage(laion_natural_features, laion_natural_subset)
            laion_subset_coverage_scores.append(coverage)
            logging.info(
                f"Iteration {i + 1} LAION-natural subset coverage: {coverage:.2f}%"
            )

        mean_laion_subset_coverage = np.mean(laion_subset_coverage_scores)
        std_laion_subset_coverage = np.std(laion_subset_coverage_scores)
        logging.info(
            f"Mean LAION-natural subset coverage: {mean_laion_subset_coverage:.2f}%"
        )
        logging.info(
            f"Std LAION-natural subset coverage: {std_laion_subset_coverage:.2f}%"
        )

        results = {
            "nsd_coverage_of_laion_natural": nsd_coverage,
            "things_coverage_of_laion_natural": things_coverage,
            "random_laion_subset_coverage_scores": laion_subset_coverage_scores,
            "mean_random_laion_subset_coverage": mean_laion_subset_coverage,
            "std_random_laion_subset_coverage": std_laion_subset_coverage,
            "params": {
                "model": args.model,
                "covering_subset_size": args.covering_subset_size,
                "iterations": args.iterations,
                "main_seed": args.seed,
            },
        }
        results_fp = (
            out_dir
            / f"coverage_scores_main_seed-{args.seed}_iterations-{args.iterations}.pkl"
        )
        with open(results_fp, "wb") as f:
            pickle.dump(results, f)
        logging.info(f"Results saved to {results_fp}")
        logging.info(f"Scores: {results}")

    elif args.task == "laion_natural_vs_laion_2b":
        logging.info("Starting task: laion_natural_vs_laion_2b")

        # Load LAION-2B (covered)
        laion_2b_features = load_laion_data(
            args.model, "laion_2b", args.n_samples, clf_thresh=0.0
        )

        # Load LAION-natural (source for covering subsets)
        laion_natural_features_source = load_laion_data(
            args.model, "laion_natural", args.n_samples, clf_thresh=0.75
        )

        all_run_coverage_scores = []
        logging.info(
            f"Running {args.iterations} iterations for LAION-natural subset coverage of LAION-2B..."
        )
        for i in range(args.iterations):
            current_iter_seed = args.seed + i
            np.random.seed(current_iter_seed)
            logging.info(
                f"Iteration {i + 1}/{args.iterations}, using seed {current_iter_seed}"
            )

            subset_indices = np.random.choice(
                np.arange(laion_natural_features_source.shape[0]),
                size=args.covering_subset_size,
                replace=False,
            )
            covering_subset = laion_natural_features_source[subset_indices]

            coverage = run_pca_coverage(laion_2b_features, covering_subset)
            all_run_coverage_scores.append(coverage)
            logging.info(f"Iteration {i + 1} coverage: {coverage:.2f}%")

        mean_coverage = np.mean(all_run_coverage_scores)
        std_coverage = np.std(all_run_coverage_scores)

        logging.info(f"All coverage scores: {all_run_coverage_scores}")
        logging.info(
            f"Mean coverage over {args.iterations} iterations: {mean_coverage:.2f}%"
        )
        logging.info(
            f"Std deviation over {args.iterations} iterations: {std_coverage:.2f}%"
        )
        print(
            f"Task: {args.task}, Model: {args.model}\n"
            f"Mean coverage of LAION-2B by LAION-natural ({args.covering_subset_size} samples) over {args.iterations} runs (main_seed: {args.seed}):\n"
            f"Mean: {mean_coverage:.2f}%, Std: {std_coverage:.2f}%"
        )

        results = {
            "scores": all_run_coverage_scores,
            "mean_coverage": mean_coverage,
            "std_coverage": std_coverage,
            "params": {
                "model": args.model,
                "covering_subset_size": args.covering_subset_size,
                "iterations": args.iterations,
                "main_seed": args.seed,
            },
        }
        results_fp = (
            out_dir
            / f"laion_natural_vs_laion_2b_scores_iterations-{args.iterations}_main_seed-{args.seed}.pkl"
        )
        with open(results_fp, "wb") as f:
            pickle.dump(results, f)
        logging.info(f"Results saved to {results_fp}")

    logging.info("Script finished.")
