import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from h5py import File
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import RidgeCV

from how_to_sample.coreset import ClusteringHelper
from how_to_sample.nsd import (
    get_available_subjects,
    get_roi,
    load_nsd_betas,
    load_nsd_ncs,
)
from how_to_sample.sampling_full import (
    sample_greedy_ed,
    sample_kcenter,
    sample_kmeans,
    sample_margin,
    sample_oracle,
    sample_random,
    sample_stratified,
)

SubjectData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
SamplingResults = Dict[str, Union[str, int, List[Dict[str, Union[int, float]]]]]


def correlation_across_voxels(preds: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """
    Calculate correlation between predictions and actual beta values for each voxel.

    Args:
        preds: Predicted beta values, shape (n_samples, n_voxels)
        betas: Actual beta values, shape (n_samples, n_voxels)

    Returns:
        Array of correlations for each voxel, shape (n_voxels,)
    """
    return np.array(
        [np.corrcoef(preds[:, i], betas[:, i])[0, 1] for i in range(preds.shape[1])]
    )


def load_subject_data(
    subject_id: int, img_features: np.ndarray, roi_id: int = 5
) -> Optional[SubjectData]:
    """
    Load and prepare data for a specific subject.

    Args:
        subject_id: NSD subject identifier (1-8)
        img_features: Precomputed image features for all NSD stimuli
        roi_id: ROI identifier (5 = ventral stream)

    Returns:
        Tuple of (betas, noise_ceilings, subject_features, stim_indices) or None if error
    """
    try:
        # Load ROI mask for ventral stream
        roi = get_roi(subject_id, "streams")
        if roi is None:
            raise ValueError(f"No ROI data found for subject {subject_id}")

        roi_indices = roi == roi_id
        if not np.any(roi_indices):
            raise ValueError(f"No ventral stream voxels found for subject {subject_id}")

        # Load beta coefficients (brain responses)
        betas, stim_indices = load_nsd_betas(
            subject_id, voxel_indices=roi_indices, max_workers=16
        )
        if betas is None or stim_indices is None:
            raise ValueError(f"No beta data found for subject {subject_id}")

        # Load noise ceilings (reliability measure)
        ncs = load_nsd_ncs(subject_id, voxel_indices=roi_indices)
        if ncs is None:
            raise ValueError(f"No noise ceiling data found for subject {subject_id}")
        ncs = ncs / 100.0  # Convert from percentage

        # Extract corresponding image features
        subject_features = img_features[stim_indices]

        return betas, ncs, subject_features, stim_indices

    except Exception as e:
        print(f"Error loading data for subject {subject_id}: {str(e)}")
        return None


def run_sampling_strategy_analysis(
    subject_data: SubjectData,
    strategy: str,
    n_select: int,
    n_repeats: int,
    random_seed: int = 42,
) -> SamplingResults:
    """
    Run sampling strategy analysis for a given subject and strategy.

    Args:
        subject_data: Tuple of (betas, noise_ceilings, features, stim_indices)
        strategy: Sampling strategy name
        n_select: Number of samples to select
        n_repeats: Number of repetitions for stochastic strategies
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing analysis results
    """
    np.random.seed(random_seed)

    results: SamplingResults = {
        "strategy": strategy,
        "n_select": n_select,
        "n_repeats": n_repeats,
        "repeats": [],
    }

    betas, ncs, subject_features, stim_indices = subject_data

    # Create train/test split (80/20)
    n_samples = len(stim_indices)
    n_test = n_samples // 5
    test_indices = np.random.choice(n_samples, n_test, replace=False)
    train_indices = np.setdiff1d(np.arange(n_samples), test_indices)

    X_pool = subject_features[train_indices]
    Y_pool = betas[train_indices]
    X_test = subject_features[test_indices]
    Y_test = betas[test_indices]

    # Handle stochastic strategies with multiple repeats
    if strategy in ["random", "stratified"]:
        for repeat in range(n_repeats):
            seed = random_seed + repeat
            np.random.seed(seed)

            if strategy == "random":
                selected_ids = sample_random(X_pool, n_select, random_state=seed)
            else:  # stratified
                # Create clusters for stratified sampling
                kmeans = MiniBatchKMeans(n_clusters=80, random_state=random_seed)
                labels = kmeans.fit_predict(X_pool)
                selected_ids = sample_stratified(
                    X_pool, labels, n_select, random_state=seed
                )

            # Train model and evaluate
            train_features = X_pool[selected_ids]
            train_betas = Y_pool[selected_ids]
            model = RidgeCV(alpha_per_target=True)
            model.fit(train_features, train_betas)

            test_preds = model.predict(X_test)
            voxel_corrs = correlation_across_voxels(test_preds, Y_test)

            # Calculate metrics only for reliable voxels
            valid_corrs = voxel_corrs[np.where(ncs > 0)]
            avg_corr = np.mean(valid_corrs)
            max_corr = np.max(valid_corrs) if np.max(valid_corrs) != 0 else 1.0
            norm_corr = avg_corr / max_corr

            results["repeats"].append(
                {
                    "repeat": repeat,
                    "avg_corr": avg_corr,
                    "norm_avg_corr": float(norm_corr),
                    "n_selected": len(selected_ids),
                }
            )
    else:
        # Handle deterministic strategies (single run)
        kmeans = MiniBatchKMeans(n_clusters=80, random_state=random_seed)
        labels = kmeans.fit_predict(X_pool)
        clustering_helper = ClusteringHelper(X_pool, kmeans.cluster_centers_, labels)

        # Select samples based on strategy
        if strategy == "kmeans":
            selected_ids = sample_kmeans(X_pool, n_select)
        elif strategy == "kcenter":
            selected_ids = sample_kcenter(X_pool, n_select)
        elif strategy in ["greedy_ed", "greedy_ed_kmeans"]:
            selected_ids = sample_greedy_ed(
                X_pool,
                n_select,
                clustering_helper,
                init_kmeans=strategy == "greedy_ed_kmeans",
            )
        elif strategy == "oracle":
            selected_ids = sample_oracle(
                X_pool,
                Y_pool,
                n_select,
                clustering_helper,
                initial_seed=random_seed,
            )
        elif strategy == "margin":
            selected_ids = sample_margin(
                X_pool,
                Y_pool,
                n_select,
                clustering_helper,
                initial_seed=random_seed,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Train model and evaluate
        train_features = X_pool[selected_ids]
        train_betas = Y_pool[selected_ids]
        model = RidgeCV(alpha_per_target=True)
        model.fit(train_features, train_betas)

        test_preds = model.predict(X_test)
        voxel_corrs = correlation_across_voxels(test_preds, Y_test)

        # Calculate metrics only for reliable voxels
        valid_corrs = voxel_corrs[np.where(ncs > 0)]
        avg_corr = np.mean(valid_corrs)
        max_corr = np.max(valid_corrs) if np.max(valid_corrs) != 0 else 1.0
        norm_corr = avg_corr / max_corr

        results["repeats"].append(
            {
                "avg_corr": avg_corr,
                "norm_avg_corr": float(norm_corr),
                "n_selected": n_select,
            }
        )

    return results


def main() -> None:
    """Main function to run the sampling strategy analysis."""
    parser = argparse.ArgumentParser(
        description="Evaluate different sampling strategies for NSD brain prediction tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    parser.add_argument(
        "--nsd_stim_dir",
        type=Path,
        default=Path("/data/natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/"),
        help="Directory containing NSD stimuli data",
    )
    parser.add_argument(
        "--nsd_stim_info_fp",
        type=Path,
        default=Path("/data/nsd_stim_info_merged.csv"),
        help="Path to NSD stimulus info CSV file",
    )
    parser.add_argument(
        "--nsd_clip_features_fp",
        type=Path,
        default=Path("/data/nsd_CLIP_ViT-B32_features.npy"),
        help="Path to precomputed CLIP features for NSD stimuli",
    )

    # Analysis parameters
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=None,
        help="Subject IDs to process (1-8). If not specified, all available subjects are used",
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
        default="random",
        help="Sampling strategy to evaluate. Use 'full' to run all strategies",
    )
    parser.add_argument(
        "--n_selects",
        type=int,
        nargs="+",
        default=[1000],
        help="Number of training samples to select for each analysis",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=10,
        help="Number of repetitions for stochastic sampling strategies",
    )
    parser.add_argument(
        "--roi_id",
        type=int,
        default=5,
        help="ROI identifier to analyze (5 = ventral stream)",
    )

    # Execution parameters
    parser.add_argument(
        "--reset", action="store_true", help="Overwrite existing results files"
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Create output directory
    out_dir = Path(__file__).parent / "outputs" / Path(__file__).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load global data files
    print("Loading NSD stimuli and features...")
    try:
        nsd_stim_fp = args.nsd_stim_dir / "nsd_stimuli.hdf5"
        nsd_stims = File(nsd_stim_fp, "r")["imgBrick"]
        img_features = np.load(str(args.nsd_clip_features_fp))
        nsd_stim_info = pd.read_csv(args.nsd_stim_info_fp)
        print(f"Loaded {len(img_features)} image features")
    except Exception as e:
        print(f"Error loading data files: {e}")
        return

    # Determine subjects to process
    if args.subjects is None:
        args.subjects = get_available_subjects()
        print(f"Processing all available subjects: {args.subjects}")
    else:
        available_subjects = get_available_subjects()
        invalid_subjects = [s for s in args.subjects if s not in available_subjects]
        if invalid_subjects:
            print(f"Error: Invalid subject IDs: {invalid_subjects}")
            print(f"Available subjects: {available_subjects}")
            return

    # Define sampling strategies
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
    strategies_to_run = all_strategies if args.strategy == "full" else [args.strategy]

    # Process each subject
    for subject_id in args.subjects:
        print(f"\n{'=' * 50}")
        print(f"Processing subject: {subject_id}")
        print(f"{'=' * 50}")

        # Load subject data once per subject
        subject_data = load_subject_data(subject_id, img_features, args.roi_id)
        if subject_data is None:
            print(f"Skipping subject {subject_id} due to data loading error")
            continue

        betas, ncs, subject_features, stim_indices = subject_data
        print(f"Loaded {len(stim_indices)} trials, {betas.shape[1]} voxels")

        # Process each n_select value
        for n_select in args.n_selects:
            print(f"\nAnalyzing with n_select = {n_select}")

            # Check which strategies already exist
            existing_strategies = []
            for strategy in strategies_to_run:
                result_fp = (
                    out_dir / f"sampling_subject_{subject_id}_strategy_{strategy}_"
                    f"nselect_{n_select}_nrepeats_{args.n_repeats}_"
                    f"seed_{args.random_seed}.json"
                )
                if result_fp.exists() and not args.reset:
                    print(f"  Strategy {strategy}: results exist, skipping")
                    existing_strategies.append(strategy)

            # Skip if all strategies already exist
            if len(existing_strategies) == len(strategies_to_run):
                print(f"  All strategies for subject {subject_id} exist, skipping")
                continue

            # Run each strategy
            for strategy in strategies_to_run:
                if strategy in existing_strategies:
                    continue

                print(f"  Running strategy: {strategy}")

                result_fp = (
                    out_dir / f"sampling_subject_{subject_id}_strategy_{strategy}_"
                    f"nselect_{n_select}_nrepeats_{args.n_repeats}_"
                    f"seed_{args.random_seed}.json"
                )

                try:
                    results = run_sampling_strategy_analysis(
                        subject_data,
                        strategy,
                        n_select,
                        args.n_repeats,
                        args.random_seed,
                    )

                    with open(result_fp, "w") as f:
                        json.dump(results, f, indent=2)

                    # Print summary
                    if results["repeats"]:
                        avg_corrs = [r["avg_corr"] for r in results["repeats"]]
                        print(
                            f"    Average correlation: {np.mean(avg_corrs):.4f} ± {np.std(avg_corrs):.4f}"
                        )

                except Exception as e:
                    print(f"    Error running strategy {strategy}: {e}")
                    continue

    print(f"\nAnalysis complete! Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
