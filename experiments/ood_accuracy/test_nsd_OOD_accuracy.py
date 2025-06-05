import argparse
import itertools
import json
import logging
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from joblib import Memory
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from how_to_sample.nsd import (
    get_available_subjects,
    get_roi,
    load_nsd_betas,
    load_nsd_ncs,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run OOD cluster analysis on NSD data."
    )
    parser.add_argument(
        "--start-k",
        type=int,
        default=15,
        help="Initial number of clusters to try for subject-specific clustering.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=500,
        help="Minimum number of images required per cluster after subject-specific clustering.",
    )
    parser.add_argument(
        "--n-train-samples",
        type=int,
        default=500,
        help="Fixed number of samples to use for training.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Max workers for loading NSD betas.",
    )
    return parser.parse_args()


def setup_paths(start_k, min_cluster_size):
    """Setup all required paths and create output directory."""
    script_dir = Path(__file__).parent
    script_name = Path(__file__).stem
    base_output_dir = script_dir / "outputs" / script_name
    output_dir = (
        base_output_dir / f"startk_{start_k}_min_cluster_size_{min_cluster_size}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = base_output_dir / "cache_subject_clustering"
    cache_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "nsd_stim_dir": Path(
            "/LOCAL/LABSHARE/natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/"
        ),
        "clip_features": Path("/home/jroth/nsd_CLIP_ViT-B32_features.npy"),
        "stim_info": Path("/home/jroth/nsd_stim_info_merged.csv"),
        "output_dir": output_dir,
        "clustering_cache_dir": cache_dir,
    }
    return paths


def cluster_subject_features(
    features, min_cluster_size, start_k=20, batch_size=1000, random_state=42
):
    """
    Cluster subject features iteratively to meet min_cluster_size criterion.

    Decreases k starting from start_k until all clusters have >= min_cluster_size samples.
    """
    logger.info(
        f"Attempting to cluster {len(features)} features into k clusters (min size {min_cluster_size}), starting with k={start_k}."
    )
    if len(features) < start_k * min_cluster_size / 5:
        logger.warning(
            f"Very few features ({len(features)}) compared to potential clusters and min size. Clustering might fail or be trivial."
        )

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    for k in range(start_k, 1, -1):
        logger.debug(f"Trying k={k}")
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=min(batch_size, len(features)),
            random_state=random_state,
            n_init="auto",
            max_iter=300,
            reassignment_ratio=0.01,
        )
        try:
            labels = kmeans.fit_predict(scaled_features)
            counts = Counter(labels)
            smallest_cluster = min(counts.values()) if counts else 0

            if smallest_cluster >= min_cluster_size:
                logger.info(
                    f"Found suitable clustering with k={k}. Smallest cluster size: {smallest_cluster}"
                )
                return kmeans, scaler, labels
            else:
                logger.debug(
                    f"k={k} failed: smallest cluster size {smallest_cluster} < {min_cluster_size}"
                )
        except Exception as e:
            logger.warning(f"MiniBatchKMeans failed for k={k}: {e}. Trying smaller k.")
            continue

    logger.error(
        f"Could not find suitable clustering meeting min_cluster_size={min_cluster_size} (tried k from {start_k} down to 2)."
    )
    return None, None, None


def sample_stratified(indices, labels, n_samples, random_state=None):
    """Sample n_samples stratified by labels."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    proportions = counts / len(labels)
    samples_per_label = np.round(proportions * n_samples).astype(int)

    diff = n_samples - samples_per_label.sum()
    if diff != 0:
        adjust_indices = np.argsort(samples_per_label)[::-1]
        for i in range(abs(diff)):
            samples_per_label[adjust_indices[i % len(adjust_indices)]] += np.sign(diff)

    final_samples_per_label = {}
    total_sampled = 0
    indices_list = []
    rng = np.random.RandomState(random_state)

    sorted_unique_labels = sorted(unique_labels)

    for i, label in enumerate(sorted_unique_labels):
        label_indices = indices[labels == label]
        n_available = len(label_indices)
        n_to_sample = min(samples_per_label[i], n_available)
        final_samples_per_label[label] = n_to_sample
        total_sampled += n_to_sample

    remaining_needed = n_samples - total_sampled
    if remaining_needed > 0:
        logger.debug(
            f"Distributing {remaining_needed} remaining samples proportionally."
        )
        available_slots = {
            lbl: len(indices[labels == lbl]) - final_samples_per_label[lbl]
            for lbl in sorted_unique_labels
        }
        available_labels = [lbl for lbl, slots in available_slots.items() if slots > 0]

        if not available_labels:
            logger.warning(
                "Cannot distribute remaining samples, no available slots left."
            )
        else:
            slot_counts = np.array([available_slots[lbl] for lbl in available_labels])
            proportions = slot_counts / slot_counts.sum()
            extra_samples_per_label = np.round(proportions * remaining_needed).astype(
                int
            )
            diff = remaining_needed - extra_samples_per_label.sum()
            if diff != 0:
                adjust_indices = np.argsort(extra_samples_per_label)[::-1]
                for i in range(abs(diff)):
                    extra_samples_per_label[
                        adjust_indices[i % len(adjust_indices)]
                    ] += np.sign(diff)

            for idx, lbl in enumerate(available_labels):
                final_samples_per_label[lbl] += extra_samples_per_label[idx]

    sampled_indices = []
    for label in sorted_unique_labels:
        n_to_sample = final_samples_per_label[label]
        if n_to_sample > 0:
            label_indices = indices[labels == label]
            sampled_indices.append(
                rng.choice(label_indices, n_to_sample, replace=False)
            )

    if not sampled_indices:
        return np.array([], dtype=int)

    return np.concatenate(sampled_indices)


def get_cluster_accuracy(cluster_centers, predicted_features_scaled, true_labels):
    """Calculate classification accuracy based on closest cluster center."""
    if predicted_features_scaled is None or len(predicted_features_scaled) == 0:
        return float(np.nan)
    if cluster_centers is None or len(cluster_centers) == 0:
        return float(np.nan)

    try:
        distances = cdist(
            predicted_features_scaled, cluster_centers, metric="euclidean"
        )
        predicted_labels = np.argmin(distances, axis=1)
        accuracy = np.mean(predicted_labels == true_labels)
        return float(accuracy)
    except Exception as e:
        logger.error(f"Error calculating cluster accuracy: {e}")
        logger.error(
            f"Shape predicted_features_scaled: {predicted_features_scaled.shape if predicted_features_scaled is not None else 'None'}"
        )
        logger.error(
            f"Shape cluster_centers: {cluster_centers.shape if cluster_centers is not None else 'None'}"
        )
        logger.error(
            f"Shape true_labels: {true_labels.shape if true_labels is not None else 'None'}"
        )
        return float(np.nan)


def correlation_across_voxels(preds, betas):
    """Calculate correlation across voxels, handling potential NaN values."""
    corrs = np.zeros(preds.shape[1])
    valid_voxel_mask = (np.std(betas, axis=0) > 1e-6) & (np.std(preds, axis=0) > 1e-6)
    valid_indices = np.where(valid_voxel_mask)[0]

    if not valid_indices.size:
        return np.array([np.nan])

    for i in valid_indices:
        corrs[i] = np.corrcoef(preds[:, i], betas[:, i])[0, 1]

    corrs[~valid_voxel_mask] = np.nan
    return corrs[valid_voxel_mask]


def run_ood_cluster_analysis(
    subject_id,
    subject_features,
    subject_betas,
    subject_stim_indices,
    subject_cluster_labels,
    cluster_centers,
    scaler,
    ncs,
    n_train_samples,
    random_seed_base,
):
    """Run OOD cluster analysis for a single subject."""
    logger.info(f"Running OOD analysis for subject {subject_id}")

    unique_subj_clusters = sorted(np.unique(subject_cluster_labels))
    n_unique_subj_clusters = len(unique_subj_clusters)
    logger.info(
        f"Subject {subject_id} has {n_unique_subj_clusters} clusters meeting criteria."
    )

    if n_unique_subj_clusters < 2:
        logger.warning(
            f"Subject {subject_id} has less than 2 clusters meeting size criteria. Skipping OOD analysis."
        )
        return {}

    results_by_ood_cluster = {}
    analysis_rng = np.random.RandomState(random_seed_base)

    for ood_cluster_idx, ood_cluster_label in enumerate(
        tqdm(unique_subj_clusters, desc="OOD Clusters", leave=True)
    ):
        results_by_ood_cluster[f"ood_{ood_cluster_label}"] = {}

        test_mask = subject_cluster_labels == ood_cluster_label
        train_pool_mask = ~test_mask

        test_indices = np.where(test_mask)[0]
        train_pool_indices = np.where(train_pool_mask)[0]

        if len(test_indices) == 0:
            logger.warning(
                f"Skipping OOD cluster {ood_cluster_label}: No test images found (this shouldn't happen if clustering worked)."
            )
            continue
        if len(train_pool_indices) == 0:
            logger.warning(
                f"Skipping OOD cluster {ood_cluster_label}: No training pool images found (only one cluster?)."
            )
            continue

        test_features = subject_features[test_indices]
        test_betas = subject_betas[test_indices]
        test_true_labels = subject_cluster_labels[test_indices]

        available_train_clusters = sorted(
            np.unique(subject_cluster_labels[train_pool_mask])
        )
        n_available_train_clusters = len(available_train_clusters)
        train_pool_labels = subject_cluster_labels[train_pool_indices]

        logger.debug(
            f"OOD Cluster: {ood_cluster_label}, N_test: {len(test_indices)}, N_train_pool: {len(train_pool_indices)}, N_avail_train_clusters: {n_available_train_clusters}"
        )

        for k_train_clusters in tqdm(
            range(1, n_available_train_clusters + 1),
            desc="Num Training Clusters (k)",
            leave=False,
        ):
            results_by_ood_cluster[f"ood_{ood_cluster_label}"][
                f"k_{k_train_clusters}"
            ] = []
            cluster_combinations = list(
                itertools.combinations(available_train_clusters, k_train_clusters)
            )

            for combo_idx, selected_train_clusters in enumerate(
                tqdm(
                    cluster_combinations,
                    desc=f"Combinations (k={k_train_clusters})",
                    leave=False,
                )
            ):
                current_training_pool_mask = np.isin(
                    train_pool_labels, list(selected_train_clusters)
                )
                current_training_pool_indices = train_pool_indices[
                    current_training_pool_mask
                ]
                current_training_pool_labels = train_pool_labels[
                    current_training_pool_mask
                ]

                if len(current_training_pool_indices) < n_train_samples:
                    logger.warning(
                        f" Combination {selected_train_clusters} for OOD {ood_cluster_label} has only {len(current_training_pool_indices)} samples, requested {n_train_samples}. Trying to use all available, but might be too few."
                    )
                    current_n_train_samples = len(current_training_pool_indices)
                    if (
                        current_n_train_samples < k_train_clusters
                        or current_n_train_samples < 10
                    ):
                        logger.warning(
                            f"  -> Too few samples ({current_n_train_samples}). Skipping combination."
                        )
                        continue
                else:
                    current_n_train_samples = n_train_samples

                if len(current_training_pool_indices) == 0:
                    logger.warning(
                        f" Combination {selected_train_clusters} for OOD {ood_cluster_label} has 0 samples. Skipping."
                    )
                    continue

                sampling_seed = analysis_rng.randint(0, 2**32 - 1)
                train_indices = sample_stratified(
                    current_training_pool_indices,
                    current_training_pool_labels,
                    current_n_train_samples,
                    random_state=sampling_seed,
                )

                if len(train_indices) == 0:
                    logger.warning(
                        f"Stratified sampling returned 0 indices for combination {selected_train_clusters}. Skipping."
                    )
                    continue

                train_features = subject_features[train_indices]
                train_betas = subject_betas[train_indices]

                try:
                    if np.any(np.std(train_features, axis=0) < 1e-6) or np.any(
                        np.std(train_betas, axis=0) < 1e-6
                    ):
                        logger.warning(
                            f"Skipping combination {selected_train_clusters} for OOD {ood_cluster_label} due to constant features or betas in training data."
                        )
                        continue

                    encoder_model = RidgeCV(
                        alpha_per_target=True, alphas=np.logspace(-3, 5, 9)
                    )
                    encoder_model.fit(train_features, train_betas)
                    test_betas_pred = encoder_model.predict(test_features)

                    valid_ncs_mask = ncs > 0
                    if not np.any(valid_ncs_mask):
                        logger.warning(
                            f"Subject {subject_id} has no voxels with ncs > 0 in the selected ROI. Storing NaN correlation."
                        )
                        avg_corr = float(np.nan)
                    else:
                        voxel_correlations = correlation_across_voxels(
                            test_betas_pred[:, valid_ncs_mask],
                            test_betas[:, valid_ncs_mask],
                        )
                        valid_corrs = voxel_correlations[~np.isnan(voxel_correlations)]
                        avg_corr = (
                            float(np.mean(valid_corrs))
                            if len(valid_corrs) > 0
                            else float(np.nan)
                        )

                    test_features_pred_scaled = None
                    cluster_accuracy = float(np.nan)
                    try:
                        decoder_model = RidgeCV(alphas=np.logspace(-3, 5, 9))
                        decoder_model.fit(train_betas, train_features)
                        test_features_pred = decoder_model.predict(test_betas)
                        test_features_pred_scaled = scaler.transform(test_features_pred)

                        cluster_accuracy = get_cluster_accuracy(
                            cluster_centers, test_features_pred_scaled, test_true_labels
                        )
                    except Exception as e_dec:
                        logger.error(
                            f"Error during DECODER training/evaluation for OOD {ood_cluster_label}, k={k_train_clusters}, combo={selected_train_clusters}: {e_dec}"
                        )
                        continue

                    results_by_ood_cluster[f"ood_{ood_cluster_label}"][
                        f"k_{k_train_clusters}"
                    ].append(
                        {
                            "avg_ood_corr": avg_corr,
                            "cluster_accuracy": cluster_accuracy,
                            "selected_train_clusters": list(selected_train_clusters),
                            "n_train_samples_used": len(train_indices),
                        }
                    )
                except Exception as e_enc:
                    logger.error(
                        f"Error during ENCODER training/evaluation for OOD {ood_cluster_label}, k={k_train_clusters}, combo={selected_train_clusters}: {e_enc}"
                    )

    return results_by_ood_cluster


def main():
    """Main function to run the OOD cluster analysis."""
    args = parse_args()
    paths = setup_paths(args.start_k, args.min_cluster_size)
    memory = Memory(paths["clustering_cache_dir"], verbose=0)

    cached_cluster_subject_features = memory.cache(cluster_subject_features)

    logger.info("Loading CLIP features for all NSD images...")
    all_clip_features = np.load(paths["clip_features"])

    all_subject_results = {}
    for subject_id in get_available_subjects():
        logger.info(
            f"================ Processing subject {subject_id} ================"
        )

        roi = get_roi(subject_id, "streams")
        if roi is None or not np.any(roi == 5):
            logger.warning(
                f"ROI 'streams' (value 5) not found or empty for subject {subject_id}. Skipping."
            )
            continue
        roi_indices = roi == 5
        betas, stim_indices = load_nsd_betas(
            subject_id, voxel_indices=roi_indices, max_workers=args.max_workers
        )
        ncs = load_nsd_ncs(subject_id, voxel_indices=roi_indices) / 100

        if betas is None or ncs is None or len(stim_indices) == 0:
            logger.warning(
                f"Could not load betas/ncs or no stimuli found for subject {subject_id} in ROI. Skipping."
            )
            continue

        logger.info(f"Loaded {betas.shape[0]} betas/stimuli for subject {subject_id}.")

        subject_features = all_clip_features[stim_indices]

        kmeans_model, scaler, subject_cluster_labels = cached_cluster_subject_features(
            subject_features,
            min_cluster_size=args.min_cluster_size,
            start_k=args.start_k,
            random_state=42,
        )

        if kmeans_model is None:
            logger.error(
                f"Failed to cluster features for subject {subject_id}. Skipping analysis for this subject."
            )
            continue

        cluster_centers = kmeans_model.cluster_centers_

        subject_base_seed = hash(subject_id) & (2**32 - 1)
        results = run_ood_cluster_analysis(
            subject_id,
            subject_features,
            betas,
            stim_indices,
            subject_cluster_labels,
            cluster_centers,
            scaler,
            ncs,
            args.n_train_samples,
            random_seed_base=subject_base_seed,
        )

        if results:
            output_file = paths["output_dir"] / f"subject_{subject_id}_results.json"
            try:

                class NpEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        if isinstance(obj, np.floating):
                            return float(obj)
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        return super(NpEncoder, self).default(obj)

                with open(output_file, "w") as f:
                    json.dump(results, f, indent=2, cls=NpEncoder)
                logger.info(f"Saved results for subject {subject_id} to {output_file}")
                all_subject_results[subject_id] = results
            except TypeError as e:
                logger.error(
                    f"Failed to serialize results for subject {subject_id}: {e}"
                )
                logger.info(
                    "Attempting to save problematic data structure for debugging..."
                )
                debug_file = (
                    paths["output_dir"] / f"subject_{subject_id}_results_DEBUG.pkl"
                )
                try:
                    with open(debug_file, "wb") as f:
                        pickle.dump(results, f)
                    logger.info(f"Problematic data saved to {debug_file}")
                except Exception as pe:
                    logger.error(f"Could not pickle debug data: {pe}")
        else:
            logger.info(
                f"No results generated for subject {subject_id}, likely skipped due to insufficient clusters or data."
            )


if __name__ == "__main__":
    main()
