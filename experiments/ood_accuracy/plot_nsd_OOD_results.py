# Filename: aggregate_plot_ood_cluster_results.py

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate and plot OOD cluster analysis results."
    )
    # Arguments to identify the input directory
    parser.add_argument(
        "--start-k",
        type=int,
        required=True,
        help="Initial number of clusters used in the analysis script (determines input dir).",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        required=True,
        help="Minimum cluster size used in the analysis script (determines input dir).",
    )
    # Base directory where the analysis script placed its outputs
    parser.add_argument(
        "--base-analysis-dir",
        type=str,
        default="outputs/nsd_OOD_accuracy",
        help="Base directory containing the analysis output folders.",
    )
    return parser.parse_args()


def setup_paths(args):
    """Setup input and output paths."""
    script_dir = Path(__file__).parent
    script_name = Path(__file__).stem

    # Construct the specific input directory based on analysis parameters
    input_dir = (
        Path(args.base_analysis_dir)
        / f"startk_{args.start_k}_min_cluster_size_{args.min_cluster_size}"
    )

    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Create output directory for plots relative to this script
    output_dir = script_dir / "outputs" / script_name
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {"input_dir": input_dir, "output_dir": output_dir}
    return paths


def aggregate_results(input_dir):
    """Load and aggregate results from JSON files."""
    all_results = []
    json_files = sorted(list(input_dir.glob("subject_*_results.json")))

    if not json_files:
        logger.warning(f"No subject result files found in {input_dir}")
        return pd.DataFrame()

    logger.info(f"Found {len(json_files)} subject result files. Aggregating...")

    for fpath in tqdm(json_files, desc="Processing subjects"):
        subject_id = fpath.stem.split("_")[1]  # Extract subject ID like 'subj01'
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not decode JSON from {fpath}. Skipping.")
            continue
        except Exception as e:
            logger.warning(f"Error reading {fpath}: {e}. Skipping.")
            continue

        if not data:
            logger.info(f"No data found for subject {subject_id} in {fpath}. Skipping.")
            continue

        # --- Aggregation Steps ---
        # 1. Average over combinations for each (subject, ood_cluster, k)
        subject_avg_by_k_ood = {}  # {(k, ood_label): {'corr': list, 'acc': list}}

        for ood_key, k_data in data.items():  # ood_key = "ood_X"
            ood_label = int(ood_key.split("_")[1])
            for k_key, combo_results in k_data.items():  # k_key = "k_Y"
                k = int(k_key.split("_")[1])

                if (
                    not combo_results
                ):  # Skip if no combinations were successful for this k
                    continue

                # Extract metrics for all combinations for this k
                k_corrs = [res.get("avg_ood_corr", np.nan) for res in combo_results]
                k_accs = [res.get("cluster_accuracy", np.nan) for res in combo_results]

                # Calculate mean, ignoring NaNs
                mean_k_corr = np.nanmean(k_corrs) if k_corrs else np.nan
                mean_k_acc = np.nanmean(k_accs) if k_accs else np.nan

                if (k, ood_label) not in subject_avg_by_k_ood:
                    subject_avg_by_k_ood[(k, ood_label)] = {"corr": [], "acc": []}

                # Store the average *across combinations* for this specific OOD cluster and k
                # Only append if the value is not NaN
                if not np.isnan(mean_k_corr):
                    subject_avg_by_k_ood[(k, ood_label)]["corr"].append(mean_k_corr)
                if not np.isnan(mean_k_acc):
                    subject_avg_by_k_ood[(k, ood_label)]["acc"].append(mean_k_acc)

        # 2. Average over OOD clusters for each (subject, k)
        for k_val in sorted(list(set(k for k, ood in subject_avg_by_k_ood.keys()))):
            k_ood_corrs = []
            k_ood_accs = []
            # Collect the already-averaged-over-combinations results for this k across all OOD clusters
            for (k, ood_label), metrics in subject_avg_by_k_ood.items():
                if k == k_val:
                    k_ood_corrs.extend(
                        metrics["corr"]
                    )  # extend as these are results from different OODs
                    k_ood_accs.extend(metrics["acc"])

            # Calculate final mean across OOD clusters, ignoring NaNs
            final_mean_corr = np.nanmean(k_ood_corrs) if k_ood_corrs else np.nan
            final_mean_acc = np.nanmean(k_ood_accs) if k_ood_accs else np.nan

            # Append final result for this subject and k
            all_results.append(
                {
                    "subject": subject_id,
                    "k_train_clusters": k_val,
                    "avg_ood_corr": final_mean_corr,
                    "cluster_accuracy": final_mean_acc,
                }
            )

    if not all_results:
        logger.warning("Aggregation resulted in no data.")
        return pd.DataFrame()

    return pd.DataFrame(all_results)


def plot_results(df, output_dir):
    """Generate and save plots using seaborn."""
    if df.empty:
        logger.warning("DataFrame is empty. Skipping plotting.")
        return

    logger.info(f"Plotting results for {df['subject'].nunique()} subjects.")
    sns.set_theme(style="ticks")

    # --- Plot 1: Average OOD Correlation ---
    plt.figure(figsize=(8, 6))
    # Individual subject lines (thin, semi-transparent)
    sns.lineplot(
        data=df,
        x="k_train_clusters",
        y="avg_ood_corr",
        units="subject",
        estimator=None,
        lw=0.5,
        alpha=0.3,
        color="grey",
    )
    # Average line across subjects (thicker, solid)
    sns.lineplot(
        data=df,
        x="k_train_clusters",
        y="avg_ood_corr",
        lw=2,
        alpha=0.9,
        label="Average across subjects",
        color="tab:blue",
        errorbar=("se"),  # Show standard error of the mean
    )
    plt.xlabel("Number of training clusters (k)")
    plt.ylabel("Average OOD correlation")
    plt.title("Encoding performance vs. number of training clusters")
    plt.legend()
    sns.despine()
    plot_path = output_dir / "ood_correlation_vs_k.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved correlation plot to {plot_path}")
    plt.close()

    # --- Plot 2: Cluster Accuracy ---
    plt.figure(figsize=(8, 6))
    # Individual subject lines
    # Average line across subjects
    sns.lineplot(
        data=df,
        x="k_train_clusters",
        y="cluster_accuracy",
        lw=2,
        alpha=0.9,
        label="Average across subjects",
        color="tab:red",
        errorbar=("se"),
    )
    plt.xlabel("Number of training clusters (k)")
    plt.ylabel("OOD cluster accuracy")
    plt.title("Cluster accuracy vs. number of training clusters")
    plt.legend()
    sns.despine()
    plot_path = output_dir / "cluster_accuracy_vs_k.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved cluster accuracy plot to {plot_path}")
    plt.close()


def main():
    """Main function to run aggregation and plotting."""
    args = parse_args()
    try:
        paths = setup_paths(args)
    except FileNotFoundError as e:
        logger.error(f"Setup failed: {e}")
        return

    # Aggregate results
    results_df = aggregate_results(paths["input_dir"])

    # Plot results
    if not results_df.empty:
        plot_results(results_df, paths["output_dir"])
    else:
        logger.warning("No data aggregated. Plots will not be generated.")

    logger.info("Aggregation and plotting finished.")


if __name__ == "__main__":
    main()
