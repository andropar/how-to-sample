import argparse
import glob
import logging
import os
from pathlib import Path
from typing import Dict, List

import clip
import numpy as np
import torch
import tqdm

from how_to_sample.data import LaionTarDataset


def run(tar_glob: str, cache_fp: str, log_fp: str) -> None:
    """
    Extract CLIP visual features from LAION tar files and save them as compressed numpy arrays.

    The resulting numpy arrays will be stored alongside each tar file, with the extension ".clip.visual.npz".

    Args:
        tar_glob (str): Glob pattern for LAION tar files.
        cache_fp (str): Path to cache file tracking processed tar files.
        log_fp (str): Path to log file.
    """
    model, transform = clip.load("ViT-B/32", device="cuda")
    model.eval()

    activations: Dict[str, List[np.ndarray]] = {}

    def hook_fn(module, input, output, key: str) -> None:
        # Collect activations for the specified key
        if key not in activations:
            activations[key] = [output.detach().cpu().numpy()]
        else:
            activations[key].append(output.detach().cpu().numpy())

    # Register a forward hook to capture visual activations
    hook = model.visual.register_forward_hook(
        lambda module, input, output: hook_fn(module, input, output, "visual")
    )

    tar_files = glob.glob(tar_glob)
    logging.basicConfig(
        filename=log_fp,
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )

    # Read cache file to skip already processed tar files
    if os.path.exists(cache_fp):
        with open(cache_fp, "r") as f:
            done = f.read().splitlines()
    else:
        done = []
    tar_files = [tar_file for tar_file in tar_files if tar_file not in done]

    for i, tar_file in enumerate(tar_files):
        logging.info(f"Processing {tar_file} ({len(tar_files) - i} remaining)")

        output_fp = tar_file.replace(".tar", ".clip.visual.npz")

        activations = {}

        try:
            ds = LaionTarDataset(tar_file, transform=transform)
            dl = torch.how_to_sample.data.DataLoader(
                ds, batch_size=512, shuffle=False, num_workers=1
            )

            for batch in tqdm.tqdm(dl):
                with torch.no_grad():
                    _ = model.encode_image(batch.cuda())

            for key, value in activations.items():
                activations[key] = np.concatenate(value)

            np.savez_compressed(output_fp, activations["visual"])

            with open(cache_fp, "a") as f:
                f.write(tar_file + "\n")
        except Exception:
            logging.exception(f"Error processing {tar_file}")

            with open(cache_fp, "a") as f:
                f.write(tar_file + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract CLIP visual features from LAION tar files."
    )
    parser.add_argument(
        "--tar_glob",
        type=str,
        required=True,
        help="Glob pattern for LAION tar files",
    )
    script_dir = Path(__file__).parent.resolve()
    default_cache_fp = script_dir / "outputs" / "clip_embeddings_tracker.txt"
    default_log_fp = script_dir / "outputs" / "clip_embeddings.log"
    parser.add_argument(
        "--cache_fp",
        type=str,
        default=str(default_cache_fp),
        help="Path to cache file tracking processed tar files",
    )
    parser.add_argument(
        "--log_fp",
        type=str,
        default=str(default_log_fp),
        help="Path to log file",
    )
    args = parser.parse_args()
    Path(args.cache_fp).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_fp).parent.mkdir(parents=True, exist_ok=True)
    run(args.tar_glob, args.cache_fp, args.log_fp)
