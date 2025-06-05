import argparse
import glob
from pathlib import Path
from typing import List

import clip
import numpy as np
import torch

from how_to_sample.data import ImageDataset
from how_to_sample.extract import extract_features


def run(
    image_glob: str,
    output_fp: str,
    model_name: str = "ViT-B/32",
) -> None:
    """
    Extract CLIP features from a set of images and save them to disk.

    Args:
        image_glob (str): Glob pattern for image files to process.
        output_fp (str): Path to save the extracted features (.npz).
        model_name (str): Name of the CLIP model to use.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Find all image file paths matching the glob pattern
    image_fps: List[str] = sorted(glob.glob(image_glob))

    # Load CLIP model and transform
    model, transform = clip.load(model_name, device=device)
    model.eval()

    # Create dataset with CLIP transform
    ds = ImageDataset(image_fps, transform=transform)

    # Extract features
    print("Extracting CLIP features...")
    features = extract_features(model, ds, None, "avg", device=device)

    # Save features to disk
    np.savez_compressed(output_fp, features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract CLIP features from a set of images and save to disk."
    )
    parser.add_argument(
        "--image_glob",
        type=str,
        required=True,
        help="Glob pattern for image files to process (e.g., 'things/images/*/*.jpg').",
    )
    script_dir = Path(__file__).parent.resolve()
    default_output_fp = script_dir / "outputs" / "embeddings_clip.npz"
    parser.add_argument(
        "--output_fp",
        type=str,
        default=str(default_output_fp),
        help="Path to save the extracted features (.npz).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B/32",
        help="Name of the CLIP model to use.",
    )
    args = parser.parse_args()
    Path(args.output_fp).parent.mkdir(parents=True, exist_ok=True)
    run(args.image_glob, args.output_fp, args.model_name)
