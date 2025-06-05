import argparse
from pathlib import Path
from typing import Optional

import clip
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from how_to_sample.extract import extract_features


class NSDDataset(Dataset):
    """
    PyTorch Dataset for loading images from an NSD HDF5 file.

    Args:
        hdf5_path (str or Path): Path to the HDF5 file containing NSD images.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, hdf5_path: str, transform: Optional[callable] = None):
        self.h5_file = h5py.File(hdf5_path, "r")
        self.images = self.h5_file["imgBrick"]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        # Convert HDF5 array to PIL Image
        image = Image.fromarray(self.images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def main(
    hdf5_path: str,
    output_fp: str,
    model_name: str = "ViT-B/32",
) -> None:
    """
    Extract CLIP features from NSD images and save them to disk.

    Args:
        hdf5_path (str): Path to the NSD HDF5 file containing images.
        output_fp (str): Path to save the extracted features (.npz).
        model_name (str): Name of the CLIP model to use.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP model and transform
    model, transform = clip.load(model_name, device=device)
    model.eval()

    # Load dataset with CLIP transform
    dataset = NSDDataset(hdf5_path, transform=transform)

    # Extract features
    print("Extracting CLIP features...")
    features = extract_features(model, dataset, None, "avg", device=device)

    # Save features
    np.savez_compressed(output_fp, features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract CLIP features from NSD HDF5 images and save to disk."
    )
    parser.add_argument(
        "--hdf5_path",
        type=str,
        required=True,
        help="Path to the NSD HDF5 file containing images.",
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
    main(args.hdf5_path, args.output_fp, args.model_name)
