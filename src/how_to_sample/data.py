import glob
import json
import os
import pickle
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import tqdm
from PIL import Image
from torch.utils.data import Dataset


def save_pickle_file(obj: Any, fp: Union[str, Path]) -> None:
    """Save an object to a pickle file.

    Args:
        obj: Object to save
        fp: File path to save to
    """
    with open(fp, "wb") as f:
        pickle.dump(obj, f)


class ImageAccessor:
    """Provides uniform access to images across different datasets.

    This class abstracts the differences between various image datasets (NSD, THINGS)
    and provides a consistent interface for accessing images by index.

    Attributes:
        h5_file: HDF5 file handle for NSD dataset (if applicable)
        images: Image data array for NSD dataset (if applicable)
        image_fps: List of image file paths for THINGS dataset (if applicable)
    """

    def __init__(
        self,
        dataset_type: str,
        nsd_path: str = "data/nsd/nsd_stimuli.hdf5",
        things_path: str = "data/things/images/*/*.jpg",
    ) -> None:
        """Initialize the ImageAccessor for a specific dataset type.

        Args:
            dataset_type: Type of dataset ('nsd' or 'things')
            nsd_path: Path to NSD HDF5 file
            things_path: Glob pattern for THINGS image files

        Raises:
            ValueError: If dataset_type is not supported
        """
        if dataset_type == "nsd":
            self.h5_file = h5py.File(nsd_path, "r")
            self.images = self.h5_file["imgBrick"]
            self._access_method = self._get_nsd_image

        elif dataset_type == "things":
            self.image_fps = sorted(glob.glob(things_path))
            self._access_method = self._get_things_image

        else:
            raise ValueError("dataset_type must be one of: 'nsd', 'things'")

    def get_image(self, idx: int) -> Image.Image:
        """Get PIL image at specified index.

        Args:
            idx: Index of the image to retrieve

        Returns:
            PIL Image object in RGB format
        """
        return self._access_method(idx)

    def _get_nsd_image(self, idx: int) -> Image.Image:
        """Get image from NSD dataset at specified index.

        Args:
            idx: Index of the image in the NSD dataset

        Returns:
            PIL Image object converted to RGB
        """
        return Image.fromarray(self.images[idx]).convert("RGB")

    def _get_things_image(self, idx: int) -> Image.Image:
        """Get image from THINGS dataset at specified index.

        Args:
            idx: Index of the image in the THINGS dataset

        Returns:
            PIL Image object converted to RGB
        """
        return Image.open(self.image_fps[idx]).convert("RGB")

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        if hasattr(self, "images"):
            return len(self.images)
        return len(self.image_fps)

    def __del__(self) -> None:
        """Clean up resources when the object is destroyed."""
        if hasattr(self, "h5_file"):
            self.h5_file.close()


NSD_FEATURES_PATHS: Dict[str, str] = {
    "barlowtwins": "data/nsd/embeddings/barlowtwins.npy",
    "alexnet": "data/nsd/embeddings/alexnet.npy",
    "clip": "data/nsd/clip_vitb32_features.npy.npz",
    "alexnet_layer3": "data/nsd/embeddings/alexnet_layer3.npy",
}


def load_nsd_features(model: str) -> np.ndarray:
    """Load NSD features for a specified model.

    Args:
        model: Name of the feature model to load

    Returns:
        Feature array with shape (n_samples, feature_dim)

    Raises:
        KeyError: If model is not found in NSD_FEATURES_PATHS
    """
    nsd_feature_fp = NSD_FEATURES_PATHS[model]
    nsd_features = (
        np.load(nsd_feature_fp)["arr_0"]
        if ".npz" in nsd_feature_fp
        else np.load(nsd_feature_fp)
    ).squeeze()

    return nsd_features


class DataAccessor:
    """Main class for accessing both images and features across different datasets.

    This class provides a unified interface for loading features and images from
    multiple datasets (NSD, THINGS) with consistent API.

    Attributes:
        feature_model: Name of the feature model to use
        paths: Dictionary mapping dataset names to their file paths
    """

    def __init__(
        self,
        feature_model: str,
        nsd_path: str = "./data/nsd/nsd_stimuli.hdf5",
        things_path: str = "./data/things/images/*/*.jpg",
        things_features_path: str = "./data/things/embeddings",
    ) -> None:
        """Initialize the DataAccessor with specified paths and feature model.

        Args:
            feature_model: Name of the feature model to use
            nsd_path: Path to NSD HDF5 file
            things_path: Glob pattern for THINGS image files
            things_features_path: Path to THINGS feature embeddings directory
        """
        self.feature_model = feature_model
        self.paths = {
            "nsd": nsd_path,
            "things": things_path,
            "things_features": things_features_path,
        }

    def load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, ImageAccessor]:
        """Load a specific dataset's features and image accessor.

        Args:
            dataset_name: One of 'nsd' or 'things'

        Returns:
            Tuple containing:
                - Feature array with shape (n_samples, feature_dim)
                - ImageAccessor instance for the dataset

        Raises:
            ValueError: If dataset_name is not supported
        """
        # Load features based on dataset
        if dataset_name == "nsd":
            features = load_nsd_features(self.feature_model)
        elif dataset_name == "things":
            features = np.load(
                f"{self.paths['things_features']}/embeddings_{self.feature_model}.npz"
            )["arr_0"]
        else:
            raise ValueError("dataset_name must be one of: 'nsd', 'things'")

        # Create image accessor
        image_accessor = ImageAccessor(
            dataset_name,
            nsd_path=self.paths["nsd"],
            things_path=self.paths["things"],
        )

        return features, image_accessor


class LAIONSampleHelper:
    """Helper class for handling LAION dataset samples and their features.

    This class manages access to image features and metadata stored across multiple files,
    providing utilities to load and organize samples from the LAION dataset.

    The class implements a global indexing system that maps across multiple feature files:
    - For example, with 2 feature files containing 100 and 200 samples respectively:
      * Indices 0-99 map to samples in first file
      * Indices 100-299 map to samples in second file
      * Index 250 would load sample 150 from the second file

    Attributes:
        used_feature_fps: List of paths to feature files
        included_indices: List of indices to include from each feature file
        mapping: Combined mapping of (feature_fp, index) pairs that enables
                the global index space across multiple files
    """

    def __init__(
        self, used_feature_fps: List[str], included_indices: List[List[int]]
    ) -> None:
        """Initialize the helper with feature files and their corresponding indices.

        Args:
            used_feature_fps: Paths to feature files to use
            included_indices: Indices to include from each feature file
        """
        self.used_feature_fps = used_feature_fps
        self.included_indices = included_indices
        self.mapping = self.get_mapping()

    def get_mapping(self) -> List[Tuple[str, int]]:
        """Create a flat mapping of all (feature_fp, index) pairs.

        Returns:
            List of tuples containing (feature_file_path, index)
        """
        mapping = []
        for feature_fp, indices in zip(self.used_feature_fps, self.included_indices):
            mapping.extend([(feature_fp, i) for i in indices])
        return mapping

    def group_by_feature_fp(
        self, indices: List[int]
    ) -> Dict[str, List[Tuple[int, int]]]:
        """Group sample indices by their source feature file.

        Args:
            indices: List of indices to group

        Returns:
            Mapping of feature file paths to lists of (within_file_index, original_index)
        """
        fps_and_indices = [self.mapping[i] for i in indices]
        grouped_by_fp = {}
        for orig_i, (fp, within_fp_i) in enumerate(fps_and_indices):
            if fp not in grouped_by_fp:
                grouped_by_fp[fp] = []
            grouped_by_fp[fp].append((within_fp_i, orig_i))
        return grouped_by_fp

    def get_sample_features(
        self,
        indices: Optional[List[int]] = None,
        n_jobs: int = 32,
        text_features: bool = False,
    ) -> np.ndarray:
        """Load features for specified samples using parallel processing.

        Args:
            indices: Indices of samples to load. If None, loads all samples
            n_jobs: Number of parallel jobs for processing
            text_features: Whether to load text instead of visual features

        Returns:
            Array of features for requested samples with shape (n_samples, feature_dim)
        """
        indices = indices if indices is not None else list(range(len(self.mapping)))
        groups = self.group_by_feature_fp(indices)

        # Process features in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(groups))) as executor:
            futures = [
                executor.submit(get_sample_features_single, fp, indices, text_features)
                for fp, indices in groups.items()
            ]
            sample_info = []
            for future in tqdm.tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Loading sample features",
            ):
                sample_info.append(future.result())

        # Combine and sort features by original indices
        all_features, all_orig_indices = [], []
        for features, orig_indices in sample_info:
            all_features.append(features)
            all_orig_indices.extend(orig_indices)

        sorted_indices = np.argsort(all_orig_indices)
        all_features = np.concatenate(all_features, axis=0)
        all_features = all_features[sorted_indices]

        return all_features

    def get_sample_info(
        self, indices: Optional[List[int]] = None, n_jobs: int = 32
    ) -> Tuple[List[Optional[Image.Image]], List[Optional[Dict]]]:
        """Load images and metadata for specified samples using parallel processing.

        Args:
            indices: Indices of samples to load. If None, loads all samples
            n_jobs: Number of parallel jobs for processing

        Returns:
            Tuple containing:
                - List of PIL Images (None if loading failed)
                - List of metadata dictionaries (None if loading failed)
        """
        indices = indices if indices is not None else list(range(len(self.mapping)))
        groups = self.group_by_feature_fp(indices)

        # Process samples in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(groups))) as executor:
            futures = [
                executor.submit(get_sample_info_single, fp, indices)
                for fp, indices in groups.items()
            ]
            sample_info = []
            for future in tqdm.tqdm(
                as_completed(futures), total=len(futures), desc="Loading sample info"
            ):
                sample_info.append(future.result())

        # Combine and sort results by original indices
        all_images, all_metadata, all_orig_indices = [], [], []
        for images, metadata, orig_indices in sample_info:
            all_images.extend(images)
            all_metadata.extend(metadata)
            all_orig_indices.extend(orig_indices)

        sorted_indices = np.argsort(all_orig_indices)
        all_images = [all_images[i] for i in sorted_indices]
        all_metadata = [all_metadata[i] for i in sorted_indices]

        return all_images, all_metadata


def get_sample_features_single(
    feature_fp: str, indices: List[Tuple[int, int]], text_features: bool = False
) -> Tuple[np.ndarray, List[int]]:
    """Load features from a single feature file for specified indices.

    Args:
        feature_fp: Path to feature file
        indices: List of (within_file_index, original_index) pairs
        text_features: Whether to load text features instead of visual features

    Returns:
        Tuple containing:
            - Feature array for the requested indices
            - List of original indices corresponding to the features
    """
    within_fp_indices, orig_indices = [i for i, _ in indices], [i for _, i in indices]

    # Switch to text features path if requested
    if text_features:
        feature_fp = feature_fp.replace(".clip.visual", ".clip.text")
    features = np.load(feature_fp)
    features = features[list(features.keys())[0]]
    return features[within_fp_indices], orig_indices


def get_sample_info_single(
    feature_fp: str, indices: List[Tuple[int, int]]
) -> Tuple[List[Optional[Image.Image]], List[Optional[Dict]], List[int]]:
    """Load images and metadata from a single tar file for specified indices.

    Args:
        feature_fp: Path to feature file (will be converted to tar path)
        indices: List of (within_file_index, original_index) pairs

    Returns:
        Tuple containing:
            - List of PIL Images (None if loading failed)
            - List of metadata dictionaries (None if loading failed)
            - List of original indices
    """
    # Convert feature file path to tar file path
    tar_fp = feature_fp.replace(".clip.visual.npz", ".tar").replace(
        ".clip.text.npz", ".tar"
    )
    tar_ds = LaionTarDataset(tar_fp)

    images, metadatas, orig_indices = [], [], []
    for within_fp_i, orig_i in indices:
        # Try to load image and metadata, handling potential errors gracefully
        try:
            image = tar_ds.get_image(within_fp_i)
        except Exception:
            image = None

        try:
            metadata = tar_ds.get_metadata(within_fp_i)
        except Exception:
            metadata = None

        images.append(image)
        metadatas.append(metadata)
        orig_indices.append(orig_i)

    return images, metadatas, orig_indices


def get_data_from_feature_fp(
    feature_fp: str,
    clf: Any,  # sklearn classifier
    text_features: bool = False,
    clf_thresh: float = 0.65,
    clf_op: str = "gt",
    clip_feature_fp: Optional[str] = None,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Extract filtered data from a single feature file based on classification and clustering criteria.

    This function applies multiple filtering steps:
    1. Classification filtering based on natural image probability
    2. Cluster-based filtering to remove duplicated content

    Args:
        feature_fp: Path to the feature file
        clf: Trained classifier for natural image detection
        text_features: Whether to load text features instead of visual features
        clf_thresh: Classification threshold for natural images
        clf_op: Classification operation ('gt' for greater than, 'lt' for less than)
        clip_feature_fp: Optional path to CLIP features for classification

    Returns:
        Tuple containing:
            - Path to the feature file (potentially modified for text features)
            - Filtered feature array
            - Indices of included features from the original file

    Raises:
        AssertionError: If clf_op is not 'gt' or 'lt'
    """
    assert clf_op in ["gt", "lt"], f"clf_op must be 'gt' or 'lt', got {clf_op}"

    # Load features from the archive
    feature_archive_key = list(np.load(feature_fp).keys())[0]
    feature_fp_data = np.load(feature_fp)[feature_archive_key]
    clip_features = (
        feature_fp_data
        if clip_feature_fp is None
        else np.load(clip_feature_fp)["arr_0"]
    )

    # Apply classification filtering
    preds = clf.predict_proba(clip_features)
    included_features_indices = (
        np.where(preds[:, 1] > clf_thresh)[0]
        if clf_op == "gt"
        else np.where(preds[:, 1] < clf_thresh)[0]
    )

    feature_fp_data = feature_fp_data[included_features_indices]
    clip_features = clip_features[included_features_indices]

    # Switch to text features if requested
    if text_features:
        feature_fp = feature_fp.replace(".clip.visual", ".clip.text")

    # Apply final filtering
    feature_fp_data = np.load(feature_fp)[feature_archive_key][
        included_features_indices
    ]

    return (
        feature_fp,
        feature_fp_data,
        included_features_indices,
    )


def load_pickle_file(file_path: Union[str, Path]) -> Any:
    """Load data from a pickle file.

    Args:
        file_path: Path to the pickle file

    Returns:
        The unpickled object
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def get_random_samples_from_LAION(
    n_samples: int,
    seed: int = 42,
    n_jobs: int = 32,
    text_features: bool = False,
    clf_thresh: float = 0.65,
    feature_model: str = "clip",
    clf_op: str = "gt",
    base_laion_dir: str = "data/laion2b",
    clf_fp: str = "data/laion_natural_img_clf.pkl",
) -> Tuple[List[str], np.ndarray, List[List[int]]]:
    """Get random samples from LAION dataset that meet natural image criteria.

    This function processes LAION dataset files in parallel to extract samples that:
    1. Pass natural image classification with specified threshold
    2. Are not in duplicated clusters
    3. Meet the requested sample count

    Args:
        n_samples: Number of samples to return
        seed: Random seed for reproducibility
        n_jobs: Number of parallel jobs for processing
        text_features: Whether to return text features instead of image features
        clf_thresh: Classification threshold for natural images
        feature_model: Feature model to use ('clip', 'alexnet', 'barlowtwins', 'alexnet_layer3')
        clf_op: Classification operation ('gt' or 'lt')
        base_laion_dir: Base directory containing LAION dataset files
        clf_fp: Path to the natural image classifier pickle file

    Returns:
        Tuple containing:
            - List of used feature file paths
            - Feature array of shape (total_samples, feature_dim)
            - List of indices of included features for each file

    Raises:
        AssertionError: If feature_model is not supported
    """
    np.random.seed(seed)

    # Load required models and data
    clf = load_pickle_file(clf_fp)

    assert feature_model in [
        "clip",
    ], f"Feature model {feature_model} not supported."

    # Map feature model names to file path components
    feature_model_fp_name_mapping = {
        "clip": "clip.visual",
    }

    # Find all feature files for the specified model
    feature_fps = glob.glob(
        f"{base_laion_dir}/*/*.{feature_model_fp_name_mapping[feature_model]}.npz"
    )

    # Filter for files that have corresponding text features if needed
    if text_features and feature_model == "clip":
        feature_fps = [
            fp
            for fp in feature_fps
            if os.path.exists(fp.replace(".clip.visual", ".clip.text"))
        ]
    np.random.shuffle(feature_fps)

    # Initialize tracking variables
    used_feature_fps = []
    feature_list = []
    cum_n_features = []
    included_indices = []
    ct = 0

    # Process files in parallel until we have enough samples
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(
                get_data_from_feature_fp,
                feature_fp,
                clf,
                text_features,
                clf_thresh,
                clf_op,
                feature_fp.replace(
                    feature_model_fp_name_mapping[feature_model], "clip.visual"
                ),
            )
            for feature_fp in feature_fps
        ]

        pbar = tqdm.tqdm(as_completed(futures), total=len(futures))
        for future in pbar:
            try:
                (
                    feature_fp,
                    feature_fp_data,
                    included_features_indices,
                ) = future.result()
            except Exception as e:
                print(f"Error in future: {e}")
                continue

            used_feature_fps.append(feature_fp)
            included_indices.append(included_features_indices)
            feature_list.append(feature_fp_data)

            # Track cumulative feature count
            if len(cum_n_features) == 0:
                cum_n_features.append(feature_fp_data.shape[0])
            else:
                cum_n_features.append(cum_n_features[-1] + feature_fp_data.shape[0])

            pbar.set_postfix(
                {
                    "n_fps": len(feature_list),
                    "cum_n_features": cum_n_features[-1],
                }
            )

            # Stop when we have enough samples
            if cum_n_features[-1] > n_samples:
                break

            ct += 1

        # Cancel remaining futures to free resources
        for future in futures:
            future.cancel()

    # Combine all features into a single array
    features = np.concatenate(feature_list, axis=0)

    return used_feature_fps, features, included_indices


class LaionTarDataset(Dataset):
    """PyTorch Dataset for accessing images and captions from LAION tar archives.

    This dataset class provides access to images and their associated metadata
    stored in tar files from the LAION dataset. It handles both image loading
    and caption extraction with optional tokenization.

    Attributes:
        tar_fp: Path to the tar file
        transform: Optional image transformation function
        return_caption: Whether to return captions instead of images
        tokenizer: Optional tokenizer for caption processing
        tar: Opened tar file handle
        tar_names: List of all file names in the tar archive
        img_ids: List of image IDs that have both .jpg and .json files
    """

    def __init__(
        self,
        tar_fp: str,
        transform: Optional[Any] = None,
        return_caption: bool = False,
        tokenizer: Optional[Any] = None,
    ) -> None:
        """Initialize the LaionTarDataset.

        Args:
            tar_fp: Path to the tar file containing images and metadata
            transform: Optional transformation to apply to images
            return_caption: If True, __getitem__ returns captions instead of images
            tokenizer: Optional tokenizer for processing captions
        """
        self.tar_fp = tar_fp
        self.transform = transform
        self.return_caption = return_caption
        self.tokenizer = tokenizer

        self.tar = tarfile.open(self.tar_fp, "r")
        self.tar_names = self.tar.getnames()

        # Find image IDs that have both image and metadata files
        self.img_ids = [
            name.split(".")[0]
            for name in self.tar_names
            if name.endswith(".jpg") and name.split(".")[0] + ".json" in self.tar_names
        ]

    def __len__(self) -> int:
        """Return the number of valid image-metadata pairs in the dataset."""
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Union[Image.Image, str, Any]:
        """Get an item from the dataset.

        Args:
            idx: Index of the item to retrieve

        Returns:
            Either an image (PIL.Image) or caption (str/tokenized) depending on return_caption
        """
        if self.return_caption:
            return self.get_caption(idx)
        else:
            return self.get_image(idx)

    def get_image(self, idx: int) -> Image.Image:
        """Load and return an image at the specified index.

        Args:
            idx: Index of the image to load

        Returns:
            PIL Image in RGB format, or a black fallback image if loading fails
        """
        img_id = self.img_ids[idx]
        try:
            image = Image.open(self.tar.extractfile(img_id + ".jpg"))
            image = image.convert("RGB")
        except Exception as e:
            print(f"Error loading/converting image {img_id}: {str(e)}")
            # Return a small black image as fallback
            image = Image.new("RGB", (224, 224), color="black")

        if self.transform:
            image = self.transform(image)

        return image

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """Load and return metadata for an image at the specified index.

        Args:
            idx: Index of the metadata to load

        Returns:
            Dictionary containing image metadata
        """
        img_id = self.img_ids[idx]
        metadata = json.load(self.tar.extractfile(img_id + ".json"))
        return metadata

    def get_caption(self, idx: int) -> Union[str, Any]:
        """Get the caption for an image at the specified index.

        Args:
            idx: Index of the caption to retrieve

        Returns:
            Caption string, or tokenized caption if tokenizer is provided
        """
        caption = self.get_metadata(idx)["caption"]

        if self.tokenizer:
            caption = self.tokenizer(caption, truncate=True).squeeze()

        return caption


class ImageDataset(Dataset):
    """Simple PyTorch Dataset for loading images from file paths.

    This dataset class provides a straightforward way to load images from
    a list of file paths with optional transformations.

    Attributes:
        image_fps: List of image file paths
        transform: Optional transformation function to apply to images
    """

    def __init__(self, image_fps: List[str], transform: Optional[Any] = None) -> None:
        """Initialize the ImageDataset.

        Args:
            image_fps: List of paths to image files
            transform: Optional transformation to apply to loaded images
        """
        self.image_fps = image_fps
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_fps)

    def __getitem__(self, idx: int) -> Image.Image:
        """Load and return an image at the specified index.

        Args:
            idx: Index of the image to load

        Returns:
            PIL Image in RGB format, optionally transformed
        """
        image = Image.open(self.image_fps[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image
