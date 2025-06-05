import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
from h5py import File
from scipy.io import loadmat
from scipy.stats import zscore
from tqdm.auto import tqdm

# Path to the Natural Scenes Dataset (NSD) data directory. This should be set to the root directory
# containing all NSD data files (betas, stimuli, etc.)
NSD_DIR = Path("REPLACE WITH PATH TO NSD DATA")


def load_betas_from_fp(
    fp: Union[str, Path],
    standardize: bool = True,
    voxel_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Load beta coefficients from a NIfTI file with optional preprocessing.

    Beta values in NSD are stored as int16 multiplied by 300 to save space.
    This function loads them, converts back to float32, and optionally
    standardizes and subsets the data.

    Args:
        fp: Path to the NIfTI file containing beta coefficients.
        standardize: Whether to z-score standardize betas across the last axis.
            Default is True.
        voxel_indices: Optional array of voxel indices to subset the data.
            If None, all voxels are returned.

    Returns:
        Beta coefficients as float32 array, optionally standardized and subsetted.

    Note:
        Betas are divided by 300 to restore original scale as per NSD documentation:
        https://cvnlab.slite.page/p/6CusMRYfk0/Functional-data-NSD
    """
    # Restore original scale from int16 storage format
    betas = nib.load(fp).get_fdata().astype(np.float32) / 300

    if voxel_indices is not None:
        betas = betas[voxel_indices]

    if standardize:
        betas = zscore(betas, axis=-1)

    return betas


def get_available_subjects(nsd_dir: Path = NSD_DIR) -> List[int]:
    """
    Get list of available subject IDs in the NSD dataset.

    Args:
        nsd_dir: Path to the NSD data directory. Defaults to global NSD_DIR.

    Returns:
        Sorted list of subject IDs (integers) found in the dataset.

    Example:
        >>> subjects = get_available_subjects()
        >>> print(subjects)  # [1, 2, 3, 4, 5, 6, 7, 8]
    """
    subject_dirs = sorted((nsd_dir / "nsddata_betas" / "ppdata").glob("subj*"))
    subject_ids = [int(Path(d).name[4:]) for d in subject_dirs]

    return subject_ids


def get_spaces(subject_id: int, nsd_dir: Path = NSD_DIR) -> List[str]:
    """
    Get available data spaces (resolutions/templates) for a subject.

    Args:
        subject_id: Subject identifier (1-8 for NSD).
        nsd_dir: Path to the NSD data directory. Defaults to global NSD_DIR.

    Returns:
        List of available space names (e.g., 'func1pt8mm', 'anat0pt8mm').

    Example:
        >>> spaces = get_spaces(1)
        >>> print(spaces)  # ['anat0pt8mm', 'func1pt8mm']
    """
    return [
        Path(fp).stem
        for fp in sorted(
            (
                nsd_dir / "nsddata_betas" / "ppdata" / f"subj{str(subject_id).zfill(2)}"
            ).glob("*")
        )
    ]


def get_available_rois(
    subject_id: int, nsd_dir: Path = NSD_DIR, space: str = "func1pt8mm"
) -> List[str]:
    """
    Get available region of interest (ROI) names for a subject in a given space.

    Args:
        subject_id: Subject identifier (1-8 for NSD).
        nsd_dir: Path to the NSD data directory. Defaults to global NSD_DIR.
        space: Data space/resolution to use. Defaults to 'func1pt8mm'.

    Returns:
        List of available ROI names without file extensions.

    Raises:
        AssertionError: If the specified space is not available for the subject.

    Example:
        >>> rois = get_available_rois(1)
        >>> print(rois[:3])  # ['V1v', 'V1d', 'V2v']
    """
    assert space in get_spaces(subject_id, nsd_dir)

    return [
        Path(fp).name.replace(".nii.gz", "")
        for fp in sorted(
            (
                nsd_dir
                / "nsddata"
                / "ppdata"
                / f"subj{str(subject_id).zfill(2)}"
                / space
                / "roi"
            ).glob("*.nii.gz")
        )
    ]


def get_roi(
    subject_id: int,
    roi_name: str,
    nsd_dir: Path = NSD_DIR,
    space: str = "func1pt8mm",
) -> np.ndarray:
    """
    Load a specific ROI mask for a subject.

    Args:
        subject_id: Subject identifier (1-8 for NSD).
        roi_name: Name of the ROI to load (without file extension).
        nsd_dir: Path to the NSD data directory. Defaults to global NSD_DIR.
        space: Data space/resolution to use. Defaults to 'func1pt8mm'.

    Returns:
        ROI mask as float32 numpy array with same dimensions as functional data.

    Example:
        >>> v1_mask = get_roi(1, 'V1v')
        >>> print(f"V1v has {np.sum(v1_mask > 0)} voxels")
    """
    roi_fp = (
        nsd_dir
        / "nsddata"
        / "ppdata"
        / f"subj{str(subject_id).zfill(2)}"
        / space
        / "roi"
        / f"{roi_name}.nii.gz"
    )
    roi = nib.load(roi_fp).get_fdata().astype(np.float32)

    return roi


def load_nsd_betas(
    subject_id: int,
    zscore_betas: bool = True,
    nsd_dir: Path = NSD_DIR,
    space: str = "func1pt8mm",
    voxel_indices: Optional[np.ndarray] = None,
    max_workers: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and process NSD beta coefficients for a subject across all sessions.

    This function loads beta coefficients from all sessions, handles trial averaging
    for repeated stimuli, and returns the processed data along with trial identifiers.
    Uses parallel processing for efficient loading of multiple session files.

    Args:
        subject_id: Subject identifier (1-8 for NSD).
        zscore_betas: Whether to z-score standardize betas. Defaults to True.
        nsd_dir: Path to the NSD data directory. Defaults to global NSD_DIR.
        space: Data space/resolution to use. Defaults to 'func1pt8mm'.
        voxel_indices: Optional array of voxel indices to subset the data.
            If None, all voxels are loaded.
        max_workers: Maximum number of parallel workers for loading. Defaults to 16.

    Returns:
        Tuple containing:
            - averaged_betas: Array of shape (n_unique_trials, n_voxels) with
              trial-averaged beta coefficients
            - trials_with_betas: Array of trial IDs corresponding to the betas

    Note:
        Repeated trials are automatically averaged. The function handles the complex
        trial ordering and stimulus mapping from the NSD experimental design.

    Example:
        >>> betas, trial_ids = load_nsd_betas(1, voxel_indices=v1_indices)
        >>> print(f"Loaded {betas.shape[0]} unique trials, {betas.shape[1]} voxels")
    """
    # Load experimental design to handle trial ordering
    experiment_design = loadmat(
        nsd_dir / "nsddata" / "experiments" / "nsd" / "nsd_expdesign.mat"
    )
    betas_dir = (
        nsd_dir
        / "nsddata_betas"
        / "ppdata"
        / f"subj{str(subject_id).zfill(2)}"
        / space
        / "betas_fithrf_GLMdenoise_RR"
    )
    sub_betas_fps = sorted(betas_dir.glob("betas_session*.nii.gz"))

    # Load beta files in parallel for efficiency
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                load_betas_from_fp,
                fp,
                zscore_betas,
                voxel_indices,
            ): i
            for i, fp in enumerate(sub_betas_fps)
        }

        betas = [None] * len(sub_betas_fps)
        for future in tqdm(
            as_completed(future_to_idx), total=len(future_to_idx), desc="Loading betas"
        ):
            idx = future_to_idx[future]
            betas[idx] = future.result()

    betas = np.concatenate(betas, axis=-1)

    # Map trials to stimulus IDs using experimental design
    trial_ordering = (
        experiment_design["subjectim"][
            subject_id - 1, experiment_design["masterordering"].squeeze() - 1
        ]
        - 1
    )

    # Group trials by stimulus ID for averaging repeated presentations
    unique_trials_and_indices = {}
    for i, trial in enumerate(trial_ordering):
        if trial not in unique_trials_and_indices:
            unique_trials_and_indices[trial] = [i]
        else:
            unique_trials_and_indices[trial].append(i)

    # Average betas across repeated presentations of the same stimulus
    averaged_betas = []
    trials_with_betas = []
    for trial, indices in unique_trials_and_indices.items():
        stim_indices = [i for i in indices if i < betas.shape[1]]
        if len(stim_indices) > 0:
            averaged_betas.append(np.nanmean(betas[:, stim_indices], axis=1))
            trials_with_betas.append(trial)
    averaged_betas = np.array(averaged_betas)

    return averaged_betas, np.array(trials_with_betas)


def load_nsd_ncs(
    subject_id: int,
    voxel_indices: Optional[np.ndarray] = None,
    space: str = "func1pt8mm",
    nsd_dir: Path = NSD_DIR,
) -> np.ndarray:
    """
    Load Noise Ceiling Scores (NCS) for a subject's voxels.

    NCS represents the maximum possible prediction accuracy for each voxel,
    accounting for measurement noise. Calculated from noise ceiling SNR (ncsnr)
    using the formula: NCS = 100 * (ncsnr² / (ncsnr² + 1/3))

    Args:
        subject_id: Subject identifier (1-8 for NSD).
        voxel_indices: Optional array of voxel indices to subset the data.
            If None, all voxels are returned.
        space: Data space/resolution to use. Defaults to 'func1pt8mm'.
        nsd_dir: Path to the NSD data directory. Defaults to global NSD_DIR.

    Returns:
        Noise ceiling scores as percentage values (0-100) for each voxel.

    Example:
        >>> ncs = load_nsd_ncs(1, voxel_indices=v1_indices)
        >>> print(f"Mean NCS in V1: {np.mean(ncs):.1f}%")
    """
    betas_dir = (
        nsd_dir
        / "nsddata_betas"
        / "ppdata"
        / f"subj{str(subject_id).zfill(2)}"
        / space
        / "betas_fithrf_GLMdenoise_RR"
    )
    ncsnr = nib.load(betas_dir / "ncsnr.nii.gz").get_fdata().astype(np.float32)
    if voxel_indices is not None:
        ncsnr = ncsnr[voxel_indices]

    # Convert noise ceiling SNR to percentage score
    ncs = 100 * ((ncsnr**2) / ((ncsnr**2) + 1 / 3))

    return ncs


def load_nsd_images(trials: List[int], nsd_dir: Path = NSD_DIR) -> np.ndarray:
    """
    Load NSD stimulus images for specified trial IDs.

    Efficiently loads images from the HDF5 stimulus file while preserving
    the original trial ordering in the returned array.

    Args:
        trials: List of trial IDs to load images for.
        nsd_dir: Path to the NSD data directory. Defaults to global NSD_DIR.

    Returns:
        Array of images with shape (n_trials, height, width, channels).
        Images are returned in the same order as the input trial list.

    Example:
        >>> images = load_nsd_images([0, 1, 2])
        >>> print(f"Loaded {images.shape[0]} images of size {images.shape[1:3]}")
    """
    images_hdf5 = File(
        nsd_dir / "nsddata_stimuli" / "stimuli" / "nsd" / "nsd_stimuli.hdf5"
    )["imgBrick"]

    # Sort indices for efficient HDF5 access, then restore original order
    sorted_indices = np.argsort(trials)
    sorted_trials = np.array(trials)[sorted_indices]

    images = images_hdf5[sorted_trials, ...]
    images = images[np.argsort(sorted_indices)]

    return images


def load_nsd_stim_hdf5(nsd_dir: Path = NSD_DIR) -> File:
    """
    Load the NSD stimulus HDF5 file handle for direct access.

    Returns a handle to the HDF5 file containing all NSD stimulus images.
    Useful for memory-efficient access when working with large subsets of stimuli.

    Args:
        nsd_dir: Path to the NSD data directory. Defaults to global NSD_DIR.

    Returns:
        HDF5 file handle to the imgBrick dataset containing stimulus images.

    Warning:
        Remember to close the file handle when done to free resources.

    Example:
        >>> stim_hdf5 = load_nsd_stim_hdf5()
        >>> print(f"Dataset contains {stim_hdf5.shape[0]} images")
        >>> # Use stim_hdf5[indices] to access specific images
    """
    return File(nsd_dir / "nsddata_stimuli" / "stimuli" / "nsd" / "nsd_stimuli.hdf5")[
        "imgBrick"
    ]
