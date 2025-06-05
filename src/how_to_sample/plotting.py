from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np


def display_images_in_grid(
    imgs: List[np.ndarray],
    saveto: Optional[Union[str, Path]] = None,
    figsize_multiplier: float = 2,
) -> None:
    """
    Display a collection of images in an automatically sized grid layout.

    The function arranges images in a square-like grid, calculating the optimal
    number of rows and columns based on the total number of images. Empty
    subplots are hidden when the number of images doesn't fill the grid perfectly.

    Args:
        imgs: List of image arrays to display. Each image should be a numpy array
            compatible with matplotlib's imshow function (2D grayscale or 3D RGB/RGBA).
        saveto: Optional path to save the figure. If None, the figure is only displayed.
            Can be a string or Path object.
        figsize_multiplier: Scaling factor for subplot size. Each subplot will be
            (figsize_multiplier x figsize_multiplier) inches. Default is 2.

    Returns:
        None: The function displays the plot and optionally saves it to disk.

    Example:
        >>> images = [img1, img2, img3, img4]
        >>> display_images_in_grid(images, saveto="grid.png", figsize_multiplier=3)
    """
    n_imgs = len(imgs)

    # Calculate grid dimensions - aim for square-like layout
    n_cols = int(np.ceil(np.sqrt(n_imgs)))
    n_rows = int(np.ceil(n_imgs / n_cols))

    # Create subplot grid with consistent 2D array structure
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * figsize_multiplier, n_rows * figsize_multiplier),
        squeeze=False,  # Always return 2D array even for single row/column
    )

    # Populate grid with images and hide empty subplots
    for i, ax in enumerate(axs.flat):
        if i < n_imgs:
            ax.imshow(imgs[i])
            ax.axis("off")  # Remove axis ticks and labels for cleaner display
        else:
            ax.axis("off")  # Hide empty subplots

    plt.tight_layout()

    if saveto:
        plt.savefig(saveto)
