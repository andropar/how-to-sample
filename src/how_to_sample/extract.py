from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def get_activation(name: str) -> Tuple[Callable, Dict[str, torch.Tensor]]:
    """Create a forward hook to capture activations from a specific layer.

    Args:
        name: Name identifier for the activation to be stored

    Returns:
        Tuple containing:
            - hook: Forward hook function to register with a layer
            - activation: Dictionary that will store the captured activations
    """
    activation: Dict[str, torch.Tensor] = {}

    def hook(model: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        activation[name] = output.detach()

    return hook, activation


def extract_features(
    model: nn.Module,
    dataset: Dataset,
    layer_name: str,
    pooling: Optional[Union[str, nn.Module]] = None,
    batch_size: int = 64,
    device: str = "cuda",
) -> np.ndarray:
    """Extract features from a specific layer of a PyTorch model.

    This function registers a forward hook on the specified layer to capture
    intermediate activations during forward passes. Optional pooling can be
    applied to reduce spatial dimensions.

    Args:
        model: PyTorch model to extract features from
        dataset: PyTorch dataset containing input data
        layer_name: Name of the layer to extract features from (must match
                   the layer name in model.named_modules())
        pooling: Pooling operation to apply to extracted features. Can be:
                - 'avg': Adaptive average pooling to 1x1
                - 'max': Adaptive max pooling to 1x1
                - nn.Module: Custom pooling layer
                - None: No pooling applied
        batch_size: Number of samples to process in each batch
        device: Device to run inference on ('cuda', 'cpu', etc.)

    Returns:
        Numpy array of extracted features with shape (n_samples, feature_dim)
        where feature_dim depends on the layer and pooling configuration

    Raises:
        ValueError: If layer_name is not found in the model or if pooling
                   type is not recognized
    """
    model.eval()
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    features = []

    # Setup pooling layer based on specification
    pool_layer: Optional[nn.Module] = None
    if isinstance(pooling, str):
        if pooling.lower() == "avg":
            pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pooling.lower() == "max":
            pool_layer = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError(f"Unknown pooling type: {pooling}")
    else:
        pool_layer = pooling

    # Find target layer and register forward hook
    hook_handle = None
    for name, layer in model.named_modules():
        if name == layer_name:
            hook, activation = get_activation(layer_name)
            hook_handle = layer.register_forward_hook(hook)
            break
    else:
        raise ValueError(f"Layer {layer_name} not found in model")

    # Extract features batch by batch
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            batch = batch.to(device)
            # Forward pass triggers the hook to capture activations
            _ = model(batch)
            features_batch = activation[layer_name]

            # Apply pooling if specified
            if pool_layer is not None:
                features_batch = pool_layer(features_batch)
                # Remove spatial dimensions for adaptive pooling layers
                if isinstance(pool_layer, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
                    features_batch = features_batch.squeeze(-1).squeeze(-1)

            features.append(features_batch.cpu().numpy())

    # Clean up the registered hook
    if hook_handle is not None:
        hook_handle.remove()

    return np.concatenate(features).squeeze()
