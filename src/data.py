"""
data.py
=======
Turns a single 2D image (a slice extracted from an MRI series) into the
normalised tensor the Pelvis-SigLIP models expect.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def preprocess_image_array(pil_or_array, img_size: int, mean, std, grayscale: bool = False) -> torch.Tensor:
    """Resize + normalise a single image for model input.

    Args:
        pil_or_array: a PIL.Image or a numpy array (grayscale or RGB).
        img_size: target side length in pixels (model-specific, see config.json).
        mean, std: per-channel normalisation (model-specific, see config.json).
        grayscale: whether the model expects a z-scored grayscale image
            replicated to 3 channels (True) or standard RGB normalisation (False).

    Returns:
        A (1, 3, img_size, img_size) float tensor, ready for model.forward().
    """
    if isinstance(pil_or_array, np.ndarray):
        img = Image.fromarray(pil_or_array)
    else:
        img = pil_or_array

    mode = "L" if grayscale else "RGB"
    img = img.convert(mode).resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)

    if grayscale:
        arr = (arr / 255.0 - arr.mean() / 255.0) / (arr.std() / 255.0 + 1e-8)
        arr = arr[None]
        arr = np.repeat(arr, 3, axis=0)
    else:
        arr = arr / 255.0
        arr = (arr - np.array(mean)) / np.array(std)
        arr = arr.transpose(2, 0, 1)

    return torch.from_numpy(arr).float().unsqueeze(0)
