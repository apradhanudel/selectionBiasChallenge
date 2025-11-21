"""Apply a block-letter mask to a stippled image to illustrate bias."""

from __future__ import annotations

import numpy as np


def create_masked_stipple(
    stipple_img: np.ndarray,
    mask_img: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Return a masked stipple image, removing points under dark mask areas.

    Parameters
    ----------
    stipple_img:
        2D numpy array containing the stippled (grayscale) source image with
        values typically in [0, 1].
    mask_img:
        2D numpy array containing the block-letter mask generated previously,
        also with values in [0, 1] where darker values indicate masked regions.
    threshold:
        Mask cutoff. Pixels with mask value strictly below this threshold are
        considered "masked" and will be set to white (1.0) in the output.
    """

    if stipple_img.shape != mask_img.shape:
        raise ValueError("stipple_img and mask_img must have the same shape")
    if not (0 <= threshold <= 1):
        raise ValueError("threshold must be within [0, 1]")

    stipple = np.asarray(stipple_img, dtype=np.float32)
    mask = np.asarray(mask_img, dtype=np.float32)

    masked = np.where(mask < threshold, 1.0, stipple)
    return masked


__all__ = ["create_masked_stipple"]

