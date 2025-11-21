"""Utilities for assembling the four-panel statistics meme."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

PanelData = Sequence[np.ndarray]
PANEL_TITLES: Tuple[str, ...] = (
    "Reality",
    "Your Model",
    "Selection Bias",
    "Estimate",
)


def _coerce_uint8(image: np.ndarray) -> np.ndarray:
    """Return an 8-bit representation scaled from [0, 1] floats if needed."""

    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise ValueError("Each image must be 2D (grayscale) or 3D (color)")

    arr = np.squeeze(arr)
    if arr.ndim == 1:
        raise ValueError("Images must have height and width dimensions")

    arr = arr.astype(np.float32)
    if np.max(arr) <= 1.0 + 1e-6:
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    else:
        arr = np.clip(arr, 0.0, 255.0)

    return arr.astype(np.uint8)


def _resize_to_target(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize array to (height, width) using Pillow for high quality."""

    target_h, target_w = size
    mode = "L" if arr.ndim == 2 else "RGB"
    pil_img = Image.fromarray(arr, mode=mode)
    resized = pil_img.resize((target_w, target_h), Image.Resampling.BICUBIC)
    return np.array(resized)


def _prepare_images(
    images: PanelData,
    border_color: str = "#111111",
    border_width: int = 6,
) -> Sequence[np.ndarray]:
    """Normalize, resize, and add borders to each panel."""

    uint8_images = [_coerce_uint8(img) for img in images]
    heights = [img.shape[0] for img in uint8_images]
    widths = [img.shape[1] for img in uint8_images]
    target_size = (max(heights), max(widths))
    resized = [_resize_to_target(img, target_size) for img in uint8_images]

    bordered: list[np.ndarray] = []
    for img in resized:
        mode = "L" if img.ndim == 2 else "RGB"
        pil_img = Image.fromarray(img, mode=mode)
        bordered_img = ImageOps.expand(
            pil_img, border=border_width, fill=border_color
        )
        bordered.append(np.array(bordered_img))

    return bordered


def create_statistics_meme(
    original_img: np.ndarray,
    stipple_img: np.ndarray,
    block_letter_img: np.ndarray,
    masked_stipple_img: np.ndarray,
    output_path: str,
    dpi: int = 150,
    background_color: str = "white",
    border_color: str = "#111111",
    border_width: int = 6,
) -> None:
    """Compose the four-panel statistics meme and save it as a PNG file."""

    panels = (
        original_img,
        stipple_img,
        block_letter_img,
        masked_stipple_img,
    )

    processed = _prepare_images(panels, border_color=border_color, border_width=border_width)
    fig_width = 4 * processed[0].shape[1] / 100
    fig_height = processed[0].shape[0] / 100

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(max(fig_width, 10), max(fig_height, 3)),
        dpi=dpi,
    )
    fig.patch.set_facecolor(background_color)

    for ax, img, title in zip(axes, processed, PANEL_TITLES):
        cmap = "gray" if img.ndim == 2 else None
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#333333")
            spine.set_linewidth(0.8)

    fig.tight_layout(pad=3.0)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, transparent=False, facecolor=background_color, bbox_inches="tight")
    plt.close(fig)


__all__ = ["create_statistics_meme"]

