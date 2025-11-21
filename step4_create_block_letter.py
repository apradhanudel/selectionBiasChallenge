"""Utilities for generating simple block-letter images."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def _load_font(size: int) -> ImageFont.ImageFont:
    """Attempt to load a bold truetype font; fallback to Pillow's default."""

    font_files: Sequence[str] = (
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "Arial Bold.ttf",
        "Arial.ttf",
        "Helvetica Bold.ttf",
        "Helvetica.ttf",
    )

    system_font_dirs: Sequence[Path] = (
        Path("/System/Library/Fonts"),  # macOS
        Path("/Library/Fonts"),
        Path("/usr/share/fonts"),  # Linux
        Path("/usr/local/share/fonts"),
        Path.home() / "Library/Fonts",
        Path("C:/Windows/Fonts"),  # Windows
    )

    checked: Iterable[Path | str] = list(font_files)
    for directory in system_font_dirs:
        if directory.exists():
            checked = [*checked, *(directory / name for name in font_files)]

    for candidate in checked:
        try:
            return ImageFont.truetype(str(candidate), size=size)
        except OSError:
            continue

    # Final fallback; Pillow default is bitmap-only so we approximate the size.
    return ImageFont.load_default()


def create_block_letter_s(
    height: int,
    width: int,
    letter: str = "S",
    font_size_ratio: float = 0.9,
    thinning_passes: int = 2,
) -> np.ndarray:
    """Return a height×width array with centered block letter values in [0, 1].

    The letter is rendered in black (0.0) on a white (1.0) background using
    Pillow, scaled so it fits snugly within the target canvas. This function is
    intended to provide the “selection bias” pattern used elsewhere in the
    project.
    """

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive integers")
    if not (0 < font_size_ratio <= 1):
        raise ValueError("font_size_ratio must be between 0 and 1")
    if not letter:
        raise ValueError("letter must be a non-empty string")

    # Use the first character and make it uppercase for consistency.
    glyph = letter[0].upper()

    # Prepare blank white canvas.
    canvas = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(canvas)

    base_size = max(1, int(min(height, width) * font_size_ratio))
    font = _load_font(base_size)

    # Reduce font size until it fits inside the canvas (with a small margin).
    max_attempts = base_size
    bbox = draw.textbbox((0, 0), glyph, font=font, anchor=None)
    while max_attempts > 0 and (
        bbox is None
        or bbox[2] - bbox[0] > width
        or bbox[3] - bbox[1] > height
    ):
        base_size = max(1, base_size - 1)
        font = _load_font(base_size)
        bbox = draw.textbbox((0, 0), glyph, font=font, anchor=None)
        max_attempts -= 1

    if bbox is None:
        raise RuntimeError("Unable to compute text bounding box")

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    offset_x = (width - text_width) / 2 - bbox[0]
    offset_y = (height - text_height) / 2 - bbox[1]

    # Keep the stroke subtle so the letter appears less thick overall.
    stroke = 0  # keep glyph edges crisp; thinning handles thickness.
    draw.text(
        (offset_x, offset_y),
        glyph,
        fill=0,
        font=font,
        stroke_width=stroke,
        stroke_fill=0,
    )

    # Apply morphological thinning by running a MaxFilter (which favors white) to
    # shave pixels from the black glyph. Each pass removes roughly one pixel.
    thinning_passes = max(0, int(thinning_passes))
    if thinning_passes:
        for _ in range(thinning_passes):
            canvas = canvas.filter(ImageFilter.MaxFilter(size=3))

    # Normalize to 0..1 floats (black letter = 0).
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return array


__all__ = ["create_block_letter_s"]

