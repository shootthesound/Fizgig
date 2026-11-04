"""Image utilities for sample generation and display."""

import numpy as np
import torch
from PIL import Image
from typing import Optional, Tuple


def preprocess_image(
    image: Image.Image, w: int, h: int, handle_alpha: bool = False
) -> Tuple[torch.Tensor, np.ndarray, Optional[np.ndarray]]:
    """Preprocess an image for model input.

    Args:
        image: Input PIL image (RGB or RGBA).
        w: Target width.
        h: Target height.
        handle_alpha: Whether to preserve alpha channel.

    Returns:
        Tuple of (image_tensor NCHW [-1,1], image_np HWC [0,255], alpha or None).
    """
    if image.mode == "RGBA":
        alpha = np.array(image.split()[-1])
    else:
        alpha = None

    if handle_alpha:
        image = image.convert("RGBA")
    else:
        image = image.convert("RGB")

    image_np = np.array(image)
    image_np = resize_image_to_bucket(image_np, (w, h))

    image_tensor = torch.from_numpy(image_np).float() / 127.5 - 1.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> NCHW
    return image_tensor, image_np, alpha


def flatten(image: Image.Image, background=(255, 255, 255)) -> Image.Image:
    """An RGB copy of a picture to look at (a captioner, a face finder, a thumbnail, a video): a see-through one is laid
    on `background` first. convert("RGB") alone shows whatever colour its transparent pixels happen to hold - often
    black, and random in a render that drew transparency."""
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        base = Image.new("RGBA", rgba.size, tuple(background) + (255,))
        return Image.alpha_composite(base, rgba).convert("RGB")
    return image.convert("RGB")


def on_white(a: np.ndarray) -> np.ndarray:
    """flatten() for a uint8 (H, W, 4) array: RGB, its see-through pixels laid on white. Any other array as it is."""
    if not (isinstance(a, np.ndarray) and a.ndim == 3 and a.shape[-1] == 4):
        return a
    al = a[..., 3:].astype(np.float32) / 255.0
    return (a[..., :3].astype(np.float32) * al + 255.0 * (1.0 - al)).round().astype(np.uint8)


def resize_image_to_bucket(image_np: np.ndarray, bucket_size: Tuple[int, int]) -> np.ndarray:
    """Resize an image (numpy HWC) to the target bucket size using PIL."""
    w, h = bucket_size
    img = Image.fromarray(image_np)
    if img.size != (w, h):
        img = img.resize((w, h), Image.LANCZOS)
    return np.array(img)


def save_images_grid(images: list, path: str, cols: int = 4):
    """Save a list of PIL images as a grid to a file."""
    if not images:
        return

    n = len(images)
    rows = (n + cols - 1) // cols
    w, h = images[0].size

    grid = Image.new("RGB", (w * cols, h * rows), (0, 0, 0))
    for i, img in enumerate(images):
        row, col = divmod(i, cols)
        grid.paste(img, (col * w, row * h))

    grid.save(path)
