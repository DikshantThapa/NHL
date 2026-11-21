"""Basic image operations with NumPy and Matplotlib.

Loads a grayscale image, smooths it with a mean filter, thresholds it and
counts the pixel values. Run `python image_basics.py [image.png]` to see the
result, and `python test_image_basics.py` to run the checks.
"""
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def load_gray(path: str) -> np.ndarray:
    """Read an image file and return a 2D float array (grayscale, 0..1)."""
    img = mpimg.imread(path)
    if img.dtype == np.uint8:  # jpg files come as 0..255 integers
        img = img / 255.0
    if img.ndim == 3:  # colour image: average the colour channels
        img = img[:, :, :3].mean(axis=2)
    return img.astype(float)


def blur(img: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean filter with a k x k window, written with loops (no scipy/cv2).

    The border is padded by repeating the edge pixels, so the output has
    the same shape as the input and edge pixels are not darkened.
    """
    if k % 2 == 0:
        raise ValueError("k must be odd so the window has a centre pixel")
    r = k // 2
    padded = np.pad(img, r, mode="edge")
    out = np.zeros_like(img, dtype=float)
    height, width = img.shape
    for y in range(height):
        for x in range(width):
            window = padded[y:y + k, x:x + k]  # k x k area around (y, x)
            out[y, x] = window.mean()
    return out


def threshold(img: np.ndarray, t: float) -> np.ndarray:
    """Return a mask of 0/1 values: 1 where img > t."""
    return (img > t).astype(int)


def histogram(img: np.ndarray, bins: int = 16) -> np.ndarray:
    """Count pixels per bin over [0, 1]. Matches np.histogram counts."""
    counts = np.zeros(bins, dtype=int)
    for value in img.ravel():
        index = int(value * bins)
        if index == bins:  # value 1.0 belongs to the last bin
            index = bins - 1
        counts[index] += 1
    return counts


def main(path: str = "sample.png") -> None:
    """Show original, blurred, thresholded and histogram in a 2x2 figure."""
    img = load_gray(path)
    blurred = blur(img, 5)
    mask = threshold(blurred, 0.5)
    counts = histogram(img, 16)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(blurred, cmap="gray")
    axes[0, 1].set_title("Mean blur 5x5")
    axes[1, 0].imshow(mask, cmap="gray")
    axes[1, 0].set_title("Threshold 0.5 (on blurred)")
    axes[1, 1].bar(range(len(counts)), counts)
    axes[1, 1].set_title("Histogram (16 bins)")
    for ax in axes.ravel()[:3]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sample.png")
