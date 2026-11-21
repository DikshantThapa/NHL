"""Checks for image_basics.py. Run: python test_image_basics.py"""
import numpy as np

from image_basics import blur, histogram, load_gray, threshold


def test_all() -> None:
    img = load_gray("sample.png")
    assert img.ndim == 2 and 0.0 <= img.min() and img.max() <= 1.0

    flat = np.full((5, 5), 0.5)
    assert np.allclose(blur(flat, 3)[1:-1, 1:-1], 0.5)  # interior stays flat
    assert blur(img, 3).shape == img.shape

    mask = threshold(img, 0.5)
    assert set(np.unique(mask)) <= {0, 1}

    expected, _ = np.histogram(img, bins=16, range=(0, 1))
    assert (histogram(img, 16) == expected).all()
    print("all tests passed")


if __name__ == "__main__":
    test_all()
