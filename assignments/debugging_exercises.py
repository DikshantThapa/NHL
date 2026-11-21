"""Debugging exercises 1-3 (CV&DS admission): bug explanation and fix.

Run `python debugging_exercises.py` to execute the self-checks.
"""
import csv
from typing import List

import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------
# Exercise 1: id_to_fruit
# BUG: a set has no order. Iterating it gives an arbitrary order (strings are
#      hash-randomised per Python run), so "index 1" is not stable.
# FIX: take an ordered list; the index then means a fixed position.
# --------------------------------------------------------------------------
def id_to_fruit(fruit_id: int, fruits: List[str]) -> str:
    """Return the fruit at position `fruit_id` of the ordered list `fruits`.

    :param fruit_id: index of the fruit to get
    :param fruits: ordered list of fruit names
    :return: the name at `fruit_id`
    :raises RuntimeError: if the id is out of range
    """
    if not 0 <= fruit_id < len(fruits):
        raise RuntimeError(f"Fruit with id {fruit_id} does not exist")
    return fruits[fruit_id]


# --------------------------------------------------------------------------
# Exercise 2: swap
# BUG 1 (obvious): the right-hand side repeats coords[:, 1] and swaps columns
#      2 and 3 the wrong way round, so it is not a x<->y swap.
# BUG 2 (hidden): column slices are numpy VIEWS. Assigning them one by one
#      overwrites data before it is read, and the input array is mutated.
# FIX: work on a copy and assign all four columns from the original at once.
# --------------------------------------------------------------------------
def swap(coords: np.ndarray) -> np.ndarray:
    """Swap x and y in every [x1, y1, x2, y2, class] row.

    :param coords: array of shape [n, 5]
    :return: new array [y1, x1, y2, x2, class]; the input is not modified
    """
    swapped = coords.copy()
    swapped[:, [0, 1, 2, 3]] = coords[:, [1, 0, 3, 2]]
    return swapped


# --------------------------------------------------------------------------
# Exercise 3: plot_data
# BUG 1: csv.reader returns strings, so np.stack builds a string array and
#        matplotlib plots categorical labels, not numbers.
# BUG 2: the call plot(results[:, 1], results[:, 0]) puts recall on x, but the
#        axis labels (and the task) say precision is x and recall is y.
# FIX: convert to float and plot precision (col 0) on x, recall (col 1) on y.
# --------------------------------------------------------------------------
def load_pr_data(csv_file_path: str) -> np.ndarray:
    """Read a precision/recall csv (with header) into a float array [n, 2]."""
    with open(csv_file_path, newline="") as result_csv:
        reader = csv.reader(result_csv, delimiter=",")
        next(reader)  # skip header
        rows: List[List[float]] = [[float(v) for v in row] for row in reader]
    return np.array(rows)


def plot_data(csv_file_path: str) -> None:
    """Plot the precision-recall curve: precision on x, recall on y."""
    results = load_pr_data(csv_file_path)
    plt.plot(results[:, 0], results[:, 1])
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.show()


def _self_check() -> None:
    fruits = ["apple", "orange", "melon", "kiwi", "strawberry"]
    assert id_to_fruit(1, fruits) == "orange"
    assert id_to_fruit(3, fruits) == "kiwi"
    assert id_to_fruit(4, fruits) == "strawberry"
    try:
        id_to_fruit(5, fruits)
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass

    coords = np.array([[10, 5, 15, 6, 0], [6, 5, 13, 16, 1]])
    original = coords.copy()
    out = swap(coords)
    assert (out == [[5, 10, 6, 15, 0], [5, 6, 16, 13, 1]]).all()
    assert (coords == original).all()  # input untouched

    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pr.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["precision", "recall"])
            w.writerows([[0.013, 0.951], [1.0, 0.0]])
        data = load_pr_data(path)
        assert data.dtype == float and data[0, 0] == 0.013
    print("all checks passed")


if __name__ == "__main__":
    _self_check()
