import numpy as np
from numpy.typing import NDArray
from pathlib import Path

from typing import Any, Tuple


DataArray = NDArray[np.float64]
LabelArray = NDArray[np.int_]

def load_data_labels(
    fname_data: str,
    load_args: dict = {},
) -> Tuple[DataArray, LabelArray]:
    """
    Load data and labels from .csv, .tsv or .npy files.

    Load classification / clustering datasets with datapoints (X) saved
    as a .csv if X is a 2D array, and as a npy otherwise. Labels (y)
    files ends with the same extension as its corresponding X (so .csv
    or .npy). The filename convention is
    - X: `{dataset_name}_data.{ext}`
    - y: `{dataset_name}_labels.{ext}`

    with matching {dataset_name} and {ext} for a given (X, y) pair.

    Parameters
    ----------
    fname_data : str
        Full filename to the data file (i.e., including the `_data.ext`)
    load_args : dict[str, Any], default={}
        Extra keyword arguments forwarded to ``numpy.load`` or
        ``numpy.loadtxt``.

    Returns
    -------
    Tuple[DataArray, LabelArray]
        Data array and labels array loaded from matching ``*_data.ext`` and
        ``*_labels.ext`` files, where `ext` can be "csv", "tsv", or "npy".

    Raises
    ------
    ValueError
        If ``fname_data`` does not use a supported extension.
    """
    # We could directly replace _data without the extension but it's a bit less
    # safe, in case the pattern "_data" appears somewhere else in the path
    ext = Path(fname_data).suffix
    fname_labels = fname_data.replace(f"_data{ext}", f"_labels{ext}")

    if ext in [".npy", "npy"]:
        data = np.load(fname_data, **load_args)
        labels = np.load(fname_labels, **load_args)
    elif ext in [".csv", "csv", ".tsv", "tsv"]:
        ext = ext.lstrip(".")
        data = np.loadtxt(fname_data, **load_args)
        labels = np.loadtxt(fname_labels, **load_args)
    else:
        raise ValueError(f"Unsupported extension: {ext}. Use 'csv', 'tsv' or 'npy'.")
    return data, labels

def save_data_labels(
    X: DataArray,
    y: LabelArray,
    fnames_root: str,
    force_npy: bool = False,
    save_args: dict[str, Any] = {},
) -> None:
    """
    Save data and labels to .csv or .npy files.

    Save classification / clustering datasets with datapoints (X) saved
    as a .csv if X is a 2D array, and as a npy otherwise. Labels (y)
    files ends with the same extension as its corresponding X (so .csv
    or .npy). The filename convention is
    - X: `{dataset_name}_data.{ext}`
    - y: `{dataset_name}_labels.{ext}`

    with matching {dataset_name} and {ext} for a given (X, y) pair.

    Here `fnames_root` should be `path/to/dataset_name`, on which the final filenames of X and y are based.

    Parameters
    ----------
    X : DataArray
        Data array to save.
    y : LabelArray
        Label array to save alongside ``X``.
    fnames_root : str
        Output path prefix, excluding the ``_data`` or ``_labels`` suffix and
        file extension.
    force_npy : bool, default=False
        Whether to always save with NumPy's binary ``.npy`` format.
    save_args : dict[str, Any], default={}
        Extra keyword arguments forwarded to ``numpy.save`` or
        ``numpy.savetxt``.

    Returns
    -------
    None
        This function writes files to disk and returns nothing.
    """
    p = Path(fnames_root)
    p.parent.mkdir(parents=True, exist_ok=True)

    shape = X.shape
    if len(shape) > 2 or force_npy:
        ext = "npy"
    else:
        ext = "csv"

    if ext == "npy":
        np.save(f"{fnames_root}_data.{ext}", X, **save_args)
        np.save(f"{fnames_root}_labels.{ext}", y, **save_args)
    else:
        np.savetxt(f"{fnames_root}_data.{ext}", X, **save_args)
        np.savetxt(f"{fnames_root}_labels.{ext}", y, **save_args)

def compute_centroids(X: DataArray, y: LabelArray) -> DataArray:
    """
    Compute the centroids of clusters in a dataset.

    Assumes that labels are integers 0..k-1, where k is the number of clusters.

    Parameters
    ----------
    X : NDArray[np.float64]
        The data points, shape (n_samples, n_features).
    y : NDArray[np.int_]
        The cluster labels for each data point, shape (n_samples,).

    Returns
    -------
    NDArray[np.float64]
        The centroids of the clusters, shape (n_clusters, n_features).
    """
    unique_labels = np.unique(y)
    centroids = np.zeros((len(unique_labels), X.shape[1]))
    for label in unique_labels:
        # Extract points belonging to the current cluster
        cluster_points = X[y == label]
        # Compute the centroid of the cluster
        centroids[int(label)] = np.mean(cluster_points, axis=0)
    return centroids
