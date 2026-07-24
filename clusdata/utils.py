import numpy as np
from pathlib import Path

def load_data_labels(
    fname_data: str,
    load_args: dict = {},
) -> Tuple[np.ndarray, np.ndarray]:
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

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Data array and labels array loaded from matching ``*_data.ext`` and
        ``*_labels.ext`` files, where `ext` can be "csv", "tsv", or "npy".
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
    X: np.ndarray,
    y: np.ndarray,
    fnames_root: str,
    force_npy: bool = False,
    save_args: dict = {},
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


    Returns
    -------
    None
        This function writes files to disk and returns nothing.
    """
    p = Path(fname_data)
    p.parent.mkdir(parents=True, exist_ok=True)

    shape = X.shape
    if len(shape) > 2 or force_npy:
        ext = "npy"
    else:
        ext = "csv"

    if ext == "npy":
        np.save(f"{fname_data}_data.{ext}", X, **save_args)
        np.save(f"{fname_data}_labels.{ext}", y, **save_args)
    else:
        np.savetxt(f"{fname_data}_data.{ext}", X, **save_args)
        np.savetxt(f"{fname_data}_labels.{ext}", y, **save_args)


