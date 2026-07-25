import pytest

import numpy as np
from sklearn.datasets import make_blobs

from clusdata.misclassify import full_random, balanced, bully

def test_full_random():

    X, y = make_blobs(n_samples=100, centers=5, n_features=2, random_state=42)

    y_wrong = full_random(X, y)

    assert isinstance(y_wrong, np.ndarray)
    assert y_wrong.shape == y.shape

def test_balanced():

    X, y = make_blobs(n_samples=100, centers=5, n_features=2, random_state=42)

    y_wrong = balanced(X, y)

    assert isinstance(y_wrong, np.ndarray)
    assert y_wrong.shape == y.shape

    y_wrong = balanced(X, y, allow_same_closest=True)

    assert isinstance(y_wrong, np.ndarray)
    assert y_wrong.shape == y.shape

    # Artificially decrease one cluster size to trigger the error
    # 50% of the point of class 0 are relabeled to 1
    y_error = np.copy(y)
    idx_y_0 = np.where(y == 0)[0]
    idx_y_0_to_change = idx_y_0[:len(idx_y_0)//2]
    y_error[idx_y_0_to_change] = 1

    with pytest.raises(ValueError):
        y_wrong = balanced(
            X, y_error, global_misclassified=0.8, error_if_not_enough=True
        )
    y_wrong = balanced(
        X, y_error, global_misclassified=0.8, error_if_not_enough=False
    )

def test_bully():

    X, y = make_blobs(n_samples=100, centers=5, n_features=2, random_state=42)
    counts_y = np.bincount(y)


    y_wrong = bully(X, y)

    # There is no cluster fully eaten
    assert isinstance(y_wrong, np.ndarray)
    assert y_wrong.shape == y.shape
    assert len(np.unique(y_wrong)) == 5


    y_wrong = bully(X, y, global_misclassified=0.5)

    # Two clusters are fully eaten, and one is half eaten
    assert isinstance(y_wrong, np.ndarray)
    assert y_wrong.shape == y.shape
    counts_y_wrong = np.bincount(y_wrong)
    print(counts_y, counts_y_wrong)
    assert len(np.unique(y_wrong)) < len(np.unique(y))
    assert len(np.unique(y_wrong)) == 3
