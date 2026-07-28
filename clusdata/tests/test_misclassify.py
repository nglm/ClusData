import pytest

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from clusdata.misclassify import (
    full_random, balanced, bully, subclustering,
    flag_misclassified, plot_misclassified, set_misclassified,
)

def test_set_misclassified():
    X, y = make_blobs(n_samples=100, centers=5, n_features=2, random_state=42)

    for r in [0.1, [0.1, 0.2], [0.1, 0.2, 0.3, 0.4, 0.5]]:
        for apply in [True, False]:
            new_r = set_misclassified(
                misclassified=r, y=y, apply_misclassification_to_all_clusters=apply
            )

            assert isinstance(new_r, list)
            assert len(new_r) == 5
            assert all(isinstance(x, float) for x in new_r)




def test_full_random():

    X, y = make_blobs(n_samples=100, centers=5, n_features=2, random_state=42)

    for r in [0.1, [0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4, 0.5]]:
        for apply in [True, False]:
            for is_global in [True, False]:
                y_wrong = full_random(
                    X, y, misclassified=r, apply_misclassification_to_all_clusters=apply,
                    global_misclassification=is_global
                )

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
            X, y_error, misclassified=0.8, error_if_not_enough=True
        )
    y_wrong = balanced(
        X, y_error, misclassified=0.8, error_if_not_enough=False
    )

def test_bully():

    X, y = make_blobs(n_samples=100, centers=5, n_features=2, random_state=42)
    counts_y = np.bincount(y)


    y_wrong = bully(X, y)

    # There is no cluster fully eaten
    assert isinstance(y_wrong, np.ndarray)
    assert y_wrong.shape == y.shape
    assert len(np.unique(y_wrong)) == 5


    y_wrong = bully(X, y, misclassified=0.5)

    # Two clusters are fully eaten, and one is half eaten
    assert isinstance(y_wrong, np.ndarray)
    assert y_wrong.shape == y.shape
    counts_y_wrong = np.bincount(y_wrong)
    print(counts_y, counts_y_wrong)
    assert len(np.unique(y_wrong)) < len(np.unique(y))
    assert len(np.unique(y_wrong)) == 3

def test_subclustering():
    X, y = make_blobs(n_samples=100, centers=5, n_features=2, random_state=42)

    for method in ["grouped", "random"]:
        for misclassified in [0.1, [0.1, 0.2, 0.3]]:
            for misclassify_minority in [True, False]:
                y_wrong = subclustering(
                    X, y, method=method, misclassified=misclassified,
                    misclassify_minority=misclassify_minority
                )

                assert isinstance(y_wrong, np.ndarray)
                assert y_wrong.shape == y.shape
                assert len(np.unique(y_wrong)) > len(np.unique(y))

                # Check the new number of clusters
                if isinstance(misclassified, list):
                    assert len(np.unique(y_wrong)) == len(np.unique(y)) + len(misclassified)
                else:
                    assert len(np.unique(y_wrong)) == len(np.unique(y)) + 1

def test_flag_misclassified():
    X, y = make_blobs(n_samples=100, centers=5, n_features=2, random_state=42)

    y_wrong = full_random(X, y)

    # Check that the misclassified points are flagged correctly
    y_flagged = flag_misclassified(y, y_wrong)
    assert isinstance(y_flagged, np.ndarray)
    assert y_flagged.shape == y.shape
    assert np.sum(y_flagged == -1) == np.sum(y != y_wrong)

def test_plot_misclassified():
    X, y = make_blobs(n_samples=100, centers=5, n_features=2, random_state=42)

    y_wrong = full_random(X, y)

    # Check when no fig and ax are provided
    fig, ax = plot_misclassified(X, y, y_wrong)
    assert fig is not None
    assert ax is not None

    # Check when fig and ax are provided with a fig having multiple subplots
    list_X = [X]*3
    list_y = [y]*3
    list_y_wrong = [y_wrong]*3
    n_plots = len(list_X)
    # Share y-axis and y-label across all subplots
    fig, axs = plt.subplots(1, n_plots, figsize=(5*n_plots,5), sharey=True)
    correct_in_grey = True

    for i in range(n_plots):
        X = list_X[i]
        y = list_y[i]
        y_wrong = list_y_wrong[i]

        fig, axs[i] = plot_misclassified(
            X, y, y_wrong, correct_in_grey=correct_in_grey,
            fig=fig, ax=axs[i], scatter_kwargs={"alpha": 0.1}
        )
        correct_in_grey = not correct_in_grey

        axs[i].set_xlabel("")


    axs[0].set_ylabel("Feature 2")