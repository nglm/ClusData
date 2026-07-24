"""
Create misclassification on purpose, for illustration purposes or to test
different cluster validity indices, etc.

- 2, 5, 20 gaussians
    - Number of datapoints
    - Given global percentage of misclassified
    - percentage of misclassified per cluster
    - For the 2 version: Failed like complete random clustering
- 3 circles, with more or less distinctions between the inner ones.
  - Failed like KMeans (have 3 attractor outside the circles)
"""

import numpy as np
from numpy.typing import NDArray
import random

from utils import compute_centroids


DataArray = NDArray[np.float64]
LabelArray = NDArray[np.int_]


def full_random(
    X: DataArray,
    y: LabelArray,
    global_misclassified: float = 0.10,
    seed: int = 42
) -> LabelArray:
    """Randomly replace a fraction of labels with incorrect ones.

    Parameters
    ----------
    X : NDArray[np.float64]
        Dataset used only to infer the number of samples.
    y : NDArray[np.int_]
        Ground-truth cluster labels encoded as integers.
    global_misclassified : float, default=0.10
        Fraction of samples whose labels should be replaced.
    seed : int, default=42
        Seed passed to Python's random generator for reproducibility.

    Returns
    -------
    NDArray[np.int_]
        Copy of ``y`` where the selected entries are assigned a label that
        differs from their original one.
    """
    random.seed(seed)

    N = len(X)
    idx = list(range(N))
    classes = set(np.unique(y))
    N_misclassified = int(N*global_misclassified)

    # ---------- Full random misclassification ---------------
    # Get random indices to misclassify
    idx_misclassified = random.sample(idx, N_misclassified)

    y_wrong = np.copy(y)
    for i in idx_misclassified:
         # Get a wrong label (anything but not the correct one, at random)
        new_label = random.sample(classes - {y[i]}, 1)
        y_wrong[i] = new_label

    return y_wrong

def balanced(
    X: DataArray,
    y: LabelArray,
    global_misclassified: float = 0.10,
    seed: int = 42,
    allow_same_closest: bool = False,
    error_if_not_enough: bool = True
) -> LabelArray:
    """Misclassify each cluster toward its nearest centroid.

    The requested number of misclassified samples is distributed evenly across
    clusters. For each cluster, the samples closest to the nearest other
    cluster centroid are relabeled to that neighboring cluster.

    Parameters
    ----------
    X : NDArray[np.float64]
        Dataset of shape ``(n_samples, n_features)``.
    y : NDArray[np.int_]
        Ground-truth cluster labels encoded as integers.
    global_misclassified : float, default=0.10
        Fraction of samples to relabel across the whole dataset.
    seed : int, default=42
        Seed passed to Python's random generator for reproducibility.
    allow_same_closest : bool, default=False
        Whether multiple source clusters may target the same closest cluster.
    error_if_not_enough : bool, default=True
        Whether to raise when a cluster contains fewer points than the number
        requested for relabeling from that cluster.

    Returns
    -------
    NDArray[np.int_]
        Copy of ``y`` containing the balanced nearest-centroid relabeling.

    Raises
    ------
    ValueError
        If a cluster does not contain enough samples to satisfy the requested
        per-cluster misclassification count and ``error_if_not_enough`` is
        ``True``.
    """

    random.seed(seed)

    N = len(X)
    idx = list(range(N))
    classes = set(np.unique(y))
    N_misclassified = int(N*global_misclassified)
    N_misclassified_c = int(N_misclassified/len(classes))

    y_wrong = np.copy(y)

    visited_clusters: list[int] = []
    visited_closest: list[int] = []

    # Compute centroids for each cluster
    centroids = compute_centroids(X, y)

    for c in classes:

        # --------- indices of the current cluster -------
        idx_c = [i for i in idx if y[i] == c]
        N_c = len(idx_c)

        # ------ Find the closest cluster to c -----------
        distances = np.linalg.norm(centroids - centroids[c], axis=1)

        # Ignore the distance to itself
        distances[c] = np.inf
        if not allow_same_closest:
            # Ignore already visited clusters
            for visited_c in visited_closest:
                distances[visited_c] = np.inf
        closest_cluster = np.argmin(distances)

        # ------- Find closest points to closest cluster ------
        # Get the distances of datapoints in c to this closest cluster
        distances_to_closest = np.linalg.norm(
            X[idx_c] - centroids[closest_cluster], axis=1
        )

        # sort the indices in idx_c
        idx_c_sorted = np.argsort(distances_to_closest)

        # Find the indices in idx from the argsort of idx_c
        if N_misclassified_c > N_c and error_if_not_enough:
            raise ValueError(
                f"Not enough points in cluster {c} to misclassify "
                f"{N_misclassified_c} points. Only {N_c} available."
            )

        idx_misclassified_c = idx_c[idx_c_sorted[:min(N_misclassified_c, N_c)]]

        # ------- Assign the wrong labels --------------
        for i in idx_misclassified_c:
            y_wrong[i] = closest_cluster

        # ---------- Update visited -------------
        visited_clusters.append(c)
        visited_closest.append(closest_cluster)


    return y_wrong

def bully(
    X: DataArray,
    y: LabelArray,
    global_misclassified: float = 0.10,
) -> LabelArray:
    """Relabel points by letting dominant clusters absorb distant ones.

    Clusters are processed from the furthest centroid norm to the smallest.
    Samples from a processed cluster are progressively reassigned to another
    cluster until the requested global number of misclassified points is met.

    Parameters
    ----------
    X : NDArray[np.float64]
        Dataset of shape ``(n_samples, n_features)``.
    y : NDArray[np.int_]
        Ground-truth cluster labels encoded as integers.
    global_misclassified : float, default=0.10
        Fraction of samples to relabel across the whole dataset.

    Returns
    -------
    NDArray[np.int_]
        Copy of ``y`` after applying the bully-style relabeling.
    """

    N = len(X)
    idx = list(range(N))
    N_misclassified = int(N*global_misclassified)

    y_wrong = np.copy(y)


    # Compute centroids for each cluster
    centroids = compute_centroids(X, y)

    # Make the furthest away clusters be the bullied clusters
    # (i.e the ones with thegreatest norm)
    sorted_class = np.argsort(
        np.linalg.norm(centroids, axis=1), descending=True
    )

    # Keep track of indices to misclassify
    idx_bullied: list[int] = []

    # Bully the first cluster as much as possible, then the second, etc.
    for i, c in enumerate(sorted_class):

        # Current bully: next cluster in the sorted list
        bully = idx_bullied[i+1]

        # --------- indices of the current cluster -------
        idx_c = [i for i in idx if y[i] == c]
        N_c = len(idx_c)

        # ------ Find datapoints to bully  -----------
        # Bully as many points as possible from this cluster,
        # but not more than the number of points left to misclassify
        n_newly_bullied = min(N_c, N_misclassified - N_bullied)

        # Find the indices of the closest datapoints to the bully
        if n_newly_bullied <= N_c:
            dist_to_bully = np.linalg.norm(
                X[idx_c] - centroids[c], axis=1
            )
            idx_closest_to_bully = np.argsort(dist_to_bully)[:n_newly_bullied]
        # Otherwise, take them all
        else:
            idx_closest_to_bully = idx_c.copy()

        # Update the list of bullied indices
        idx_bullied += idx_closest_to_bully
        N_bullied = len(idx_bullied)

        # Check whether we have enough points to misclassify
        finish = N_bullied >= N_misclassified
        if finish:
            for i in idx_bullied:
                y_wrong[i] = bully

            # Stop
            break


    return y_wrong
