"""
Create misclassification on purpose, for illustration purposes or to test
different cluster validity indices, etc.

- 2, 5, 20 gaussians
    - Number of datapoints
    - Given global percentage of misclassified
    - percentage of misclassified per cluster
    - For the dataset with 2 clusters: Failed like complete random clustering
- 3 circles, with more or less distinctions between the inner ones.
  - Failed like KMeans (have 3 attractor outside the circles)
"""

from typing import Union, List

import numpy as np
from numpy.typing import NDArray
import random
import matplotlib
from matplotlib import pyplot as plt

from .utils import compute_centroids


DataArray = NDArray[np.float64]
LabelArray = NDArray[np.int_]

def set_misclassified(
    misclassified: Union[float, List[float]],
    y: LabelArray,
    apply_misclassification_to_all_clusters: bool = True,
) -> List[int]:
    """
    Set the number of misclassified samples per cluster.

    Parameters
    ----------
    misclassified : Union[float, List[float]]
        Fraction of samples within a cluster (or a list of clusters if a
        list is provided) whose labels should be replaced. If a list is provided, the length of the list must be less than or equal to the number of clusters in ``y``. If fewer values are provided than the number of clusters, the remaining clusters will have a misclassification rate of 0.
    y : NDArray[np.int_]
        Ground-truth cluster labels encoded as integers.
    apply_misclassification_to_all_clusters : bool, default=True
        If ``True`` and if ``misclassified`` is a float, then applies
        the same misclassification rate to all clusters. If ``False``,
        only applies the misclassification strategy to the first
        (smallest label) cluster.

    Returns
    -------
    List[int]
        List of misclassified fractions for each cluster.

    Raises
    ------
    ValueError
        If the length of the misclassified list is greater than the
        number of clusters in ``y`` or if any value in the list is not
        between 0 and 1.
    TypeError
        If ``misclassified`` is not a float or a list of floats.
    """

    N_clusters = len(np.unique(y))

    if isinstance(misclassified, float):
        # All clusters get the same rate
        if apply_misclassification_to_all_clusters:
            misclassified = [misclassified] * N_clusters
        # Only the first cluster get a non null misclassification rate
        else:
            misclassified = [misclassified] + [0.]*(N_clusters-1)
    elif isinstance(misclassified, list):

        # Raise error if more rates than clusters
        if len(misclassified) > N_clusters:
            raise ValueError(
                f"Length of misclassified list ({len(misclassified)}) "
                f"greater than the number of clusters ({N_clusters})."
            )

        # Complete with zeros if needed
        elif len(misclassified) < N_clusters:
            misclassified = misclassified + [0.] * (N_clusters - len(misclassified))

    # Raise error if wrong type
    else:
        raise TypeError(
            f"misclassified must be a float or a list of floats, "
            f"got {type(misclassified)}."
        )
    if any([r < 0 or r > 1 for r in misclassified]):
        raise ValueError(
            f"All values in misclassified list must be between 0 and 1. "
            f"Got {misclassified}."
        )
    return misclassified


def full_random(
    X: DataArray,
    y: LabelArray,
    misclassified: Union[float, List[float]] = 0.10,
    seed: int = 42,
    apply_misclassification_to_all_clusters: bool = True,
    global_misclassification: bool = False,
) -> LabelArray:
    """
    Randomly mislabel a fraction of datapoints within some clusters.

    Parameters
    ----------
    X : NDArray[np.float64]
        Dataset used only to infer the number of samples.
    y : NDArray[np.int_]
        Ground-truth cluster labels encoded as integers.
    misclassified : Union[float, List[float]]
        Fraction of samples within a cluster (or a list of clusters if a
        list is provided) whose labels should be replaced. If a list is provided, the length of the list must be less than or equal to the number of clusters in ``y``. If fewer values are provided than the number of clusters, the remaining clusters will have a misclassification rate of 0.
    seed : int, default=42
        Seed passed to Python's random generator for reproducibility.
    apply_misclassification_to_all_clusters : bool, default=True
        If ``True`` and if ``misclassified`` is a float, then applies
        the same misclassification rate to all clusters. If ``False``,
        only applies the misclassification strategy to the first
        (smallest label) cluster.
    global_misclassification : bool, default=False
        If ``True`` and if ``misclassified`` was a float, randomly
        re-assign labels independently of the cluster. The result will
        be very similar to applying the same misclassification rate to
        all clusters, except that there will be small variations in the
        precise number of misclassified datapoint within each cluster.
        This parameter is ignored if ``misclassified`` is not a float.

    Returns
    -------
    NDArray[np.int_]
        Copy of ``y`` where the selected entries are assigned a label that
        differs from their original one.
    """
    random.seed(seed)

    N = len(X)
    idx = list(range(N))
    classes = sorted(np.unique(y).tolist())
    y_wrong = np.copy(y)

     # ---------- Full random misclassification ---------------
    if isinstance(misclassified, float) and global_misclassification:

        # Number of samples to misclassify in this cluster
        N_misclassified = int(N * misclassified)

        # Get random indices to misclassify
        idx_misclassified = random.sample(idx, N_misclassified)

        y_wrong = np.copy(y)
        for i in idx_misclassified:
            # Get a wrong label (anything but not the correct one, at random)
            new_label = random.sample(list(set(classes) - {y[i]}), 1)[0]
            y_wrong[i] = new_label

    # --------- Random misclassification within clusters ---------------
    else:

        misclassified = set_misclassified(
            misclassified=misclassified, y=y,
            apply_misclassification_to_all_clusters=apply_misclassification_to_all_clusters
        )

        for c, misclassified_c in zip(classes, misclassified):

            # Indices of the current cluster
            idx_c = [i for i in idx if y[i] == c]
            N_c = len(idx_c)
            # Number of samples to misclassify in this cluster
            N_misclassified = int(N_c * misclassified_c)

            # ------- Random misclassification within cluster -------
            # Get random indices to misclassify
            idx_misclassified = random.sample(idx_c, N_misclassified)

            for i in idx_misclassified:
                # Get new label (anything but not the correct one, at random)
                new_label = random.sample(list(set(classes) - {y[i]}), 1)[0]
                y_wrong[i] = new_label

    return y_wrong


def balanced(
    X: DataArray,
    y: LabelArray,
    misclassified: float = 0.10,
    seed: int = 42,
    allow_same_closest: bool = False,
    apply_misclassification_to_all_clusters: bool = True,
) -> LabelArray:
    """
    Misclassify each cluster evenly toward its nearest centroid.

    The requested number of misclassified samples is distributed evenly
    across clusters. For each cluster, the samples closest to the
    nearest other cluster centroid are relabeled to that neighboring
    cluster.

    Parameters
    ----------
    X : NDArray[np.float64]
        Dataset of shape ``(n_samples, n_features)``.
    y : NDArray[np.int_]
        Ground-truth cluster labels encoded as integers.
    misclassified : Union[float, List[float]]
        Fraction of samples within a cluster (or a list of clusters if a
        list is provided) whose labels should be replaced. If a list is provided, the length of the list must be less than or equal to the number of clusters in ``y``. If fewer values are provided than the number of clusters, the remaining clusters will have a misclassification rate of 0.
    seed : int, default=42
        Seed passed to Python's random generator for reproducibility.
    allow_same_closest : bool, default=False
        Whether multiple source clusters may target the same closest cluster.
    apply_misclassification_to_all_clusters : bool, default=True
        If ``True`` and if ``misclassified`` is a float, then applies
        the same misclassification rate to all clusters. If ``False``,
        only applies the misclassification strategy to the first
        (smallest label) cluster.

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
    classes = sorted(np.unique(y).tolist())

    misclassified = set_misclassified(
        misclassified=misclassified, y=y,
        apply_misclassification_to_all_clusters=apply_misclassification_to_all_clusters
    )

    y_wrong = np.copy(y)

    visited_clusters: list[int] = []
    visited_closest: list[int] = []

    # Compute centroids for each cluster
    centroids = compute_centroids(X, y)

    for c, misclassified_c in zip(classes, misclassified):

        # --------- indices of the current cluster -------
        idx_c = [i for i in idx if y[i] == c]
        N_c = len(idx_c)
        N_misclassified_c = int(N_c * misclassified_c)

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

        idx_misclassified_c = [
            idx_c[i]
            for i in idx_c_sorted[:min(N_misclassified_c, N_c)]
        ]

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
    misclassified: float = 0.10,
    error_if_not_enough: bool = True,
) -> LabelArray:
    """
    Misclassify datapoints by letting some clusters being absorbed.

    Clusters are processed from the furthest centroid norm and then to
    its neighbors. Samples from a processed cluster are progressively
    reassigned to another cluster until the requested global number of
    misclassified points is met.

    Parameters
    ----------
    X : NDArray[np.float64]
        Dataset of shape ``(n_samples, n_features)``.
    y : NDArray[np.int_]
        Ground-truth cluster labels encoded as integers.
    misclassified : float, default=0.10
        Fraction of samples to relabel across the whole dataset.
    error_if_not_enough : bool, default=True
        Whether to raise a ValueError if there are not enough points in a cluster to satisfy the requested misclassification count.

    Returns
    -------
    NDArray[np.int_]
        Copy of ``y`` after applying the bully-style relabeling.

    Raises
    ------
    ValueError
        If the total number of datapoints minus the number of datapoints
        in the last bully cluster is smaller than the number of samples
        needed to satisfy the requested misclassification count and
        ``error_if_not_enough`` is ``True``.
    """

    N = len(X)
    idx = list(range(N))
    N_misclassified = int(N*misclassified)

    y_wrong = np.copy(y)


    # Compute centroids for each cluster
    centroids = compute_centroids(X, y)

    # Make the furthest away clusters be the first bullied cluster
    # (i.e the one with the greatest norm)
    first_bullied = np.argmax(np.linalg.norm(centroids, axis=1))

    # ------ Find the closest cluster to the first bullied -----------
    distances_to_first = np.linalg.norm(
        centroids - centroids[first_bullied], axis=1
    )

    # Sort the clusters by distance to the first bullied cluster
    # descending order appeared in numpy > 2.5
    if np.__version__ >= "2.5":
        sorted_class = np.argsort(distances_to_first, descending=False)
    else:
        sorted_class = np.argsort(distances_to_first)[::-1]

    # Keep track of indices to misclassify
    idx_bullied: list[int] = []
    N_bullied = 0

    # Bully the first cluster as much as possible, then the second, etc.
    for i, c in enumerate(sorted_class):

        # --------- indices of the current cluster -------
        idx_c = [i for i in idx if y[i] == c]
        N_c = len(idx_c)

        # Current bully: next cluster in the sorted list
        # If we reached the last bully and still want more misclassified
        # points than N- N_bully, then i+1 will yield an error. This
        # case means that we only have one cluster in the dataset, and
        # we might raise an error if error_if_not_enough is True
        i_bully = i+1
        if i+1 == len(sorted_class):
            if error_if_not_enough:
                raise ValueError(
                    f"Not enough points to misclassify {N_misclassified} points. "
                    f"Only {N - N_bullied} available."
                )
            else:
                bully = c
                idx_bullied += idx_c
        # Regular case, where we have a next cluster to bully
        else:
            bully = sorted_class[i_bully]

            # ------ Find datapoints to bully  -----------
            # Bully as many points as possible from this cluster,
            # but not more than the number of points left to misclassify
            n_newly_bullied = min(N_c, N_misclassified - N_bullied)

            # Find the indices of the closest datapoints to the bully
            if n_newly_bullied < N_c:
                dist_to_bully = np.linalg.norm(
                    X[idx_c] - centroids[bully], axis=1
                )
                # Find in idx_c the indices of the closest points to the bully
                idx_closest_to_bully = np.argsort(dist_to_bully)
                # Find them in idx and keep only the first ones
                idx_closest_to_bully = [
                    idx_c[i] for i in idx_closest_to_bully[:n_newly_bullied]
                ]
            # Otherwise, take them all
            else:
                idx_closest_to_bully = idx_c.copy()

            # Update the list of bullied indices
            idx_bullied += idx_closest_to_bully

        # Update labels if we are done
        N_bullied = len(idx_bullied)

        # Check whether we have enough points to misclassify
        finish = N_bullied >= N_misclassified
        if finish:
            for i in idx_bullied:
                y_wrong[i] = bully

            # Stop
            break

    return y_wrong

def subclustering(
    X: DataArray,
    y: LabelArray,
    misclassified: Union[float, List[float]] = 0.10,
    method: str = "grouped",
    seed: int = 42,
    misclassify_minority: bool = False,
    apply_misclassification_to_all_clusters: bool = True
) -> LabelArray:
    """
    Misclassify datapoints by creating subclusters within some clusters.

    Can create random subclusters or grouped subclusters that are close
    to a random point within the cluster.

    Note that this function a priori keeps the clusters aligned, meaning
    that the labels in ``y`` and ``y_wrong`` will be the same for the
    correctly classified samples.

    If ``misclassify_minority`` is set to ``True``, then the minority of
    the subcluster will be misclassified instead of the majority. More
    specifically, if a subcluster within a cluster becomes the majority
    then the new label of the subcluster will be the same as the
    original cluster label and the minority will be the misclassified
    samples.

    Parameters
    ----------
    X : NDArray[np.float64]
        Dataset of shape ``(n_samples, n_features)``.
    y : NDArray[np.int_]
        Ground-truth cluster labels encoded as integers.
    misclassified : Union[float, List[float]]
        Fraction of samples within a cluster (or a list of clusters if a
        list is provided) whose labels should be replaced. If a list is provided, the length of the list must be less than or equal to the number of clusters in ``y``. If fewer values are provided than the number of clusters, the remaining clusters will have a misclassification rate of 0.
    method : str, default="grouped"
        Method to use for creating subclusters. Can be either "grouped" or "random". If "grouped", the misclassified samples will be grouped together in a subcluster that is close to a random point within the cluster. If "random", the misclassified samples will be randomly selected within the cluster.
    seed : int, default=42
        Seed passed to Python's random generator for reproducibility.
    misclassify_minority : bool, default=False
        Whether to misclassify the minority of the subcluster instead of the majority.
    apply_misclassification_to_all_clusters : bool, default=True
        If ``True`` and if ``misclassified`` is a float, then applies
        the same misclassification rate to all clusters. If ``False``,
        only applies the misclassification strategy to the first
        (smallest label) cluster.

    Returns
    -------
    NDArray[np.int_]
        Copy of ``y`` after applying the relabeling.
    """
    random.seed(seed)

    misclassified = set_misclassified(
        misclassified=misclassified, y=y,
        apply_misclassification_to_all_clusters=apply_misclassification_to_all_clusters
    )

    N = len(X)
    idx = list(range(N))
    # We use set to be able to perform set operations later on
    classes = sorted(np.unique(y).tolist())
    last_label = int(max(classes))
    y_wrong = np.copy(y)

    # Group the misclassified points together in a subcluster
    for i, (c, misclassified_c) in enumerate(zip(classes, misclassified)):

        # Indices of the current cluster
        idx_c = [i for i in idx if y[i] == c]
        N_c = len(idx_c)
        # Number of samples to misclassify in this cluster
        N_misclassified_c = int(N_c * misclassified_c)

        if method == "grouped":

            # Get a random point within the cluster to be the center of the subcluster
            center_idx = random.choice(idx_c)
            center_point = X[center_idx]

            # Compute distances to the center point
            distances_to_center = np.linalg.norm(X[idx_c] - center_point, axis=1)

            # Sort indices by distance to the center point
            idx_c_sorted = np.argsort(distances_to_center)

            # Select the closest points to form the subcluster
            idx_misclassified = [
                idx_c[i]
                for i in idx_c_sorted[:min(N_misclassified_c, N_c)]
            ]
        elif method == "random":
            # Randomly select indices to misclassify
            idx_misclassified = random.sample(idx_c, min(N_misclassified_c, N_c))

        # Assign a new label for the subcluster
        new_label = last_label + i + 1

        # Case where the subcluster becomes the majority
        if misclassify_minority and len(idx_misclassified) > N_c / 2:
            idx_misclassified = [i for i in idx_c if i not in idx_misclassified]

        for i in idx_misclassified:
            y_wrong[i] = new_label

    return y_wrong

def flag_misclassified(
        y_true: LabelArray,
        y_wrong: LabelArray
    ) -> LabelArray:
    """
    Flag the misclassified samples between two label arrays.

    Keeps all labels from the true array as is except for datapoints that were misclassified in y_wrong for which a new label `-1` is added to the returned array.,

    Note that this function assumes that the labels in ``y_true`` and ``y_wrong`` are "aligned".

    Note that ``y_true`` and ``y_wrong`` can have different number of clusters, but the labels must be aligned for the correctly classified samples.

    Parameters
    ----------
    y_true : NDArray[np.int_]
        Ground-truth cluster labels encoded as integers.
    y_wrong : NDArray[np.int_]
        Predicted cluster labels encoded as integers.

    Returns
    -------
    LabelArray
        Copy of ``y_true`` except that the misclassified entries are assigned a label `-1`.
    """
    y_flagged = np.copy(y_true)
    y_flagged[y_true != y_wrong] = -1
    return y_flagged

def plot_misclassified(
    X: DataArray,
    y_true: LabelArray,
    y_wrong: LabelArray,
    correct_in_grey: bool = True,
    fig: matplotlib.figure.Figure = None,
    ax: matplotlib.axes.Axes = None,
    scatter_kwargs: dict = {"alpha": 0.3}
):
    """
    Plot the misclassified samples between two label arrays.

    Note that this function assumes that the labels in ``y_true`` and ``y_wrong`` are "aligned".

    Note that ``y_true`` and ``y_wrong`` can have different number of clusters, but the labels must be aligned for the correctly classified samples.

    Parameters
    ----------
    X : NDArray[np.float64]
        Dataset of shape ``(n_samples, n_features)``.
    y_true : NDArray[np.int_]
        Ground-truth cluster labels encoded as integers.
    y_wrong : NDArray[np.int_]
        Predicted cluster labels encoded as integers.
    correct_in_grey : bool, default=True
        Whether to plot the correctly classified samples in grey or not.
    fig : matplotlib.figure.Figure, optional
        Matplotlib figure object. If not provided, a new figure will be created.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object. If not provided, a new axes will be created.
    scatter_kwargs : dict, optional
        Additional keyword arguments to pass to the scatter plot function.

    Returns
    -------
    fig, axs
        Matplotlib figure and axes objects.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    y_flagged = flag_misclassified(y_true, y_wrong)
    unique_clusters = np.unique(y_flagged)

    if not correct_in_grey:
        color_map = {
            cluster: plt.cm.tab20(i)
            for i, cluster in enumerate(unique_clusters)
        }

    i_correct = 0
    for c in unique_clusters:
        # Plot the correctly classified points
        if c != -1:
            # Either in grey
            if correct_in_grey:
                if i_correct == 0:
                    ax.scatter(
                        X[y_flagged == c, 0], X[y_flagged == c, 1],
                        color='grey', marker='o', label=f'Correctly classified',
                        **scatter_kwargs
                    )
                    i_correct += 1
                else:
                    ax.scatter(
                        X[y_flagged == c, 0], X[y_flagged == c, 1],
                        color='grey', marker='o', **scatter_kwargs
                    )
            # Or a different color for each cluster
            else:
                ax.scatter(
                    X[y_flagged == c, 0], X[y_flagged == c, 1],
                    color=color_map[c], marker='o', **scatter_kwargs
                )

    # Plot the misclassified points in red with a different marker
    # We plot them at the end so that they are on top of the correctly classified points
    ax.scatter(
        X[y_flagged == -1, 0], X[y_flagged == -1, 1],
        color='red', marker='x', label='Misclassified',
        **scatter_kwargs
    )

    ax.set_xlabel("Feature 1")
    ax.legend()
    ax.set_ylabel("Feature 2")
    return fig, ax