import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from .utils import save_data_labels

def showcase(
    path_data: str = "./datasets/sklearn_showcase",
    N = 500,
    seed: int = 30,
    standardise: bool = True
):
    """
    Courtesy to https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
    """

    np.random.seed(seed)

    # ============
    # Generate datasets. We choose the size big enough to see the scalability
    # of the algorithms, but not too big to avoid too long running times
    # ============
    noisy_circles = datasets.make_circles(n_samples=N, factor=0.5, noise=0.05)
    noisy_moons = datasets.make_moons(n_samples=N, noise=0.05)
    blobs = datasets.make_blobs(n_samples=N, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=N, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(
        n_samples=N, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
    )

    list_datasets = [
        ( noisy_circles, "noisy_circles", ),
        ( noisy_moons, "noisy_moons", ),
        ( varied, "varied", ),
        ( aniso, "aniso", ),
        ( blobs, "blobs", ),
        ( no_structure, "no_structure", ),
    ]

    for dataset, name in list_datasets:

        X, y = dataset

        # normalize dataset for easier parameter selection
        if standardise:
            X = StandardScaler().fit_transform(X)

        save_data_labels(X, y, f"{path_data}/{name}")







