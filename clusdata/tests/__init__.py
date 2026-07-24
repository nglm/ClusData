import pytest

from ..sklearn import showcase

def test_showcase():

    # Original dataset but not standardized
    path_data = "clusdata/datasets/sklearn_showcase_no_std"
    showcase(path_data, standardise=False)

    # Original dataset
    path_data_std = f"{path_data}"
    showcase(path_data_std)

    # More datapoints
    path_data_std = f"{path_data}_5000"
    showcase(path_data_std)
