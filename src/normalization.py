# -----------------------------------------------------------------------------
# (C) 2023 Higor Grigorio (higorgrigorio@gmail.com)  (MIT License)
# -----------------------------------------------------------------------------

import numpy as np


def _normalize_min_max(data: np.array) -> np.array:
    """
    Normalize data using min-max method

    Args:
        data: np.array
            The data to be normalized

    Returns:
        np.array
            The normalized data
    """
    return (data - data.min()) / (data.max() - data.min())


def _normalize_z_score(data: np.array) -> np.array:
    """
    Normalize data using z-score method

    Args:
        data: np.array
            The data to be normalized

    Returns:
        np.array
            The normalized data
    """
    return (data - data.mean()) / data.std()


def normalize(data, method='min-max'):
    """
    Normalize data using the specified method

    Args:
        data: np.array
            The data to be normalized
        method: str
            The method to be used for normalization. It can be 'min-max' or 'z-score'

    Returns:
        np.array
            The normalized data
    """

    if method == 'min-max':
        return _normalize_min_max(data)
    elif method == 'z-score':
        return _normalize_z_score(data)
    else:
        raise ValueError('Invalid method')


_min = 5
_max = 10

#print(f'Min-Max: {normalize(np.array([10, 5, 60, 55]), "min-max")}')
#print(f'Z-Score: {normalize(np.array([10, 5, 60, 55]), "z-score")}')