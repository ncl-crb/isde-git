from pandas import read_csv
import numpy as np


def load_data(filename):

    """
    Load data from a csv file

    Parameters
    ----------
    filename : string
        Filename to be loaded.

    Returns
    -------
    X : ndarray
        the data matrix.

    y : ndarray
        the labels of each sample.
    """
    data = read_csv(filename)
    z = np.array(data)
    y = z[:, 0]
    X = z[:, 1:]
    return X, y


def split_data(x, y, tr_fraction=0.5):
    """
    Split the data x, y into two random subsets

    """
    idx = np.linspace(0, y.size-1, num=y.size, dtype=int)
    np.random.shuffle(idx)
    n_tr = int(y.size*tr_fraction)
    n_ts = y.size - n_tr
    print("[CHECK]... ",n_tr+n_ts)
    x_tr = x[idx[: n_tr], :]
    y_tr = y[idx[: n_tr]]

    x_ts = x[idx[n_tr : n_tr + n_ts], :]
    y_ts = y[idx[n_tr: n_tr + n_ts]]
    return x_tr, y_tr, x_ts, y_ts
