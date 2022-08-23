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
    """Split the data X,y into two random subsets.

    input:
        x: set of images
        y: labels
        fract_tr: float, percentage of samples to put in the training set.
            If necessary, number of samples in the training set is rounded to
            the lowest integer number.

    output:
        Xtr: set of images (numpy array, training set)
        Xts: set of images (numpy array, test set)
        ytr: labels (numpy array, training set)
        yts: labels (numpy array, test set)

    """
    idx = np.linspace(0, y.size-1, num=y.size, dtype=int)
    np.random.shuffle(idx)
    n_tr = int(y.size*tr_fraction)
    n_ts = y.size - n_tr
    x_tr = x[idx[: n_tr], :]
    y_tr = y[idx[: n_tr]]

    x_ts = x[idx[n_tr : n_tr + n_ts], :]
    y_ts = y[idx[n_tr: n_tr + n_ts]]
    return x_tr, y_tr, x_ts, y_ts
