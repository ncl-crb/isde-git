import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class NMC(object):
    """
    Class implementing the Nearest Mean Centroid (NMC) classifier.

    This classifier estimates one centroid per class from the training data,
    and predicts the label of a never-before-seen (test) point based on its
    closest centroid.

    Attributes
    -----------------
    - centroids: read-only attribute containing the centroid values estimated
        after training

    Methods
    -----------------
    - fit(x,y) estimates centroids from the training data
    - predict(x) predicts the class labels on testing points

    """

    def __init__(self):
        self._centroids = None
        self._class_labels = None  # class labels may not be contiguous indices

    @property
    def centroids(self):
        return self._centroids

    @property
    def class_labels(self):
        return self._class_labels

    def fit(self, xtr, ytr):
        """Compute the average centroid for each class.

        This function should populate the `._centroids` attribute
        with a numpy array of shape (num_classes, num_features).

        input:
            x: set of images (training set, numpy array)
            y: labels (training set, numpy array)

        """
        n_classes = np.unique(ytr).size
        n_feature = xtr.shape[1]
        self._centroids = np.zeros(shape=(n_classes, n_feature))
        for k in range(n_classes):
            xk = xtr[ytr == k]
            self._centroids[k] = np.mean(xk)
        return self

    def predict(self, xts):
        """Predict the class of each input.

        input:
            x: set of images (test set, numpy array)

        output:
            y: labels (numpy array)

        """
        if self._centroids is None:
            raise ValueError("Classifier is not fit. Call fit()!")
        dist = euclidean_distances(xts, self._centroids)
        y_pred = np.argmin(dist, axis=1)
        return y_pred
