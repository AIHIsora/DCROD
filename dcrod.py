"""
Author:
Kangsheng Li <likangsheng125@outlook.com>
https://github.com/AIHIsora/DCROD

Cite:
Kangsheng Li, Xin Gao, Shiyuan Fu, Xinping Diao, Ping Ye, Bing Xue, Jiahao Yu, Zijian Huang,
Robust outlier detection based on the changing rate of directed density ratio,
Expert Systems with Applications,
Volume 207,
2022,
117988,
https://doi.org/10.1016/j.eswa.2022.117988.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.neighbors import NearestNeighbors

class DCROD(NearestNeighbors):
    """
    Unsupervised Outlier Detection using changing rate of
    directed density ratio

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    radius : float, default=1.0
        Range of parameter space to use by default for :meth:`radius_neighbors`
        queries.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    metric : str or callable, default='minkowski'
        the distance metric to use for the tree.  The default metric is
        minkowski, and with p=2 is equivalent to the standard Euclidean
        metric. See the documentation of :class:`DistanceMetric` for a
        list of available metrics.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a :term:`sparse graph`,
        in which case only "nonzero" elements may be considered neighbors.

    p : int, default=2
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    decision_scores_ : ndarray of shape (n_samples,)
        The outlier scores of input samples. The higher, the more abnormal.

    n_neighbors_ : int
        The actual number of neighbors used for :meth:`kneighbors` queries.

    Examples
    --------
    from dcrod import DCROD

    detector = DCROD(n_neighbors=30)
    detector.fit(X)
    y_outlier_score = detect.decision_scores_
    # y_outlier_score is the outlier scores of samples in X.
    # You can use it to calculate AUC, or detect outliers by a threshold theta.

    """
    def __init__(self, *, n_neighbors=5, radius=1.0,
                 algorithm='auto', leaf_size=30, metric='minkowski',
                 p=2, metric_params=None, n_jobs=None):
        super().__init__(
            n_neighbors=n_neighbors,
            radius=radius,
            algorithm=algorithm,
            leaf_size=leaf_size, metric=metric, p=p,
            metric_params=metric_params, n_jobs=n_jobs)

    def fit(self, X, y=None):
        """
        Fit the model using X as training data.
        And predict the outlier score of X.

        :param X:
            Supposed to be DataFrame in pandas.
        :param y:
            Ignored
            Not used, present for API consistency by convention.

        :return:
            self : object
        """
        X = pd.DataFrame(X)
        self.dim = len(X.columns)
        # min-max normalization for data
        for denmi in X.columns:
            xx = normalization(X.loc[:, denmi])
            X[denmi] = xx

        super().fit(X)

        n_samples = self.n_samples_fit_
        if self.n_neighbors > n_samples:
            warnings.warn("n_neighbors (%s) is greater than the "
                          "total number of samples (%s). n_neighbors "
                          "will be set to (n_samples - 1) for estimation."
                          % (self.n_neighbors, n_samples))
        self.n_neighbors_ = max(1, min(self.n_neighbors, n_samples - 1))

        self._distances_fit_X_, self._neighbors_indices_fit_X_ = self.kneighbors(
            n_neighbors=self.n_neighbors_)

        # calculating the bandwidth of Gaussian kernel
        h = np.mean(self._distances_fit_X_)

        # get the kNN Graph
        kNN_G = self.kneighbors_graph(mode='connectivity')
        # get the reverse kNN Graph
        rNN_G = kNN_G.getH()
        # get the extended kNN Graph
        krNN_G = kNN_G + rNN_G

        self.rho = list()
        self._neighbors_indices_krNN = list()

        for i in range(0, len(X)):
            krNN_i = krNN_G[i].nonzero()[1]
            distlist = np.linalg.norm(self._fit_X[i] - self._fit_X[krNN_i], axis=1)
            krNN_i = krNN_i[np.argsort(distlist)]
            self._neighbors_indices_krNN.append(krNN_i)
            # calculate the local density
            self.rho.append(np.mean(np.exp(-distlist**2/(2*h**2))))

        self.rho = np.array(self.rho)
        # calculate the DCR score
        self.DCR = self._density_changing_rate(np.array(X), self.rho, self._neighbors_indices_fit_X_)

        self.decision_scores_ = self.DCR

        return self

    def _density_changing_rate(self, X, all_rol, neighbor_indices):
        """

        :param X:
            The input data X.
        :param all_rol:
            The density calculated before.
        :param neighbor_indices:
            The neighbor_indices of all samples.

        :return:
            DCR scores of X.
        """
        neg = X[neighbor_indices]
        Xneg = X[:, np.newaxis, :]

        dire = neg - Xneg
        rolneg = all_rol[neighbor_indices]
        all_rol = all_rol + 1e-5
        rateofrol = rolneg / all_rol[:, np.newaxis]

        DRD = dire * rateofrol[:, :, np.newaxis]
        DCR = np.zeros(len(X))

        for tempk in range(1, self.n_neighbors_ - 1):
            DRDk = np.sum(DRD[:, 0:tempk, :], axis=1)
            DRDk1 = np.sum(DRD[:, 0:(tempk+1), :], axis=1)
            deltaDRD = abs(np.linalg.norm(DRDk1, axis=1)-np.linalg.norm(DRDk, axis=1))
            DCR = DCR + deltaDRD
        return DCR


# min-max normalization function
def normalization(data):
    _range = max(data) - min(data)
    if _range != 0:
        return (data - min(data))/_range
    else:
        return np.zeros(len(data))
