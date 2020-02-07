"""
K-prototypes clustering for mixed categorical and numerical data
"""

# pylint: disable=super-on-old-class

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array

from .util import encode_features, get_unique_rows, decode_centroids
from . import _k_proto
from .k_proto_proc import k_prototypes_single
from .k_proto_proc import preprocess
from .util import _util

from joblib import Parallel, delayed

def k_prototypes(X, n_clusters, max_iter,
                 gamma, gamma_method, init,
                 n_init, verbose, n_jobs, random_state, enc_map):
    """k-prototypes algorithm"""

    if sparse.issparse(X):
        raise TypeError("k-prototypes does not support sparse data.")

    # FIX: allow to pass a random state
    random_state = check_random_state(random_state)

    Xnum, Xcat = preprocess(X)
    Xnum, Xcat = check_array(Xnum), check_array(Xcat, dtype=None)

    ncatattrs = Xcat.shape[1]
    nnumattrs = Xnum.shape[1]

    if ncatattrs == 0:
        ValueError("No categorical attributes in input data, use K-Means instead.")

    n_points = X.shape[0]
    assert n_clusters <= n_points, "Cannot have more clusters ({}) " \
                                   "than data points ({}).".format(n_clusters, n_points)

    # Convert the categorical values in Xcat to integers for speed.
    # Based on the unique values in Xcat, we can make a mapping to achieve this.

    Xcat, enc_map = encode_features(Xcat, enc_map)

    # TODO: Fix the below.
    # Are there more n_clusters than unique rows? Then set the unique
    # rows as initial values and skip iteration.
    # unique = get_unique_rows(X)
    # n_unique = unique.shape[0]
    # if n_unique <= n_clusters:
    #     max_iter = 0
    #     n_init = 1
    #     n_clusters = n_unique
    #     init = list(_split_num_cat(unique, categorical))
    #     init[1], _ = encode_features(init[1], enc_map)

    # Estimate a good value for gamma, which determines the weighing of
    # categorical values in clusters (see clustMixType R package docs and
    # Huang [1997]).
    if gamma is None:
        if gamma_method.lower() == 'clustmixtype':
            import functools
            # Transpose in order to get means and counts on a per column basis
            counts = list(map(functools.partial(np.unique, return_counts=True), Xcat.T))
            scored_sum = lambda x: 1 - sum((x[1] / sum(x[1]))**2)
            vcat = np.mean(list(map(scored_sum, counts)))
            vnum = np.mean(list(map(np.var, Xnum.T)))
            gamma = vnum / vcat
        # 'default' would probably be better here and disallow every other value(?)
        else:
            gamma = 0.5 * Xnum.std()

    all_centroids = []
    all_labels = []
    all_costs = []
    all_n_iters = []

    results = []
    seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)

    if n_jobs == 1:
        for init_no in range(n_init):
            results.append(k_prototypes_single(Xnum, Xcat, nnumattrs, ncatattrs,
                                               n_clusters, n_points, max_iter, gamma, init,
                                               init_no, random_state, verbose))
    else:
        # Assignation instead of append cause no longer returning atomic values
        results = Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=0)(
            delayed(k_prototypes_single)(Xnum, Xcat, nnumattrs, ncatattrs,
                                         n_clusters, n_points, max_iter, gamma, init,
                                         init_no, seed, verbose)
            for init_no, seed in enumerate(seeds))

    all_centroids, all_labels, all_costs, all_n_iters = zip(*results)

    best = np.argmin(all_costs)
    if n_init > 1 and verbose:
        print("Best run was number {}".format(best + 1))

    # Note: return gamma in case it was automatically determined.
    return all_centroids[best], enc_map, all_labels[best], \
        all_costs[best], all_n_iters[best], gamma


class KPrototypes(BaseEstimator, ClusterMixin, TransformerMixin):
    """k-protoypes clustering algorithm for mixed numerical/categorical data.

    Parameters
    -----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, default: 300
        Maximum number of iterations of the k-modes algorithm for a
        single run.

    n_init : int, default: 10
        Number of time the k-modes algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of cost.

    init : {'Huang', 'Cao', 'random' or a list of ndarrays}, default: 'Cao'
        Method for initialization:
        'Huang': Method in Huang [1997, 1998]
        'Cao': Method in Cao et al. [2009]
        'random': choose 'n_clusters' observations (rows) at random from
        data for the initial centroids.
        If a list of ndarrays is passed, it should be of length 2, with
        shapes (n_clusters, n_features) for numerical and categorical
        data respectively. These are the initial centroids.

    gamma : float, default: None
        Weighing factor that determines relative importance of numerical vs.
        categorical attributes (see discussion in Huang [1997]). By default,
        automatically calculated from data.

    verbose : integer, optional
        Verbosity mode.

    Attributes
    ----------
    centroids_ : array, [n_clusters, n_features]
        Categories of cluster centroids

    labels_ :
        Labels of each point

    cost_ : float
        Clustering cost, defined as the sum distance of all points to
        their respective cluster centroids.

    n_iter_ : int
        The number of iterations the algorithm ran for.

    gamma : float
        The (potentially calculated) weighing factor.

    Notes
    -----
    See:
    Huang, Z.: Extensions to the k-modes algorithm for clustering large
    data sets with categorical values, Data Mining and Knowledge
    Discovery 2(3), 1998.

    """

    def __init__(self, n_clusters=8, max_iter=100, init='Huang', n_init=10, gamma=None,
                 verbose=0, n_jobs=-1, random_state=42, gamma_method='clustmixtype'):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init
        self.n_init = n_init
        self.verbose = verbose
        self.gamma_method = gamma_method
        self.random_state = random_state
        if ((isinstance(self.init, str) and self.init == 'Cao') or
                hasattr(self.init, '__array__')) and self.n_init > 1:
            if self.verbose:
                print("Initialization method and algorithm are deterministic. "
                      "Setting n_init to 1.")
            self.n_init = 1

        self.gamma = gamma
        self.n_jobs = n_jobs


    def fit(self, X, y=None, enc_map=None):
        """Compute k-prototypes clustering.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
        categorical : Index of columns that contain categorical data
        """

        # If self.gamma is None, gamma will be automatically determined from
        # the data. The function below returns its value.
        self._enc_centroids, self._enc_map, self.labels_, self.cost_,\
            self.n_iter_, self.gamma = k_prototypes(X,
                                                    self.n_clusters,
                                                    self.max_iter,
                                                    self.gamma,
                                                    self.gamma_method,
                                                    self.init,
                                                    self.n_init,
                                                    self.verbose,
                                                    self.n_jobs,
                                                    self.random_state,
                                                    enc_map)
        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.
        categorical : Index of columns that contain categorical data

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        assert hasattr(self, '_enc_centroids'), "Model not yet fitted."

        Xnum, Xcat = preprocess(X)
        Xnum, Xcat = check_array(Xnum), check_array(Xcat, dtype=None)
        Xcat, _ = encode_features(Xcat, enc_map=self._enc_map)
        return _k_proto._labels_cost(Xnum, Xcat, self._enc_centroids, self.gamma)[0]

    @property
    def centroids_(self):
        if hasattr(self, '_enc_centroids'):
            return (
                self._enc_centroids[0],
                decode_centroids(self._enc_centroids[1], self._enc_map)
            )
        else:
            raise AttributeError("'{}' object has no attribute 'centroids_' "
                                 "because the model is not yet fitted.")

