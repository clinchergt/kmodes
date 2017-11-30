"""
K-prototypes clustering for mixed categorical and numerical data
"""

# pylint: disable=super-on-old-class

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils.validation import check_array

from .util import encode_features, get_unique_rows, decode_centroids
from .util.init import init_cao, init_huang
from . import _k_proto
from .util import _util


# Number of tries we give the initialization methods to find non-empty
# clusters before we switch to random initialization.
MAX_INIT_TRIES = 10
# Number of tries we give the initialization before we raise an
# initialization error.
RAISE_INIT_TRIES = 100


def _split_num_cat(X, categorical):
    """Extract numerical and categorical columns.
    Convert to numpy arrays, if needed.

    :param X: Feature matrix
    :param categorical: Indices of categorical columns
    """
    Xnum = np.ascontiguousarray(X[:, [ii for ii in range(X.shape[1])
                               if ii not in categorical]]).astype(np.float64)
    Xcat = np.ascontiguousarray(X[:, categorical])
    return Xnum, Xcat


def build_cat_index(x):
    cat_offsets = np.zeros(x.shape[1], dtype=np.int64)
    for cat in range(x.shape[1] - 1):
        cat_offsets[cat + 1] = cat_offsets[cat] + len(np.unique(x[:,cat]))
    return cat_offsets


def k_prototypes(X, categorical, n_clusters, max_iter,
                 gamma, init, n_init, verbose):
    """k-prototypes algorithm"""

    if sparse.issparse(X):
        raise TypeError("k-prototypes does not support sparse data.")

    # Convert pandas objects to numpy arrays.
    if 'pandas' in str(X.__class__):
        X = X.values

    if categorical is None or not categorical:
        raise NotImplementedError(
            "No categorical data selected, effectively doing k-means. "
            "Present a list of categorical columns, or use scikit-learn's "
            "KMeans instead."
        )
    if isinstance(categorical, int):
        categorical = [categorical]
    assert len(categorical) != X.shape[1], \
        "All columns are categorical, use k-modes instead of k-prototypes."
    assert max(categorical) < X.shape[1], \
        "Categorical index larger than number of columns."

    ncatattrs = len(categorical)
    nnumattrs = X.shape[1] - ncatattrs
    n_points = X.shape[0]
    assert n_clusters <= n_points, "Cannot have more clusters ({}) " \
                                   "than data points ({}).".format(n_clusters, n_points)

    Xnum, Xcat = _split_num_cat(X, categorical)
    Xnum, Xcat = check_array(Xnum), check_array(Xcat, dtype=None)

    # Convert the categorical values in Xcat to integers for speed.
    # Based on the unique values in Xcat, we can make a mapping to achieve this.
    Xcat, enc_map = encode_features(Xcat)
    cat_offsets = build_cat_index(Xcat)

    # Are there more n_clusters than unique rows? Then set the unique
    # rows as initial values and skip iteration.
    unique = get_unique_rows(X)
    n_unique = unique.shape[0]
    if n_unique <= n_clusters:
        max_iter = 0
        n_init = 1
        n_clusters = n_unique
        init = list(_split_num_cat(unique, categorical))
        init[1], _ = encode_features(init[1], enc_map)

    # Estimate a good value for gamma, which determines the weighing of
    # categorical values in clusters (see Huang [1997]).
    if gamma is None:
        gamma = 0.5 * Xnum.std()

    all_centroids = []
    all_labels = []
    all_costs = []
    all_n_iters = []
    for init_no in range(n_init):

        # For numerical part of initialization, we don't have a guarantee
        # that there is not an empty cluster, so we need to retry until
        # there is none.
        init_tries = 0
        while True:
            init_tries += 1
            # _____ INIT _____
            if verbose:
                print("Init: initializing centroids")
            if isinstance(init, str) and init.lower() == 'huang':
                centroids = init_huang(Xcat, n_clusters)
            elif isinstance(init, str) and init.lower() == 'cao':
                centroids = init_cao(Xcat, n_clusters)
            elif isinstance(init, str) and init.lower() == 'random':
                seeds = np.random.choice(range(n_points), n_clusters)
                centroids = Xcat[seeds]
            elif isinstance(init, list):
                # Make sure inits are 2D arrays.
                init = [np.atleast_2d(cur_init).T if len(cur_init.shape) == 1
                        else cur_init
                        for cur_init in init]
                assert init[0].shape[0] == n_clusters, \
                    "Wrong number of initial numerical centroids in init " \
                    "({}, should be {}).".format(init[0].shape[0], n_clusters)
                assert init[0].shape[1] == nnumattrs, \
                    "Wrong number of numerical attributes in init ({}, should be {})."\
                    .format(init[0].shape[1], nnumattrs)
                assert init[1].shape[0] == n_clusters, \
                    "Wrong number of initial categorical centroids in init ({}, " \
                    "should be {}).".format(init[1].shape[0], n_clusters)
                assert init[1].shape[1] == ncatattrs, \
                    "Wrong number of categorical attributes in init ({}, should be {})."\
                    .format(init[1].shape[1], ncatattrs)
                centroids = [np.asarray(init[0], dtype=np.float64),
                             np.asarray(init[1], dtype=np.uint8)]
            else:
                raise NotImplementedError("Initialization method not supported.")

            if not isinstance(init, list):
                # Numerical is initialized by drawing from normal distribution,
                # categorical following the k-modes methods.
                meanx = np.mean(Xnum, axis=0)
                stdx = np.std(Xnum, axis=0)
                centroids = (
                    meanx + np.random.randn(n_clusters, nnumattrs) * stdx,
                    centroids
                )

            if verbose:
                print("Init: initializing clusters")
            membship = np.zeros(n_points, dtype=np.int64)
            # Keep track of the sum of attribute values per cluster so that we
            # can do k-means on the numerical attributes.
            cl_attr_sum = np.zeros((n_clusters, nnumattrs), dtype=np.float64)
            # Same for the membership sum per cluster
            cl_memb_sum = np.zeros(n_clusters, dtype=np.int64)
            # cl_attr_freq is a list of lists with dictionaries that contain
            # the frequencies of values per cluster and attribute.
            cl_attr_freq = np.zeros((n_clusters, cat_offsets.sum()), dtype=np.int64)
            for ipoint in range(n_points):
                # Initial assignment to clusters
                c = _k_proto._get_clust(Xnum[ipoint], centroids[0], Xcat[ipoint],
                                        centroids[1], centroids[1].shape[0], gamma)
                clust = c['clust']
                membship[ipoint] = clust
                cl_memb_sum[clust] += 1
                # Count attribute values per cluster.
                for iattr, curattr in enumerate(Xnum[ipoint]):
                    cl_attr_sum[clust, iattr] += curattr
                for iattr, curattr in enumerate(Xcat[ipoint]):
                    # print(iattr, curattr)
                    cl_attr_freq[clust, cat_offsets[iattr] + curattr] += 1

            # If no empty clusters, then consider initialization finalized.
            if cl_memb_sum.all():
                break

            if init_tries == MAX_INIT_TRIES:
                # Could not get rid of empty clusters. Randomly
                # initialize instead.
                init = 'random'
            elif init_tries == RAISE_INIT_TRIES:
                raise ValueError(
                    "Clustering algorithm could not initialize. "
                    "Consider assigning the initial clusters manually."
                )

        # Perform an initial centroid update.
        for ik in range(n_clusters):
            for iattr in range(nnumattrs):
                centroids[0][ik, iattr] = cl_attr_sum[ik, iattr] / cl_memb_sum[ik]
            for iattr in range(ncatattrs):
                centroids[1][ik, iattr] = _util._get_max_value_key(cl_attr_freq, cat_offsets, ik, iattr)

        # _____ ITERATION _____
        if verbose:
            print("Starting iterations...")
        itr = 0
        converged = False
        cost = np.Inf
        # centroids[1] = centroids[1].astype('int64')
        while itr <= max_iter and not converged:
            itr += 1

            centroids, moves = _k_proto._k_prototypes_iter(Xnum, Xcat, centroids,
                                                  cl_attr_sum, cl_memb_sum, cl_attr_freq, cat_offsets,
                                                  membship, gamma)

            converged = (moves == 0)

            # TODO: Verify why we need to calculate labels and cost for every iteration as opposed to every run
            # labels, ncost = _k_proto._labels_cost(Xnum, Xcat, centroids, gamma)
            #
            # converged = (moves == 0) or (ncost >= cost)
            # cost = ncost
            # if verbose:
            #     print("Run: {}, iteration: {}/{}, moves: {}, ncost: {}"
            #           .format(init_no + 1, itr, max_iter, moves, ncost))

        labels, cost = _k_proto._labels_cost(Xnum, Xcat, centroids, gamma)

        # Store results of current run.
        all_centroids.append(centroids)
        all_labels.append(labels)
        all_costs.append(cost)
        all_n_iters.append(itr)

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
                 verbose=0):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init
        self.n_init = n_init
        self.verbose = verbose
        if ((isinstance(self.init, str) and self.init == 'Cao') or
                hasattr(self.init, '__array__')) and self.n_init > 1:
            if self.verbose:
                print("Initialization method and algorithm are deterministic. "
                      "Setting n_init to 1.")
            self.n_init = 1

        self.gamma = gamma


    def fit(self, X, y=None, categorical=None):
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
                                                    categorical,
                                                    self.n_clusters,
                                                    self.max_iter,
                                                    self.gamma,
                                                    self.init,
                                                    self.n_init,
                                                    self.verbose)
        return self

    def predict(self, X, categorical=None):
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

        # Convert pandas objects to numpy arrays.
        if 'pandas' in str(X.__class__):
            X = X.values

        Xnum, Xcat = _split_num_cat(X, categorical)
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
