import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import stable_cumsum

from .util import encode_features, get_unique_rows, decode_centroids
from .util.init import init_cao, init_huang
from . import _k_proto
from .util import _util

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


def cat_sizes(X):
    cat_sizes = np.zeros(X.shape[1], dtype=np.int64)
    for cat in range(X.shape[1]):
        cat_sizes[cat] = np.max(X[:, cat]) + 1
    return cat_sizes


def cat_offsets(cat_sizes):
    cat_cumsum = cat_sizes.cumsum()
    cat_offsets = np.zeros(cat_sizes.shape[0], dtype=np.int64)
    for cat in range(cat_sizes.shape[0] - 1):
        cat_offsets[cat + 1] = cat_cumsum[cat]
    return cat_offsets


def cat_index(data):
    cat_cols = data.select_dtypes("category")
    return [data.columns.get_loc(c) for c in cat_cols]


def preprocess(data):
    if type(data) == tuple:
        return data

    if type(data) == dict:
        if not ('num' in data and 'cat' in data):
            raise ValueError("Input data in dict form should contain a 'num' and 'cat' key,\
             pointing to numerical and categorical data, respectively.")
        return data['num'], data['cat']

    if 'pandas' in str(data.__class__):
        categorical = cat_index(data)
        data = data.values
        return _split_num_cat(data, categorical)

    raise ValueError("Input data should be in the form of a tuple, dictionary or Pandas DataFrame.")

def k_prototypes_single(Xnum, Xcat, nnumattrs, ncatattrs,
                        n_clusters, n_points, max_iter, gamma, init,
                        init_no, random_state, verbose):
    # For numerical part of initialization, we don't have a guarantee
    # that there is not an empty cluster, so we need to retry until
    # there is none.
    init_tries = 0
    sizes = cat_sizes(Xcat)
    offsets = cat_offsets(sizes)

    random_state = check_random_state(random_state)

    while True:
        init_tries += 1
        # _____ INIT _____
        if verbose:
            print("Init: initializing centroids")
        if isinstance(init, str) and init.lower() == 'huang':
            centroids = init_huang(Xcat, n_clusters, random_state)
        elif isinstance(init, str) and init.lower() == 'cao':
            centroids = init_cao(Xcat, n_clusters)
        elif isinstance(init, str) and init.lower() == 'random':
            seeds = random_state.choice(range(n_points), n_clusters)
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

            # Initialize instead using kmeans++ which makes kmeans
            # more efficient and gets rid of the initialization problem
            # where empty clusters result

            # Faster than np.sum(Xnum * Xnum, axis=1) and uses up less memory
            # less readable tho

            n_samples, n_features = Xnum.shape
            centers = np.empty((n_clusters, n_features), dtype=Xnum.dtype)
            row_norms = np.einsum('ij,ij->i', Xnum, Xnum)

            # Recommended by Arthur and Vassilvitskii
            n_local_trials = 2 + int(np.log(n_clusters))

            # kmeans++ picks first centroid at random from the data
            center_id = random_state.randint(n_samples)
            centers[0] = Xnum[center_id]

            # kmeans++ uses distances from the current center to the
            # rest of the points in the data to pick the next center
            closest_dist_sq = euclidean_distances(
                    centers[0, np.newaxis], Xnum, Y_norm_squared=row_norms,
                    squared=True)
            current_pot = closest_dist_sq.sum()

            # Pick the remaining n_clusters-1 points
            for c in range(1, n_clusters):
                # Choose center candidates by sampling with probability proportional
                # to the squared distance to the closest existing center
                rand_vals = random_state.random_sample(n_local_trials) * current_pot
                candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                                rand_vals)
                # XXX: numerical imprecision can result in a candidate_id out of range
                np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                        out=candidate_ids)

                # Compute distances to center candidates
                distance_to_candidates = euclidean_distances(
                    Xnum[candidate_ids], Xnum, Y_norm_squared=row_norms, squared=True)

                # update closest distances squared and potential for each candidate
                np.minimum(closest_dist_sq, distance_to_candidates,
                           out=distance_to_candidates)
                candidates_pot = distance_to_candidates.sum(axis=1)

                # Decide which candidate is the best
                best_candidate = np.argmin(candidates_pot)
                current_pot = candidates_pot[best_candidate]
                closest_dist_sq = distance_to_candidates[best_candidate]
                best_candidate = candidate_ids[best_candidate]

                # Permanently add best center candidate found in local tries
                centers[c] = Xnum[best_candidate]

            centroids = (centers, centroids)

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
        cl_attr_freq = np.zeros((n_clusters, sizes.sum()), dtype=np.int64)
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
                offset = offsets[iattr]
                cl_attr_freq[clust, offset + curattr] += 1

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
            centroids[1][ik, iattr] = _util._get_max_value_key(cl_attr_freq, offsets, ik, iattr)

    # _____ ITERATION _____
    if verbose:
        print("Starting iterations...")
    itr = 0
    converged = False
    cost = np.Inf
    while itr <= max_iter and not converged:
        itr += 1

        centroids, moves = _k_proto._k_prototypes_iter(Xnum, Xcat, centroids,
                                              cl_attr_sum, cl_memb_sum, cl_attr_freq, offsets,
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

    if not converged and verbose:
        print("Did not converge in the specified amount of iterations.")

    labels, cost = _k_proto._labels_cost(Xnum, Xcat, centroids, gamma)

    # Return results to be concatenated with the rest
    return centroids, labels, cost, itr
