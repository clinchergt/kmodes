# cython: language_level=3
# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False


import numpy as np
cimport numpy as np
from .util cimport _util
from .util cimport _dissim


ctypedef struct ClustInfo:
    int clust
    double min_dist


cpdef ClustInfo _get_clust(double[:] x_num,
                           double[:, :] centroids_num,
                           long[:] x_cat,
                           long[:, :] centroids_cat,
                           int num_clusters,
                           double gamma):
    cdef:
        int iclust, clust, b
        double curr_dist, a
        ClustInfo c

    c.min_dist = 999999
    for iclust in range(num_clusters):
        a = _dissim._euclidean_dissim(centroids_num[iclust], x_num)
        b = _dissim._matching_dissim(centroids_cat[iclust], x_cat)
        curr_dist = a + gamma * b

        if curr_dist < c.min_dist:
            c.min_dist = curr_dist
            c.clust = iclust

    return c


def _init_centroids(np.ndarray[long, ndim=2, mode='c'] X,
                    np.ndarray[long, ndim=2, mode='c'] centroids):
    cdef:
        long iclust, ipoint, ipoint_centroid

        long[:] _dissims = np.zeros(X.shape[0], dtype='int64')

        long[:, :] _X = X
        long[:, :] _centroids = centroids

        np.ndarray[long, ndim=1, mode='c'] ndx, dissims

    dissims = np.asarray(_dissims)

    for iclust in range(centroids.shape[0]):
        ipoint_centroid = 0

        for ipoint in range(X.shape[0]):
            _dissims[ipoint] = _dissim._matching_dissim(_X[ipoint], _centroids[iclust])

        ndx = np.argsort(dissims)

        while ipoint_centroid + 1 < X.shape[0] and np.all(X[ndx[ipoint_centroid]] == centroids, axis=1).any():
            ipoint_centroid += 1
        centroids[iclust] = X[ndx[ipoint_centroid]]

    return centroids


def _labels_cost(np.ndarray[double, ndim=2, mode='c'] Xnum,
                 np.ndarray[long, ndim=2, mode='c'] Xcat,
                 centroids,
                 double gamma):
    """Calculate labels and cost function given a matrix of points and
    a list of centroids for the k-prototypes algorithm.
    """
    cdef:
        int ipoint, clust
        int n_points = Xnum.shape[0]
        int n_clusters = len(centroids[0])

        double min_dist
        double cost = 0.0
        double[:, :] _Xnum = Xnum
        double[:] x_num
        double[:, :] _centroids_num = centroids[0]

        long[:, :] _Xcat = Xcat
        long[:] x_cat
        long[:, :] _centroids_cat = centroids[1]

        ClustInfo c

    labels = np.empty(n_points, dtype=np.uint8)
    for ipoint in range(n_points):
        x_num = _Xnum[ipoint]
        x_cat = _Xcat[ipoint]
        c = _get_clust(x_num, _centroids_num, x_cat, _centroids_cat, n_clusters, gamma)

        labels[ipoint] = c.clust
        cost += c.min_dist

    return labels, cost


def _k_prototypes_iter(np.ndarray[double, ndim=2, mode='c'] Xnum,
                       np.ndarray[long, ndim=2, mode='c'] Xcat,
                       centroids,
                       np.ndarray[double, ndim=2, mode='c'] cl_attr_sum,
                       np.ndarray[long, ndim=1, mode='c'] cl_memb_sum,
                       np.ndarray[long, ndim=2, mode='c'] cl_attr_freq,
                       np.ndarray[long, ndim=1, mode='c'] cat_offsets,
                       np.ndarray[long, ndim=1, mode='c'] membship,
                       double gamma):
    """Single iteration of the k-prototypes algorithm"""
    cdef:
        int ipoint, clust, curc, iattr
        int n_points = Xnum.shape[0]
        int n_clusters = centroids[0].shape[0]
        int n_num_attr = Xnum.shape[1]
        int moves = 0

        double min_dist
        double[:, :] _Xnum = Xnum
        double[:] x_num
        double[:, :] _cl_attr_sum = cl_attr_sum
        double[:, :] _centroids_num = centroids[0]

        long[:, :] _Xcat = Xcat
        long[:] x_cat
        long[:] _membship = membship
        long[:] _cl_memb_sum = cl_memb_sum
        long[:, :] _centroids_cat = centroids[1]

        ClustInfo c

    for ipoint in range(n_points):
        # Get numeric and categorical attribute values for the current point.
        x_num = _Xnum[ipoint]
        x_cat = _Xcat[ipoint]

        old_clust = _membship[ipoint]
        c = _get_clust(x_num, _centroids_num, x_cat, _centroids_cat, n_clusters, gamma)
        clust = c.clust

        if clust == old_clust:
            # Point is already in its right place.
            continue

        # Move point, and update old/new cluster frequencies and centroids.
        moves += 1

        # Note that membship gets updated by kmodes.move_point_cat.
        # move_point_num only updates things specific to the k-means part.
        _util._move_point_num(
            x_num, clust, old_clust, _cl_attr_sum, _cl_memb_sum
        )
        _util._move_point_cat(
            x_cat, ipoint, clust, old_clust,
            cl_attr_freq, cat_offsets, _membship, _centroids_cat
        )

        # Update old and new centroids for numerical attributes using
        # the means and sums of all values
        for iattr in range(n_num_attr):
            for curc in (clust, old_clust):
                if cl_memb_sum[curc]:
                    _centroids_num[curc, iattr] = cl_attr_sum[curc, iattr] / cl_memb_sum[curc]
                else:
                    _centroids_num[curc, iattr] = 0.

        # In case of an empty cluster, reinitialize with a random point
        # from largest cluster.
        if not cl_memb_sum[old_clust]:
            from_clust = cl_memb_sum.argmax()
            choices = [ii for ii in range(n_points) if _membship[ii] == from_clust]
            rindx = np.random.choice(choices)

            _util._move_point_num(
                Xnum[rindx], old_clust, from_clust, cl_attr_sum, cl_memb_sum
            )
            _util._move_point_cat(
                Xcat[rindx], rindx, old_clust, from_clust,
                cl_attr_freq, cat_offsets, membship, _centroids_cat
            )

    return centroids, moves
