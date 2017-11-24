# cython: language_level=3
# cython: profile=True
# cython: boundscheck=True

cimport cython
from cython cimport double, long

import numpy as np
cimport numpy as np

from .util import get_max_value_key


@cython.profile(False)
cdef inline double _euclidean_dissim(double[:] a, double[:] b, int n):
    cdef double result, tmp
    result = 0.0

    cdef int i
    for i in range(n):
        tmp = (a[i] - b[i])
        result += tmp * tmp

    return result


@cython.profile(False)
cdef inline long _matching_dissim(long[:] a, long[:] b, int n):
    cdef long result, tmp
    result = 0

    cdef int i
    for i in range(n):
        tmp = (a[i] != b[i])
        result += tmp

    return result


cdef move_point_num(double[:] point, int to_clust, int from_clust, double[:, :] cl_attr_sum, long[:] cl_memb_sum):
    """Move point between clusters, numerical attributes."""
    cdef:
        int iattr = 0
        double curattr

    # Update sum of attributes in cluster.
    for iattr in range(point.shape[0]):
        curattr = point[iattr]
        cl_attr_sum[to_clust, iattr] += curattr
        cl_attr_sum[from_clust, iattr] -= curattr
    # Update sums of memberships in cluster
    cl_memb_sum[to_clust] += 1
    cl_memb_sum[from_clust] -= 1


def move_point_cat(long[:] point, int ipoint, int to_clust, int from_clust,
                   cl_attr_freq,
                   membship,
                   centroids):
    """Move point between clusters, categorical attributes."""
    cdef:
        int iattr
        double curattr

    membship[to_clust, ipoint] = 1
    membship[from_clust, ipoint] = 0
    # Update frequencies of attributes in cluster.
    for iattr, curattr in enumerate(point):
        to_attr_counts = cl_attr_freq[to_clust][iattr]
        from_attr_counts = cl_attr_freq[from_clust][iattr]

        # Increment the attribute count for the new "to" cluster
        to_attr_counts[curattr] += 1

        current_attribute_value_freq = to_attr_counts[curattr]
        current_centroid_value = centroids[to_clust][iattr]
        current_centroid_freq = to_attr_counts[current_centroid_value]
        if current_centroid_freq < current_attribute_value_freq:
            # We have incremented this value to the new mode. Update the centroid.
            centroids[to_clust][iattr] = curattr

        # Decrement the attribute count for the old "from" cluster
        from_attr_counts[curattr] -= 1

        old_centroid_value = centroids[from_clust][iattr]
        if old_centroid_value == curattr:
            # We have just removed a count from the old centroid value. We need to
            # recalculate the centroid as it may no longer be the maximum
            centroids[from_clust][iattr] = get_max_value_key(from_attr_counts)

    return cl_attr_freq, membship, centroids


cdef _get_clust(double[:] x_num,
                double[:, :] centroids_num,
                long[:] x_cat,
                long[:, :] centroids_cat,
                int num_clusters,
                double gamma):
    cdef int iclust, clust
    cdef double curr_dist, min_dist
    min_dist = 9999999

    cdef double a
    cdef long b
    for iclust in range(num_clusters):
        a = _euclidean_dissim(centroids_num[iclust], x_num, x_num.shape[0])
        b = _matching_dissim(centroids_cat[iclust], x_cat, x_cat.shape[0])
        curr_dist = a + gamma * b

        if curr_dist < min_dist:
            min_dist = curr_dist
            clust = iclust

    return min_dist, clust


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

    labels = np.empty(n_points, dtype=np.uint8)
    for ipoint in range(n_points):
        x_num = _Xnum[ipoint]
        x_cat = _Xcat[ipoint]
        min_dist, clust = _get_clust(x_num, _centroids_num, x_cat, _centroids_cat, n_clusters, gamma)

        labels[ipoint] = clust
        cost += min_dist

    return labels, cost


def _k_prototypes_iter(np.ndarray[double, ndim=2, mode='c'] Xnum,
                       np.ndarray[long, ndim=2, mode='c'] Xcat,
                       centroids,
                       np.ndarray[double, ndim=2, mode='c'] cl_attr_sum,
                       np.ndarray[long, ndim=1, mode='c'] cl_memb_sum,
                       cl_attr_freq,
                       np.ndarray[np.uint8_t, ndim=2, mode='c'] membship,
                       double gamma):
    """Single iteration of the k-prototypes algorithm"""
    cdef:
        int ipoint, clust, curc, iattr
        int n_points = Xnum.shape[0]
        int n_clusters = centroids[0].shape[0]
        int n_num_attr = Xnum.shape[1]

        double min_dist
        double[:, :] _Xnum = Xnum
        double[:] x_num
        double[:, :] _cl_attr_sum = cl_attr_sum
        double[:, :] _centroids_num = centroids[0]

        long[:, :] _Xcat = Xcat
        long[:] x_cat
        long[:] _cl_memb_sum = cl_memb_sum
        long[:, :] _centroids_cat = centroids[1]

    moves = 0

    for ipoint in range(n_points):
        # Get numeric and categorical attribute values for the current point.
        x_num = _Xnum[ipoint]
        x_cat = _Xcat[ipoint]

        _, clust = _get_clust(x_num, _centroids_num, x_cat, _centroids_cat, n_clusters, gamma)

        if membship[clust, ipoint]:
            # Point is already in its right place.
            continue

        # Move point, and update old/new cluster frequencies and centroids.
        moves += 1
        old_clust = np.argwhere(membship[:, ipoint])[0][0]

        # Note that membship gets updated by kmodes.move_point_cat.
        # move_point_num only updates things specific to the k-means part.
        move_point_num(
            x_num, clust, old_clust, _cl_attr_sum, _cl_memb_sum
        )
        cl_attr_freq, membship, _ = move_point_cat(
            x_cat, ipoint, clust, old_clust,
            cl_attr_freq, membship, _centroids_cat
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
            from_clust = membship.sum(axis=1).argmax()
            choices = [ii for ii, ch in enumerate(membship[from_clust, :]) if ch]
            rindx = np.random.choice(choices)

            move_point_num(
                Xnum[rindx], old_clust, from_clust, cl_attr_sum, cl_memb_sum
            )
            cl_attr_freq, membship, _ = move_point_cat(
                Xcat[rindx], rindx, old_clust, from_clust,
                cl_attr_freq, membship, _centroids_cat
            )

    return centroids, moves
