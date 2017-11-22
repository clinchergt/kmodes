# cython: profile=False
# cython: language_level=3
cimport cython
from cython cimport double, long

import numpy as np
cimport numpy as np

from . import kmodes


cdef inline double _euclidean_dissim(double[:] a, double[:] b, int n):
    cdef double result, tmp
    result = 0.0

    cdef int i
    for i in range(n):
        tmp = (a[i] - b[i])
        result += tmp * tmp
    return result


cdef inline long _matching_dissim(long[:] a, long[:] b, int n):
    cdef long result, tmp
    result = 0

    cdef int i
    for i in range(n):
        tmp = (a[i] != b[i])
        result += tmp

    return result


cdef inline int _get_clust(double[:] x_num,
                           double[:, :] centroids_num,
                           long[:] x_cat,
                           long[:, :] centroids_cat,
                           int num_clusters,
                    double gamma):
    cdef int iclust, clust
    cdef double curr_dist, min_dist
    min_dist = 9999999

    cdef double a, b
    for iclust in range(num_clusters):
        a = _euclidean_dissim(centroids_num[iclust], x_num, x_num.shape[0])
        b = _matching_dissim(centroids_cat[iclust], x_cat, x_cat.shape[0])
        curr_dist = a + gamma * b

        if curr_dist < min_dist:
            min_dist = curr_dist
            clust = iclust

    return clust


def _labels_cost(np.ndarray[double, ndim=2, mode='c'] Xnum,
                       np.ndarray[long, ndim=2, mode='c'] Xcat,
                       centroids,
                       membship,
                       double gamma):
    """Calculate labels and cost function given a matrix of points and
    a list of centroids for the k-prototypes algorithm.
    """

    n_points = Xnum.shape[0]
    cdef double[:, :] _Xnum = Xnum
    cdef long[:, :] _Xcat = Xcat

    # cdef int moves = 0
    cdef int ipoint, iclust
    cdef int clust
    cdef double curr_dist
    cdef double min_dist
    cdef int num_clusters = len(centroids[0])
    cdef double a, b

    cdef double[:, :] centroids_num = centroids[0]
    cdef long[:, :] centroids_cat = centroids[1].astype('int64')
    cdef double[:] x_num
    cdef long[:] x_cat

    cdef double cost = 0.0
    # n_points = 6
    labels = np.empty(n_points, dtype=np.uint8)
    for ipoint in range(n_points):
        x_num = _Xnum[ipoint]
        x_cat = _Xcat[ipoint]
        min_dist = 99999999999

        for iclust in range(num_clusters):
            a = _euclidean_dissim(centroids_num[iclust], x_num, x_num.shape[0])
            b = _matching_dissim(centroids_cat[iclust], x_cat, x_cat.shape[0])
            curr_dist = a + gamma * b

            if curr_dist < min_dist:
                min_dist = curr_dist
                clust = iclust
        #     print(curr_dist, end=" ")

        # print("\n=============values=================")
        labels[ipoint] = clust
        cost += min_dist
        # print(clust)
        # print(cost)
        # print("-------------------------------------")


    return labels, cost


def move_point_num(point, to_clust, from_clust, cl_attr_sum, cl_memb_sum):
    """Move point between clusters, numerical attributes."""
    # Update sum of attributes in cluster.
    for iattr, curattr in enumerate(point):
        cl_attr_sum[to_clust][iattr] += curattr
        cl_attr_sum[from_clust][iattr] -= curattr
    # Update sums of memberships in cluster
    cl_memb_sum[to_clust] += 1
    cl_memb_sum[from_clust] -= 1
    return cl_attr_sum, cl_memb_sum


def _k_prototypes_iter(np.ndarray[double, ndim=2, mode='c'] Xnum,
                       np.ndarray[long, ndim=2, mode='c'] Xcat,
                       centroids,
                       cl_attr_sum,
                       cl_memb_sum,
                       cl_attr_freq,
                       membship,
                       double gamma):
    """Single iteration of the k-prototypes algorithm"""
    cdef double[:, :] _Xnum = Xnum
    cdef long[:, :] _Xcat = Xcat

    # cdef int moves = 0
    cdef int ipoint, iclust
    cdef int clust
    cdef double curr_dist
    cdef double min_dist
    cdef int num_clusters = len(centroids[0])

    cdef double[:, :] centroids_num = centroids[0]
    cdef long[:, :] centroids_cat = centroids[1].astype('int64')
    cdef double[:] x_num
    cdef long[:] x_cat

    cdef double a, b

    moves = 0
    # print(centroids_cat)
    for ipoint in range(Xnum.shape[0]):
        x_num = _Xnum[ipoint]
        x_cat = _Xcat[ipoint]

        clust = _get_clust(x_num, centroids_num, x_cat, centroids_cat, num_clusters, gamma)

        if membship[clust, ipoint]:
            # Point is already in its right place.
            continue

        # Move point, and update old/new cluster frequencies and centroids.
        moves += 1
        old_clust = np.argwhere(membship[:, ipoint])[0][0]

        # Note that membship gets updated by kmodes.move_point_cat.
        # move_point_num only updates things specific to the k-means part.
        cl_attr_sum, cl_memb_sum = move_point_num(
            Xnum[ipoint], clust, old_clust, cl_attr_sum, cl_memb_sum
        )
        cl_attr_freq, membship, centroids[1] = kmodes.move_point_cat(
            Xcat[ipoint], ipoint, clust, old_clust,
            cl_attr_freq, membship, centroids[1]
        )

        # Update old and new centroids for numerical attributes using
        # the means and sums of all values
        for iattr in range(len(Xnum[ipoint])):
            for curc in (clust, old_clust):
                if cl_memb_sum[curc]:
                    centroids[0][curc, iattr] = cl_attr_sum[curc, iattr] / cl_memb_sum[curc]
                else:
                    centroids[0][curc, iattr] = 0.

        # In case of an empty cluster, reinitialize with a random point
        # from largest cluster.
        if not cl_memb_sum[old_clust]:
            from_clust = membship.sum(axis=1).argmax()
            choices = [ii for ii, ch in enumerate(membship[from_clust, :]) if ch]
            rindx = np.random.choice(choices)

            cl_attr_sum, cl_memb_sum = move_point_num(
                Xnum[rindx], old_clust, from_clust, cl_attr_sum, cl_memb_sum
            )
            cl_attr_freq, membship, centroids[1] = kmodes.move_point_cat(
                Xcat[rindx], rindx, old_clust, from_clust,
                cl_attr_freq, membship, centroids[1]
            )

    # print(moves)
    return centroids, moves
