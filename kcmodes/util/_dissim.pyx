# cython: language_level=3
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport cython
cimport numpy as np

@cython.profile(False)
cdef double _euclidean_dissim(double[:] centroid, double[:] x):
    cdef double result, tmp
    result = 0.0

    cdef int32_t i
    for i in range(x.shape[0]):
        tmp = (centroid[i] - x[i])
        result += tmp * tmp

    return result


@cython.profile(False)
cdef int32_t _matching_dissim(int64_t[:] centroid, int64_t[:] x):
    cdef int32_t result, tmp
    result = 0

    cdef int32_t i
    for i in range(x.shape[0]):
        tmp = (centroid[i] != x[i])
        result += tmp

    return result

cpdef pairwise_dissim(double[:, :] x_num, int64_t[:, :] x_cat, double[:, :] y_num, int64_t[:, :] y_cat, double gamma):
    result = np.zeros((x_num.shape[0], y_num.shape[0]), dtype=np.float)

    cdef int32_t i, j
    cdef double a
    cdef int64_t b
    for i in range(x_num.shape[0]):
        for j in range(y_num.shape[0]):
            a = _euclidean_dissim(y_num[j], x_num[i])
            b = _matching_dissim(y_cat[j], x_cat[i])
            result[i, j] = a + gamma * b

    return result
