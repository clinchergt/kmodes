# cython: language_level=3
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False

cimport cython

@cython.profile(False)
cdef double _euclidean_dissim(double[:] centroid, double[:] x):
    cdef double result, tmp
    result = 0.0

    cdef int i
    for i in range(x.shape[0]):
        tmp = (centroid[i] - x[i])
        result += tmp * tmp

    return result


@cython.profile(False)
cdef int _matching_dissim(long[:] centroid, long[:] x):
    cdef int result, tmp
    result = 0

    cdef int i
    for i in range(x.shape[0]):
        tmp = (centroid[i] != x[i])
        result += tmp

    return result
