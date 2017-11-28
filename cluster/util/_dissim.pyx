# cython: language_level=3
# cython: profile=True
# cython: boundscheck=True
cimport cython
from cython cimport double, long

@cython.profile(False)
cdef double _euclidean_dissim(double[:] a, double[:] b, int n):
    cdef double result, tmp
    result = 0.0

    cdef int i
    for i in range(n):
        tmp = (a[i] - b[i])
        result += tmp * tmp

    return result


@cython.profile(False)
cdef long _matching_dissim(long[:] a, long[:] b, int n):
    cdef long result, tmp
    result = 0

    cdef int i
    for i in range(n):
        tmp = (a[i] != b[i])
        result += tmp

    return result
