from libc.stdint cimport int32_t, int64_t

cdef double _euclidean_dissim(double[:], double[:])
cdef int32_t _matching_dissim(int64_t[:], int64_t[:])
cpdef pairwise_dissim(double[:, :], int64_t[:, :], double[:, :], int64_t[:, :], double)
