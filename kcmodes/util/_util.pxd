from libc.stdint cimport int32_t, int64_t

cdef void _move_point_num(double[:], int32_t, int32_t, double[:, :], int64_t[:])
cdef void _move_point_cat(int64_t[:], int64_t, int64_t, int64_t, int64_t[:, :], int64_t[:], int64_t[:], int64_t[:, :])
cpdef int64_t _get_max_value_key(int64_t[:, :], int64_t[:], int64_t, int64_t)
