cdef void _move_point_num(double[:], int, int, double[:, :], long[:])
cdef void _move_point_cat(long[:], long, long, long, long[:, :], long[:], long[:], long[:, :])
cpdef long _get_max_value_key(long[:, :], long[:], long, long)