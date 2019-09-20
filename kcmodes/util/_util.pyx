# cython: language_level=3
# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.stdint cimport int32_t, int64_t

cdef void _move_point_num(double[:] point, int32_t to_clust, int32_t from_clust, double[:, :] cl_attr_sum, int64_t[:] cl_memb_sum):
    """Move point between clusters, numerical attributes."""
    cdef:
        int32_t iattr = 0
        double curattr

    # Update sum of attributes in cluster.
    for iattr in range(point.shape[0]):
        curattr = point[iattr]
        cl_attr_sum[to_clust, iattr] += curattr
        cl_attr_sum[from_clust, iattr] -= curattr
    # Update sums of memberships in cluster
    cl_memb_sum[to_clust] += 1
    cl_memb_sum[from_clust] -= 1


cdef void _move_point_cat(int64_t[:] point, int64_t ipoint, int64_t to_clust, int64_t from_clust,
                   int64_t[:, :] cl_attr_freq,
                   int64_t[:] cat_offsets,
                   int64_t[:] membship,
                   int64_t[:, :] centroids):
    """Move point between clusters, categorical attributes."""
    cdef:
        int32_t iattr, offset, max_index
        int64_t max_val
        int64_t curattr
        int64_t current_centroid_value, current_centroid_freq, current_attribute_value_freq

    membship[ipoint] = to_clust
    # Update frequencies of attributes in cluster.
    for iattr in range(point.shape[0]):
        offset = cat_offsets[iattr]
        curattr = point[iattr]
        to_attr_counts = cl_attr_freq[to_clust, iattr + offset]
        from_attr_counts = cl_attr_freq[from_clust, iattr + offset]

        # Increment the attribute count for the new "to" cluster
        cl_attr_freq[from_clust, curattr + offset] -= 1
        cl_attr_freq[to_clust, curattr + offset] += 1

        current_attribute_value_freq = cl_attr_freq[to_clust, curattr + offset]
        current_centroid_value = centroids[to_clust, iattr]
        current_centroid_freq = cl_attr_freq[to_clust, offset + current_centroid_value]
        if current_centroid_freq < current_attribute_value_freq:
            # We have incremented this value to the new mode. Update the centroid.
            centroids[to_clust, iattr] = curattr

        # Decrement the attribute count for the old "from" cluster

        old_centroid_value = centroids[from_clust, iattr]
        if old_centroid_value == curattr:
            # We have just removed a count from the old centroid value. We need to
            # recalculate the centroid as it may no longer be the maximum
            centroids[from_clust, iattr] = _get_max_value_key(cl_attr_freq, cat_offsets, from_clust, iattr)


cpdef int64_t _get_max_value_key(int64_t[:, :] cl_attr_freq, int64_t[:] cat_offsets, int64_t clust, int64_t iattr):
    cdef:
        int64_t max_val = 0
        int64_t index = 0
        int64_t i
        int64_t offset
        int64_t[:] cats
    offset = cat_offsets[iattr]
    if iattr == <long>len(cat_offsets) - 1:
        cats = cl_attr_freq[clust, offset:]
    else:
        cats = cl_attr_freq[clust, offset:cat_offsets[iattr+1]]
    for i in range(len(cats)):
        if cats[i] > max_val:
            max_val = cats[i]
            max_index = i
    return max_index
