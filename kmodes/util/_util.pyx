# cython: language_level=3
# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False


cdef void _move_point_num(double[:] point, int to_clust, int from_clust, double[:, :] cl_attr_sum, long[:] cl_memb_sum):
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


cdef void _move_point_cat(long[:] point, long ipoint, long to_clust, long from_clust,
                   long[:, :] cl_attr_freq,
                   long[:] cat_offsets,
                   long[:] membship,
                   long[:, :] centroids):
    """Move point between clusters, categorical attributes."""
    cdef:
        int iattr, offset, max_index
        long max_val
        long curattr
        long current_centroid_value, current_centroid_freq, current_attribute_value_freq

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


cpdef long _get_max_value_key(long[:, :] cl_attr_freq, long[:] cat_offsets, long clust, long iattr):
    cdef:
        long max_val = 0
        long index = 0
        unsigned long i
        long offset
        long[:] cats
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
