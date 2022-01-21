from tapr.main.utils import NULL

def assert_ntable_equivalent(ntbl1, ntbl2):
    reflist1 = ntbl1.reflist
    refmap1 = ntbl1.refmap
    reflist2 = ntbl2.reflist
    refmap2 = ntbl2.refmap
    for dim1, dim2 in zip(refmap1.dims, refmap2.dims):
        if (refmap1.coords[dim1] != refmap2.coords[dim2]).any():
            raise ValueError("NTable objects are not equivalent")
    for index1, index2 in zip(refmap1.values.flat, refmap2.values.flat):
        value1 = reflist1[index1]
        value2 = reflist2[index2]
        if value1 != value2:
            # check to see if values are NULL, which are falsey
            if isinstance(value1, NULL) and isinstance(value2, NULL):
                continue
            raise ValueError("NTable objects are not equivalent")