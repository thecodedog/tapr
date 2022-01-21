import itertools as it

import numpy as np

from .ntable import NTable
from .tabularization import tabularize
from .utils import full, NULL, default_refmap, xarray_coords_to_dict, basic_refmap


def blank(coords, dims, engine=None, ttype=None):
    """
    Creates an NTable with given coordinates and dimensions
    whose elements are all NULL()

    Parameters
    ----------
    coords:
        The coordinates for the desired blank NTable
    dims:
        The dimensions for the desired blank NTable
    engine:
        The engine for the resulting NTable to use. Default is None.
    ttype:
        The ttype for the resulting NTable to use. Default is None.

    Returns
    -------
    ntable (NTable): The desired NTable

    """
    return full(NULL(), coords=coords, dims=dims, engine=engine, ttype=ttype)


def sblank(*shape, engine=None, ttype=None):
    """
    Similar to blank, but based on a given shape instead of
    coordinates and dimensions.

    Parameters
    ----------
    shape : tuple
        The shape for the desired blank NTable
    engine:
        The engine for the resulting NTable to use. Default is None.
    ttype:
        The ttype for the resulting NTable to use. Default is None.

    Returns
    -------
    ntable (NTable): The desired NTable

    """
    total_size = 1
    for size in shape:
        total_size *= size

    reflist = [NULL()] * total_size
    refmap = default_refmap(*shape)
    return NTable(reflist, refmap, engine=engine, ttype=ttype)


def cartograph(ntbl, engine=None, ttype=None):
    """
    Creates an NTable whose elements are the intersection (tuple) of the
    corresponding coordinates of the input NTable

    Parameters
    ----------
    ntbl: NTable
        The NTable to base the resulting NTable on
    engine:
        The engine for the resulting NTable to use. Default is None.
    ttype:
        The ttype for the resulting NTable to use. Default is None.

    Returns
    -------
    ntable (NTable): The desired NTable

    """
    coords = ntbl.struct.coords
    dims = ntbl.struct.dims
    coords_dict = xarray_coords_to_dict(coords, dims)
    reflist = list(it.product(*coords_dict.values()))
    refmap = basic_refmap(coords, dims)
    return NTable(reflist, refmap, engine=engine, ttype=ttype)

def count(ntbl, engine=None, ttype=None):
    coords = ntbl.struct.coords
    dims = ntbl.struct.dims
    coords_dict = xarray_coords_to_dict(coords, dims)
    reflist = list(i for i,_ in enumerate(it.product(*coords_dict.values())))
    refmap = ntbl.refmap.copy()
    return NTable(reflist, refmap, engine=engine, ttype=ttype)


# def itertable(iterable, engine=None, ttype=None):
#     """
#     Converts potentially nested iterables into a representative
#     NTable.
#
#     Parameters
#     ----------
#     iterable: iterable
#         the iterable to generate the NTable from
#     engine:
#         The engine for the resulting NTable to use. Default is None.
#     ttype:
#         The ttype for the resulting NTable to use. Default is None.
#
#     Returns
#     -------
#     ntable: NTable
#         The desired NTable
#
#     """
#     expanded = expand(iterable)
#     refarray = np.array(expanded, dtype="object")
#     reflist = []
#     for item in refarray.flat:
#         try:
#             # see if the item was wrapped in a shell
#             reflist.append(item.__TAPR_PEEL__())
#         except AttributeError:
#             reflist.append(item)
#
#     refmap = default_refmap(*refarray.shape)
#     return NTable(reflist, refmap, engine=engine, ttype=ttype)
