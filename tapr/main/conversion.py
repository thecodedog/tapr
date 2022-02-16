from collections.abc import Mapping
import operator as op
import functools as ft
import itertools as it

import numpy as np
import pandas as pd
import xarray as xr

from .utils import basic_refmap, NULL
from .processing import broadcast_tables, tabular_map
from .engines import StandardEngine
from .ttypes import STANDARD_TTYPE


def _tuplefy(*args):
    return args


def _tabulate_tuple(tup):
    tds = broadcast_tables(*tup)
    return tabular_map(_tuplefy, *tds)


def _listify(*args):
    return list(args)


def _tabulate_list(lst):
    tds = broadcast_tables(*lst)
    return tabular_map(_listify, *tds)


def _slicefy(*args):
    return slice(args[0], args[1], args[2])


def _tabulate_slice(slc):
    slice_tup = (slc.start, slc.stop, slc.step)
    tds = broadcast_tables(*slice_tup)
    return tabular_map(_slicefy, *tds)


def _dictify_tuple(tup):
    return {t[0]: t[1] for t in tup}


def _tabulate_dictionary(dictionary):
    items_tup = tuple(dictionary.items())
    compressed_tuple = tabulate(items_tup)
    result = tabular_map(_dictify_tuple, compressed_tuple)
    return result


def tabulate(con):
    if isinstance(con, dict):
        new_con = {}
        for k, v in con.items():
            try:
                new_con[k] = tabulate(v)
            except Exception:
                new_con[k] = v
        return _tabulate_dictionary(new_con)

    if isinstance(con, tuple):
        new_con = tuple()
        for item in con:
            try:
                new_con += (tabulate(item),)
            except Exception:
                new_con += (item,)
        return _tabulate_tuple(new_con)

    if isinstance(con, list):
        new_con = list()
        for item in con:
            try:
                new_con.append(tabulate(item))
            except Exception:
                new_con.append(item)
        return _tabulate_list(new_con)

    if isinstance(con, slice):
        slice_tup = (con.start, con.stop, con.step)
        new_slice_tup = tuple()
        for item in slice_tup:
            try:
                new_slice_tup += (tabulate(item),)
            except Exception:
                new_slice_tup += (item,)
        new_slice = slice(*new_slice_tup)
        return _tabulate_slice(new_slice)

    raise TypeError(f"unable to tabulate object of type {type(con)}")


def _mapping_depth(mapping):
    if isinstance(mapping, Mapping):
        return (
            max(map(_mapping_depth, mapping.values())) if mapping else 0
        ) + 1
    return 0


# def _flatten_mapping_values(mapping):
#     result = []
#     for item in mapping.values():
#         if not isinstance(item, Mapping):
#             result.append(item)
#         else:
#             result.extend(_flatten_mapping_values(item))
#     return result


def _get_nested_value(data, index, default=None):
    try:
        return ft.reduce(op.getitem, index, data)
    except (KeyError, IndexError):
        return default


def _extract_mapping_coords(mapping, dest=None, depth=0, dims=None):
    retflag = False
    if dest is None:
        # if dest is None we are in the top-most call
        maxdepth = _mapping_depth(mapping)
        dest = {i: [] for i in range(maxdepth)}
        retflag = True
    for k, v in mapping.items():
        if k not in dest[depth]:
            dest[depth].append(k)
        if isinstance(v, Mapping):
            _extract_mapping_coords(v, dest=dest, depth=depth + 1)
    if retflag:
        if dims is None:
            return {f"dim{k}": v for k, v in dest.items()}
        else:
            return {dims[k]: v for k, v in dest.items()}


def _mapping_to_ntable(mapping, dims=None, engine=None, ttype=None):
    from .ntable import NTable

    coords = _extract_mapping_coords(mapping, dims=dims)
    dmap = basic_refmap(coords, tuple(coords.keys()))
    data_keys = it.product(*coords.values())
    dlist = [_get_nested_value(mapping, index, NULL()) for index in data_keys]

    return NTable(dlist, dmap, engine=engine, ttype=ttype)


def _data_array_to_ntable(data_array, engine=None, ttype=None):
    from .ntable import NTable

    dlist = list(data_array.values.flat)
    dmap_values = np.arange(len(dlist)).reshape(*(data_array.shape))
    dmap = xr.DataArray(dmap_values, data_array.coords, data_array.dims)
    return NTable(dlist, dmap, engine=engine, ttype=ttype)


def _pandas_to_ntable(pds, dims=None, engine=None, ttype=None):
    from .ntable import NTable

    reflist = list(pds.values.flat)
    if isinstance(pds, pd.DataFrame):
        if dims is None:
            dims = ("rows", "cols")
        coords = {dims[0]: list(pds.index), dims[1]: list(pds.columns)}
    elif isinstance(pds, pd.Series):
        if dims is None:
            dims = ("rows",)
        coords = {dims[0]: list(pds.index)}
    refmap = basic_refmap(coords, dims)
    return NTable(reflist, refmap, engine=engine, ttype=ttype)


def ntable(obj, dims=None, engine=None, ttype=None):
    """
    Parameters
    ----------
    obj : Mapping, xr.DataArray, pd.DataFrame, NTable or container of NTable objects
        The object to create a NTable object from.
    dims : Sequence, optional
        The desired dimension names. If None, dim names will be dim0, dim1,
        dim2, etc. The default is None.
    engine : Engine, optional
        The engine to be used when doing computations. If None, a standard
        serial engine will be used. The default is None.
    ttype : set, optional
        Set of objects that the NTable may contain. If None, a set of common
        objects is used. The default is None.

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    ntbl : TYPE
        DESCRIPTION.

    """
    from .ntable import NTable

    if isinstance(obj, Mapping):
        ntbl = _mapping_to_ntable(obj, dims, engine=engine, ttype=ttype)
        return ntbl
    if isinstance(obj, xr.DataArray):
        ntbl = _data_array_to_ntable(obj, engine=engine, ttype=ttype)
        return ntbl
    if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
        ntbl = _pandas_to_ntable(obj, dims, engine=engine, ttype=ttype)
        return ntbl
    if isinstance(obj, NTable):
        ntbl = NTable(obj.reflist, obj.refmap, engine=engine, ttype=ttype)
        return ntbl
    try:
        ntbl = tabulate(obj)
        return ntbl
    except TypeError:
        raise TypeError(f"Unable to convert {type(obj)} to NTable")
