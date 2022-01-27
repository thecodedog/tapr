import inspect
from collections.abc import MutableMapping

import h5py
import numpy as np
import xarray as xr

from .defs import PRINTABLE_TYPES


def _dummy_func(x, y):
    return x + y


def validate_engine(engine):
    try:
        dummy_result = list(engine(_dummy_func, [1, 2, 3, 4], [4, 5, 6]))
        if len(dummy_result) != 3:
            raise Exception
    except Exception:
        raise ValueError("The input must follow the same interface as map")


def validate_ttype(ttype):
    for t in ttype:
        if not isinstance(t, type):
            raise ValueError("Every value in ttype must be of type type")


def validate_ntable_init(reflist, refmap, engine, ttype):
    if not isinstance(refmap, xr.DataArray):
        raise TypeError("refmap must be an xarray DataArray")
    if len(list(refmap.coords.keys())) == 0:
        raise ValueError("Data map must have non empty coordinates")
    if not isinstance(reflist, list):
        raise TypeError("reflist must be a list")

    for item in refmap.data.flat:
        try:
            reflist[item]
        except IndexError:
            raise ValueError(
                "every element of refmap must be a valid index of reflist"
            )

    # if len(np.unique(refmap)) < refmap.values.size:
    #     raise ValueError("Every element of refmap must be unique")

    validate_engine(engine)
    validate_ttype(ttype)


def ttype_to_attrs(ttype):
    attrs = {}
    for type_ in ttype:
        attrs.update({item[0]:item[1] for item in inspect.getmembers(type_)})
    return attrs


def xarray_coords_to_dict(coords, dims=None):
    coords_dict = {}
    if dims is None:
        keys = coords.keys()
    else:
        keys = dims
    for k in keys:
        try:
            coords_dict[k] = list(coords[k].values)
        except TypeError:
            coords_dict[k] = [coords[k].values.item()]
    return coords_dict


def coords_dims_to_shape(coords, dims):
    if isinstance(coords, xr.DataArray):
        coords = xarray_coords_to_dict(coords)
    return tuple(len(coords[d]) for d in dims)


def basic_refmap(coords, dims):
    shape = coords_dims_to_shape(coords, dims)
    size = np.prod(shape)
    refmap = xr.DataArray(
        np.arange(size, dtype="int").reshape(shape), coords, dims
    )
    return refmap


def full(value, coords, dims, engine=None, ttype=None):
    from .ntable import NTable

    shape = coords_dims_to_shape(coords, dims)
    size = int(np.prod(shape))
    dlist = [value] * size
    dmap = xr.DataArray(
        np.arange(size, dtype="int").reshape(shape), coords, dims
    )
    # TODO: Turn off validation once this has been tested thoroughly
    return NTable(dlist, dmap, engine, ttype)

def full_like(value, ntbl, engine=None, ttype=None, lite=False):
    if lite:
        return full_lite(value, ntbl.struct.coords, ntbl.struct.dims, engine=engine, ttype=ttype)
    return full(value, ntbl.struct.coords, ntbl.struct.dims, engine=engine, ttype=ttype)

def full_lite(value, coords, dims, engine=None, ttype=None):
    from .ntable import NTable

    shape = coords_dims_to_shape(coords, dims)
    size = int(np.prod(shape))
    dlist = [value]
    dmap = xr.DataArray(np.zeros(shape, dtype="int"), coords, dims)
    # TODO: Turn off validation once this has been tested thoroughly
    return NTable(dlist, dmap, engine, ttype)


def str_ntable_element(val):
    try:
        return val.__ntable_element__str__()
    except AttributeError:
        if isinstance(val, np.ndarray):
            return f"ndarray,{val.shape},{val.dtype}"
        elif isinstance(val, xr.DataArray):
            return f"data array,{val.shape},{val.dtype}"
        elif type(val) in PRINTABLE_TYPES:
            return str(val)
        elif isinstance(val, str):
            if len(val) < 15:
                return f'"{val}"'
            else:
                return f"{val[:5]}...{val[-5:]}"
        elif isinstance(val, (list, tuple)):
            if len(val) < 5:
                return f"{val}"
            else:
                return f"{val[:2]}...{val[-2:]}"
        else:
            return type(val).__name__


def str_ntable(ntbl):
    from .tabularization import tabularize

    max_rows = 100 + 25
    max_cols = 20 + 5
    max_other = 10 + 3

    dims = ntbl.struct.dims

    reduce_index = {}

    try:
        if len(ntbl.struct.coords[dims[0]]) > max_rows:
            reduce_index[dims[0]] = np.r_[0:50, -50:0]
    except IndexError:
        pass

    try:
        if len(ntbl.struct.coords[dims[1]]) > max_cols:
            reduce_index[dims[0]] = np.r_[0:10, -10:0]
    except IndexError:
        pass

    print_ellipsises = {}

    for dim in dims[2:]:
        if len(ntbl.struct.coords[dim]) > max_other:
            reduce_index[dim] = np.r_[0:5, -5:0]
            print_ellipsises[dim] = True
        else:
            print_ellipsises[dim] = False

    ntbl = ntbl.struct[reduce_index]

    ntbl = tabularize(str_ntable_element)(ntbl)

    if ntbl.struct.ndim > 2:
        string = ""
        dim = ntbl.struct.dims[-1]
        ntbl_map = ntbl.ntable_map(dim)
        for i, (k, v) in enumerate(ntbl_map.items()):
            sep = "#" * 79
            string += f"{k}:\n\n{str_ntable(v)}\n\n{sep}\n\n"
            if (i == 4) and (print_ellipsises[dim]):
                string += f". . .\n\n{sep}\n\n"

    elif ntbl.struct.ndim == 0:
        string = str(ntbl.struct.item())
    else:
        string = str(ntbl.to_pandas())
    return string


def any_ntables(iterable):
    from .ntable import NTable

    return any(isinstance(item, NTable) for item in iterable)


def call_args_kwargs(func, *args):
    return func(*args[0], **args[1])


def call(func, *args, **kwargs):
    return func(*args, **kwargs)

def get_method_and_call(object, methodname, *args, **kwargs):
    method = getattr(object, methodname)
    return method(*args, **kwargs)

def setitem(obj, index, value):
    obj[index] = value


class _Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]


# class SUPREMUM(metaclass=_Singleton):
#     def __gt__(self, other):
#         return True
#
#     def __ge__(self, other):
#         return True
#
#     def __lt__(self, other):
#         return False
#
#     def __le__(self, other):
#         return False

# class INFIMUM(metaclass=_Singleton):
#     def __gt__(self, other):
#         return False
#
#     def __ge__(self, other):
#         return False
#
#     def __lt__(self, other):
#         return True
#
#     def __le__(self, other):
#         return True


class NULL(np.lib.mixins.NDArrayOperatorsMixin, metaclass=_Singleton):
    def __str__(self):
        return "NULL"

    def __repr__(self):
        return "NULL"

    def __ntable_element__str__(self):
        return "NULL"

    def __getitem__(self, index):
        return NULL()

    def __call__(self, *args, **kwargs):
        return NULL()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return NULL()

    # def __array_function__(self, func, types, args, kwargs):
    #     return NULL()


def handle_improper_broadcast(bdata, bmap):
    bdata.append(NULL())
    new_bmap_array = np.nan_to_num(bmap, nan=len(bdata) - 1).astype("int")
    new_bmap = xr.DataArray(new_bmap_array, bmap.coords, bmap.dims)
    return bdata, new_bmap


def concatenate_ntables(objs, dim, coords=None):
    from .ntable import NTable

    dmaps = tuple(obj.refmap for obj in objs)
    dlists = tuple(obj.reflist for obj in objs)
    true_sizes = tuple(len(dlist) for dlist in dlists)

    new_engine = objs[0].engine

    new_ttype = set()
    for obj in objs:
        new_ttype |= obj.ttype

    new_dmaps = tuple(
        dmap + sum(true_sizes[0:i]) for i, dmap in enumerate(dmaps)
    )

    # TODO: handle when dim is new and assign new coordinates
    new_dmap = xr.concat(new_dmaps, dim)
    dim_is_new = True
    for new_dmap_ in new_dmaps:
        if dim in new_dmap_.dims:
            dim_is_new = False
            break

    if dim_is_new:
        # if dim is new, then dim was added along axis=0 per xarray.concate
        # for NTables we want the behavior so that it is a new dimension
        # created at the end so we need to rearange...
        new_dmap = new_dmap.transpose(
            *(new_dmap.dims[1:] + (new_dmap.dims[0],))
        )
        if coords is None:
            # if dim is new then we need to assign it coordinates. If not
            # supplied, create them as list of integers ranging from 0 to
            # the number of input ntables
            coords = list(range(0, len(objs)))
        # if dim is new, we need to assign new coordinates which have either
        # been supplied by the caller or set to be a simple integer index
        new_dmap = new_dmap.assign_coords({dim: coords})

    new_dlist = []
    for dlist in dlists:
        new_dlist.extend(dlist)

    try:
        return NTable(new_dlist, new_dmap)
    except:
        new_dlist, new_dmap = handle_improper_broadcast(new_dlist, new_dmap)
        return NTable(new_dlist, new_dmap, engine=new_engine, ttype=new_ttype)


def default_refmap(*shape):
    dims = tuple(f"dim{i}" for i in range(len(shape)))
    coords = {dim: [] for dim in dims}
    for dim, size in zip(dims, shape):
        coords[dim] = [f"coord{i}" for i in range(size)]

    refarray = np.arange(np.prod(shape)).reshape(shape)
    refmap = xr.DataArray(refarray, coords, dims)
    return refmap


# class _Shell:
#     def __init__(self, obj):
#         self._obj = obj
#
#     def __TAPR_PEEL__(self):
#         return self._obj
#
#
# def shell(obj, layers=1):
#     return _Shell(obj)


def expand(iterable):
    result = []
    for item in iterable:
        try:
            result.append(expand(item))
        except TypeError:
            result.append(item)

    return result


def flatten(iterable):
    result = []
    for item in iterable:
        try:
            result.extend(flatten(item))
        except TypeError:
            result.append(item)

    return result


class _H5PYGroupDictifier(MutableMapping):
    def __init__(self, group=None):
        self._group = group

    def __getitem__(self, key):
        return self._group[key]

    def __setitem__(self, key, value):
        if isinstance(value, _H5PYGroupDictifier):
            if value._group is None:
                # if group is None, assume that we are meant to
                # just create a new child group with the name of
                # key.
                self._group.create_group(key)
            else:
                self._group[key] = value._group
        else:
            self._group[key] = value

    def __delitem__(self, key):
        raise NotImplementedError

    def __iter__(self):
        return iter(self._group)

    def __len__(self):
        return len(self._group)


def dictifyh5(group_or_file):
    if isinstance(group_or_file, h5py.File):
        return _H5PYGroupDictifier(group_or_file["/"])
    elif isinstance(group_or_file, h5py.Group):
        return _H5PYGroupDictifier(group_or_file)
    else:
        raise TypeError


# class Counter:
#     def __init__(self, start=0, stop=None, step=1):
#         self._start = start
#         self._stop = stop
#         self._step = step
#         self._current = start
#
#     def __call__(self, *args, **kwargs):
#         next_ = self._current + self._step
#         if self._stop is not None:
#             if next_ >= self._stop:
#                 raise ValueError("counter exceeded stop value")
#         self._current = next_
#         return self._current


class NTableStopIteration(Exception):
    pass
