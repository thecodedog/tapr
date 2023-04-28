import operator as op
import mmap

import numpy as np
import xarray as xr
import h5py

from ..main.utils import xarray_coords_to_dict


def save_ntable(ntbl, fname, allow_pickle=False):
    """
    Save NTable object to a file.

    Parameters
    ----------
    ntbl : NTable
        The NTable to save.
    fname : str
        The name of the file to save it as.

    Returns
    -------
    None.

    """
    from .serialization import serialize
    from ..main.tabularization import tabularize

    bytes_ntable, type_id_ntable = tabularize()(serialize)(ntbl, allow_pickle=allow_pickle)
    len_ntable = tabularize()(len)(bytes_ntable)

    engine_bytes, engine_type_id = serialize(ntbl.engine)

    total_bytes = bytearray()
    for bytes_ in bytes_ntable.struct.flat:
        total_bytes += bytes_

    total_bytes = total_bytes + engine_bytes

    len_array = len_ntable.to_data_array(dtype=int)

    stop_array = xr.DataArray(
        np.cumsum(len_array.values).reshape(len_array.shape),
        coords=len_array.coords,
        dims=len_array.dims,
    )
    start_array = stop_array - len_array
    type_id_array = type_id_ntable.to_data_array()

    ublock_size = 1 << (len(total_bytes) - 1).bit_length()
    if ublock_size < 512:
        ublock_size = 512
    coords_dict = xarray_coords_to_dict(ntbl.struct.coords)
    with h5py.File(fname, "w", userblock_size=ublock_size) as fo:
        fo["/start"] = start_array.values
        fo["/stop"] = stop_array.values
        fo["/type"] = type_id_array.values
        fo.attrs["engine_length"] = len(engine_bytes)
        fo.attrs["engine_type_id"] = engine_type_id
        for i, dim in enumerate(ntbl.struct.dims):
            labels = coords_dict[dim]
            fo[f"/coords/{dim}"] = np.string_(labels)
            fo[f"/coords/{dim}"].attrs["axis"] = i

    with open(fname, "r+b") as fo:
        fo.write(total_bytes)


def load_ntable(fname, filter={}, allow_pickle=False):
    """

    Parameters
    ----------
    fname : str
        The name file to load.

    Returns
    -------
    loaded_ntable : NTable
        The loaded NTable object.

    """
    from .serialization import deserialize
    from ..main.conversion import ntable
    from ..main.tabularization import tabularize

    with h5py.File(fname, "r") as fo:
        userblock_size = fo.userblock_size
        start_ndarray = fo["/start"][...]
        stop_ndarray = fo["/stop"][...]
        type_id_ndarray = fo["/type"].asstr()[...]
        data_stop_loc = stop_ndarray.max()
        engine_length = fo.attrs["engine_length"]
        engine_type_id = fo.attrs["engine_type_id"]
        dims = []
        axes = []
        coords_dict = {}
        for dim, coord_dset in fo["/coords/"].items():
            dims.append(dim)
            axes.append(coord_dset.attrs["axis"])
            coords_dict[dim] = [
                item.decode() for item in list(coord_dset[...])
            ]

        dims = [dim for _, dim in sorted(zip(axes, dims))]

    with open(fname, "rb") as fo:
        total_bytes = mmap.mmap(fo.fileno(),userblock_size, access=mmap.ACCESS_READ)
        engine_bytes = total_bytes[
            data_stop_loc : data_stop_loc + engine_length
        ]

    start_ntable = ntable(xr.DataArray(start_ndarray, coords_dict, dims)).filter[filter]
    stop_ntable = ntable(xr.DataArray(stop_ndarray, coords_dict, dims)).filter[filter]
    type_id_ntable = ntable(xr.DataArray(type_id_ndarray, coords_dict, dims)).filter[filter]

    bytes_ntable = tabularize()(op.getitem)(
        total_bytes, slice(start_ntable, stop_ntable)
    )

    loaded_ntable = tabularize()(deserialize)(bytes_ntable, type_id_ntable, allow_pickle=allow_pickle)
    loaded_ntable.engine = deserialize(engine_bytes, engine_type_id)
    return loaded_ntable
