import warnings as wn
import itertools as it

import numpy as np
import xarray as xr

from .utils import concatenate_ntables, xarray_coords_to_dict, basic_refmap


class _LocIndexer:
    """
    Index by labels rather than position.
    """

    def __init__(self, struct):
        self._struct = struct

    def __getitem__(self, index):
        from .ntable import NTable

        new_refmap = self._struct.ntable.refmap.loc[index]
        return NTable(self._struct.ntable.reflist, new_refmap, validate=False)

    def __setitem__(self, index, value):
        from .ntable import NTable

        index_map = self._struct.ntable.refmap.loc[index]

        if len(np.unique(index_map)) < index_map.values.size:
            wn.warn(
                """Warning: The index corresponds to a part of the
                    structure that is multi-referential. This means that there 
                    are multiple items in the refmap that point to the same 
                    item in the reflist. As such, assignment operations could result in some
                    unexpected behavior."""
            )

        def _setreflist(i, v):
            self._struct.ntable.ttype.add(type(v))
            self._struct.ntable.reflist[i] = v

        if not isinstance(value, NTable):
            list(
                map(
                    _setreflist,
                    index_map.values.flat,
                    (value for item in index_map.values.flat),
                )
            )
        else:
            # should numpy or xarray broadcasting be used here?
            bi, bv = xr.broadcast(index_map, value.refmap)
            if np.isnan(bi.values).any() or np.isnan(bv.values).any():
                raise ValueError("Unable to assign input value")
            list(
                map(
                    _setreflist,
                    bi.values.flat,
                    (value.reflist[vi] for vi in bv.values.flat),
                )
            )


class NTableStructure:
    """
    NTable structure object. Used for structure-oriented operations such as
    indexing, assigning, transposing, iterating over elements, etc.

    Parameters
    ----------
    ntbl : NTable
        The ntable object whose structure is to be worked with.

    """

    def __init__(self, ntbl):
        self._ntbl = ntbl

    @property
    def ntable(self):
        """The corresponding NTable object."""
        return self._ntbl

    @property
    def flat(self):
        """An iterable that flattens the structure of the NTable object."""
        return (self._ntbl.reflist[i] for i in self._ntbl.refmap.values.flat)

    @property
    def dims(self):
        """The dimensions of the NTable object"""
        return self._ntbl.refmap.dims

    @property
    def coords(self):
        """The coords of the NTable object"""
        return self._ntbl.refmap.coords

    @property
    def name(self):
        """The name of the NTable object"""
        return self._ntbl.refmap.name

    @property
    def ndim(self):
        """The number of dimensions in the NTable object"""
        return self._ntbl.refmap.ndim

    @property
    def shape(self):
        """The shape of the NTable object"""
        return self._ntbl.refmap.shape

    @property
    def size(self):
        """The size of the NTable object"""
        return self._ntbl.refmap.size

    @property
    def loc(self):
        """Coordinate name based indexer"""
        return _LocIndexer(self)

    @property
    def T(self):
        """The transposed NTable object."""
        from .ntable import NTable

        return NTable(
            self._ntbl.reflist,
            self._ntbl.refmap.T,
            self._ntbl.engine,
            self._ntbl.ttype,
        )

    def __getitem__(self, index):
        from .ntable import NTable

        new_refmap = self._ntbl.refmap[index]
        return NTable(self._ntbl.reflist, new_refmap)

    def __setitem__(self, index, value):
        from .ntable import NTable

        index_map = self._ntbl.refmap[index]

        if len(np.unique(index_map)) < index_map.values.size:
            wn.warn(
                """Warning: The index corresponds to a part of the
                    structure that is multi-referential. This means that there 
                    are multiple items in the refmap that point to the same 
                    item in the reflist. As such, assignment operations could result in some
                    unexpected behavior."""
            )

        def _setreflist(i, v):
            self._ntbl.ttype.add(type(v))
            self._ntbl.reflist[i] = v

        if not isinstance(value, NTable):
            list(
                map(
                    _setreflist,
                    index_map.values.flat,
                    (value for item in index_map.values.flat),
                )
            )
        else:
            # should numpy or xarray broadcasting be used here?
            bi, bv = xr.broadcast(index_map, value.refmap)
            if np.isnan(bi.values).any() or np.isnan(bv.values).any():
                raise ValueError("Unable to assign input value")
            list(
                map(
                    _setreflist,
                    bi.values.flat,
                    (value.reflist[vi] for vi in bv.values.flat),
                )
            )

    def __add__(self, right):
        return concatenate_ntables(
            (self.ntable, right.ntable), dim=self.dims[0]
        )

    def transpose(self, *refmap_args, **refmap_kwargs):
        from .ntable import NTable

        return NTable(
            self._ntbl.reflist,
            self._ntbl.refmap.transpose(*refmap_args, **refmap_kwargs),
            self._ntbl.engine,
            self._ntbl.ttype,
        )

    def item(self):
        """
        If the NTable object has just a single element (regardless of its
        dimensionality), returns that item.
        """
        if self.size > 1:
            raise ValueError(
                f"Cannot return a single item for NTable Structure of size greater than 1 ({self.size})"
            )
        return next(self.flat)

    def relabel(self, **kwargs):
        """
        Relabel coordinates.

        Parameters
        ----------
        **kwargs : dictionary-like
            Keys specify which dimension to apply the relabel to, values
            should be mappings themselves that map old labels along the
            corresponding dimension to new labels.

        Returns
        -------
        NTable
            The relabled NTable.

        """
        from .ntable import NTable

        refmap = self._ntbl.refmap
        for k, v in kwargs.items():
            refmap = refmap.to_dataset(k).rename(**v).to_array(k)

        return NTable(self._ntbl.reflist, refmap)

    def flatter(self, dims=None):
        if dims is None:
            dims = tuple()
        for dim in dims:
            if dim not in self.dims:
                raise ValueError(f"dim {dim} not found")
        dims = tuple(dim for dim in self.dims if dim not in dims)
        coord_dict = xarray_coords_to_dict(self.coords)
        indexes = it.product(*(coord_dict[dim] for dim in dims))
        for index in indexes:
            yield self.loc[{dim: coord for dim, coord in zip(dims, index)}]

    def compress(self, dims=None):
        from .ntable import NTable

        new_reflist = [ntbl for ntbl in self.flatter(dims)]
        new_dims = tuple(dim for dim in self.dims if dim not in dims)
        coords_dict = xarray_coords_to_dict(self.coords)
        new_coords = {dim: coords_dict[dim] for dim in new_dims}

        new_refmap = basic_refmap(new_coords, new_dims)

        return NTable(new_reflist, new_refmap, engine=self.ntable.engine)
