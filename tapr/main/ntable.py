from collections.abc import MutableMapping
import itertools as it
import operator as op

import numpy as np
import xarray as xr

from .defs import UFUNC_TO_OP
from .utils import (
    validate_ntable_init,
    str_ntable,
    xarray_coords_to_dict,
    ttype_to_attrs,
    call,
    setitem,
    validate_engine,
    validate_ttype,
    concatenate_ntables,
    full,
    NULL,
    NTableStopIteration,
    get_method_and_call,
    full_like,
)
from .tabularization import tabularize
from .engines import StandardEngine, ProcessEngine, ThreadEngine
from .handling import (
    handled_by,
    FunctionError,
    print_warning_return_function_error,
)
from .structure import NTableStructure
from .filtering import NTableFilter, contains, matches
from .alchemy import NTableAlchemy, NTableMapAlchemy


class NTableMap(MutableMapping):
    """
    A dictionary-like interface for working with a NTable object along a
    specific dimension.

    Parameters
    ----------
    dim : str
        The dimension to create a ntable map for.

    Raises
    ------
    ValueError
        Raised if the given dimension does not exist.
    """

    def __init__(self, ntable, dim):
        self._ntable = ntable
        self._dim = dim

    @property
    def ntable(self):
        return self._ntable

    @property
    def dim(self):
        return self._dim

    @property
    def alchemy(self):
        return NTableMapAlchemy(self)

    def __dir__(self):
        result = list(self.__dict__)
        result.extend(list(self.keys()))
        return result

    # def _ipython_key_completions_(self, incomplete_key):
    #     if not isinstance(incomplete_key, str):
    #         return []
    #     coords_dict = xarray_coords_to_dict(self._ntable.struct.coords)
    #     dim_coords = coords_dict[self._dim]
    #     result = []
    #     for coord in dim_coords:
    #         if isinstance(coord, str):
    #             if coord.startswith(incomplete_key):
    #                 result.append(coord)
    #
    #     return [coord for coord in dim_coords if coord.startswith(incomplete_key)]

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(f"{attr} is not an attribute of NTable")

    def __str__(self):
        return str(dict(**self))

    def __repr__(self):
        return str(self)

    def __getitem__(self, key):
        return self._ntable.struct.loc[{self._dim: key}]

    def __setitem__(self, key, value):
        if isinstance(key, list):
            if isinstance(value, NTable):
                if len(key) != len(value.ntable_map(self._dim)):
                    raise ValueError(
                        f"value.{self._dim} must have same length as key"
                    )
                for k, v in zip(key, value.ntable_map(self.dim).values()):
                    self[k] = v
            else:
                for k in key:
                    self[k] = value
        else:
            try:
                self._ntable.struct.loc[{self._dim: key}] = value
            except KeyError:
                sample_ntable = next(value for value in self.values())
                sample_coords = xarray_coords_to_dict(
                    sample_ntable.struct.coords
                )
                sample_coords[self._dim][0] = key
                extension = full(
                    NULL(), sample_coords, self._ntable.struct.dims
                )
                intermediate = concatenate_ntables(
                    (self._ntable, extension), dim=self.dim
                )
                intermediate.ntable_map(self.dim)[key] = value
                self._ntable._reflist = intermediate.reflist
                self._ntable._refmap = intermediate.refmap

    def __delitem__(self, key):
        raise NotImplementedError

    def __iter__(self):
        coords_dict = xarray_coords_to_dict(self._ntable.struct.coords)
        return iter(coords_dict[self._dim])

    def __len__(self):
        coords_dict = xarray_coords_to_dict(self._ntable.struct.coords)
        return len(coords_dict[self._dim])

    def contains(self, string):
        return self._ntable.filter[{self._dim: contains(string)}]

    def matches(self, pattern):
        return self._ntable.filter[{self._dim: matches(pattern)}]

    def relabel(self, **kwargs):
        return self._ntable.struct.relabel(**{self._dim: kwargs})

class TabularizedMethod:
    """
    Represents a method whose call should be tabularized.
    """

    def __init__(self, ntbl, method_name):
        self._ntbl = ntbl
        self._method_name = method_name

    def __call__(self,*args, **kwargs):
        handled_ = handled_by(FunctionError)(get_method_and_call)
        return tabularize(engine=self._ntbl.engine)(handled_)(self._ntbl, self._method_name, *args, **kwargs)

    def __str__(self):
        method_name_ntbl = full_like(self._method_name, self._ntbl, lite=True)
        return str(method_name_ntbl)

    def __repr__(self):
        return str(self)


class TabularizedAttributes:
    """
    Used to interface with the attributes of the ELEMENTS of a NTable object
    as opposed to the attributes of the NTable object itself.
    """

    def __init__(self, ntable):
        super().__setattr__("_ntable", ntable)

    def __getattr__(self, attr):
        handled_ = handled_by(FunctionError)(getattr)
        return tabularize(engine=self._ntable.engine)(handled_)(
            self._ntable, attr
        )

    def __setattr__(self, attr, value):
        handled_ = handled_by(FunctionError)(setattr)
        tabularize(engine=self._ntable.engine)(handled_)(
            self._ntable, attr, value
        )


def _next(iterator):
    # this function exists to raise a different type of exception
    # than StopIteration so that the tabularization proces doesn't
    # continue after StopIteration is raised (since it wont be
    # handled until the end of the end of the tabularized call) and
    # issues arise when trying to complete the tabularized call.
    # Throwing an error here prevents the process from finishing.
    try:
        next_item = next(iterator)
        return next_item
    except StopIteration:
        raise NTableStopIteration


class _NTable_Iterator:
    def __init__(self, ntbl):
        self._ntbl = ntbl

    def __next__(self):
        try:
            if isinstance(self._ntbl.engine, ProcessEngine):
                # if the engine is a ProcessEngine, then the StopIteration will
                # be masked by the processes and cannot stop the iteration. As
                # such, the tabularized next will be done with threads.
                engine = ThreadEngine(self._ntbl.engine.processes)
            else:
                engine = self._ntbl.engine
            return tabularize(engine=engine)(_next)(self._ntbl)
        except NTableStopIteration:
            raise StopIteration


class NTable(np.lib.mixins.NDArrayOperatorsMixin):
    """
    N-dimensional, tabular representation of a set of heterogeneous
    python data

    Parameters
    ----------
    reflist : list
        List containing the data to represent.
    refmap : DataArray
        A dataarray whose elements correspond to indexes in the reflist.
        Describes the layout of the data.
    engine : callable, optional
        A map-like callable. Must take in a function as the first argument
        and iterables whose elements will be passed into the function.
        If None, a serial-mapping engine will be used.
    ttype : set, optional
        A set of types. These types define what the NTable object MIGHT contain.
        This set is used to determine if a failed getattr operation should be
        tabularized or not. If None, a set of common types is used.
    validate : bool, optional
        Whether or not validate the NTable constructor inputs. Users making NTable
        objects manually should always set this to True. The default is True.

    Methods
    -------


    """

    def __init__(
        self, reflist, refmap, engine=None, ttype=None, validate=True
    ):

        if engine is None:
            engine = StandardEngine()
        orig_ttype = ttype
        if ttype is None:
            ttype = set()
        if validate:
            validate_ntable_init(reflist, refmap, engine, ttype)
        if orig_ttype is None:
            # if ttype was originally None, needed to define it as
            # something that will pass the validation step. Once
            # passed, we can assign it types based on the contents
            # of the intended NTable
            for item in (reflist[i] for i in refmap.values.flat):
                ttype.add(type(item))
        self._reflist = reflist
        self._refmap = refmap
        self._engine = engine
        self._ttype = ttype

    @property
    def reflist(self):
        """The list of data that the NTable object represents"""
        return self._reflist

    @property
    def refmap(self):
        """The dataarray that describes the layout of the data"""
        return self._refmap

    @property
    def struct(self):
        """
        An object that allows the user to work with the structure of a
        NTable object

        """
        return NTableStructure(self)

    @property
    def tattr(self):
        """
        An object where getting and setting attributes on it results in the
        tabularization of the get or set call

        """
        return TabularizedAttributes(self)

    @property
    def engine(self):
        """The engine that gets used for processing"""
        return self._engine

    @engine.setter
    def engine(self, e):
        validate_engine(e)
        self._engine = e

    @property
    def ttype(self):
        """The data types the NTable object MIGHT contain"""
        return self._ttype

    @ttype.setter
    def ttype(self, t):
        validate_ttype(t)
        self._ttype = t

    @property
    def filter(self):
        """An object used for filtering the NTable object."""
        return NTableFilter(self)

    @property
    def alchemy(self):
        """An object used for special types of lookups/operations."""
        return NTableAlchemy(self)

    def __dir__(self):
        result = list(self.__dict__)
        result.extend(self.struct.dims)
        result.extend(ttype_to_attrs(self.ttype))
        return result

    def __getattr__(self, attr):
        try:
            return self.ntable_map(attr)
        except ValueError:
            attr_dict = ttype_to_attrs(self._ttype)
            if attr in attr_dict:
                if callable(attr_dict[attr]):
                    # If the attribute is callable, save on overhead
                    # by returning a TabularizedMethod object instead
                    # of calling tabularized getattr since
                    # TabularizedMethod will reduce on tabularization
                    # overhead.
                    return TabularizedMethod(self, attr)
                return getattr(self.tattr, attr)
            raise AttributeError(f"{attr} is not an attribute of NTable")

    def __str__(self):
        ttype_strings = sorted([type_.__name__ for type_ in self._ttype])
        ttype_str = "|".join(ttype_strings)
        return f"{str_ntable(self)}\n{self.struct.coords}\nEngine:\n{self._engine}\nTtype:\n{ttype_str}"

    def __repr__(self):
        return str(self)

    def __iter__(self):
        ntable_of_iterators = tabularize(engine=self._engine)(iter)(self)
        return _NTable_Iterator(ntable_of_iterators)

    def __getitem__(self, index):
        handled_ = handled_by(FunctionError)(op.getitem)
        return tabularize(engine=self._engine)(handled_)(self, index)

    def __setitem__(self, index, value):
        handled_ = handled_by(print_warning_return_function_error)(setitem)
        result = tabularize(engine=self._engine)(handled_)(self, index, value)

    def __call__(self, *args, **kwargs):
        handled_ = handled_by(FunctionError)(call)
        return tabularize(engine=self._engine)(handled_)(self, *args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        ufunc = UFUNC_TO_OP.get(ufunc, ufunc)
        handled_ = handled_by(FunctionError)(ufunc)
        return tabularize(engine=self._engine)(handled_)(*inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        handled_ = handled_by(FunctionError)(func)
        return tabularize(engine=self._engine)(handled_)(*args, **kwargs)

    def item(self):
        return self.struct.item()

    def ntable_map(self, dim):
        """
        Returns a ntable map object for the given dimension

        Parameters
        ----------
        dim : str
            The dimension to create a ntable map for.

        Raises
        ------
        ValueError
            Raised if the given dimension does not exist.

        Returns
        -------
        NTableMap
            An object that allows the user the treat a given dimension object
            as a dictionary-like object.

        """
        if dim not in self.struct.dims:
            raise ValueError(f"{dim} dimension does not exist")
        return NTableMap(self, dim)

    def to_dictionary(self, into=None, enforce_nested_typing=True):
        """
        Convert a NTable object into a dictionary

        Parameters
        ----------
        into : MutableMapping, optional
            A dictionary-like object to store elements into. If None,
            a new dictionary is created and returned.

        enforce_nested_typing : bool
            Flag indicating whether or not nested elements of into
            should be made the same type as into. Default is True.

        Returns
        -------
        into : MutableMapping
            The resulting dictionary-like object.

        """
        if into is None:
            into = {}
        dim = self.struct.dims[0]
        for k, v in self.ntable_map(dim).items():
            if isinstance(v, NTable):
                if v.struct.ndim > 0:
                    if enforce_nested_typing:
                        into[k] = type(into)()
                    else:
                        into[k] = {}
                    v.to_dictionary(into[k])
                else:
                    into[k] = v.struct.item()
            else:
                into[k] = v

        return into

    def to_pandas(self, dtype="object"):
        """
        Convert a NTable object into a Pandas object
        (Series or DataFrame depending on the shape of the NTable object.)

        Parameters
        ----------
        dtype : str, np.dtype, optional
            Data type that the resulting Pandas object should be.
            The default is "object".

        Raises
        ------
        ValueError
            Raised when called on a NTable object with more than 2 dimensions.

        Returns
        -------
        Series or DataFrame
            The resulting Pandas object.

        """
        if self.struct.ndim < 3 and self.struct.ndim > 0:
            return self.to_data_array(dtype=dtype).to_pandas()
        else:
            raise ValueError(
                f"Unable to convert {self.struct.ndim} dimensional NTable to pandas object"
            )

    def to_data_array(self, dtype="object"):
        """
        Convert a NTable object into a DataArray

        Parameters
        ----------
        dtype : str, np.dtype, optional
            Data type that the resulting DataArray should be.
            The default is "object".

        Returns
        -------
        DataArray
            The resulting DataArray.

        """
        array = np.empty(self.refmap.size, dtype="object")
        array[:] = list(self.struct.flat)
        array = array.reshape(self.refmap.shape)
        return xr.DataArray(
            array, self.refmap.coords, self.refmap.dims
        ).astype(dtype)
