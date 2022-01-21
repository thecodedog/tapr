import functools as ft

from .processing import tabular_map, broadcast_tables
from .utils import any_ntables, call_args_kwargs
from .conversion import tabulate
from .engines import StandardEngine


class _Tabularized:
    def __init__(self, func, engine):
        self._func = func
        self._engine = engine

    def __call__(self, *args, **kwargs):
        from .ntable import NTable

        try:
            targs = tabulate(args)
        except ValueError:
            targs = args
        try:
            tkwargs = tabulate(kwargs)
        except ValueError:
            tkwargs = kwargs
        if not isinstance(targs, NTable) and not isinstance(tkwargs, NTable):
            # if there are no NTable objects in args or kwargs, just call the
            # function normally on the inputs
            return self._func(*args, **kwargs)

        bfunc, bargs, bkwargs = broadcast_tables(
            self._func, targs, tkwargs, lite=True
        )
        # bfunc engine will always be just a StandardEngine since it is
        # generated via a (scalar, NTable, NTable or scalar) broadcast. We want
        # the resulting engine to be that of bargs. This can be assured by
        # setting the engine of the first argument to tabular_map (bfunc) to
        # be that of bargs. Same for ttype being STANDARD_TTYPE
        bfunc.engine = bargs.engine
        bfunc.ttype = bargs.ttype
        return tabular_map(
            (call_args_kwargs, self._engine), bfunc, bargs, bkwargs
        )

    def __str__(self):
        return f"Tabularized:\nfunc: {self._func.__name__}\nengine: {self._engine}"

    def __repr__(self):
        return str(self)

def tabularize(func, engine=None):
    """
    Decorates the given function so that it can work on NTables.
    Parameters
    ----------
    func (callable): The function to decorate
    engine: The engine to be used in the tabularized call. Default is None.

    Returns
    -------

    """
    if engine is None:
        engine = StandardEngine()
    return _Tabularized(func, engine)
