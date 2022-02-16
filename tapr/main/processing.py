import itertools as it

import numpy as np
import xarray as xr

from .utils import basic_refmap, full, full_lite, handle_improper_broadcast
from .engines import StandardEngine


def broadcast_tables(*args, lite=False):
    from .ntable import NTable

    ntbls = [arg for arg in args if isinstance(arg, NTable)]
    nons = [arg for arg in args if not isinstance(arg, NTable)]
    # ensure there is at least one ntable object
    if len(ntbls) == 0:
        raise ValueError("args must contain at least one NTable object")

    dmaps = [ntbl.refmap for ntbl in ntbls]
    dlists = [ntbl.reflist for ntbl in ntbls]
    engines = [ntbl.engine for ntbl in ntbls]
    ttypes = [ntbl.ttype for ntbl in ntbls]
    new_dmaps = xr.broadcast(*dmaps)
    new_ntbls = []
    for dlist, dmap, engine, ttype in zip(dlists, new_dmaps, engines, ttypes):
        try:
            new_ntbls.append(NTable(dlist, dmap, engine, ttype))
        except:
            # if the above fails, perhaps it was because an improper broadcast
            # took place. Try handling it.
            dlist, dmap = handle_improper_broadcast(dlist, dmap)
            new_ntbls.append(NTable(dlist, dmap, engine, ttype))

    if lite:
        non_ntbls = [
            full_lite(non, new_dmaps[0].coords, new_dmaps[0].dims)
            for non in nons
        ]
    else:
        non_ntbls = [
            full(non, new_dmaps[0].coords, new_dmaps[0].dims) for non in nons
        ]

    # return results in the right order
    result = []
    for item in args:
        if isinstance(item, NTable):
            result.append(new_ntbls.pop(0))
        else:
            result.append(non_ntbls.pop(0))

    return result


def tabular_map(func_engine, *ntable_args):
    from .ntable import NTable

    if isinstance(func_engine, tuple):
        func = func_engine[0]
        engine = func_engine[1]
    else:
        func = func_engine
        engine = StandardEngine()
    new_reflist = list(
        engine.__tapr_engine__map__(func, *(ntbl.struct.flat for ntbl in ntable_args))
    )
    new_refmap = basic_refmap(
        ntable_args[0].refmap.coords, ntable_args[0].refmap.dims
    )

    # TODO: Change validate to False once new_reflist and new_refmap
    # consistency is proven to be guaranteed
    result_engine = ntable_args[0].engine
    return NTable(new_reflist, new_refmap, result_engine, validate=True)
