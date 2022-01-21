import operator as op
import numpy as np
import pandas as pd
import datetime as dt

TAPR_RESERVED_KEYWORD = "__TAPR_RESERVED_COORDS__"

# TODO: Ensure that all operators are handled

UFUNC_TO_OP = {
    np.add: op.add,
    np.subtract: op.sub,
    np.multiply: op.mul,
    np.divide: op.truediv,
    np.equal: op.eq,
}

PRINTABLE_TYPES = {bool, int, float, pd.Timestamp, pd.Timedelta, dt.datetime}
PRINTABLE_TYPES.update(tuple(np.sctypeDict.values()))
