import importlib as il
alchemy = il.import_module(".main.alchemy", package=__name__)
conversion = il.import_module(".main.conversion", package=__name__)
defs = il.import_module(".main.defs", package=__name__)
engines = il.import_module(".main.engines", package=__name__)
filtering = il.import_module(".main.filtering", package=__name__)
handling = il.import_module(".main.handling", package=__name__)
ntable = il.import_module(".main.ntable", package=__name__)
processing = il.import_module(".main.processing", package=__name__)
qol = il.import_module(".main.qol", package=__name__)
structure = il.import_module(".main.structure", package=__name__)
tabularization = il.import_module(".main.tabularization", package=__name__)
ttypes = il.import_module(".main.ttypes", package=__name__)
utils = il.import_module(".main.utils", package=__name__)

io = il.import_module(".io", package=__name__)

from .main.conversion import ntable
from .main.qol import blank, sblank, cartograph, count
from .main.tabularization import tabularize
from .main.utils import full, full_lite, full_like, concatenate_ntables as concatenate

from .io.ntableio import save_ntable, load_ntable

_px = il.import_module(".visualization.plotly.express", package=__name__)
# look for any tabularized plotly express functions and expose them here
for name in dir(_px):
    value = getattr(_px,name)
    if isinstance(value, tabularization._Tabularized):
        globals()[name] = value