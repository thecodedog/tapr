import plotly.offline as module_

from tapr.main.tabularization import tabularize

# tabularize any callables in plotly.express
for name in dir(module_):
    value = getattr(module_,name)
    if callable(value):
        globals()[name] = tabularize()(value)
    else:
        globals()[name] = value