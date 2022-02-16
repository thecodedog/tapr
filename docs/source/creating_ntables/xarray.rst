Xarray
******

As with pandas data, chances are if the data you have fits nicely into Xarray
then you should consider why exactly you need to convert into an N-table; Xarray
uses numpy in its processing and will therefore be faster than Tapr's pure python
implementation. That being said if part of your workflow is best done with tapr,
then converting from xarray DataArrays to N-tables is possible:

.. code-block:: python

    import numpy as np
    import xarray as xr
    import tapr as tp

    darray = xr.DataArray(
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape((4, 3)),
        coords={"rows": [f"row{i}" for i in range(4)], "cols": [f"col{i}" for i in range(3)]},
        dims=("rows", "cols"),
    )

.. code-block:: python

    darray
    Out[11]: 
    <xarray.DataArray (rows: 4, cols: 3)>
    array([[ 1,  2,  3],
        [ 4,  5,  6],
        [ 7,  8,  9],
        [10, 11, 12]])
    Coordinates:
    * rows     (rows) <U4 'row0' 'row1' 'row2' 'row3'
    * cols     (cols) <U4 'col0' 'col1' 'col2'


.. code-block:: python

    ntbl
    Out[12]: 
    cols col0 col1 col2
    rows               
    row0    1    2    3
    row1    4    5    6
    row2    7    8    9
    row3   10   11   12
    Coordinates:
    * rows     (rows) <U4 'row0' 'row1' 'row2' 'row3'
    * cols     (cols) <U4 'col0' 'col1' 'col2'
    Engine:
    Standard (serial) Engine
    Ttype:
    int64
