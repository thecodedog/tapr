Pandas
******

If the data to be worked with is already in a Pandas structure, then
chances are that Pandas already offers the functionality that is needed
to work with the data in question. That being said, there are times where
Tapr is the desired library to work with because other sets of data need
to be approached with Tapr, but some of data arrises in the form of Pandas
structures (perhaps configuraiton data loaded in from a csv). In such a
scenario, Tapr's ntable conversion function will nicely convert from pandas
DataFrames/Series to N-tables:

.. code-block:: python

    import pandas as pd
    import tapr as tp
    dframe = pd.DataFrame({"col1":{"row1": 3, "row2":"3"}, "col2":{"row1": 3.0, "row2":"three"}})
    series = dframe["col1"]

    ntbl_dframe = tp.ntable(dframe)
    ntbl_series = tp.ntable(series)

DataFrame:

.. code-block:: python

    dframe
    Out[10]: 
        col1   col2
    row1    3    3.0
    row2    3  three

.. code-block:: python

    ntbl_dframe
    Out[11]: 
    cols col1     col2
    rows              
    row1    3      3.0
    row2  "3"  "three"
    Coordinates:
    * rows     (rows) <U4 'row1' 'row2'
    * cols     (cols) <U4 'col1' 'col2'
    Engine:
    Standard (serial) Engine
    Ttype:
    float|int|str


Series:

.. code-block:: python

    series
    Out[4]: 
    row1    3
    row2    3
    Name: col1, dtype: object

.. code-block:: python

    ntbl_series
    Out[5]: 
    rows
    row1      3
    row2    "3"
    dtype: object
    Coordinates:
    * rows     (rows) <U4 'row1' 'row2'
    Engine:
    Standard (serial) Engine
    Ttype:
    int|str

