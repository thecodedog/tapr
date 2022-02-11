Dictionaries
************

The most common and default form of labeled, structured data is of course
the dictionary. Tapr allows for easy conversion from this nested structure
to a tabular N-table with the ntable conversion function:

.. code-block:: python

    import tapr as tp

    ntbl = tp.ntable({"row1":{"col1": 3, "col2":"3"}, "row2":{"col1": 3.0, "col2":"three"}})
    
    ntbl
    Out[5]: 
    dim1 col1     col2
    dim0              
    row1    3      "3"
    row2  3.0  "three"
    Coordinates:
    * dim0     (dim0) <U4 'row1' 'row2'
    * dim1     (dim1) <U4 'col1' 'col2'
    Engine:
    Standard (serial) Engine
    Ttype:
    float|int|str

Note that the outer most entries are assumed to be the rows, the next outer most
keys the columns, and so on.

The ntable conversion function can also handle dictionaries that are "missing"
data:

.. code-block:: python

    ntbl = tp.ntable({"row1":{"col1": 3}, "row2":{"col1": 3.0, "col2":"three"}})
    ntbl
    Out[7]: 
    dim1 col1     col2
    dim0              
    row1    3     NULL
    row2  3.0  "three"
    Coordinates:
    * dim0     (dim0) <U4 'row1' 'row2'
    * dim1     (dim1) <U4 'col1' 'col2'
    Engine:
    Standard (serial) Engine
    Ttype:
    NULL|float|int|str

Since N-tables must be tabular, it substitutes a Tapr defined object called NULL.
NULL can be thought of as being a generalized NaN; while NaN has the property
of causing mathematical operations on it to return NaN, NULL has the property
of returning NULL when ANY operation is applied to it.

.. code-block:: python

    null = tp.utils.NULL()

    null + 1
    Out[13]: NULL

.. code-block:: python

    null[0]
    Out[14]: NULL

.. code-block:: python

    null("abcdefg")
    Out[15]: NULL


