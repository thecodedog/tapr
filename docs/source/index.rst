.. Tapr documentation master file, created by
   sphinx-quickstart on Tue Nov 16 17:44:40 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Tapr's documentation!
================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   overview
   creating_ntables
   tutorials
   changelog
   modules
   



What is |project|?
==================

|project| is a library to facilitate TAbular PRogramming in python.
Tabular programming is a programming paradigm that revolves around structuring
not only data, but also functionality into a table or table-like fashion.
For those that are familiar with libraries like numpy, pandas, and xarray,
tabular programming extends the concept of broadcasting to **ANY** python
object or operation; the same way that these libraries get rid of the need
for loops when working with primitive data types, |project| gets
rid of the need for loops when working with any type of python object.
For example, the task of extracting the 2nd character of each element in
a dictionary of strings:

.. code-block:: python

    data = {'row1': {'col1': "1234", 'col2': "5678"}, 'row2': {'col1': "3000", 'col2': "heyo"}}
    new_data = {}
    for k, v in data.items():
        new_data[k] = {}
        for k1, v1 in v.items():
            new_data[k][k1] = v1[1]
    new_data
    Out[3]: {'row1': {'col1': '2', 'col2': '6'}, 'row2': {'col1': '0', 'col2': 'e'}}

becomes:

.. code-block:: python

    ntbl = ntable({'row1': {'col1': "1234", 'col2': "5678"}, 'row2': {'col1': "3000", 'col2': "heyo"}})
    
    new_ntbl = ntbl[1]

    new_ntbl
    Out[4]: 
    dim1 col1 col2
    dim0          
    row1  "2"  "6"
    row2  "0"  "e"
    Coordinates:
    * dim0     (dim0) <U4 'row1' 'row2'
    * dim1     (dim1) <U4 'col1' 'col2'
    Engine:
    Standard (serial) Engine
    Ttype:
    str


Why use |project|?
------------------
Programmers, developers, and data scientists will all find |project| to be of 
use not only when they need to work with data that is inherently tabular but 
also when the logic to be applied to said data has an inherently tabular
structure. While libraries like numpy, pandas, and xarray facilitate similar
patterns, not all operations and data types are handled as nicely as they are
in |project|.

For a better idea of what |project| has to offer, checkout the |project| :doc:`Overview <./overview>`.

Why NOT use |project|?
----------------------
While |project| can do more with more types of data than its python peers, it
ultimately does so at a cost to its speed. The reason is that the ability to 
work with mixed, non-primitive data types means that the processing needs to
take place in pure python. If your data fits nicely into numpy, pandas, or
xarray, and they are capable of supporting the operations you need to do, then
you should use them instead.

Comparison to other libraries:
------------------------------
=======  ============= =========== ============ ================== 
Library  N-Dimensional Label Based Broadcasting Arbitrary Elements 
=======  ============= =========== ============ ==================
tapr           X            X           X               X 
numpy          X                        X                   
pandas                      X           X                    
xarray         X            X           X                    
=======  ============= =========== ============ ==================

Installation
------------
|project| can be installed via pip

``pip install tapr``


License
-------
|project| is licensed under the `MIT license. <https://bitbucket.org/elspacedoge/tapr/src/master/LICENSE>`_