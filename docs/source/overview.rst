Overview
********

N-Tables
--------
The core data structure that |project| has to offer is the N-table. The
N-table data structure is an N-dimensional table-like structure
(hence the name) that can contain any python object as one of its
elements. Rather than have it explained in detail, let's simply take
a look at one:

.. code-block:: python

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
    int|float|str

Above is the representation of an N-table object in the python console. it
has two rows and two columns and contains various representations of the
number three (varying in data type). The representation also describes bits
of meta-data that the N-table makes use of.

.. code-block:: python

    dim1 col1     col2
    dim0              
    row1    3      "3"
    row2  3.0  "three"

This section describes the overall structure. In row1, col2 exists
the element "3". Note that these labels are just that. It could easily
be that the labels were as follows:

.. code-block:: python

    dim1 row1     row2
    dim0              
    col1    3      "3"
    col2  3.0  "three"

with col1 and col2 being where one would expect the rows and vice versa. These
labels are essentially arbitrary and typically created by the user when 
creating N-tables from other data types.

The dim0 and dim1 text indicates which dimension the labels belong to.
This is further made clear in the coordinates section of the output:

.. code-block:: python

    Coordinates:
    * dim0     (dim0) <U4 'row1' 'row2'
    * dim1     (dim1) <U4 'col1' 'col2'

As with the indices along each dimension, the dimensions themselves
are labled as well. Note that in most cases **the order of the dimensions
is less important than the names of the dimensions.** That is to say 
under most circumstances (especially operations between N-tables)
the previous N-table behaves the same as the following one:

.. code-block:: python

    dim0 row1     row2
    dim1              
    col1    3      3.0
    col2  "3"  "three"
    Coordinates:
    * dim0     (dim0) <U4 'row1' 'row2'
    * dim1     (dim1) <U4 'col1' 'col2'
    Engine:
    Standard (serial) Engine
    Ttype:
    int|float|str

----

.. code-block:: python

    Engine:
    Standard (serial) Engine

The above portion of the N-table representation indicates the type of
engine that will be used for processing. In this case the engine assigned
to the N-table is a standard serial engine. This means that unless specifically
told otherwise, processing on the N-table will be done in serial.

Technically, the engine assigned to the N-table can be any callable that
behaves as the map built-in function i.e. one that takes in a function
to be called, followed by n arguments to be passed to the input function.
That being said, it is recommended to use those offered by the |project|.engines
module where one can find the standard serial engine as well as a thread engine
and a process engine that can be used to process an N-table in a multi-threaded
or multi-process way.

.. code-block:: python

    Ttype:
    int|float|str

This final portion of the output indicates the type of objects the N-table expects
to contain, specifically with respect to get-attribute operations. What this means
is that trying to access attributes of the N-table, the N-table will check to see
if the requested attribute exists in the ttype i.e. exists in one of the types
defined by the ttype. If the attribute exists in the ttype (and not in the N-table),
then the get-attribute call will return an N-table whose elements are the requested
attribute of the initial N-table's elements:

.. code-block:: python

    In [22]: ntbl
    Out[22]: 

    sims                           sim0  ...                      sim2
    variables                            ...                          
    var0       ndarray,(100, 3),float64  ...  ndarray,(100, 3),float64
    var1       ndarray,(100, 3),float64  ...  ndarray,(100, 3),float64
    var2       ndarray,(100, 3),float64  ...  ndarray,(100, 3),float64
    var3       ndarray,(100, 3),float64  ...  ndarray,(100, 3),float64
    var4       ndarray,(100, 3),float64  ...  ndarray,(100, 3),float64
    [5 rows x 3 columns]
    Coordinates:
    * sims       (sims) <U4 'sim0' 'sim1' 'sim2'
    * variables  (variables) <U4 'var0' 'var1' 'var2' 'var3' 'var4'
    Engine:
    Standard (serial) Engine
    Ttype:
    ndarray



    In [23]: ntbl.T
    Out[23]: 
    sims                           sim0  ...                      sim2
    variables                            ...                          
    var0       ndarray,(3, 100),float64  ...  ndarray,(3, 100),float64
    var1       ndarray,(3, 100),float64  ...  ndarray,(3, 100),float64
    var2       ndarray,(3, 100),float64  ...  ndarray,(3, 100),float64
    var3       ndarray,(3, 100),float64  ...  ndarray,(3, 100),float64
    var4       ndarray,(3, 100),float64  ...  ndarray,(3, 100),float64

    [5 rows x 3 columns]
    Coordinates:
    * sims       (sims) <U4 'sim0' 'sim1' 'sim2'
    * variables  (variables) <U4 'var0' 'var1' 'var2' 'var3' 'var4'
    Engine:
    Standard (serial) Engine
    Ttype:
    ndarray

Tabularization
--------------
If N-tables are the core data structure of |project|, then tabularization is
its core functionality. Tabularization is the process converting functions
that work on regular python data types into functions that "just know"
how to work with N-table objects. Let's look at an example. Consider
an N-table like so:

.. code-block:: python

    cols      col0 col1 col2
    rows               
    row0         0    0    0
    row1         0    1    2
    row2         0    2    4
    row3         0    3    6
    row4         0    4    8
    Coordinates:
    * cols     (cols) <U4 'col0' 'col1' 'col2'
    * rows     (rows) <U4 'row0' 'row1' 'row2' 'row3' 'row4'
    Engine:
    Standard (serial) Engine
    Ttype:
    int

Suppose we wanted to apply a basic function to every element of the N-table.
Such a function would look like this:

.. code-block:: python

    def add1(x):
        return x + 1

Tabularization wraps the function so that it can take N-tables as arguments
with the result being that of the function broadcasted to every element in
the N-table:

.. code-block:: python

    tabularized_add1 = tabularize(add1)

    tabularized_add1(ntbl)

    Out[10]: 
    cols      col0 col1 col2
    rows               
    row0         1    1    1
    row1         1    2    3
    row2         1    3    5
    row3         1    4    7
    row4         1    5    9
    Coordinates:
    * cols     (cols) <U4 'col0' 'col1' 'col2'
    * rows     (rows) <U4 'row0' 'row1' 'row2' 'row3' 'row4'
    Engine:
    Standard (serial) Engine
    Ttype:
    int

Note that the add1 function is for example only. The NTable class
overrides many built in operations including __add__ (and all other
arithmetic operations) to automatically tabularize the process for you.
You can simply do the following and be done with it:

.. code-block:: python

    ntbl + 1
    Out[12]: 
    cols      col0 col1 col2
    rows               
    row0         1    1    1
    row1         1    2    3
    row2         1    3    5
    row3         1    4    7
    row4         1    5    9
    Coordinates:
    * cols     (cols) <U4 'col0' 'col1' 'col2'
    * rows     (rows) <U4 'row0' 'row1' 'row2' 'row3' 'row4'
    Engine:
    Standard (serial) Engine
    Ttype:
    int

Furthermore, the tabularize function can be used in python wrapper syntax:

.. code-block:: python

    @tabularize
    def add1(x):
        return x + 1

----

Now, consider a slightly more complicated function of adding two different values
together and returning the result:

.. code-block:: python

    @tabularize
    def add(x,y):
        return x + y

To help see what goes on we will use the following N-table:

.. code-block:: python

    cols    col0    col1    col2
    rows                        
    row0  "r0c0"  "r0c1"  "r0c2"
    row1  "r1c0"  "r1c1"  "r1c2"
    row2  "r2c0"  "r2c1"  "r2c2"
    row3  "r3c0"  "r3c1"  "r3c2"
    row4  "r4c0"  "r4c1"  "r4c2"
    Coordinates:
    * cols     (cols) <U4 'col0' 'col1' 'col2'
    * rows     (rows) <U4 'row0' 'row1' 'row2' 'row3' 'row4'
    Engine:
    Standard (serial) Engine
    Ttype:
    str


Calling the tabularized add function with both arguments as the above N-table
we get:

.. code-block:: python

    add(ntbl, ntbl)
    Out[17]: 
    cols        col0        col1        col2
    rows                                    
    row0  "r0c0r0c0"  "r0c1r0c1"  "r0c2r0c2"
    row1  "r1c0r1c0"  "r1c1r1c1"  "r1c2r1c2"
    row2  "r2c0r2c0"  "r2c1r2c1"  "r2c2r2c2"
    row3  "r3c0r3c0"  "r3c1r3c1"  "r3c2r3c2"
    row4  "r4c0r4c0"  "r4c1r4c1"  "r4c2r4c2"
    Coordinates:
    * cols     (cols) <U4 'col0' 'col1' 'col2'
    * rows     (rows) <U4 'row0' 'row1' 'row2' 'row3' 'row4'
    Engine:
    Standard (serial) Engine
    Ttype:
    str

Which is straight forward (elements rowi,colj from each table are added together).

The following is much more interesting:

.. code-block:: python

    add(ntbl, ntbl.cols["col1"])
    Out[22]: 
    cols        col0        col1        col2
    rows                                    
    row0  "r0c0r0c1"  "r0c1r0c1"  "r0c2r0c1"
    row1  "r1c0r1c1"  "r1c1r1c1"  "r1c2r1c1"
    row2  "r2c0r2c1"  "r2c1r2c1"  "r2c2r2c1"
    row3  "r3c0r3c1"  "r3c1r3c1"  "r3c2r3c1"
    row4  "r4c0r4c1"  "r4c1r4c1"  "r4c2r4c1"
    Coordinates:
    * cols     (cols) <U4 'col0' 'col1' 'col2'
    * rows     (rows) <U4 'row0' 'row1' 'row2' 'row3' 'row4'
    Engine:
    Standard (serial) Engine
    Ttype:
    str

As you can see calls to tabularized functions with inputs of different coordinates
attempts to do label based broadcasting. In this example this results in an
N-table whose elements are the result of adding elements rowi,colj from the first
N-table with elements rowi,col1 from the second N-table.

N-table Tabularized Operations
------------------------------
The NTable class defines many methods that override operations so that their
behavior is tabularized. The operations that are overridden in such a way are:

#. __getattr__
#. __getitem__
#. __setitem__
#. __iter__
#. __call__

NTable also inherits from numpy.lib.mixins.NDArrayOperatorsMixin and implements
__array_ufunc__ and __array_func__. This means that all arithmetic operations
are tabularized by default as well as numpy funcs and ufuncs. For those
unfamiliar, ufuncs are numpy functions that act on numpy arrays in a broadcasted
manner. Numpy funcs on the other hand act between numpy arrays but are for things
like manipulating the array structures (like stacking/concatenation operations).

Let's go over what each of these mean for N-table operations.

**__getattr__**

The implementation of __getattr__ is such that it will attempt to pass the act
of getting the requested attribute onto the elements of the N-table. This only
happens if the following two conditions are met:

#. The requested attribute is does not share its name with one of the dimensions
#. The requested attribute exists as an element in one of the types defined in the N-table's ttype.

Note that __getattr__ is only called if the requested attribute is not found
in the NTable class. This means that attributes in the NTable class will shadow
those found in the N-table's elements. This ultimately places a third constraint
on when the tabularized __getattr__ behavior will occur: when the requested
attribute is not also an attribute of the NTable class.

Here is an example of what this looks like using the previous N-table with
numpy arrays as elements:

.. code-block:: python

    ntbl
    Out[29]: 
    cols                      sim0  ...                      sim2
    rows                            ...                          
    var0  ndarray,(100, 3),float64  ...  ndarray,(100, 3),float64
    var1  ndarray,(100, 3),float64  ...  ndarray,(100, 3),float64
    var2  ndarray,(100, 3),float64  ...  ndarray,(100, 3),float64
    var3  ndarray,(100, 3),float64  ...  ndarray,(100, 3),float64
    var4  ndarray,(100, 3),float64  ...  ndarray,(100, 3),float64
    [5 rows x 3 columns]
    Coordinates:
    * cols     (cols) <U4 'sim0' 'sim1' 'sim2'
    * rows     (rows) <U4 'var0' 'var1' 'var2' 'var3' 'var4'
    Engine:
    Standard (serial) Engine
    Ttype:
    ndarray

Requesting the "T" attribute will return the transpose of all arrays:

.. code-block:: python

    ntbl.T
    Out[30]: 
    cols                      sim0  ...                      sim2
    rows                            ...                          
    var0  ndarray,(3, 100),float64  ...  ndarray,(3, 100),float64
    var1  ndarray,(3, 100),float64  ...  ndarray,(3, 100),float64
    var2  ndarray,(3, 100),float64  ...  ndarray,(3, 100),float64
    var3  ndarray,(3, 100),float64  ...  ndarray,(3, 100),float64
    var4  ndarray,(3, 100),float64  ...  ndarray,(3, 100),float64
    [5 rows x 3 columns]
    Coordinates:
    * cols     (cols) <U4 'sim0' 'sim1' 'sim2'
    * rows     (rows) <U4 'var0' 'var1' 'var2' 'var3' 'var4'
    Engine:
    Standard (serial) Engine
    Ttype:
    ndarray

Requesting the "shape" attribute will return the shape of all arrays:

.. code-block:: python

    ntbl.shape
    Out[5]: 
    cols      sim0      sim1      sim2
    rows                              
    var0  (100, 3)  (100, 3)  (100, 3)
    var1  (100, 3)  (100, 3)  (100, 3)
    var2  (100, 3)  (100, 3)  (100, 3)
    var3  (100, 3)  (100, 3)  (100, 3)
    var4  (100, 3)  (100, 3)  (100, 3)
    Coordinates:
    * cols     (cols) <U4 'sim0' 'sim1' 'sim2'
    * rows     (rows) <U4 'var0' 'var1' 'var2' 'var3' 'var4'
    Engine:
    Standard (serial) Engine
    Ttype:
    tuple

**__getitem__**

The implementation of __getitem__ is extremely straight forward. It is roughly
equivalent to ``tabularize(op.getitem)(self, index)`` which behaves like so:

.. code-block:: python

    ntbl[3:10,1]
    Out[33]: 
    cols                  sim0                  sim1                  sim2
    rows                                                                  
    var0  ndarray,(7,),float64  ndarray,(7,),float64  ndarray,(7,),float64
    var1  ndarray,(7,),float64  ndarray,(7,),float64  ndarray,(7,),float64
    var2  ndarray,(7,),float64  ndarray,(7,),float64  ndarray,(7,),float64
    var3  ndarray,(7,),float64  ndarray,(7,),float64  ndarray,(7,),float64
    var4  ndarray,(7,),float64  ndarray,(7,),float64  ndarray,(7,),float64
    Coordinates:
    * cols     (cols) <U4 'sim0' 'sim1' 'sim2'
    * rows     (rows) <U4 'var0' 'var1' 'var2' 'var3' 'var4'
    Engine:
    Standard (serial) Engine
    Ttype:
    ndarray

Note that ANY python object can be used as the index for the __getitem__ call.
As long as the underlying elements know how to handle it, it will succeed.

**__setitem__**

Like __getitem__, __setitem__ is also straight forward. It's implementation
is roughly equivalent to ``tabularize(setitem)(self, index, value)`` where
setitem is a function defined in tapr.utils to assign value at index:

.. code-block:: python

    def setitem(obj, index, value):
        obj[index] = value

Calling setitem on a N-table mutates the elements instead of returning
a new N-table. The the array representation in the N-table representation
does not show us the elements of the arrays, so let's see what a single
element looks like and mutate that:

.. code-block:: python

    ntbl[32,1]
    Out[36]: 
    cols                sim0                sim1                sim2
    rows                                                            
    var0  12.660963148004477   8.323231508004868  10.997459741977162
    var1    5.17516835334136   12.35698720254154   8.020847850459557
    var2   4.922614186527418   9.445586347458322   6.293137027630474
    var3  12.379655781877489  0.8503742699718885  2.8541487997373807
    var4  2.4158605415331804   6.822124587744828  12.428677900672248
    Coordinates:
    * cols     (cols) <U4 'sim0' 'sim1' 'sim2'
    * rows     (rows) <U4 'var0' 'var1' 'var2' 'var3' 'var4'
    Engine:
    Standard (serial) Engine
    Ttype:
    float64

Setting the value to be 3...

.. code-block:: python

    ntbl[32,1] = 3
    ntbl[32,1]
    Out[38]: 
    cols sim0 sim1 sim2
    rows               
    var0  3.0  3.0  3.0
    var1  3.0  3.0  3.0
    var2  3.0  3.0  3.0
    var3  3.0  3.0  3.0
    var4  3.0  3.0  3.0
    Coordinates:
    * cols     (cols) <U4 'sim0' 'sim1' 'sim2'
    * rows     (rows) <U4 'var0' 'var1' 'var2' 'var3' 'var4'
    Engine:
    Standard (serial) Engine
    Ttype:
    float64

Again, ANY python object can be used as the index (and in this case the value) 
for the __setitem__ call. As long as the underlying elements know how to
handle it, it will succeed.

**__iter__**

The __iter__ override allows users to iterate over N-tables in a tabularized
way. What this means is that iterating over an N-table yields N-tables
whose elements are the result of a tabularized next call on an N-table
of iterators (generated by a tabularized iter call on the original N-table):

.. code-block:: python

    for ntbl_ in ntbl[0]:
        print(ntbl_)
    cols                sim0                sim1                sim2
    rows                                                            
    var0  4.6370182423803925    7.43727940466936  11.555725707020242
    var1   8.569378576522295  1.2896458719847574  12.214520775830003
    var2   5.594850992330632  11.434225783682663  11.346364921189943
    var3   8.336795515669248   4.852632215426621   8.909488743969845
    var4  2.0886620524994957   8.873996466747679  10.294609882384023
    Coordinates:
    * cols     (cols) <U4 'sim0' 'sim1' 'sim2'
    * rows     (rows) <U4 'var0' 'var1' 'var2' 'var3' 'var4'
    Engine:
    Standard (serial) Engine
    Ttype:
    float64
    cols                sim0                sim1                sim2
    rows                                                            
    var0    2.18269163824821   1.644767178903004     8.1464273583552
    var1  2.3118542125361277  13.147541697016267  0.5620385150049827
    var2  11.533282359428394   8.103574131869937   10.83730842522449
    var3   2.261930164372926    4.09758853305501   4.874004598382216
    var4   5.675965717903967   12.90618772911868  1.5858550186317601
    Coordinates:
    * cols     (cols) <U4 'sim0' 'sim1' 'sim2'
    * rows     (rows) <U4 'var0' 'var1' 'var2' 'var3' 'var4'
    Engine:
    Standard (serial) Engine
    Ttype:
    float64
    cols                sim0                sim1                sim2
    rows                                                            
    var0  10.598092574139145   5.668263266504647   7.876175580599507
    var1  13.249840422945903   5.120160504001887   5.592196157556078
    var2  10.060509315306659  12.872704062538952   5.015425179453149
    var3  5.9139017658535655  2.5073854946740144  1.7369189578005717
    var4   5.414148016751149   8.804168665664582   11.06059335848008
    Coordinates:
    * cols     (cols) <U4 'sim0' 'sim1' 'sim2'
    * rows     (rows) <U4 'var0' 'var1' 'var2' 'var3' 'var4'
    Engine:
    Standard (serial) Engine
    Ttype:
    float64


**__call__**

Finally, the __call__ override will tabularize the call function
and apply it to the originating N-table. This is roughly equivalent
to ``tabularize(call)(self, *args, **kwargs)`` where call is defined in tapr.utils
as 

.. code-block:: python

    def call(func, *args, **kwargs):
        return func(*args, **kwargs)

This of course will work only if the elements of the N-table are callables.
While there are multiple ways you might end up with callables as elements,
the most likely one is probably when trying to call a method you expect 
to find in your N-table elements. Using the N-table with strings from
earlier:

.. code-block:: python

    cols    col0    col1    col2
    rows                        
    row0  "r0c0"  "r0c1"  "r0c2"
    row1  "r1c0"  "r1c1"  "r1c2"
    row2  "r2c0"  "r2c1"  "r2c2"
    row3  "r3c0"  "r3c1"  "r3c2"
    row4  "r4c0"  "r4c1"  "r4c2"
    Coordinates:
    * cols     (cols) <U4 'col0' 'col1' 'col2'
    * rows     (rows) <U4 'row0' 'row1' 'row2' 'row3' 'row4'
    Engine:
    Standard (serial) Engine
    Ttype:
    str

Attempting to call the .upper() method on the N-table happens in two parts.
First, the tabularized N-table __getattr__ method is called for .upper:

.. code-block:: python

    ntbl.upper
    Out[7]: 
    cols                        col0  ...                        col2
    rows                              ...                            
    row0  builtin_function_or_method  ...  builtin_function_or_method
    row1  builtin_function_or_method  ...  builtin_function_or_method
    row2  builtin_function_or_method  ...  builtin_function_or_method
    row3  builtin_function_or_method  ...  builtin_function_or_method
    row4  builtin_function_or_method  ...  builtin_function_or_method
    [5 rows x 3 columns]
    Coordinates:
    * cols     (cols) <U4 'col0' 'col1' 'col2'
    * rows     (rows) <U4 'row0' 'row1' 'row2' 'row3' 'row4'
    Engine:
    Standard (serial) Engine
    Ttype:
    builtin_function_or_method

Finally, the tabularized __call__ method takes place on the N-table of
upper methods:

.. code-block:: python

    ntbl.upper()
    Out[8]: 
    cols    col0    col1    col2
    rows                        
    row0  "R0C0"  "R0C1"  "R0C2"
    row1  "R1C0"  "R1C1"  "R1C2"
    row2  "R2C0"  "R2C1"  "R2C2"
    row3  "R3C0"  "R3C1"  "R3C2"
    row4  "R4C0"  "R4C1"  "R4C2"
    Coordinates:
    * cols     (cols) <U4 'col0' 'col1' 'col2'
    * rows     (rows) <U4 'row0' 'row1' 'row2' 'row3' 'row4'
    Engine:
    Standard (serial) Engine
    Ttype:
    str


Tabulation (and un-Tabulation)
------------------------------
The final concept that is worth discussing at the overview level is that of
tabulation. Tabulation is the process of "tabulating" collections of N-table
(and potentially non-N-table) objects into a single N-table whose elements are
collections of the same type. Here is what that looks like for a collection of
N-tables:

.. code-block:: python

    ntbl = ntable({"row1":{"col1": 3, "col2":"3"}, "row2":{"col1": 3.0, "col2":"three"}})

    tuple_of_ntables = (ntbl, ntbl)

    ntable_of_tuples = tabulate(tuple_of_ntables)

.. code-block:: python

    tuple_of_ntables
    Out[5]: 
    (dim1 col1     col2
    dim0              
    row1    3      "3"
    row2  3.0  "three"
    Coordinates:
    * dim0     (dim0) <U4 'row1' 'row2'
    * dim1     (dim1) <U4 'col1' 'col2'
    Engine:
    Standard (serial) Engine
    Ttype:
    int|float|str,
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
    int|float|str)

.. code-block:: python

    ntable_of_tuples
    Out[7]: 
    dim1        col1                col2
    dim0                                
    row1      (3, 3)          ('3', '3')
    row2  (3.0, 3.0)  ('three', 'three')
    Coordinates:
    * dim0     (dim0) <U4 'row1' 'row2'
    * dim1     (dim1) <U4 'col1' 'col2'
    Engine:
    Standard (serial) Engine
    Ttype:
    int|float|str

And for a tuple of an N-table and another object:

.. code-block:: python

    tabulate((ntbl, 10))
    Out[9]: 
    dim1       col1           col2
    dim0                          
    row1    (3, 10)      ('3', 10)
    row2  (3.0, 10)  ('three', 10)
    Coordinates:
    * dim0     (dim0) <U4 'row1' 'row2'
    * dim1     (dim1) <U4 'col1' 'col2'
    Engine:
    Standard (serial) Engine
    Ttype:
    int|float|str

But why is this needed? Well, from the user's perspective it admittedly might
not come up often. |project| mostly makes use of it internally in order to
prepare arguments into tabular functions so that the underlying routine
can just be called on each element of the N-table generated by tabulating the
arguments.

Un-tabulation of N-tables (the act of taking an N-table of collections and
converting it into a collection of N-tables) is implemented by simply
iterating over the N-table of collections:


.. code-block:: python

    tuple(nt for nt in tabulate((ntbl, 10)))
    Out[10]: 
    (dim1 col1     col2
    dim0              
    row1    3      "3"
    row2  3.0  "three"
    Coordinates:
    * dim0     (dim0) <U4 'row1' 'row2'
    * dim1     (dim1) <U4 'col1' 'col2'
    Engine:
    Standard (serial) Engine
    Ttype:
    int|float|str,
    dim1 col1 col2
    dim0          
    row1   10   10
    row2   10   10
    Coordinates:
    * dim0     (dim0) <U4 'row1' 'row2'
    * dim1     (dim1) <U4 'col1' 'col2'
    Engine:
    Standard (serial) Engine
    Ttype:
    int)

This is particularly useful when working with tabularized functions that usually
return multiple objects. The tabular version of the function will always return
a single object (an N-table of tuples):

.. code-block:: python

    def multi_output_function(value1, value2):
        return value2, value1
    result = tabularize(multi_output_function)(ntbl, 21)
    result
    Out[5]: 
    dim1       col1           col2
    dim0                          
    row1    (21, 3)      (21, '3')
    row2  (21, 3.0)  (21, 'three')
    Coordinates:
    * dim0     (dim0) <U4 'row1' 'row2'
    * dim1     (dim1) <U4 'col1' 'col2'
    Engine:
    Standard (serial) Engine
    Ttype:
    tuple

But the base function would typically be called like so:

.. code-block:: python

    result1, result2 = multi_output_function(3,4)

It would be nice if we can use the same syntax when calling a tabular function
that we use when calling the base function would it not? Fortunately because
of how the __iter__ method is defined for the NTable class, this is totally
possible:

.. code-block:: python

    ntbl1, ntbl2 = tabularize(multi_output_function)(ntbl, 21)
    ntbl1
    Out[8]: 
    dim1 col1 col2
    dim0          
    row1   21   21
    row2   21   21
    Coordinates:
    * dim0     (dim0) <U4 'row1' 'row2'
    * dim1     (dim1) <U4 'col1' 'col2'
    Engine:
    Standard (serial) Engine
    Ttype:
    int
    ntbl2
    Out[9]: 
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
    int|float|str