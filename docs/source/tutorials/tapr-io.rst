Tapr IO
*******
|project| understands that most tabular forms of data already have plenty
of very well written libraries that support IO for them. As a result,
|project| focuses on conversion from in-memory objects like dataframes and
data-arrays (pandas and xarray cover a large amount of IO for tabular data)
to N-tables, ultimately relying on the user to load in data using existing
libraries. That being said, |project| needs a way to offload more types of
data than can be done with other libraries. Because of this, |project| also
provides its own file type which is also discussed here.

Loading in a .csv
-----------------

.. code-block:: python

    import pandas as pd
    from tapr.main.conversion import ntable

    # Load in the csv as a dataframe. Note that in this example
    # it is expected that the 0th column in the csv corresponds
    # to the indexes (row labels) and so the read_csv function
    # must be informed of that.
    dframe = pd.read_csv("data.csv", index_col=0)
    # Pass the data frame into the ntable conversion function.
    # Note that the input data frame is transposed. This is
    # because data frames are column oriented (first order
    # indexing accesses columns) while N-tables are row oriented
    ntbl = ntable(dframe.T)



Saving an N-table to .csv
-------------------------

.. code-block:: python

    # Saving a compatible N-table to a csv is as simple as
    # converting it to a data frame and imediately calling
    # the to_csv() data frame method.
    ntbl.to_pandas().to_csv("data1.csv")


Loading in an hdf5 file
-----------------------

The h5py File object behaves like a hdf5 group which
in turn behaves like a dictionary. As such, the ntable
conversion method treats it like a dictionary, returning
an N-table of hdf5 dataset objects. Indexing said N-table
with the ellipses object actually loads the data into
in-memory numpy arrays.

.. code-block:: python

    import h5py
    from tapr.main.conversion import ntable

    with h5py.File("data.hdf5", "r") as h5file:
        ntbl = ntable(h5file)[...]


Saving an N-table to an hdf5 file
---------------------------------

As is the case with loading in hdf5 files, saving hdf5
files also takes advantage of the h5py File object's
dictionary-like behavior. However, the to_dictionary
NTable method needs its input to behave a bit more
like a dictionary than the h5py File object actually
does. As a result, saving to an hdf5 file requires
one small wrapper around the File object to force it
to dictionary-like enough for the to_dictionary
method to work.

.. code-block:: python

    import h5py
    from tapr.main.utils import dictifyh5

    with h5py.File(fname, "w") as h5file:
        ntbl.to_dictionary(dictifyh5(h5file))

The .ntbl file type
-------------------
Since common file formats aren't able to support ALL types of python data,
|project| provides its own file type that allows any N-table containing
any python data type to be written to disk. In order to do so however,
|project| makes use of python's pickle library which is inherently unsafe.
As such, N-table io functions make you explicitly state that you are fine
with using pickle:

.. code-block:: python

    from tapr.io.ntableio import save_ntable, load_ntable
    # save an N-table
    save_ntable(ntbl, "data.ntbl", allow_pickle=True)
    # load an N-table
    ntbl1 = load_ntable("data.ntbl", allow_pickle=True)

Failing to do so will cause an exception to be raised if the N-table contains
any data types that do not have registered serlializers:

.. code-block:: python

    class MyClass:
        def __init__(self):
            self._a = "hello darkness my old friend"
    ntbl = ntable({"row1":{"col3": 3, "col4":"3"}, "row2":{"col3": 3.0, "col4":MyClass()}})
    save_ntable(ntbl, "data.ntbl")
    ntbl1 = load_ntable("data.ntbl")

.. code-block::

    ValueError: Non-pickle serializer for type <class '__main__.MyClass'> not found and allow_pickle flag was False. Either define a non-pickle serializer or set the flag to True.

Fortunately |project| defines and registers serializers and deserializers for
many common data types:

.. code-block:: python

    ntbl = ntable({"row1":{"col3": 3, "col4":"3"}, "row2":{"col3": 3.0, "col4":np.array(3)}})
    save_ntable(ntbl, "data.ntbl")
    ntbl1 = load_ntable("data.ntbl")

The above code will run without error even though the allow_pickle flag is not set.
|project| will not try to pickle or un-pickle unless the N-table contains a data
type that does not have a registered serializer or deserializer. Check out the
:ref:`Custom serialization for N-table IO` section below.


Custom serialization for N-table IO
-----------------------------------
While |project| implements serialization/deserialization for many common
data types, it is likely a user will find themselves needing to save/load
an N-table containing some other data type that is not already handled.
This could be the result of a user creating their own data type, pulling
in data types from another library, or simply one that was not forseen
as being necessary. Either way, |project| provides a method for users
to register their own serializer/deserializer that will be used by the
N-table load/save functions:

.. code-block:: python

    from tapr.io.ntableio import save_ntable, load_ntable
    from tapr.io.serialization import serializer, deserializer

    # user defined class
    class MyClass:
        def __init__(self, info="hello darkness my old friend"):
            self._info = info
        @property
        def info(self):
            return self._info

    ntbl = ntable({"row1":{"col3": 3, "col4":"3"}, "row2":{"col3": 3.0, "col4":MyClass()}})

    # register serializer
    @serializer(MyClass, "__my_class__")
    def my_class_serializer(myclass_obj):
        return myclass_obj.info.encode()

    # register deserializer
    @deserializer("__my_class__")
    def my_class_deserializer(bytes_):
        return MyClass(bytes_.decode())

    # save and load
    save_ntable(ntbl, "data.ntbl")

    ntbl1 = load_ntable("data.ntbl")

Let's break down what exactly is happening. The user defines a class called
MyClass. It doesn't do much, but it is not a type that is handled by the
default serialization/deseralization capabilities, meaning that trying to
save and load such a class would require pickle. This can be avoided by
making use of the serializer and deserializer decorators to decorate
user defined functions that know how to serialize and deserialize MyClass
objects. The inputs into the serializer decorator are the type that the
decorated function should expect to serialize followed by a user definied
identifier. The first argument tells the N-table save function to use the
decorated function when it comes across an object of that type. The second
argument is stored in the .ntbl file. It tells the load function that it
needs to look for a deserializer with the same identifier. This is what
the argument in the deserializer decorator is for. It ultimately links
the serializer and deserializer methods; Objects serialized with identifier
id will be deserialized by deserializer with identifier id.