Example Workflow
****************

As with any library, it might not be obvious at first what exactly |project|
offers, or at least how users would go about adopting it in their own
projects/workflows. Here we offer an example work flow that is akin to the
use case that inspired the creation of |project|: working with groups of 
arrays of various length.


In this example we will be using a years worth of minute by minute stock data
for various ETFs. We will use |project| to do some basic analysis on the data
and visualize it. We will finally use |project| to save the data so that
in the future it is automatically in a structure we want to use.

Necessary Imports
-----------------

In order to perform the various steps there are a few libraries that need
to be imported:

.. code-block:: python

    import pathlib as pl
    import glob


    import numpy as np
    import pandas as pd

    import plotly.io as pio

    import tapr as tp

    pio.renderers.default = "browser"

Loading
-------

.. code-block:: python

    filepath = pl.Path(__file__)
    path = filepath.parent / "data/stocks/"
    files = [file for file in glob.glob(str(path / "*"))]

    dframes = [pd.read_csv(file, parse_dates=["timestamp"]) for file in files]

Let's process what is happening here. We first define the path to the csv
files. We then create file names for each file in the directory. Then we
use the pandas library to load in each of the files as pandas DataFrames.
There is datetime information in the file under the timestamp column, so we
let read_csv know that we would like the strings parsed as such.

The resulting dataframes look like this:

.. code-block:: python

    dframes[0]
    Out[4]: 
                        timestamp      open      high       low     close  volume
    0     2020-12-28 12:00:00+00:00  118.0000  118.0000  118.0000  118.0000    1913
    1     2020-12-28 14:30:00+00:00  118.0100  118.1900  117.9500  118.0250   49152
    2     2020-12-28 14:31:00+00:00  117.9700  117.9800  117.9500  117.9650    2961
    3     2020-12-28 14:32:00+00:00  117.9107  117.9290  117.8900  117.9000   22499
    4     2020-12-28 14:33:00+00:00  117.8900  117.9631  117.8800  117.9631    3345
                            ...       ...       ...       ...       ...     ...
    98671 2021-12-23 20:57:00+00:00  146.1660  146.1660  146.1100  146.1100   10025
    98672 2021-12-23 20:58:00+00:00  146.1200  146.1200  146.0892  146.0892    7859
    98673 2021-12-23 20:59:00+00:00  146.0700  146.0700  145.9500  145.9600   10721
    98674 2021-12-23 21:00:00+00:00  145.9700  145.9700  145.9700  145.9700    3980
    98675 2021-12-23 21:40:00+00:00  146.2700  146.2700  146.2700  146.2700     200
    [98676 rows x 6 columns]


.. code-block:: python

    dframes[1]
    Out[5]: 
                        timestamp      open      high       low     close  volume
    0     2020-12-28 14:30:00+00:00  197.7400  197.8900  197.6500  197.8800   28078
    1     2020-12-28 14:31:00+00:00  197.6982  197.9400  197.6982  197.9400     532
    2     2020-12-28 14:32:00+00:00  197.9399  197.9399  197.4950  197.7162    2439
    3     2020-12-28 14:33:00+00:00  197.2910  197.5300  197.2910  197.3500    2130
    4     2020-12-28 14:34:00+00:00  197.2500  197.2879  197.1700  197.1700    1292
                            ...       ...       ...       ...       ...     ...
    83615 2021-12-23 20:56:00+00:00  225.4400  225.4400  225.3600  225.4100    1112
    83616 2021-12-23 20:57:00+00:00  225.3400  225.3400  225.2900  225.2900    1263
    83617 2021-12-23 20:58:00+00:00  225.3000  225.3000  225.2400  225.2400    1043
    83618 2021-12-23 20:59:00+00:00  225.2800  225.2800  225.1100  225.2000   12651
    83619 2021-12-23 21:00:00+00:00  225.1800  225.1800  225.1800  225.1800    2684
    [83620 rows x 6 columns]

.. code-block:: python

    dframes[2]
    Out[6]: 
                        timestamp      open    high       low   close  volume
    0     2020-12-28 13:49:00+00:00   86.4900   86.49   86.4900   86.49     100
    1     2020-12-28 14:30:00+00:00   86.0500   86.06   86.0000   86.00    3676
    2     2020-12-28 14:31:00+00:00   85.9400   85.94   85.9400   85.94     100
    3     2020-12-28 14:34:00+00:00   85.8904   85.92   85.8701   85.91    1674
    4     2020-12-28 14:35:00+00:00   85.9100   85.91   85.9100   85.91     100
                            ...       ...     ...       ...     ...     ...
    43090 2021-12-23 20:55:00+00:00  106.3300  106.33  106.3300  106.33     289
    43091 2021-12-23 20:56:00+00:00  106.3100  106.32  106.3100  106.32     419
    43092 2021-12-23 20:58:00+00:00  106.2599  106.26  106.2400  106.24    1849
    43093 2021-12-23 20:59:00+00:00  106.2400  106.24  106.1400  106.14    1726
    43094 2021-12-23 21:00:00+00:00  106.1500  106.15  106.1500  106.15    2455
    [43095 rows x 6 columns]

and so on.


Restructuring
-------------
While pandas offers a ton of functionality to support just about any type of
processing imaginable (including the type of processing we are about to do),
|project| can as well, and is arguably a bit more clear as to what is going
on as it does it. Either way, this is a |project| example so we must work
with |project| data types. The goal is to have a single N-table that looks
something like this:

.. code-block:: python

    ticker                                VTV  ...                             VOE
    value                                      ...                                
    timestamp  ndarray,(98676,),datetime64[D]  ...  ndarray,(67932,),datetime64[D]
    open             ndarray,(98676,),float64  ...        ndarray,(67932,),float64
    high             ndarray,(98676,),float64  ...        ndarray,(67932,),float64
    low              ndarray,(98676,),float64  ...        ndarray,(67932,),float64
    close            ndarray,(98676,),float64  ...        ndarray,(67932,),float64
    volume             ndarray,(98676,),int64  ...          ndarray,(67932,),int64
    [6 rows x 12 columns]
    Coordinates:
    * value    (value) <U9 'timestamp' 'open' 'high' 'low' 'close' 'volume'
    * ticker   (ticker) <U3 'VTV' 'VB' 'MGV' 'VO' 'VOT' ... 'VV' 'VUG' 'MGC' 'VOE'
    Engine:
    Standard (serial) Engine
    Ttype:
    ndarray

It is an N-table that contains arrays representing various stock values info
at different times (including the timestamps themselves). The rows indicate
the specific piece of information while the columns indicate the specific
stock ticker the information is for.

To get to this from a list of dataframes there is some restructuring that needs
to occur. Fortunately this is quite easy to do in |project|:

The first step is to convert the dataframes to dictionaries of arrays. This
allows us to then call the ntable conversion function on each dictionary
to get a list of N-tables whose elements are the arrays we seek to have:

.. code-block:: python

    # convert dataframes into dictionaries of arrays

    dictionaries = [{k:np.array(v) for k,v in dframe.items()} for dframe in dframes]

    # create an ntable for each individual dataset.
    ntbls = [
        tp.ntable(dictionary, dims=("value",))
        for dictionary in dictionaries
    ]

    ntbls[0]
    Out[11]: 
    value
    timestamp     ndarray,(98676,),object
    open         ndarray,(98676,),float64
    high         ndarray,(98676,),float64
    low          ndarray,(98676,),float64
    close        ndarray,(98676,),float64
    volume         ndarray,(98676,),int64
    dtype: object
    Coordinates:
    * value    (value) <U9 'timestamp' 'open' 'high' 'low' 'close' 'volume'
    Engine:
    Standard (serial) Engine
    Ttype:
    ndarray

We then concatenate the list of N-tables into a two dimensional N-table
with the new dimension representing the individual tickers that the
data is for. This gets us close to the desired N-table with one small issue.
While in the dataframes the timestamp column was indeed of dtype datetime64,
for one reason or another extracting that column and converting it to an
array gives an array of pandas Timestamp objects. As such we must add a step
to convert the timestamp into datetime64 type and finally have the desired
N-table:

.. code-block:: python

    # concatenate to have the data in one place
    ntbl = tp.concatenate(
        ntbls, "ticker", [pl.Path(file).stem for file in files]
    )

    # pandas outputs things as Timestamp objects instead of datetime64
    # even if they are datetime64 data types. We need to correct for this.
    ntbl.value["timestamp"] = ntbl.value["timestamp"].astype("datetime64[D]")

    ntbl
    Out[12]: 
    ticker                                VTV  ...                             VOE
    value                                      ...                                
    timestamp  ndarray,(98676,),datetime64[D]  ...  ndarray,(67932,),datetime64[D]
    open             ndarray,(98676,),float64  ...        ndarray,(67932,),float64
    high             ndarray,(98676,),float64  ...        ndarray,(67932,),float64
    low              ndarray,(98676,),float64  ...        ndarray,(67932,),float64
    close            ndarray,(98676,),float64  ...        ndarray,(67932,),float64
    volume             ndarray,(98676,),int64  ...          ndarray,(67932,),int64
    [6 rows x 12 columns]
    Coordinates:
    * value    (value) <U9 'timestamp' 'open' 'high' 'low' 'close' 'volume'
    * ticker   (ticker) <U3 'VTV' 'VB' 'MGV' 'VO' 'VOT' ... 'VV' 'VUG' 'MGC' 'VOE'
    Engine:
    Standard (serial) Engine
    Ttype:
    ndarray


Processing
----------

The goal in this example will be to produce moving-average plots of stock
prices. This is one way technical analysts determine when to buy or sell
a stock. It should be mentioned that this of course is *NOT* financial advice,
simply a statement about what other people do.

Here we will specifically take a look at the 50-day rolling average of our
stocks and compare that to the daily values. Immediately we run into
an issue: the data is minute by minute, not daily. We can however come up
with daily values with some simple interpolation. First we come up with the
time values we want data for:


.. code-block:: python

    # get daily timestamp
    start_time = ntbl.value.timestamp[0]
    end_time = ntbl.value.timestamp[-1]
    timestamp_daily = np.arange(
        start_time, end_time, np.timedelta64(1, "D"), dtype="datetime64[D]", like=start_time
    )
    ntbl.value["timestamp_daily"] = timestamp_daily

    ntbl.value["timestamp_daily"]
    Out[4]: 
    ticker
    VTV    ndarray,(360,),datetime64[D]
    VB     ndarray,(360,),datetime64[D]
    MGV    ndarray,(360,),datetime64[D]
    VO     ndarray,(360,),datetime64[D]
    VOT    ndarray,(360,),datetime64[D]
    VBK    ndarray,(360,),datetime64[D]
    VBR    ndarray,(360,),datetime64[D]
    MGK    ndarray,(360,),datetime64[D]
    VV     ndarray,(360,),datetime64[D]
    VUG    ndarray,(360,),datetime64[D]
    MGC    ndarray,(360,),datetime64[D]
    VOE    ndarray,(360,),datetime64[D]
    dtype: object
    Coordinates:
        value    <U15 'timestamp_daily'
    * ticker   (ticker) <U3 'VTV' 'VB' 'MGV' 'VO' 'VOT' ... 'VV' 'VUG' 'MGC' 'VOE'
    Engine:
    Standard (serial) Engine
    Ttype:
    ndarray

With these generated we can now interpolate with respect to the original time
stamps to get the daily stock values:

.. code-block:: python

    # interpolate values

    ntbl.value["close_daily"] = np.interp(
        ntbl.value.timestamp_daily.astype("float"),
        ntbl.value.timestamp.astype("float"),
        ntbl.value.close,
    )

    ntbl.value.close_daily
    Out[4]: 
    ticker
    VTV    ndarray,(360,),float64
    VB     ndarray,(360,),float64
    MGV    ndarray,(360,),float64
    VO     ndarray,(360,),float64
    VOT    ndarray,(360,),float64
    VBK    ndarray,(360,),float64
    VBR    ndarray,(360,),float64
    MGK    ndarray,(360,),float64
    VV     ndarray,(360,),float64
    VUG    ndarray,(360,),float64
    MGC    ndarray,(360,),float64
    VOE    ndarray,(360,),float64
    dtype: object
    Coordinates:
        value    <U15 'close_daily'
    * ticker   (ticker) <U3 'VTV' 'VB' 'MGV' 'VO' 'VOT' ... 'VV' 'VUG' 'MGC' 'VOE'
    Engine:
    Standard (serial) Engine
    Ttype:
    ndarray

And finally we generate the rolling average:

.. code-block:: python

    # the following convolution and division combo is equivalent to a 50 day rolling average
    ntbl.value["rolling_50_day"] = np.convolve(ntbl.value.close_daily, np.ones(50), "valid") / 50


Take a minute to note how simple the synatx is. No for loops. No extreme
adaptation of methods or functions to fit the |project| paradigm. You
simply code what you would for a single array and get the result for 
multiple arrays. This is what |project| aims to achieve in all data
workflows; code for one thing, get results for all.


Visualizing
-----------

It's great that we have produced interesting data that can be used by a user
to make decisions, but unless that user is an auto-trader (possible), a 
visual representation of the data might be more useful. |project| makes
generating nice plots for multiple datasets an extremely simple process by
leveraging plotly.py in tabularized fashion right out of the box. Note that
the imports from earlier import a |project| version of plotly.express.
This version exposes already tabularized plotly.express functions so you
don't have to.

To start, we use the cartograph function to get what we want the titles to
be:

.. code-block:: python

    title = tp.cartograph(ntbl.value.timestamp_daily)[0]

cartograph is a quality of life function (obtained from the |project| qol module)
that returns an N-table whose elements are the multi-dimensional index that
would return itself under an ntbl.struct.loc getitem operation. That is perhaps
a bit confusing. Let's take a look at what it returns:

.. code-block:: python

    cartograph(ntbl)
    Out[17]: 
    ticker                                  VTV  ...                         VOE
    value                                        ...                            
    timestamp              ('timestamp', 'VTV')  ...        ('timestamp', 'VOE')
    open                        ('open', 'VTV')  ...             ('open', 'VOE')
    high                        ('high', 'VTV')  ...             ('high', 'VOE')
    low                          ('low', 'VTV')  ...              ('low', 'VOE')
    close                      ('close', 'VTV')  ...            ('close', 'VOE')
    volume                    ('volume', 'VTV')  ...           ('volume', 'VOE')
    timestamp_daily  ('timestamp_daily', 'VTV')  ...  ('timestamp_daily', 'VOE')
    close_daily          ('close_daily', 'VTV')  ...      ('close_daily', 'VOE')
    [8 rows x 12 columns]
    Coordinates:
    * value    (value) <U15 'timestamp' 'open' ... 'timestamp_daily' 'close_daily'
    * ticker   (ticker) <U3 'VTV' 'VB' 'MGV' 'VO' 'VOT' ... 'VV' 'VUG' 'MGC' 'VOE'
    Engine:
    Standard (serial) Engine
    Ttype:
    tuple

This is useful when we want meta-data found in the N-table coordinates to appear
in results we want such as creating a title for the plots. cartograph returns
an N-table of tuples containing the tickers:

.. code-block:: python

    cartograph(ntbl.value.timestamp_daily)
    Out[18]: 
    ticker
    VTV    ('VTV',)
    VB      ('VB',)
    MGV    ('MGV',)
    VO      ('VO',)
    VOT    ('VOT',)
    VBK    ('VBK',)
    VBR    ('VBR',)
    MGK    ('MGK',)
    VV      ('VV',)
    VUG    ('VUG',)
    MGC    ('MGC',)
    VOE    ('VOE',)
    dtype: object
    Coordinates:
        value    <U15 'timestamp_daily'
    * ticker   (ticker) <U3 'VTV' 'VB' 'MGV' 'VO' 'VOT' ... 'VV' 'VUG' 'MGC' 'VOE'
    Engine:
    Standard (serial) Engine
    Ttype:
    tuple

Which was than indexed for the title:

.. code-block:: python

    title
    Out[20]: 
    ticker
    VTV    VTV
    VB      VB
    MGV    MGV
    VO      VO
    VOT    VOT
    VBK    VBK
    VBR    VBR
    MGK    MGK
    VV      VV
    VUG    VUG
    MGC    MGC
    VOE    VOE
    dtype: object
    Coordinates:
        value    <U15 'timestamp_daily'
    * ticker   (ticker) <U3 'VTV' 'VB' 'MGV' 'VO' 'VOT' ... 'VV' 'VUG' 'MGC' 'VOE'
    Engine:
    Standard (serial) Engine
    Ttype:
    str_


With the ntbl representing the title for each plot made, it is time to create the figures

.. code-block:: python

    figure = tp.line(
        x=ntbl.value.timestamp_daily[49:],
        y=[ntbl.value.close_daily[49:], ntbl.value.rolling_50_day],
        title=title,
    )

It's as simple as calling the plotly express functions the same way you normally would.
Instead of passing arrays you pass N-tables of arrays and it will generate the figures
in a tabularized manner.

From here we could in theory just call figure.show() and we would get plots that
are for the most part what we want. There is one small problem however. The traces
(the plots in the figure) will be named some default names and not what we want.
Arguably this is a limitation of the plotly express module for not allowing us to
define the trace names during the call to the line function. As a result we have
to iterate over the traces and rename them ourselves:

.. code-block:: python

    for idx, name in enumerate(["Close", "Rolling 50-day Average"]):
        oldname = figure.data[idx].name
        figure.data[idx].update(name=name)
        figure.data[idx].update(
            hovertemplate=figure.data[idx].hovertemplate.replace(
                oldname, name
            )
        )

and finally show the figures:

.. code-block:: python
    
    figure.show()


This final command should open up a tab in your default browser for each ticker.


Offloading
----------

With the processing done it is time to offload the data. There are two sets
of data to save, the core data (what we loaded in and the additional data
we calculated) and the figures; |project| defines a serializer and 
deserializer for plotly figure objects and so we can save figure objects
to be used at a later time or by a different user.

.. code-block:: python

    # save core data as a .ntbl

    tp.save_ntable(ntbl, "./stock_data.ntbl")

    # save figures as .ntbl

    tp.save_ntable(figure, "./rolling_avg_plots.ntbl")


We can verify that they were saved properly by loading them back in and
taking a look


.. code-block:: python

    stock_data = tp.load_ntable("./stock_data.ntbl")

    plots = tp.load_ntable("./plots.ntbl")

    stock_data
    Out[3]: 
    ticker                                      VTV  ...                             VOE
    value                                            ...                                
    timestamp        ndarray,(98676,),datetime64[D]  ...  ndarray,(67932,),datetime64[D]
    open                   ndarray,(98676,),float64  ...        ndarray,(67932,),float64
    high                   ndarray,(98676,),float64  ...        ndarray,(67932,),float64
    low                    ndarray,(98676,),float64  ...        ndarray,(67932,),float64
    close                  ndarray,(98676,),float64  ...        ndarray,(67932,),float64
    volume                   ndarray,(98676,),int64  ...          ndarray,(67932,),int64
    timestamp_daily    ndarray,(360,),datetime64[D]  ...    ndarray,(360,),datetime64[D]
    close_daily              ndarray,(360,),float64  ...          ndarray,(360,),float64
    rolling_50_day           ndarray,(311,),float64  ...          ndarray,(311,),float64
    [9 rows x 12 columns]
    Coordinates:
    * ticker   (ticker) <U3 'VTV' 'VB' 'MGV' 'VO' 'VOT' ... 'VV' 'VUG' 'MGC' 'VOE'
    * value    (value) <U15 'timestamp' 'open' ... 'close_daily' 'rolling_50_day'
    Engine:
    Standard (serial) Engine
    Ttype:
    ndarray

    plots
    Out[4]: 
    ticker
    VTV    Figure
    VB     Figure
    MGV    Figure
    VO     Figure
    VOT    Figure
    VBK    Figure
    VBR    Figure
    MGK    Figure
    VV     Figure
    VUG    Figure
    MGC    Figure
    VOE    Figure
    dtype: object
    Coordinates:
    * ticker   (ticker) <U3 'VTV' 'VB' 'MGV' 'VO' 'VOT' ... 'VV' 'VUG' 'MGC' 'VOE'
    Engine:
    Standard (serial) Engine
    Ttype:
    Figure

