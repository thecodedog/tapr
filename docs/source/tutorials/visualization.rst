Visualization
*************

These days there are several visualization libraries for python, the most
popular being of course matplotlib. While popular and quite capable, there
is a reason more and more visualization libraries are cropping up: matplotlib
is confusing at first, demands more effort to make things look nice, and was
ultimately designed to mimic plotting in matlab. Despite its claims that
it can be used in a "Pythonic, object oriented way", it does so in a way
that lacks the simplicity of other libraries. As a result, the recommended
library for visualization in |project| work flows is plotly.py. Plotly.py
simply feels much better to use and exceeds matplotlib in achieving
pythonic, object oriented visualization.

As a result of all of this, |project| exposes versions of plotly.py
(due to its recommended use) and matplotlib (due to its extreme popularity)
that work on N-tables right out of the box. For examples on how to use
them see the documentation below:

plotly
------

Using plotly to generate basic data plots is mostly straight forward.

.. code-block:: python

    import plotly.io as pio
    import plotly.express as px

    pio.renderers.default = "browser"

    fig = px.scatter(x=[1,2,3,4,5,6],y=[10,12,14,16,18,20])

    fig.show()


However this is likely somewhat different from what most users are familiar
with. Let's walk through each step after the imports:

Step 1 is to assign the default renderer. This tells plotly how to display
the figures once the "show" method is called. In this example we make use
of the "browser" renderer, and as a result will use a browser (the user's 
default) to display plots. See plotly.py documentation for more info.

Step 2 is to create a plotly Figure object using one of the many basic
chart/plot functions, in this case the scatter plot. Such functions
are essentially used to construct Figure objects. Figure objects contain
the information needed to create a visualization, not the visualization
itself. The representation of the figure created above is this:


.. code-block:: python

    fig
    Out[9]: 
    Figure({
        'data': [{'hovertemplate': 'x=%{x}<br>y=%{y}<extra></extra>',
                'legendgroup': '',
                'marker': {'color': '#636efa', 'symbol': 'circle'},
                'mode': 'markers',
                'name': '',
                'orientation': 'v',
                'showlegend': False,
                'type': 'scatter',
                'x': array([1, 2, 3, 4, 5, 6]),
                'xaxis': 'x',
                'y': array([10, 12, 14, 16, 18, 20]),
                'yaxis': 'y'}],
        'layout': {'legend': {'tracegroupgap': 0},
                'margin': {'t': 60},
                'template': '...',
                'xaxis': {'anchor': 'y', 'domain': [0.0, 1.0], 'title': {'text': 'x'}},
                'yaxis': {'anchor': 'x', 'domain': [0.0, 1.0], 'title': {'text': 'y'}}}
    })

If it looks like a dictionary, that's because at a low level plotly treats
dictionares as figures as long as they have the necessary values (see
plotly documentation).

Step 3 is to actually show the plot. Where/how this happens depends on the
renderer being used. Since we set the default renderer to "browser", plotly
opens up a tab in the default browser and displays the plot in there.

----

|project| exposes a tabularized version of the plotly.express module and
does so in a way that mimics base plotly. As a result, the creation of
scatter plots based on N-tables looks markedly similar (other than what
is necessary to generate an N-table):

.. code-block:: python

    import numpy as np
    from tapr.main.conversion import ntable
    import plotly.io as pio
    import tapr.visualization.plotly.express as px

    pio.renderers.default = "browser"

    # create a demo N-table
    data = {}
    for i in range(3):
        data[f"sim{i}"] = {}
        for j in range(5):
            data[f"sim{i}"][f"var{j}"] = np.random.uniform(low=0.5, high=13.3, size=(100,))
    ntbl = ntable(data, dims=("cols", "rows")).struct.T

    fig = px.scatter(x=ntbl.rows.var1, y=ntbl.rows.var2)

    fig.show()


Let's again walk through each step and take a look at what is happening/being
produced:

Step 1 is again to assign the default renderer to be "browser". This step
is identical to the non-|project| work flow.

Step 2 just creates a demo N-table with random arrays. It looks like this:

.. code-block:: python

    ntbl
    Out[6]: 
    cols                    sim0                    sim1                    sim2
    rows                                                                        
    var0  ndarray,(100,),float64  ndarray,(100,),float64  ndarray,(100,),float64
    var1  ndarray,(100,),float64  ndarray,(100,),float64  ndarray,(100,),float64
    var2  ndarray,(100,),float64  ndarray,(100,),float64  ndarray,(100,),float64
    var3  ndarray,(100,),float64  ndarray,(100,),float64  ndarray,(100,),float64
    var4  ndarray,(100,),float64  ndarray,(100,),float64  ndarray,(100,),float64
    Coordinates:
    * cols     (cols) <U4 'sim0' 'sim1' 'sim2'
    * rows     (rows) <U4 'var0' 'var1' 'var2' 'var3' 'var4'
    Engine:
    Standard (serial) Engine
    Ttype:
    ndarray

Step 3 makes use of the tabularized scatter plot function. Doing so produces
an N-table of Figure objects like so:


.. code-block:: python

    cols
    sim0    Figure
    sim1    Figure
    sim2    Figure
    dtype: object
    Coordinates:
    * cols     (cols) <U4 'sim0' 'sim1' 'sim2'
        rows     <U4 'var1'
    Engine:
    Standard (serial) Engine
    Ttype:
    Figure

Finally step 4 is to show the figures. Since show is an attribute of the
figure N-table (via its ttype), and since __call__ on N-table is
tabularized, the show call is identical to that of the non-|project| version.
It will create 3 distinct plots in the default browser (each getting its own
tab).


matplotlib
----------

For basic plotting scenarios, matplotlib is simple enough but is less
object oriented than plotly

.. code-block:: python

    import matplotlib.pyplot as plt

    lines = plt.plot([1,2,3,4,5,6],[10,12,14,16,18,20])

    plt.show()

As with plotly, let's break down what is happening hear so we can identify
what needs to happen in the |project| version.

Step 1 (after the import) is to simply call the plot function of the pyplot
module. Unlike plotly however it doesn't return an object that encapsulates
the plot that gets generated, but rather a list of something called a Line2D
object:

.. code-block:: python

    lines
    Out[7]: [<matplotlib.lines.Line2D at 0x7fb4f8e8d370>]

For information on what this is, checkout the matplotlib documentation.


Step 2 renders the plot. Notice that it isn't a method being called on the
lines or a figure object of sorts, but rather is a function of the pyplot
module.

----

Unfortunately the |project| matplotlib usage doesn't mirror the base usage
quite as well as it does for plotly. To see why let's try and mimic the
the base usage to the extent that we did for plotly:


.. code-block:: python

    import numpy as np
    from tapr.main.conversion import ntable
    import tapr.visualization.matplotlib.pyplot as plt

    # create a demo N-table
    data = {}
    for i in range(3):
        data[f"sim{i}"] = {}
        for j in range(5):
            data[f"sim{i}"][f"var{j}"] = np.random.uniform(low=0.5, high=13.3, size=(100,))
    ntbl = ntable(data, dims=("cols", "rows")).struct.T

    lines = plt.plot(ntbl.rows.var1, ntbl.rows.var2, "o")

    plt.show()

While this certainly works, the result is one figure with mutliple scatter plots
on it instead of an individual figure for each plot as was the case in plotly.
It is indeed up to the user which one they would like, but the difference is worth
pointing out. Let's see what it takes to get a figure for each individual plot:

Let us first see what that looks like for the non-|project| case:

.. code-block:: python

    import matplotlib.pyplot as plt

    fig = plt.figure()
    lines = plt.plot([1, 2, 3, 4, 5, 6], [10, 12, 14, 16, 18, 20])

    fig1 = plt.figure()
    lines1 = plt.plot([1, 2, 3, 4, 5, 6], [11, 15, 18, 20, 21, 25])

    plt.show()


In matplotlib it is necessary to manually create a figure object each time before
calling plot. The plot will get generated on the figure object created before the
plot function is called. To do this with N-tables, simply define a custom plot
function that does this for you:

.. code-block:: python

    import numpy as np
    from tapr.main.conversion import ntable
    from tapr.main.tabularization import tabularize
    import matplotlib.pyplot as plt

    # create a demo N-table
    data = {}
    for i in range(3):
        data[f"sim{i}"] = {}
        for j in range(5):
            data[f"sim{i}"][f"var{j}"] = np.random.uniform(low=0.5, high=13.3, size=(100,))
    ntbl = ntable(data, dims=("cols", "rows")).struct.T

    @tabularize
    def figplot(*plotargs, **plotkwargs):
        fig = plt.figure()
        return plt.plot(*plotargs, **plotkwargs)


    lines = figplot(ntbl.rows.var1, ntbl.rows.var2, "o")

    plt.show()
