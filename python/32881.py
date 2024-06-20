# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [Visualizing Errors](04.03-Errorbars.ipynb) | [Contents](Index.ipynb) | [Histograms, Binnings, and Density](04.05-Histograms-and-Binnings.ipynb) >
# 

# # Density and Contour Plots
# 

# Sometimes it is useful to display three-dimensional data in two dimensions using contours or color-coded regions.
# There are three Matplotlib functions that can be helpful for this task: ``plt.contour`` for contour plots, ``plt.contourf`` for filled contour plots, and ``plt.imshow`` for showing images.
# This section looks at several examples of using these. We'll start by setting up the notebook for plotting and importing the functions we will use: 
# 

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np


# ## Visualizing a Three-Dimensional Function
# 

# We'll start by demonstrating a contour plot using a function $z = f(x, y)$, using the following particular choice for $f$ (we've seen this before in [Computation on Arrays: Broadcasting](02.05-Computation-on-arrays-broadcasting.ipynb), when we used it as a motivating example for array broadcasting):
# 

def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)


# A contour plot can be created with the ``plt.contour`` function.
# It takes three arguments: a grid of *x* values, a grid of *y* values, and a grid of *z* values.
# The *x* and *y* values represent positions on the plot, and the *z* values will be represented by the contour levels.
# Perhaps the most straightforward way to prepare such data is to use the ``np.meshgrid`` function, which builds two-dimensional grids from one-dimensional arrays:
# 

x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)


# Now let's look at this with a standard line-only contour plot:
# 

plt.contour(X, Y, Z, colors='black');


# Notice that by default when a single color is used, negative values are represented by dashed lines, and positive values by solid lines.
# Alternatively, the lines can be color-coded by specifying a colormap with the ``cmap`` argument.
# Here, we'll also specify that we want more lines to be drawn—20 equally spaced intervals within the data range:
# 

plt.contour(X, Y, Z, 20, cmap='RdGy');


# Here we chose the ``RdGy`` (short for *Red-Gray*) colormap, which is a good choice for centered data.
# Matplotlib has a wide range of colormaps available, which you can easily browse in IPython by doing a tab completion on the ``plt.cm`` module:
# ```
# plt.cm.<TAB>
# ```
# 
# Our plot is looking nicer, but the spaces between the lines may be a bit distracting.
# We can change this by switching to a filled contour plot using the ``plt.contourf()`` function (notice the ``f`` at the end), which uses largely the same syntax as ``plt.contour()``.
# 
# Additionally, we'll add a ``plt.colorbar()`` command, which automatically creates an additional axis with labeled color information for the plot:
# 

plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.colorbar();


# The colorbar makes it clear that the black regions are "peaks," while the red regions are "valleys."
# 
# One potential issue with this plot is that it is a bit "splotchy." That is, the color steps are discrete rather than continuous, which is not always what is desired.
# This could be remedied by setting the number of contours to a very high number, but this results in a rather inefficient plot: Matplotlib must render a new polygon for each step in the level.
# A better way to handle this is to use the ``plt.imshow()`` function, which interprets a two-dimensional grid of data as an image.
# 
# The following code shows this:
# 

plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
           cmap='RdGy')
plt.colorbar()
plt.axis(aspect='image');


# There are a few potential gotchas with ``imshow()``, however:
# 
# - ``plt.imshow()`` doesn't accept an *x* and *y* grid, so you must manually specify the *extent* [*xmin*, *xmax*, *ymin*, *ymax*] of the image on the plot.
# - ``plt.imshow()`` by default follows the standard image array definition where the origin is in the upper left, not in the lower left as in most contour plots. This must be changed when showing gridded data.
# - ``plt.imshow()`` will automatically adjust the axis aspect ratio to match the input data; this can be changed by setting, for example, ``plt.axis(aspect='image')`` to make *x* and *y* units match.
# 

# Finally, it can sometimes be useful to combine contour plots and image plots.
# For example, here we'll use a partially transparent background image (with transparency set via the ``alpha`` parameter) and overplot contours with labels on the contours themselves (using the ``plt.clabel()`` function):
# 

contours = plt.contour(X, Y, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8)

plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
           cmap='RdGy', alpha=0.5)
plt.colorbar();


# The combination of these three functions—``plt.contour``, ``plt.contourf``, and ``plt.imshow``—gives nearly limitless possibilities for displaying this sort of three-dimensional data within a two-dimensional plot.
# For more information on the options available in these functions, refer to their docstrings.
# If you are interested in three-dimensional visualizations of this type of data, see [Three-dimensional Plotting in Matplotlib](04.12-Three-Dimensional-Plotting.ipynb).
# 

# <!--NAVIGATION-->
# < [Visualizing Errors](04.03-Errorbars.ipynb) | [Contents](Index.ipynb) | [Histograms, Binnings, and Density](04.05-Histograms-and-Binnings.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [Introduction to NumPy](02.00-Introduction-to-NumPy.ipynb) | [Contents](Index.ipynb) | [The Basics of NumPy Arrays](02.02-The-Basics-Of-NumPy-Arrays.ipynb) >
# 

# # Understanding Data Types in Python
# 

# Effective data-driven science and computation requires understanding how data is stored and manipulated.
# This section outlines and contrasts how arrays of data are handled in the Python language itself, and how NumPy improves on this.
# Understanding this difference is fundamental to understanding much of the material throughout the rest of the book.
# 
# Users of Python are often drawn-in by its ease of use, one piece of which is dynamic typing.
# While a statically-typed language like C or Java requires each variable to be explicitly declared, a dynamically-typed language like Python skips this specification. For example, in C you might specify a particular operation as follows:
# 
# ```C
# *(C, code, */)
# int result = 0;
# for(int i=0; i<100; i++){
#     result += i;
# }
# ```
# 
# While in Python the equivalent operation could be written this way:
# 
# ```python
# # Python code
# result = 0
# for i in range(100):
#     result += i
# ```
# 
# Notice the main difference: in C, the data types of each variable are explicitly declared, while in Python the types are dynamically inferred. This means, for example, that we can assign any kind of data to any variable:
# 
# ```python
# # Python code
# x = 4
# x = "four"
# ```
# 
# Here we've switched the contents of ``x`` from an integer to a string. The same thing in C would lead (depending on compiler settings) to a compilation error or other unintented consequences:
# 
# ```C
# *(C, code, */)
# int x = 4;
# x = "four";  // FAILS
# ```
# 
# This sort of flexibility is one piece that makes Python and other dynamically-typed languages convenient and easy to use.
# Understanding *how* this works is an important piece of learning to analyze data efficiently and effectively with Python.
# But what this type-flexibility also points to is the fact that Python variables are more than just their value; they also contain extra information about the type of the value. We'll explore this more in the sections that follow.
# 

# ## A Python Integer Is More Than Just an Integer
# 
# The standard Python implementation is written in C.
# This means that every Python object is simply a cleverly-disguised C structure, which contains not only its value, but other information as well. For example, when we define an integer in Python, such as ``x = 10000``, ``x`` is not just a "raw" integer. It's actually a pointer to a compound C structure, which contains several values.
# Looking through the Python 3.4 source code, we find that the integer (long) type definition effectively looks like this (once the C macros are expanded):
# 
# ```C
# struct _longobject {
#     long ob_refcnt;
#     PyTypeObject *ob_type;
#     size_t ob_size;
#     long ob_digit[1];
# };
# ```
# 
# A single integer in Python 3.4 actually contains four pieces:
# 
# - ``ob_refcnt``, a reference count that helps Python silently handle memory allocation and deallocation
# - ``ob_type``, which encodes the type of the variable
# - ``ob_size``, which specifies the size of the following data members
# - ``ob_digit``, which contains the actual integer value that we expect the Python variable to represent.
# 
# This means that there is some overhead in storing an integer in Python as compared to an integer in a compiled language like C, as illustrated in the following figure:
# 

# ![Integer Memory Layout](figures/cint_vs_pyint.png)

# Here ``PyObject_HEAD`` is the part of the structure containing the reference count, type code, and other pieces mentioned before.
# 
# Notice the difference here: a C integer is essentially a label for a position in memory whose bytes encode an integer value.
# A Python integer is a pointer to a position in memory containing all the Python object information, including the bytes that contain the integer value.
# This extra information in the Python integer structure is what allows Python to be coded so freely and dynamically.
# All this additional information in Python types comes at a cost, however, which becomes especially apparent in structures that combine many of these objects.
# 

# ## A Python List Is More Than Just a List
# 
# Let's consider now what happens when we use a Python data structure that holds many Python objects.
# The standard mutable multi-element container in Python is the list.
# We can create a list of integers as follows:
# 

L = list(range(10))
L


type(L[0])


# Or, similarly, a list of strings:
# 

L2 = [str(c) for c in L]
L2


type(L2[0])


# Because of Python's dynamic typing, we can even create heterogeneous lists:
# 

L3 = [True, "2", 3.0, 4]
[type(item) for item in L3]


# But this flexibility comes at a cost: to allow these flexible types, each item in the list must contain its own type info, reference count, and other information–that is, each item is a complete Python object.
# In the special case that all variables are of the same type, much of this information is redundant: it can be much more efficient to store data in a fixed-type array.
# The difference between a dynamic-type list and a fixed-type (NumPy-style) array is illustrated in the following figure:
# 

# ![Array Memory Layout](figures/array_vs_list.png)

# At the implementation level, the array essentially contains a single pointer to one contiguous block of data.
# The Python list, on the other hand, contains a pointer to a block of pointers, each of which in turn points to a full Python object like the Python integer we saw earlier.
# Again, the advantage of the list is flexibility: because each list element is a full structure containing both data and type information, the list can be filled with data of any desired type.
# Fixed-type NumPy-style arrays lack this flexibility, but are much more efficient for storing and manipulating data.
# 

# ## Fixed-Type Arrays in Python
# 
# Python offers several different options for storing data in efficient, fixed-type data buffers.
# The built-in ``array`` module (available since Python 3.3) can be used to create dense arrays of a uniform type:
# 

import array
L = list(range(10))
A = array.array('i', L)
A


# Here ``'i'`` is a type code indicating the contents are integers.
# 
# Much more useful, however, is the ``ndarray`` object of the NumPy package.
# While Python's ``array`` object provides efficient storage of array-based data, NumPy adds to this efficient *operations* on that data.
# We will explore these operations in later sections; here we'll demonstrate several ways of creating a NumPy array.
# 
# We'll start with the standard NumPy import, under the alias ``np``:
# 

import numpy as np


# ## Creating Arrays from Python Lists
# 
# First, we can use ``np.array`` to create arrays from Python lists:
# 

# integer array:
np.array([1, 4, 2, 5, 3])


# Remember that unlike Python lists, NumPy is constrained to arrays that all contain the same type.
# If types do not match, NumPy will upcast if possible (here, integers are up-cast to floating point):
# 

np.array([3.14, 4, 2, 3])


# If we want to explicitly set the data type of the resulting array, we can use the ``dtype`` keyword:
# 

np.array([1, 2, 3, 4], dtype='float32')


# Finally, unlike Python lists, NumPy arrays can explicitly be multi-dimensional; here's one way of initializing a multidimensional array using a list of lists:
# 

# nested lists result in multi-dimensional arrays
np.array([range(i, i + 3) for i in [2, 4, 6]])


# The inner lists are treated as rows of the resulting two-dimensional array.
# 

# ## Creating Arrays from Scratch
# 
# Especially for larger arrays, it is more efficient to create arrays from scratch using routines built into NumPy.
# Here are several examples:
# 

# Create a length-10 integer array filled with zeros
np.zeros(10, dtype=int)


# Create a 3x5 floating-point array filled with ones
np.ones((3, 5), dtype=float)


# Create a 3x5 array filled with 3.14
np.full((3, 5), 3.14)


# Create an array filled with a linear sequence
# Starting at 0, ending at 20, stepping by 2
# (this is similar to the built-in range() function)
np.arange(0, 20, 2)


# Create an array of five values evenly spaced between 0 and 1
np.linspace(0, 1, 5)


# Create a 3x3 array of uniformly distributed
# random values between 0 and 1
np.random.random((3, 3))


# Create a 3x3 array of normally distributed random values
# with mean 0 and standard deviation 1
np.random.normal(0, 1, (3, 3))


# Create a 3x3 array of random integers in the interval [0, 10)
np.random.randint(0, 10, (3, 3))


# Create a 3x3 identity matrix
np.eye(3)


# Create an uninitialized array of three integers
# The values will be whatever happens to already exist at that memory location
np.empty(3)


# ## NumPy Standard Data Types
# 
# NumPy arrays contain values of a single type, so it is important to have detailed knowledge of those types and their limitations.
# Because NumPy is built in C, the types will be familiar to users of C, Fortran, and other related languages.
# 
# The standard NumPy data types are listed in the following table.
# Note that when constructing an array, they can be specified using a string:
# 
# ```python
# np.zeros(10, dtype='int16')
# ```
# 
# Or using the associated NumPy object:
# 
# ```python
# np.zeros(10, dtype=np.int16)
# ```
# 

# | Data type	    | Description |
# |---------------|-------------|
# | ``bool_``     | Boolean (True or False) stored as a byte |
# | ``int_``      | Default integer type (same as C ``long``; normally either ``int64`` or ``int32``)| 
# | ``intc``      | Identical to C ``int`` (normally ``int32`` or ``int64``)| 
# | ``intp``      | Integer used for indexing (same as C ``ssize_t``; normally either ``int32`` or ``int64``)| 
# | ``int8``      | Byte (-128 to 127)| 
# | ``int16``     | Integer (-32768 to 32767)|
# | ``int32``     | Integer (-2147483648 to 2147483647)|
# | ``int64``     | Integer (-9223372036854775808 to 9223372036854775807)| 
# | ``uint8``     | Unsigned integer (0 to 255)| 
# | ``uint16``    | Unsigned integer (0 to 65535)| 
# | ``uint32``    | Unsigned integer (0 to 4294967295)| 
# | ``uint64``    | Unsigned integer (0 to 18446744073709551615)| 
# | ``float_``    | Shorthand for ``float64``.| 
# | ``float16``   | Half precision float: sign bit, 5 bits exponent, 10 bits mantissa| 
# | ``float32``   | Single precision float: sign bit, 8 bits exponent, 23 bits mantissa| 
# | ``float64``   | Double precision float: sign bit, 11 bits exponent, 52 bits mantissa| 
# | ``complex_``  | Shorthand for ``complex128``.| 
# | ``complex64`` | Complex number, represented by two 32-bit floats| 
# | ``complex128``| Complex number, represented by two 64-bit floats| 
# 

# More advanced type specification is possible, such as specifying big or little endian numbers; for more information, refer to the [NumPy documentation](http://numpy.org/).
# NumPy also supports compound data types, which will be covered in [Structured Data: NumPy's Structured Arrays](02.09-Structured-Data-NumPy.ipynb).
# 

# <!--NAVIGATION-->
# < [Introduction to NumPy](02.00-Introduction-to-NumPy.ipynb) | [Contents](Index.ipynb) | [The Basics of NumPy Arrays](02.02-The-Basics-Of-NumPy-Arrays.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*

# <!--NAVIGATION-->
# < [Keyboard Shortcuts in the IPython Shell](01.02-Shell-Keyboard-Shortcuts.ipynb) | [Contents](Index.ipynb) | [Input and Output History](01.04-Input-Output-History.ipynb) >

# # IPython Magic Commands
# 
# The previous two sections showed how IPython lets you use and explore Python efficiently and interactively.
# Here we'll begin discussing some of the enhancements that IPython adds on top of the normal Python syntax.
# These are known in IPython as *magic commands*, and are prefixed by the ``%`` character.
# These magic commands are designed to succinctly solve various common problems in standard data analysis.
# Magic commands come in two flavors: *line magics*, which are denoted by a single ``%`` prefix and operate on a single line of input, and *cell magics*, which are denoted by a double ``%%`` prefix and operate on multiple lines of input.
# We'll demonstrate and discuss a few brief examples here, and come back to more focused discussion of several useful magic commands later in the chapter.

# ## Pasting Code Blocks: ``%paste`` and ``%cpaste``
# 
# When working in the IPython interpreter, one common gotcha is that pasting multi-line code blocks can lead to unexpected errors, especially when indentation and interpreter markers are involved.
# A common case is that you find some example code on a website and want to paste it into your interpreter.
# Consider the following simple function:
# 
# ``` python
# >>> def donothing(x):
# ...     return x
# 
# ```
# The code is formatted as it would appear in the Python interpreter, and if you copy and paste this directly into IPython you get an error:
# 
# ```ipython
# In [2]: >>> def donothing(x):
#    ...:     ...     return x
#    ...:     
#   File "<ipython-input-20-5a66c8964687>", line 2
#     ...     return x
#                  ^
# SyntaxError: invalid syntax
# ```
# 
# In the direct paste, the interpreter is confused by the additional prompt characters.
# But never fear–IPython's ``%paste`` magic function is designed to handle this exact type of multi-line, marked-up input:
# 
# ```ipython
# In [3]: %paste
# >>> def donothing(x):
# ...     return x
# 
# ## -- End pasted text --
# ```
# 
# The ``%paste`` command both enters and executes the code, so now the function is ready to be used:
# 
# ```ipython
# In [4]: donothing(10)
# Out[4]: 10
# ```
# 
# A command with a similar intent is ``%cpaste``, which opens up an interactive multiline prompt in which you can paste one or more chunks of code to be executed in a batch:
# 
# ```ipython
# In [5]: %cpaste
# Pasting code; enter '--' alone on the line to stop or use Ctrl-D.
# :>>> def donothing(x):
# :...     return x
# :--
# ```
# 
# These magic commands, like others we'll see, make available functionality that would be difficult or impossible in a standard Python interpreter.

# ## Running External Code: ``%run``
# As you begin developing more extensive code, you will likely find yourself working in both IPython for interactive exploration, as well as a text editor to store code that you want to reuse.
# Rather than running this code in a new window, it can be convenient to run it within your IPython session.
# This can be done with the ``%run`` magic.
# 
# For example, imagine you've created a ``myscript.py`` file with the following contents:
# 
# ```python
# #-------------------------------------
# # file: myscript.py
# 
# def square(x):
#     """square a number"""
#     return x ** 2
# 
# for N in range(1, 4):
#     print(N, "squared is", square(N))
# ```
# 
# You can execute this from your IPython session as follows:
# 
# ```ipython
# In [6]: %run myscript.py
# 1 squared is 1
# 2 squared is 4
# 3 squared is 9
# ```
# 
# Note also that after you've run this script, any functions defined within it are available for use in your IPython session:
# 
# ```ipython
# In [7]: square(5)
# Out[7]: 25
# ```
# 
# There are several options to fine-tune how your code is run; you can see the documentation in the normal way, by typing **``%run?``** in the IPython interpreter.

# ## Timing Code Execution: ``%timeit``
# Another example of a useful magic function is ``%timeit``, which will automatically determine the execution time of the single-line Python statement that follows it.
# For example, we may want to check the performance of a list comprehension:
# 
# ```ipython
# In [8]: %timeit L = [n ** 2 for n in range(1000)]
# 1000 loops, best of 3: 325 µs per loop
# ```
# 
# The benefit of ``%timeit`` is that for short commands it will automatically perform multiple runs in order to attain more robust results.
# For multi line statements, adding a second ``%`` sign will turn this into a cell magic that can handle multiple lines of input.
# For example, here's the equivalent construction with a ``for``-loop:
# 
# ```ipython
# In [9]: %%timeit
#    ...: L = []
#    ...: for n in range(1000):
#    ...:     L.append(n ** 2)
#    ...: 
# 1000 loops, best of 3: 373 µs per loop
# ```
# 
# We can immediately see that list comprehensions are about 10% faster than the equivalent ``for``-loop construction in this case.
# We'll explore ``%timeit`` and other approaches to timing and profiling code in [Profiling and Timing Code](01.07-Timing-and-Profiling.ipynb).

# ## Help on Magic Functions: ``?``, ``%magic``, and ``%lsmagic``
# 
# Like normal Python functions, IPython magic functions have docstrings, and this useful
# documentation can be accessed in the standard manner.
# So, for example, to read the documentation of the ``%timeit`` magic simply type this:
# 
# ```ipython
# In [10]: %timeit?
# ```
# 
# Documentation for other functions can be accessed similarly.
# To access a general description of available magic functions, including some examples, you can type this:
# 
# ```ipython
# In [11]: %magic
# ```
# 
# For a quick and simple list of all available magic functions, type this:
# 
# ```ipython
# In [12]: %lsmagic
# ```
# 
# Finally, I'll mention that it is quite straightforward to define your own magic functions if you wish.
# We won't discuss it here, but if you are interested, see the references listed in [More IPython Resources](01.08-More-IPython-Resources.ipynb).

# <!--NAVIGATION-->
# < [Keyboard Shortcuts in the IPython Shell](01.02-Shell-Keyboard-Shortcuts.ipynb) | [Contents](Index.ipynb) | [Input and Output History](01.04-Input-Output-History.ipynb) >

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [What Is Machine Learning?](05.01-What-Is-Machine-Learning.ipynb) | [Contents](Index.ipynb) | [Hyperparameters and Model Validation](05.03-Hyperparameters-and-Model-Validation.ipynb) >
# 

# # Introducing Scikit-Learn
# 
# There are several Python libraries which provide solid implementations of a range of machine learning algorithms.
# One of the best known is [Scikit-Learn](http://scikit-learn.org), a package that provides efficient versions of a large number of common algorithms.
# Scikit-Learn is characterized by a clean, uniform, and streamlined API, as well as by very useful and complete online documentation.
# A benefit of this uniformity is that once you understand the basic use and syntax of Scikit-Learn for one type of model, switching to a new model or algorithm is very straightforward.
# 
# This section provides an overview of the Scikit-Learn API; a solid understanding of these API elements will form the foundation for understanding the deeper practical discussion of machine learning algorithms and approaches in the following chapters.
# 
# We will start by covering *data representation* in Scikit-Learn, followed by covering the *Estimator* API, and finally go through a more interesting example of using these tools for exploring a set of images of hand-written digits.
# 

# ## Data Representation in Scikit-Learn
# 

# Machine learning is about creating models from data: for that reason, we'll start by discussing how data can be represented in order to be understood by the computer.
# The best way to think about data within Scikit-Learn is in terms of tables of data.
# 

# ### Data as table
# 
# A basic table is a two-dimensional grid of data, in which the rows represent individual elements of the dataset, and the columns represent quantities related to each of these elements.
# For example, consider the [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set), famously analyzed by Ronald Fisher in 1936.
# We can download this dataset in the form of a Pandas ``DataFrame`` using the [seaborn](http://seaborn.pydata.org/) library:
# 

import seaborn as sns
iris = sns.load_dataset('iris')
iris.head()


# Here each row of the data refers to a single observed flower, and the number of rows is the total number of flowers in the dataset.
# In general, we will refer to the rows of the matrix as *samples*, and the number of rows as ``n_samples``.
# 
# Likewise, each column of the data refers to a particular quantitative piece of information that describes each sample.
# In general, we will refer to the columns of the matrix as *features*, and the number of columns as ``n_features``.
# 

# #### Features matrix
# 
# This table layout makes clear that the information can be thought of as a two-dimensional numerical array or matrix, which we will call the *features matrix*.
# By convention, this features matrix is often stored in a variable named ``X``.
# The features matrix is assumed to be two-dimensional, with shape ``[n_samples, n_features]``, and is most often contained in a NumPy array or a Pandas ``DataFrame``, though some Scikit-Learn models also accept SciPy sparse matrices.
# 
# The samples (i.e., rows) always refer to the individual objects described by the dataset.
# For example, the sample might be a flower, a person, a document, an image, a sound file, a video, an astronomical object, or anything else you can describe with a set of quantitative measurements.
# 
# The features (i.e., columns) always refer to the distinct observations that describe each sample in a quantitative manner.
# Features are generally real-valued, but may be Boolean or discrete-valued in some cases.
# 

# #### Target array
# 
# In addition to the feature matrix ``X``, we also generally work with a *label* or *target* array, which by convention we will usually call ``y``.
# The target array is usually one dimensional, with length ``n_samples``, and is generally contained in a NumPy array or Pandas ``Series``.
# The target array may have continuous numerical values, or discrete classes/labels.
# While some Scikit-Learn estimators do handle multiple target values in the form of a two-dimensional, ``[n_samples, n_targets]`` target array, we will primarily be working with the common case of a one-dimensional target array.
# 
# Often one point of confusion is how the target array differs from the other features columns. The distinguishing feature of the target array is that it is usually the quantity we want to *predict from the data*: in statistical terms, it is the dependent variable.
# For example, in the preceding data we may wish to construct a model that can predict the species of flower based on the other measurements; in this case, the ``species`` column would be considered the target array.
# 
# With this target array in mind, we can use Seaborn (see [Visualization With Seaborn](04.14-Visualization-With-Seaborn.ipynb)) to conveniently visualize the data:
# 

get_ipython().magic('matplotlib inline')
import seaborn as sns; sns.set()
sns.pairplot(iris, hue='species', size=1.5);


# For use in Scikit-Learn, we will extract the features matrix and target array from the ``DataFrame``, which we can do using some of the Pandas ``DataFrame`` operations discussed in the [Chapter 3](03.00-Introduction-to-Pandas.ipynb):
# 

X_iris = iris.drop('species', axis=1)
X_iris.shape


y_iris = iris['species']
y_iris.shape


# To summarize, the expected layout of features and target values is visualized in the following diagram:
# 

# ![](figures/05.02-samples-features.png)
# [figure source in Appendix](06.00-Figure-Code.ipynb#Features-and-Labels-Grid)

# With this data properly formatted, we can move on to consider the *estimator* API of Scikit-Learn:
# 

# ## Scikit-Learn's Estimator API
# 

# The Scikit-Learn API is designed with the following guiding principles in mind, as outlined in the [Scikit-Learn API paper](http://arxiv.org/abs/1309.0238):
# 
# - *Consistency*: All objects share a common interface drawn from a limited set of methods, with consistent documentation.
# 
# - *Inspection*: All specified parameter values are exposed as public attributes.
# 
# - *Limited object hierarchy*: Only algorithms are represented by Python classes; datasets are represented
#   in standard formats (NumPy arrays, Pandas ``DataFrame``s, SciPy sparse matrices) and parameter
#   names use standard Python strings.
# 
# - *Composition*: Many machine learning tasks can be expressed as sequences of more fundamental algorithms,
#   and Scikit-Learn makes use of this wherever possible.
# 
# - *Sensible defaults*: When models require user-specified parameters, the library defines an appropriate default value.
# 
# In practice, these principles make Scikit-Learn very easy to use, once the basic principles are understood.
# Every machine learning algorithm in Scikit-Learn is implemented via the Estimator API, which provides a consistent interface for a wide range of machine learning applications.
# 

# ### Basics of the API
# 
# Most commonly, the steps in using the Scikit-Learn estimator API are as follows
# (we will step through a handful of detailed examples in the sections that follow).
# 
# 1. Choose a class of model by importing the appropriate estimator class from Scikit-Learn.
# 2. Choose model hyperparameters by instantiating this class with desired values.
# 3. Arrange data into a features matrix and target vector following the discussion above.
# 4. Fit the model to your data by calling the ``fit()`` method of the model instance.
# 5. Apply the Model to new data:
#    - For supervised learning, often we predict labels for unknown data using the ``predict()`` method.
#    - For unsupervised learning, we often transform or infer properties of the data using the ``transform()`` or ``predict()`` method.
# 
# We will now step through several simple examples of applying supervised and unsupervised learning methods.
# 

# ### Supervised learning example: Simple linear regression
# 
# As an example of this process, let's consider a simple linear regression—that is, the common case of fitting a line to $(x, y)$ data.
# We will use the following simple data for our regression example:
# 

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x, y);


# With this data in place, we can use the recipe outlined earlier. Let's walk through the process: 
# 

# #### 1. Choose a class of model
# 
# In Scikit-Learn, every class of model is represented by a Python class.
# So, for example, if we would like to compute a simple linear regression model, we can import the linear regression class:
# 

from sklearn.linear_model import LinearRegression


# Note that other more general linear regression models exist as well; you can read more about them in the [``sklearn.linear_model`` module documentation](http://Scikit-Learn.org/stable/modules/linear_model.html).
# 

# #### 2. Choose model hyperparameters
# 
# An important point is that *a class of model is not the same as an instance of a model*.
# 
# Once we have decided on our model class, there are still some options open to us.
# Depending on the model class we are working with, we might need to answer one or more questions like the following:
# 
# - Would we like to fit for the offset (i.e., *y*-intercept)?
# - Would we like the model to be normalized?
# - Would we like to preprocess our features to add model flexibility?
# - What degree of regularization would we like to use in our model?
# - How many model components would we like to use?
# 
# These are examples of the important choices that must be made *once the model class is selected*.
# These choices are often represented as *hyperparameters*, or parameters that must be set before the model is fit to data.
# In Scikit-Learn, hyperparameters are chosen by passing values at model instantiation.
# We will explore how you can quantitatively motivate the choice of hyperparameters in [Hyperparameters and Model Validation](05.03-Hyperparameters-and-Model-Validation.ipynb).
# 
# For our linear regression example, we can instantiate the ``LinearRegression`` class and specify that we would like to fit the intercept using the ``fit_intercept`` hyperparameter:

model = LinearRegression(fit_intercept=True)
model


# Keep in mind that when the model is instantiated, the only action is the storing of these hyperparameter values.
# In particular, we have not yet applied the model to any data: the Scikit-Learn API makes very clear the distinction between *choice of model* and *application of model to data*.
# 

# #### 3. Arrange data into a features matrix and target vector
# 
# Previously we detailed the Scikit-Learn data representation, which requires a two-dimensional features matrix and a one-dimensional target array.
# Here our target variable ``y`` is already in the correct form (a length-``n_samples`` array), but we need to massage the data ``x`` to make it a matrix of size ``[n_samples, n_features]``.
# In this case, this amounts to a simple reshaping of the one-dimensional array:
# 

X = x[:, np.newaxis]
X.shape


# #### 4. Fit the model to your data
# 
# Now it is time to apply our model to data.
# This can be done with the ``fit()`` method of the model:
# 

model.fit(X, y)


# This ``fit()`` command causes a number of model-dependent internal computations to take place, and the results of these computations are stored in model-specific attributes that the user can explore.
# In Scikit-Learn, by convention all model parameters that were learned during the ``fit()`` process have trailing underscores; for example in this linear model, we have the following:
# 

model.coef_


model.intercept_


# These two parameters represent the slope and intercept of the simple linear fit to the data.
# Comparing to the data definition, we see that they are very close to the input slope of 2 and intercept of -1.
# 
# One question that frequently comes up regards the uncertainty in such internal model parameters.
# In general, Scikit-Learn does not provide tools to draw conclusions from internal model parameters themselves: interpreting model parameters is much more a *statistical modeling* question than a *machine learning* question.
# Machine learning rather focuses on what the model *predicts*.
# If you would like to dive into the meaning of fit parameters within the model, other tools are available, including the [Statsmodels Python package](http://statsmodels.sourceforge.net/).
# 

# #### 5. Predict labels for unknown data
# 
# Once the model is trained, the main task of supervised machine learning is to evaluate it based on what it says about new data that was not part of the training set.
# In Scikit-Learn, this can be done using the ``predict()`` method.
# For the sake of this example, our "new data" will be a grid of *x* values, and we will ask what *y* values the model predicts:
# 

xfit = np.linspace(-1, 11)


# As before, we need to coerce these *x* values into a ``[n_samples, n_features]`` features matrix, after which we can feed it to the model:
# 

Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)


# Finally, let's visualize the results by plotting first the raw data, and then this model fit:
# 

plt.scatter(x, y)
plt.plot(xfit, yfit);


# Typically the efficacy of the model is evaluated by comparing its results to some known baseline, as we will see in the next example
# 

# ### Supervised learning example: Iris classification
# 
# Let's take a look at another example of this process, using the Iris dataset we discussed earlier.
# Our question will be this: given a model trained on a portion of the Iris data, how well can we predict the remaining labels?
# 
# For this task, we will use an extremely simple generative model known as Gaussian naive Bayes, which proceeds by assuming each class is drawn from an axis-aligned Gaussian distribution (see [In Depth: Naive Bayes Classification](05.05-Naive-Bayes.ipynb) for more details).
# Because it is so fast and has no hyperparameters to choose, Gaussian naive Bayes is often a good model to use as a baseline classification, before exploring whether improvements can be found through more sophisticated models.
# 
# We would like to evaluate the model on data it has not seen before, and so we will split the data into a *training set* and a *testing set*.
# This could be done by hand, but it is more convenient to use the ``train_test_split`` utility function:

from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,
                                                random_state=1)


# With the data arranged, we can follow our recipe to predict the labels:
# 

from sklearn.naive_bayes import GaussianNB # 1. choose model class
model = GaussianNB()                       # 2. instantiate model
model.fit(Xtrain, ytrain)                  # 3. fit model to data
y_model = model.predict(Xtest)             # 4. predict on new data


# Finally, we can use the ``accuracy_score`` utility to see the fraction of predicted labels that match their true value:
# 

from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)


# With an accuracy topping 97%, we see that even this very naive classification algorithm is effective for this particular dataset!
# 

# ### Unsupervised learning example: Iris dimensionality
# 
# As an example of an unsupervised learning problem, let's take a look at reducing the dimensionality of the Iris data so as to more easily visualize it.
# Recall that the Iris data is four dimensional: there are four features recorded for each sample.
# 
# The task of dimensionality reduction is to ask whether there is a suitable lower-dimensional representation that retains the essential features of the data.
# Often dimensionality reduction is used as an aid to visualizing data: after all, it is much easier to plot data in two dimensions than in four dimensions or higher!
# 
# Here we will use principal component analysis (PCA; see [In Depth: Principal Component Analysis](05.09-Principal-Component-Analysis.ipynb)), which is a fast linear dimensionality reduction technique.
# We will ask the model to return two components—that is, a two-dimensional representation of the data.
# 
# Following the sequence of steps outlined earlier, we have:
# 

from sklearn.decomposition import PCA  # 1. Choose the model class
model = PCA(n_components=2)            # 2. Instantiate the model with hyperparameters
model.fit(X_iris)                      # 3. Fit to data. Notice y is not specified!
X_2D = model.transform(X_iris)         # 4. Transform the data to two dimensions


# Now let's plot the results. A quick way to do this is to insert the results into the original Iris ``DataFrame``, and use Seaborn's ``lmplot`` to show the results:
# 

iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False);


# We see that in the two-dimensional representation, the species are fairly well separated, even though the PCA algorithm had no knowledge of the species labels!
# This indicates to us that a relatively straightforward classification will probably be effective on the dataset, as we saw before.
# 

# ### Unsupervised learning: Iris clustering
# 
# Let's next look at applying clustering to the Iris data.
# A clustering algorithm attempts to find distinct groups of data without reference to any labels.
# Here we will use a powerful clustering method called a Gaussian mixture model (GMM), discussed in more detail in [In Depth: Gaussian Mixture Models](05.12-Gaussian-Mixtures.ipynb).
# A GMM attempts to model the data as a collection of Gaussian blobs.
# 
# We can fit the Gaussian mixture model as follows:
# 

from sklearn.mixture import GMM      # 1. Choose the model class
model = GMM(n_components=3,
            covariance_type='full')  # 2. Instantiate the model with hyperparameters
model.fit(X_iris)                    # 3. Fit to data. Notice y is not specified!
y_gmm = model.predict(X_iris)        # 4. Determine cluster labels


# As before, we will add the cluster label to the Iris ``DataFrame`` and use Seaborn to plot the results:
# 

iris['cluster'] = y_gmm
sns.lmplot("PCA1", "PCA2", data=iris, hue='species',
           col='cluster', fit_reg=False);


# By splitting the data by cluster number, we see exactly how well the GMM algorithm has recovered the underlying label: the *setosa* species is separated perfectly within cluster 0, while there remains a small amount of mixing between *versicolor* and *virginica*.
# This means that even without an expert to tell us the species labels of the individual flowers, the measurements of these flowers are distinct enough that we could *automatically* identify the presence of these different groups of species with a simple clustering algorithm!
# This sort of algorithm might further give experts in the field clues as to the relationship between the samples they are observing.
# 

# ## Application: Exploring Hand-written Digits
# 

# To demonstrate these principles on a more interesting problem, let's consider one piece of the optical character recognition problem: the identification of hand-written digits.
# In the wild, this problem involves both locating and identifying characters in an image. Here we'll take a shortcut and use Scikit-Learn's set of pre-formatted digits, which is built into the library.
# 

# ### Loading and visualizing the digits data
# 
# We'll use Scikit-Learn's data access interface and take a look at this data:
# 

from sklearn.datasets import load_digits
digits = load_digits()
digits.images.shape


# The images data is a three-dimensional array: 1,797 samples each consisting of an 8 × 8 grid of pixels.
# Let's visualize the first hundred of these:
# 

import matplotlib.pyplot as plt

fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]),
            transform=ax.transAxes, color='green')


# In order to work with this data within Scikit-Learn, we need a two-dimensional, ``[n_samples, n_features]`` representation.
# We can accomplish this by treating each pixel in the image as a feature: that is, by flattening out the pixel arrays so that we have a length-64 array of pixel values representing each digit.
# Additionally, we need the target array, which gives the previously determined label for each digit.
# These two quantities are built into the digits dataset under the ``data`` and ``target`` attributes, respectively:
# 

X = digits.data
X.shape


y = digits.target
y.shape


# We see here that there are 1,797 samples and 64 features.
# 

# ### Unsupervised learning: Dimensionality reduction
# 
# We'd like to visualize our points within the 64-dimensional parameter space, but it's difficult to effectively visualize points in such a high-dimensional space.
# Instead we'll reduce the dimensions to 2, using an unsupervised method.
# Here, we'll make use of a manifold learning algorithm called *Isomap* (see [In-Depth: Manifold Learning](05.10-Manifold-Learning.ipynb)), and transform the data to two dimensions:
# 

from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)
data_projected.shape


# We see that the projected data is now two-dimensional.
# Let's plot this data to see if we can learn anything from its structure:
# 

plt.scatter(data_projected[:, 0], data_projected[:, 1], c=digits.target,
            edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', 10))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5);


# This plot gives us some good intuition into how well various numbers are separated in the larger 64-dimensional space. For example, zeros (in black) and ones (in purple) have very little overlap in parameter space.
# Intuitively, this makes sense: a zero is empty in the middle of the image, while a one will generally have ink in the middle.
# On the other hand, there seems to be a more or less continuous spectrum between ones and fours: we can understand this by realizing that some people draw ones with "hats" on them, which cause them to look similar to fours.
# 
# Overall, however, the different groups appear to be fairly well separated in the parameter space: this tells us that even a very straightforward supervised classification algorithm should perform suitably on this data.
# Let's give it a try.
# 

# ### Classification on digits
# 
# Let's apply a classification algorithm to the digits.
# As with the Iris data previously, we will split the data into a training and testing set, and fit a Gaussian naive Bayes model:
# 

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)


# Now that we have predicted our model, we can gauge its accuracy by comparing the true values of the test set to the predictions:
# 

from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)


# With even this extremely simple model, we find about 80% accuracy for classification of the digits!
# However, this single number doesn't tell us *where* we've gone wrong—one nice way to do this is to use the *confusion matrix*, which we can compute with Scikit-Learn and plot with Seaborn:
# 

from sklearn.metrics import confusion_matrix

mat = confusion_matrix(ytest, y_model)

sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value');


# This shows us where the mis-labeled points tend to be: for example, a large number of twos here are mis-classified as either ones or eights.
# Another way to gain intuition into the characteristics of the model is to plot the inputs again, with their predicted labels.
# We'll use green for correct labels, and red for incorrect labels:
# 

fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

test_images = Xtest.reshape(-1, 8, 8)

for i, ax in enumerate(axes.flat):
    ax.imshow(test_images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(y_model[i]),
            transform=ax.transAxes,
            color='green' if (ytest[i] == y_model[i]) else 'red')


# Examining this subset of the data, we can gain insight regarding where the algorithm might be not performing optimally.
# To go beyond our 80% classification rate, we might move to a more sophisticated algorithm such as support vector machines (see [In-Depth: Support Vector Machines](05.07-Support-Vector-Machines.ipynb)), random forests (see [In-Depth: Decision Trees and Random Forests](05.08-Random-Forests.ipynb)) or another classification approach.
# 

# ## Summary
# 

# In this section we have covered the essential features of the Scikit-Learn data representation, and the estimator API.
# Regardless of the type of estimator, the same import/instantiate/fit/predict pattern holds.
# Armed with this information about the estimator API, you can explore the Scikit-Learn documentation and begin trying out various models on your data.
# 
# In the next section, we will explore perhaps the most important topic in machine learning: how to select and validate your model.
# 

# <!--NAVIGATION-->
# < [What Is Machine Learning?](05.01-What-Is-Machine-Learning.ipynb) | [Contents](Index.ipynb) | [Hyperparameters and Model Validation](05.03-Hyperparameters-and-Model-Validation.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [IPython Magic Commands](01.03-Magic-Commands.ipynb) | [Contents](Index.ipynb) | [IPython and Shell Commands](01.05-IPython-And-Shell-Commands.ipynb) >
# 

# # Input and Output History
# 
# Previously we saw that the IPython shell allows you to access previous commands with the up and down arrow keys, or equivalently the Ctrl-p/Ctrl-n shortcuts.
# Additionally, in both the shell and the notebook, IPython exposes several ways to obtain the output of previous commands, as well as string versions of the commands themselves.
# We'll explore those here.
# 

# ## IPython's ``In`` and ``Out`` Objects
# 
# By now I imagine you're quite familiar with the ``In [1]:``/``Out[1]:`` style prompts used by IPython.
# But it turns out that these are not just pretty decoration: they give a clue as to how you can access previous inputs and outputs in your current session.
# Imagine you start a session that looks like this:
# 
# ```ipython
# In [1]: import math
# 
# In [2]: math.sin(2)
# Out[2]: 0.9092974268256817
# 
# In [3]: math.cos(2)
# Out[3]: -0.4161468365471424
# ```
# 

# We've imported the built-in ``math`` package, then computed the sine and the cosine of the number 2.
# These inputs and outputs are displayed in the shell with ``In``/``Out`` labels, but there's more–IPython actually creates some Python variables called ``In`` and ``Out`` that are automatically updated to reflect this history:
# 
# ```ipython
# In [4]: print(In)
# ['', 'import math', 'math.sin(2)', 'math.cos(2)', 'print(In)']
# 
# In [5]: Out
# Out[5]: {2: 0.9092974268256817, 3: -0.4161468365471424}
# ```
# 

# The ``In`` object is a list, which keeps track of the commands in order (the first item in the list is a place-holder so that ``In[1]`` can refer to the first command):
# 
# ```ipython
# In [6]: print(In[1])
# import math
# ```
# 
# The ``Out`` object is not a list but a dictionary mapping input numbers to their outputs (if any):
# 
# ```ipython
# In [7]: print(Out[2])
# 0.9092974268256817
# ```
# 
# Note that not all operations have outputs: for example, ``import`` statements and ``print`` statements don't affect the output.
# The latter may be surprising, but makes sense if you consider that ``print`` is a function that returns ``None``; for brevity, any command that returns ``None`` is not added to ``Out``.
# 
# Where this can be useful is if you want to interact with past results.
# For example, let's check the sum of ``sin(2) ** 2`` and ``cos(2) ** 2`` using the previously-computed results:
# 
# ```ipython
# In [8]: Out[2] ** 2 + Out[3] ** 2
# Out[8]: 1.0
# ```
# 
# The result is ``1.0`` as we'd expect from the well-known trigonometric identity.
# In this case, using these previous results probably is not necessary, but it can become very handy if you execute a very expensive computation and want to reuse the result!
# 

# ## Underscore Shortcuts and Previous Outputs
# 
# The standard Python shell contains just one simple shortcut for accessing previous output; the variable ``_`` (i.e., a single underscore) is kept updated with the previous output; this works in IPython as well:
# 
# ```ipython
# In [9]: print(_)
# 1.0
# ```
# 
# But IPython takes this a bit further—you can use a double underscore to access the second-to-last output, and a triple underscore to access the third-to-last output (skipping any commands with no output):
# 
# ```ipython
# In [10]: print(__)
# -0.4161468365471424
# 
# In [11]: print(___)
# 0.9092974268256817
# ```
# 
# IPython stops there: more than three underscores starts to get a bit hard to count, and at that point it's easier to refer to the output by line number.
# 
# There is one more shortcut we should mention, however–a shorthand for ``Out[X]`` is ``_X`` (i.e., a single underscore followed by the line number):
# 
# ```ipython
# In [12]: Out[2]
# Out[12]: 0.9092974268256817
# 
# In [13]: _2
# Out[13]: 0.9092974268256817
# ```
# 

# ## Suppressing Output
# Sometimes you might wish to suppress the output of a statement (this is perhaps most common with the plotting commands that we'll explore in [Introduction to Matplotlib](04.00-Introduction-To-Matplotlib.ipynb)).
# Or maybe the command you're executing produces a result that you'd prefer not like to store in your output history, perhaps so that it can be deallocated when other references are removed.
# The easiest way to suppress the output of a command is to add a semicolon to the end of the line:
# 
# ```ipython
# In [14]: math.sin(2) + math.cos(2);
# ```
# 
# Note that the result is computed silently, and the output is neither displayed on the screen or stored in the ``Out`` dictionary:
# 
# ```ipython
# In [15]: 14 in Out
# Out[15]: False
# ```
# 

# ## Related Magic Commands
# For accessing a batch of previous inputs at once, the ``%history`` magic command is very helpful.
# Here is how you can print the first four inputs:
# 
# ```ipython
# In [16]: %history -n 1-4
#    1: import math
#    2: math.sin(2)
#    3: math.cos(2)
#    4: print(In)
# ```
# 
# As usual, you can type ``%history?`` for more information and a description of options available.
# Other similar magic commands are ``%rerun`` (which will re-execute some portion of the command history) and ``%save`` (which saves some set of the command history to a file).
# For more information, I suggest exploring these using the ``?`` help functionality discussed in [Help and Documentation in IPython](01.01-Help-And-Documentation.ipynb).
# 

# <!--NAVIGATION-->
# < [IPython Magic Commands](01.03-Magic-Commands.ipynb) | [Contents](Index.ipynb) | [IPython and Shell Commands](01.05-IPython-And-Shell-Commands.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [Sorting Arrays](02.08-Sorting.ipynb) | [Contents](Index.ipynb) | [Data Manipulation with Pandas](03.00-Introduction-to-Pandas.ipynb) >
# 

# # Structured Data: NumPy's Structured Arrays
# 

# While often our data can be well represented by a homogeneous array of values, sometimes this is not the case. This section demonstrates the use of NumPy's *structured arrays* and *record arrays*, which provide efficient storage for compound, heterogeneous data.  While the patterns shown here are useful for simple operations, scenarios like this often lend themselves to the use of Pandas ``Dataframe``s, which we'll explore in [Chapter 3](03.00-Introduction-to-Pandas.ipynb).
# 

import numpy as np


# Imagine that we have several categories of data on a number of people (say, name, age, and weight), and we'd like to store these values for use in a Python program.
# It would be possible to store these in three separate arrays:
# 

name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]


# But this is a bit clumsy. There's nothing here that tells us that the three arrays are related; it would be more natural if we could use a single structure to store all of this data.
# NumPy can handle this through structured arrays, which are arrays with compound data types.
# 
# Recall that previously we created a simple array using an expression like this:
# 

x = np.zeros(4, dtype=int)


# We can similarly create a structured array using a compound data type specification:
# 

# Use a compound data type for structured arrays
data = np.zeros(4, dtype={'names':('name', 'age', 'weight'),
                          'formats':('U10', 'i4', 'f8')})
print(data.dtype)


# Here ``'U10'`` translates to "Unicode string of maximum length 10," ``'i4'`` translates to "4-byte (i.e., 32 bit) integer," and ``'f8'`` translates to "8-byte (i.e., 64 bit) float."
# We'll discuss other options for these type codes in the following section.
# 
# Now that we've created an empty container array, we can fill the array with our lists of values:
# 

data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)


# As we had hoped, the data is now arranged together in one convenient block of memory.
# 
# The handy thing with structured arrays is that you can now refer to values either by index or by name:
# 

# Get all names
data['name']


# Get first row of data
data[0]


# Get the name from the last row
data[-1]['name']


# Using Boolean masking, this even allows you to do some more sophisticated operations such as filtering on age:
# 

# Get names where age is under 30
data[data['age'] < 30]['name']


# Note that if you'd like to do any operations that are any more complicated than these, you should probably consider the Pandas package, covered in the next chapter.
# As we'll see, Pandas provides a ``Dataframe`` object, which is a structure built on NumPy arrays that offers a variety of useful data manipulation functionality similar to what we've shown here, as well as much, much more.
# 

# ## Creating Structured Arrays
# 
# Structured array data types can be specified in a number of ways.
# Earlier, we saw the dictionary method:
# 

np.dtype({'names':('name', 'age', 'weight'),
          'formats':('U10', 'i4', 'f8')})


# For clarity, numerical types can be specified using Python types or NumPy ``dtype``s instead:
# 

np.dtype({'names':('name', 'age', 'weight'),
          'formats':((np.str_, 10), int, np.float32)})


# A compound type can also be specified as a list of tuples:
# 

np.dtype([('name', 'S10'), ('age', 'i4'), ('weight', 'f8')])


# If the names of the types do not matter to you, you can specify the types alone in a comma-separated string:
# 

np.dtype('S10,i4,f8')


# The shortened string format codes may seem confusing, but they are built on simple principles.
# The first (optional) character is ``<`` or ``>``, which means "little endian" or "big endian," respectively, and specifies the ordering convention for significant bits.
# The next character specifies the type of data: characters, bytes, ints, floating points, and so on (see the table below).
# The last character or characters represents the size of the object in bytes.
# 
# | Character        | Description           | Example                             |
# | ---------        | -----------           | -------                             | 
# | ``'b'``          | Byte                  | ``np.dtype('b')``                   |
# | ``'i'``          | Signed integer        | ``np.dtype('i4') == np.int32``      |
# | ``'u'``          | Unsigned integer      | ``np.dtype('u1') == np.uint8``      |
# | ``'f'``          | Floating point        | ``np.dtype('f8') == np.int64``      |
# | ``'c'``          | Complex floating point| ``np.dtype('c16') == np.complex128``|
# | ``'S'``, ``'a'`` | String                | ``np.dtype('S5')``                  |
# | ``'U'``          | Unicode string        | ``np.dtype('U') == np.str_``        |
# | ``'V'``          | Raw data (void)       | ``np.dtype('V') == np.void``        |
# 

# ## More Advanced Compound Types
# 
# It is possible to define even more advanced compound types.
# For example, you can create a type where each element contains an array or matrix of values.
# Here, we'll create a data type with a ``mat`` component consisting of a $3\times 3$ floating-point matrix:
# 

tp = np.dtype([('id', 'i8'), ('mat', 'f8', (3, 3))])
X = np.zeros(1, dtype=tp)
print(X[0])
print(X['mat'][0])


# Now each element in the ``X`` array consists of an ``id`` and a $3\times 3$ matrix.
# Why would you use this rather than a simple multidimensional array, or perhaps a Python dictionary?
# The reason is that this NumPy ``dtype`` directly maps onto a C structure definition, so the buffer containing the array content can be accessed directly within an appropriately written C program.
# If you find yourself writing a Python interface to a legacy C or Fortran library that manipulates structured data, you'll probably find structured arrays quite useful!

# ## RecordArrays: Structured Arrays with a Twist
# 
# NumPy also provides the ``np.recarray`` class, which is almost identical to the structured arrays just described, but with one additional feature: fields can be accessed as attributes rather than as dictionary keys.
# Recall that we previously accessed the ages by writing:
# 

data['age']


# If we view our data as a record array instead, we can access this with slightly fewer keystrokes:
# 

data_rec = data.view(np.recarray)
data_rec.age


# The downside is that for record arrays, there is some extra overhead involved in accessing the fields, even when using the same syntax. We can see this here:
# 

get_ipython().magic("timeit data['age']")
get_ipython().magic("timeit data_rec['age']")
get_ipython().magic('timeit data_rec.age')


# Whether the more convenient notation is worth the additional overhead will depend on your own application.
# 

# ## On to Pandas
# 
# This section on structured and record arrays is purposely at the end of this chapter, because it leads so well into the next package we will cover: Pandas.
# Structured arrays like the ones discussed here are good to know about for certain situations, especially in case you're using NumPy arrays to map onto binary data formats in C, Fortran, or another language.
# For day-to-day use of structured data, the Pandas package is a much better choice, and we'll dive into a full discussion of it in the chapter that follows.
# 

# <!--NAVIGATION-->
# < [Sorting Arrays](02.08-Sorting.ipynb) | [Contents](Index.ipynb) | [Data Manipulation with Pandas](03.00-Introduction-to-Pandas.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [Preface](00.00-Preface.ipynb) | [Contents](Index.ipynb) | [Help and Documentation in IPython](01.01-Help-And-Documentation.ipynb) >
# 

# # IPython: Beyond Normal Python
# 

# There are many options for development environments for Python, and I'm often asked which one I use in my own work.
# My answer sometimes surprises people: my preferred environment is [IPython](http://ipython.org/) plus a text editor (in my case, Emacs or Atom depending on my mood).
# IPython (short for *Interactive Python*) was started in 2001 by Fernando Perez as an enhanced Python interpreter, and has since grown into a project aiming to provide, in Perez's words, "Tools for the entire life cycle of research computing."
# If Python is the engine of our data science task, you might think of IPython as the interactive control panel.
# 
# As well as being a useful interactive interface to Python, IPython also provides a number of useful syntactic additions to the language; we'll cover the most useful of these additions here.
# In addition, IPython is closely tied with the [Jupyter project](http://jupyter.org), which provides a browser-based notebook that is useful for development, collaboration, sharing, and even publication of data science results.
# The IPython notebook is actually a special case of the broader Jupyter notebook structure, which encompasses notebooks for Julia, R, and other programming languages.
# As an example of the usefulness of the notebook format, look no further than the page you are reading: the entire manuscript for this book was composed as a set of IPython notebooks.
# 
# IPython is about using Python effectively for interactive scientific and data-intensive computing.
# This chapter will start by stepping through some of the IPython features that are useful to the practice of data science, focusing especially on the syntax it offers beyond the standard features of Python.
# Next, we will go into a bit more depth on some of the more useful "magic commands" that can speed-up common tasks in creating and using data science code.
# Finally, we will touch on some of the features of the notebook that make it useful in understanding data and sharing results.
# 

# ## Shell or Notebook?
# 
# There are two primary means of using IPython that we'll discuss in this chapter: the IPython shell and the IPython notebook.
# The bulk of the material in this chapter is relevant to both, and the examples will switch between them depending on what is most convenient.
# In the few sections that are relevant to just one or the other, we will explicitly state that fact.
# Before we start, some words on how to launch the IPython shell and IPython notebook.
# 

# ### Launching the IPython Shell
# 
# This chapter, like most of this book, is not designed to be absorbed passively.
# I recommend that as you read through it, you follow along and experiment with the tools and syntax we cover: the muscle-memory you build through doing this will be far more useful than the simple act of reading about it.
# Start by launching the IPython interpreter by typing **``ipython``** on the command-line; alternatively, if you've installed a distribution like Anaconda or EPD, there may be a launcher specific to your system (we'll discuss this more fully in [Help and Documentation in IPython](01.01-Help-And-Documentation.ipynb)).
# 
# Once you do this, you should see a prompt like the following:
# ```
# IPython 4.0.1 -- An enhanced Interactive Python.
# ?         -> Introduction and overview of IPython's features.
# %quickref -> Quick reference.
# help      -> Python's own help system.
# object?   -> Details about 'object', use 'object??' for extra details.
# In [1]:
# ```
# With that, you're ready to follow along.

# ### Launching the Jupyter Notebook
# 
# The Jupyter notebook is a browser-based graphical interface to the IPython shell, and builds on it a rich set of dynamic display capabilities.
# As well as executing Python/IPython statements, the notebook allows the user to include formatted text, static and dynamic visualizations, mathematical equations, JavaScript widgets, and much more.
# Furthermore, these documents can be saved in a way that lets other people open them and execute the code on their own systems.
# 
# Though the IPython notebook is viewed and edited through your web browser window, it must connect to a running Python process in order to execute code.
# This process (known as a "kernel") can be started by running the following command in your system shell:
# 
# ```
# $ jupyter notebook
# ```
# 
# This command will launch a local web server that will be visible to your browser.
# It immediately spits out a log showing what it is doing; that log will look something like this:
# 
# ```
# $ jupyter notebook
# [NotebookApp] Serving notebooks from local directory: /Users/jakevdp/PythonDataScienceHandbook
# [NotebookApp] 0 active kernels 
# [NotebookApp] The IPython Notebook is running at: http://localhost:8888/
# [NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
# ```
# 
# Upon issuing the command, your default browser should automatically open and navigate to the listed local URL;
# the exact address will depend on your system.
# If the browser does not open automatically, you can open a window and manually open this address (*http://localhost:8888/* in this example).
# 

# <!--NAVIGATION-->
# < [Preface](00.00-Preface.ipynb) | [Contents](Index.ipynb) | [Help and Documentation in IPython](01.01-Help-And-Documentation.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [Visualization with Seaborn](04.14-Visualization-With-Seaborn.ipynb) | [Contents](Index.ipynb) | [Machine Learning](05.00-Machine-Learning.ipynb) >
# 

# # Further Resources
# 

# ## Matplotlib Resources
# 
# A single chapter in a book can never hope to cover all the available features and plot types available in Matplotlib.
# As with other packages we've seen, liberal use of IPython's tab-completion and help functions (see [Help and Documentation in IPython](01.01-Help-And-Documentation.ipynb)) can be very helpful when exploring Matplotlib's API.
# In addition, Matplotlib’s [online documentation](http://matplotlib.org/) can be a helpful reference.
# See in particular the [Matplotlib gallery](http://matplotlib.org/gallery.html) linked on that page: it shows thumbnails of hundreds of different plot types, each one linked to a page with the Python code snippet used to generate it.
# In this way, you can visually inspect and learn about a wide range of different plotting styles and visualization techniques.
# 
# For a book-length treatment of Matplotlib, I would recommend [*Interactive Applications Using Matplotlib*](https://www.packtpub.com/application-development/interactive-applications-using-matplotlib), written by Matplotlib core developer Ben Root.
# 

# ## Other Python Graphics Libraries
# 
# Although Matplotlib is the most prominent Python visualization library, there are other more modern tools that are worth exploring as well.
# I'll mention a few of them briefly here:
# 
# - [Bokeh](http://bokeh.pydata.org) is a JavaScript visualization library with a Python frontend that creates highly interactive visualizations capable of handling very large and/or streaming datasets. The Python front-end outputs a JSON data structure that can be interpreted by the Bokeh JS engine.
# - [Plotly](http://plot.ly) is the eponymous open source product of the Plotly company, and is similar in spirit to Bokeh. Because Plotly is the main product of a startup, it is receiving a high level of development effort. Use of the library is entirely free.
# - [Vispy](http://vispy.org/) is an actively developed project focused on dynamic visualizations of very large datasets. Because it is built to target OpenGL and make use of efficient graphics processors in your computer, it is able to render some quite large and stunning visualizations.
# - [Vega](https://vega.github.io/) and [Vega-Lite](https://vega.github.io/vega-lite) are declarative graphics representations, and are the product of years of research into the fundamental language of data visualization. The reference rendering implementation is JavaScript, but the API is language agnostic. There is a Python API under development in the [Altair](https://altair-viz.github.io/) package. Though as of summer 2016 it's not yet fully mature, I'm quite excited for the possibilities of this project to provide a common reference point for visualization in Python and other languages.
# 
# The visualization space in the Python community is very dynamic, and I fully expect this list to be out of date as soon as it is published.
# Keep an eye out for what's coming in the future!
# 

# <!--NAVIGATION-->
# < [Visualization with Seaborn](04.14-Visualization-With-Seaborn.ipynb) | [Contents](Index.ipynb) | [Machine Learning](05.00-Machine-Learning.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [High-Performance Pandas: eval() and query()](03.12-Performance-Eval-and-Query.ipynb) | [Contents](Index.ipynb) | [Visualization with Matplotlib](04.00-Introduction-To-Matplotlib.ipynb) >
# 

# # Further Resources
# 
# In this chapter, we've covered many of the basics of using Pandas effectively for data analysis.
# Still, much has been omitted from our discussion.
# To learn more about Pandas, I recommend the following resources:
# 
# - [Pandas online documentation](http://pandas.pydata.org/): This is the go-to source for complete documentation of the package. While the examples in the documentation tend to be small generated datasets, the description of the options is complete and generally very useful for understanding the use of various functions.
# 
# - [*Python for Data Analysis*](http://shop.oreilly.com/product/0636920023784.do) Written by Wes McKinney (the original creator of Pandas), this book contains much more detail on the Pandas package than we had room for in this chapter. In particular, he takes a deep dive into tools for time series, which were his bread and butter as a financial consultant. The book also has many entertaining examples of applying Pandas to gain insight from real-world datasets. Keep in mind, though, that the book is now several years old, and the Pandas package has quite a few new features that this book does not cover (but be on the lookout for a new edition in 2017).
# 
# - [Stack Overflow](http://stackoverflow.com/questions/tagged/pandas): Pandas has so many users that any question you have has likely been asked and answered on Stack Overflow. Using Pandas is a case where some Google-Fu is your best friend. Simply go to your favorite search engine and type in the question, problem, or error you're coming across–more than likely you'll find your answer on a Stack Overflow page.
# 
# - [Pandas on PyVideo](http://pyvideo.org/search?q=pandas): From PyCon to SciPy to PyData, many conferences have featured tutorials from Pandas developers and power users. The PyCon tutorials in particular tend to be given by very well-vetted presenters.
# 
# Using these resources, combined with the walk-through given in this chapter, my hope is that you'll be poised to use Pandas to tackle any data analysis problem you come across!
# 

# <!--NAVIGATION-->
# < [High-Performance Pandas: eval() and query()](03.12-Performance-Eval-and-Query.ipynb) | [Contents](Index.ipynb) | [Visualization with Matplotlib](04.00-Introduction-To-Matplotlib.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [Errors and Debugging](01.06-Errors-and-Debugging.ipynb) | [Contents](Index.ipynb) | [More IPython Resources](01.08-More-IPython-Resources.ipynb) >
# 

# # Profiling and Timing Code
# 
# In the process of developing code and creating data processing pipelines, there are often trade-offs you can make between various implementations.
# Early in developing your algorithm, it can be counterproductive to worry about such things. As Donald Knuth famously quipped, "We should forget about small efficiencies, say about 97% of the time: premature optimization is the root of all evil."
# 
# But once you have your code working, it can be useful to dig into its efficiency a bit.
# Sometimes it's useful to check the execution time of a given command or set of commands; other times it's useful to dig into a multiline process and determine where the bottleneck lies in some complicated series of operations.
# IPython provides access to a wide array of functionality for this kind of timing and profiling of code.
# Here we'll discuss the following IPython magic commands:
# 
# - ``%time``: Time the execution of a single statement
# - ``%timeit``: Time repeated execution of a single statement for more accuracy
# - ``%prun``: Run code with the profiler
# - ``%lprun``: Run code with the line-by-line profiler
# - ``%memit``: Measure the memory use of a single statement
# - ``%mprun``: Run code with the line-by-line memory profiler
# 
# The last four commands are not bundled with IPython–you'll need to get the ``line_profiler`` and ``memory_profiler`` extensions, which we will discuss in the following sections.
# 

# ## Timing Code Snippets: ``%timeit`` and ``%time``
# 
# We saw the ``%timeit`` line-magic and ``%%timeit`` cell-magic in the introduction to magic functions in [IPython Magic Commands](01.03-Magic-Commands.ipynb); it can be used to time the repeated execution of snippets of code:
# 

get_ipython().magic('timeit sum(range(100))')


# Note that because this operation is so fast, ``%timeit`` automatically does a large number of repetitions.
# For slower commands, ``%timeit`` will automatically adjust and perform fewer repetitions:
# 

get_ipython().run_cell_magic('timeit', '', 'total = 0\nfor i in range(1000):\n    for j in range(1000):\n        total += i * (-1) ** j')


# Sometimes repeating an operation is not the best option.
# For example, if we have a list that we'd like to sort, we might be misled by a repeated operation.
# Sorting a pre-sorted list is much faster than sorting an unsorted list, so the repetition will skew the result:
# 

import random
L = [random.random() for i in range(100000)]
get_ipython().magic('timeit L.sort()')


# For this, the ``%time`` magic function may be a better choice. It also is a good choice for longer-running commands, when short, system-related delays are unlikely to affect the result.
# Let's time the sorting of an unsorted and a presorted list:
# 

import random
L = [random.random() for i in range(100000)]
print("sorting an unsorted list:")
get_ipython().magic('time L.sort()')


print("sorting an already sorted list:")
get_ipython().magic('time L.sort()')


# Notice how much faster the presorted list is to sort, but notice also how much longer the timing takes with ``%time`` versus ``%timeit``, even for the presorted list!
# This is a result of the fact that ``%timeit`` does some clever things under the hood to prevent system calls from interfering with the timing.
# For example, it prevents cleanup of unused Python objects (known as *garbage collection*) which might otherwise affect the timing.
# For this reason, ``%timeit`` results are usually noticeably faster than ``%time`` results.
# 
# For ``%time`` as with ``%timeit``, using the double-percent-sign cell magic syntax allows timing of multiline scripts:
# 

get_ipython().run_cell_magic('time', '', 'total = 0\nfor i in range(1000):\n    for j in range(1000):\n        total += i * (-1) ** j')


# For more information on ``%time`` and ``%timeit``, as well as their available options, use the IPython help functionality (i.e., type ``%time?`` at the IPython prompt).
# 

# ## Profiling Full Scripts: ``%prun``
# 
# A program is made of many single statements, and sometimes timing these statements in context is more important than timing them on their own.
# Python contains a built-in code profiler (which you can read about in the Python documentation), but IPython offers a much more convenient way to use this profiler, in the form of the magic function ``%prun``.
# 
# By way of example, we'll define a simple function that does some calculations:
# 

def sum_of_lists(N):
    total = 0
    for i in range(5):
        L = [j ^ (j >> i) for j in range(N)]
        total += sum(L)
    return total


# Now we can call ``%prun`` with a function call to see the profiled results:
# 

get_ipython().magic('prun sum_of_lists(1000000)')


# In the notebook, the output is printed to the pager, and looks something like this:
# 
# ```
# 14 function calls in 0.714 seconds
# 
#    Ordered by: internal time
# 
#    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#         5    0.599    0.120    0.599    0.120 <ipython-input-19>:4(<listcomp>)
#         5    0.064    0.013    0.064    0.013 {built-in method sum}
#         1    0.036    0.036    0.699    0.699 <ipython-input-19>:1(sum_of_lists)
#         1    0.014    0.014    0.714    0.714 <string>:1(<module>)
#         1    0.000    0.000    0.714    0.714 {built-in method exec}
# ```
# 
# The result is a table that indicates, in order of total time on each function call, where the execution is spending the most time. In this case, the bulk of execution time is in the list comprehension inside ``sum_of_lists``.
# From here, we could start thinking about what changes we might make to improve the performance in the algorithm.
# 
# For more information on ``%prun``, as well as its available options, use the IPython help functionality (i.e., type ``%prun?`` at the IPython prompt).
# 

# ## Line-By-Line Profiling with ``%lprun``
# 
# The function-by-function profiling of ``%prun`` is useful, but sometimes it's more convenient to have a line-by-line profile report.
# This is not built into Python or IPython, but there is a ``line_profiler`` package available for installation that can do this.
# Start by using Python's packaging tool, ``pip``, to install the ``line_profiler`` package:
# 
# ```
# $ pip install line_profiler
# ```
# 
# Next, you can use IPython to load the ``line_profiler`` IPython extension, offered as part of this package:
# 

get_ipython().magic('load_ext line_profiler')


# Now the ``%lprun`` command will do a line-by-line profiling of any function–in this case, we need to tell it explicitly which functions we're interested in profiling:
# 

get_ipython().magic('lprun -f sum_of_lists sum_of_lists(5000)')


# As before, the notebook sends the result to the pager, but it looks something like this:
# 
# ```
# Timer unit: 1e-06 s
# 
# Total time: 0.009382 s
# File: <ipython-input-19-fa2be176cc3e>
# Function: sum_of_lists at line 1
# 
# Line #      Hits         Time  Per Hit   % Time  Line Contents
# ==============================================================
#      1                                           def sum_of_lists(N):
#      2         1            2      2.0      0.0      total = 0
#      3         6            8      1.3      0.1      for i in range(5):
#      4         5         9001   1800.2     95.9          L = [j ^ (j >> i) for j in range(N)]
#      5         5          371     74.2      4.0          total += sum(L)
#      6         1            0      0.0      0.0      return total
# ```
# 
# The information at the top gives us the key to reading the results: the time is reported in microseconds and we can see where the program is spending the most time.
# At this point, we may be able to use this information to modify aspects of the script and make it perform better for our desired use case.
# 
# For more information on ``%lprun``, as well as its available options, use the IPython help functionality (i.e., type ``%lprun?`` at the IPython prompt).
# 

# ## Profiling Memory Use: ``%memit`` and ``%mprun``
# 
# Another aspect of profiling is the amount of memory an operation uses.
# This can be evaluated with another IPython extension, the ``memory_profiler``.
# As with the ``line_profiler``, we start by ``pip``-installing the extension:
# 
# ```
# $ pip install memory_profiler
# ```
# 
# Then we can use IPython to load the extension:
# 

get_ipython().magic('load_ext memory_profiler')


# The memory profiler extension contains two useful magic functions: the ``%memit`` magic (which offers a memory-measuring equivalent of ``%timeit``) and the ``%mprun`` function (which offers a memory-measuring equivalent of ``%lprun``).
# The ``%memit`` function can be used rather simply:
# 

get_ipython().magic('memit sum_of_lists(1000000)')


# We see that this function uses about 100 MB of memory.
# 
# For a line-by-line description of memory use, we can use the ``%mprun`` magic.
# Unfortunately, this magic works only for functions defined in separate modules rather than the notebook itself, so we'll start by using the ``%%file`` magic to create a simple module called ``mprun_demo.py``, which contains our ``sum_of_lists`` function, with one addition that will make our memory profiling results more clear:
# 

get_ipython().run_cell_magic('file', 'mprun_demo.py', 'def sum_of_lists(N):\n    total = 0\n    for i in range(5):\n        L = [j ^ (j >> i) for j in range(N)]\n        total += sum(L)\n        del L # remove reference to L\n    return total')


# We can now import the new version of this function and run the memory line profiler:
# 

from mprun_demo import sum_of_lists
get_ipython().magic('mprun -f sum_of_lists sum_of_lists(1000000)')


# The result, printed to the pager, gives us a summary of the memory use of the function, and looks something like this:
# ```
# Filename: ./mprun_demo.py
# 
# Line #    Mem usage    Increment   Line Contents
# ================================================
#      4     71.9 MiB      0.0 MiB           L = [j ^ (j >> i) for j in range(N)]
# 
# 
# Filename: ./mprun_demo.py
# 
# Line #    Mem usage    Increment   Line Contents
# ================================================
#      1     39.0 MiB      0.0 MiB   def sum_of_lists(N):
#      2     39.0 MiB      0.0 MiB       total = 0
#      3     46.5 MiB      7.5 MiB       for i in range(5):
#      4     71.9 MiB     25.4 MiB           L = [j ^ (j >> i) for j in range(N)]
#      5     71.9 MiB      0.0 MiB           total += sum(L)
#      6     46.5 MiB    -25.4 MiB           del L # remove reference to L
#      7     39.1 MiB     -7.4 MiB       return total
# ```
# Here the ``Increment`` column tells us how much each line affects the total memory budget: observe that when we create and delete the list ``L``, we are adding about 25 MB of memory usage.
# This is on top of the background memory usage from the Python interpreter itself.
# 
# For more information on ``%memit`` and ``%mprun``, as well as their available options, use the IPython help functionality (i.e., type ``%memit?`` at the IPython prompt).
# 

# <!--NAVIGATION-->
# < [Errors and Debugging](01.06-Errors-and-Debugging.ipynb) | [Contents](Index.ipynb) | [More IPython Resources](01.08-More-IPython-Resources.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [More IPython Resources](01.08-More-IPython-Resources.ipynb) | [Contents](Index.ipynb) | [Understanding Data Types in Python](02.01-Understanding-Data-Types.ipynb) >
# 

# # Introduction to NumPy
# 

# This chapter, along with chapter 3, outlines techniques for effectively loading, storing, and manipulating in-memory data in Python.
# The topic is very broad: datasets can come from a wide range of sources and a wide range of formats, including be collections of documents, collections of images, collections of sound clips, collections of numerical measurements, or nearly anything else.
# Despite this apparent heterogeneity, it will help us to think of all data fundamentally as arrays of numbers.
# 
# For example, images–particularly digital images–can be thought of as simply two-dimensional arrays of numbers representing pixel brightness across the area.
# Sound clips can be thought of as one-dimensional arrays of intensity versus time.
# Text can be converted in various ways into numerical representations, perhaps binary digits representing the frequency of certain words or pairs of words.
# No matter what the data are, the first step in making it analyzable will be to transform them into arrays of numbers.
# (We will discuss some specific examples of this process later in [Feature Engineering](05.04-Feature-Engineering.ipynb))
# 
# For this reason, efficient storage and manipulation of numerical arrays is absolutely fundamental to the process of doing data science.
# We'll now take a look at the specialized tools that Python has for handling such numerical arrays: the NumPy package, and the Pandas package (discussed in Chapter 3).
# 
# This chapter will cover NumPy in detail. NumPy (short for *Numerical Python*) provides an efficient interface to store and operate on dense data buffers.
# In some ways, NumPy arrays are like Python's built-in ``list`` type, but NumPy arrays provide much more efficient storage and data operations as the arrays grow larger in size.
# NumPy arrays form the core of nearly the entire ecosystem of data science tools in Python, so time spent learning to use NumPy effectively will be valuable no matter what aspect of data science interests you.
# 
# If you followed the advice outlined in the Preface and installed the Anaconda stack, you already have NumPy installed and ready to go.
# If you're more the do-it-yourself type, you can go to http://www.numpy.org/ and follow the installation instructions found there.
# Once you do, you can import NumPy and double-check the version:
# 

import numpy
numpy.__version__


# For the pieces of the package discussed here, I'd recommend NumPy version 1.8 or later.
# By convention, you'll find that most people in the SciPy/PyData world will import NumPy using ``np`` as an alias:
# 

import numpy as np


# Throughout this chapter, and indeed the rest of the book, you'll find that this is the way we will import and use NumPy.
# 

# ## Reminder about Built In Documentation
# 
# As you read through this chapter, don't forget that IPython gives you the ability to quickly explore the contents of a package (by using the tab-completion feature), as well as the documentation of various functions (using the ``?`` character – Refer back to [Help and Documentation in IPython](01.01-Help-And-Documentation.ipynb)).
# 
# For example, to display all the contents of the numpy namespace, you can type this:
# 
# ```ipython
# In [3]: np.<TAB>
# ```
# 
# And to display NumPy's built-in documentation, you can use this:
# 
# ```ipython
# In [4]: np?
# ```
# 
# More detailed documentation, along with tutorials and other resources, can be found at http://www.numpy.org.

# <!--NAVIGATION-->
# < [More IPython Resources](01.08-More-IPython-Resources.ipynb) | [Contents](Index.ipynb) | [Understanding Data Types in Python](02.01-Understanding-Data-Types.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [Customizing Ticks](04.10-Customizing-Ticks.ipynb) | [Contents](Index.ipynb) | [Three-Dimensional Plotting in Matplotlib](04.12-Three-Dimensional-Plotting.ipynb) >
# 

# # Customizing Matplotlib: Configurations and Stylesheets
# 

# Matplotlib's default plot settings are often the subject of complaint among its users.
# While much is slated to change in the 2.0 Matplotlib release in late 2016, the ability to customize default settings helps bring the package inline with your own aesthetic preferences.
# 
# Here we'll walk through some of Matplotlib's runtime configuration (rc) options, and take a look at the newer *stylesheets* feature, which contains some nice sets of default configurations.
# 

# ## Plot Customization by Hand
# 
# Through this chapter, we've seen how it is possible to tweak individual plot settings to end up with something that looks a little bit nicer than the default.
# It's possible to do these customizations for each individual plot.
# For example, here is a fairly drab default histogram:
# 

import matplotlib.pyplot as plt
plt.style.use('classic')
import numpy as np

get_ipython().magic('matplotlib inline')


x = np.random.randn(1000)
plt.hist(x);


# We can adjust this by hand to make it a much more visually pleasing plot:
# 

# use a gray background
ax = plt.axes(axisbg='#E6E6E6')
ax.set_axisbelow(True)

# draw solid white grid lines
plt.grid(color='w', linestyle='solid')

# hide axis spines
for spine in ax.spines.values():
    spine.set_visible(False)
    
# hide top and right ticks
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

# lighten ticks and labels
ax.tick_params(colors='gray', direction='out')
for tick in ax.get_xticklabels():
    tick.set_color('gray')
for tick in ax.get_yticklabels():
    tick.set_color('gray')
    
# control face and edge color of histogram
ax.hist(x, edgecolor='#E6E6E6', color='#EE6666');


# This looks better, and you may recognize the look as inspired by the look of the R language's ggplot visualization package.
# But this took a whole lot of effort!
# We definitely do not want to have to do all that tweaking each time we create a plot.
# Fortunately, there is a way to adjust these defaults once in a way that will work for all plots.
# 

# ## Changing the Defaults: ``rcParams``
# 
# Each time Matplotlib loads, it defines a runtime configuration (rc) containing the default styles for every plot element you create.
# This configuration can be adjusted at any time using the ``plt.rc`` convenience routine.
# Let's see what it looks like to modify the rc parameters so that our default plot will look similar to what we did before.
# 
# We'll start by saving a copy of the current ``rcParams`` dictionary, so we can easily reset these changes in the current session:
# 

IPython_default = plt.rcParams.copy()


# Now we can use the ``plt.rc`` function to change some of these settings:
# 

from matplotlib import cycler
colors = cycler('color',
                ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)


# With these settings defined, we can now create a plot and see our settings in action:
# 

plt.hist(x);


# Let's see what simple line plots look like with these rc parameters:
# 

for i in range(4):
    plt.plot(np.random.rand(10))


# I find this much more aesthetically pleasing than the default styling.
# If you disagree with my aesthetic sense, the good news is that you can adjust the rc parameters to suit your own tastes!
# These settings can be saved in a *.matplotlibrc* file, which you can read about in the [Matplotlib documentation](http://Matplotlib.org/users/customizing.html).
# That said, I prefer to customize Matplotlib using its stylesheets instead.
# 

# ## Stylesheets
# 
# The version 1.4 release of Matplotlib in August 2014 added a very convenient ``style`` module, which includes a number of new default stylesheets, as well as the ability to create and package your own styles. These stylesheets are formatted similarly to the *.matplotlibrc* files mentioned earlier, but must be named with a *.mplstyle* extension.
# 
# Even if you don't create your own style, the stylesheets included by default are extremely useful.
# The available styles are listed in ``plt.style.available``—here I'll list only the first five for brevity:
# 

plt.style.available[:5]


# The basic way to switch to a stylesheet is to call
# 
# ``` python
# plt.style.use('stylename')
# ```
# 
# But keep in mind that this will change the style for the rest of the session!
# Alternatively, you can use the style context manager, which sets a style temporarily:
# 
# ``` python
# with plt.style.context('stylename'):
#     make_a_plot()
# ```
# 

# Let's create a function that will make two basic types of plot:
# 

def hist_and_lines():
    np.random.seed(0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(np.random.randn(1000))
    for i in range(3):
        ax[1].plot(np.random.rand(10))
    ax[1].legend(['a', 'b', 'c'], loc='lower left')


# We'll use this to explore how these plots look using the various built-in styles.
# 

# ### Default style
# 
# The default style is what we've been seeing so far throughout the book; we'll start with that.
# First, let's reset our runtime configuration to the notebook default:
# 

# reset rcParams
plt.rcParams.update(IPython_default);


# Now let's see how it looks:
# 

hist_and_lines()


# ### FiveThiryEight style
# 
# The ``fivethirtyeight`` style mimics the graphics found on the popular [FiveThirtyEight website](https://fivethirtyeight.com).
# As you can see here, it is typified by bold colors, thick lines, and transparent axes:
# 

with plt.style.context('fivethirtyeight'):
    hist_and_lines()


# ### ggplot
# 
# The ``ggplot`` package in the R language is a very popular visualization tool.
# Matplotlib's ``ggplot`` style mimics the default styles from that package:
# 

with plt.style.context('ggplot'):
    hist_and_lines()


# ### *Bayesian Methods for Hackers( style
# 
# There is a very nice short online book called [*Probabilistic Programming and Bayesian Methods for Hackers*](http://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/); it features figures created with Matplotlib, and uses a nice set of rc parameters to create a consistent and visually-appealing style throughout the book.
# This style is reproduced in the ``bmh`` stylesheet:
# 

with plt.style.context('bmh'):
    hist_and_lines()


# ### Dark background
# 
# For figures used within presentations, it is often useful to have a dark rather than light background.
# The ``dark_background`` style provides this:
# 

with plt.style.context('dark_background'):
    hist_and_lines()


# ### Grayscale
# 
# Sometimes you might find yourself preparing figures for a print publication that does not accept color figures.
# For this, the ``grayscale`` style, shown here, can be very useful:
# 

with plt.style.context('grayscale'):
    hist_and_lines()


# ### Seaborn style
# 
# Matplotlib also has stylesheets inspired by the Seaborn library (discussed more fully in [Visualization With Seaborn](04.14-Visualization-With-Seaborn.ipynb)).
# As we will see, these styles are loaded automatically when Seaborn is imported into a notebook.
# I've found these settings to be very nice, and tend to use them as defaults in my own data exploration.
# 

import seaborn
hist_and_lines()


# With all of these built-in options for various plot styles, Matplotlib becomes much more useful for both interactive visualization and creation of figures for publication.
# Throughout this book, I will generally use one or more of these style conventions when creating plots.
# 

# <!--NAVIGATION-->
# < [Customizing Ticks](04.10-Customizing-Ticks.ipynb) | [Contents](Index.ipynb) | [Three-Dimensional Plotting in Matplotlib](04.12-Three-Dimensional-Plotting.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [Structured Data: NumPy's Structured Arrays](02.09-Structured-Data-NumPy.ipynb) | [Contents](Index.ipynb) | [Introducing Pandas Objects](03.01-Introducing-Pandas-Objects.ipynb) >
# 

# # Data Manipulation with Pandas
# 

# In the previous chapter, we dove into detail on NumPy and its ``ndarray`` object, which provides efficient storage and manipulation of dense typed arrays in Python.
# Here we'll build on this knowledge by looking in detail at the data structures provided by the Pandas library.
# Pandas is a newer package built on top of NumPy, and provides an efficient implementation of a ``DataFrame``.
# ``DataFrame``s are essentially multidimensional arrays with attached row and column labels, and often with heterogeneous types and/or missing data.
# As well as offering a convenient storage interface for labeled data, Pandas implements a number of powerful data operations familiar to users of both database frameworks and spreadsheet programs.
# 
# As we saw, NumPy's ``ndarray`` data structure provides essential features for the type of clean, well-organized data typically seen in numerical computing tasks.
# While it serves this purpose very well, its limitations become clear when we need more flexibility (e.g., attaching labels to data, working with missing data, etc.) and when attempting operations that do not map well to element-wise broadcasting (e.g., groupings, pivots, etc.), each of which is an important piece of analyzing the less structured data available in many forms in the world around us.
# Pandas, and in particular its ``Series`` and ``DataFrame`` objects, builds on the NumPy array structure and provides efficient access to these sorts of "data munging" tasks that occupy much of a data scientist's time.
# 
# In this chapter, we will focus on the mechanics of using ``Series``, ``DataFrame``, and related structures effectively.
# We will use examples drawn from real datasets where appropriate, but these examples are not necessarily the focus.
# 

# ## Installing and Using Pandas
# 
# Installation of Pandas on your system requires NumPy to be installed, and if building the library from source, requires the appropriate tools to compile the C and Cython sources on which Pandas is built.
# Details on this installation can be found in the [Pandas documentation](http://pandas.pydata.org/).
# If you followed the advice outlined in the [Preface](00.00-Preface.ipynb) and used the Anaconda stack, you already have Pandas installed.
# 
# Once Pandas is installed, you can import it and check the version:
# 

import pandas
pandas.__version__


# Just as we generally import NumPy under the alias ``np``, we will import Pandas under the alias ``pd``:
# 

import pandas as pd


# This import convention will be used throughout the remainder of this book.
# 

# ## Reminder about Built-In Documentation
# 
# As you read through this chapter, don't forget that IPython gives you the ability to quickly explore the contents of a package (by using the tab-completion feature) as well as the documentation of various functions (using the ``?`` character). (Refer back to [Help and Documentation in IPython](01.01-Help-And-Documentation.ipynb) if you need a refresher on this.)
# 
# For example, to display all the contents of the pandas namespace, you can type
# 
# ```ipython
# In [3]: pd.<TAB>
# ```
# 
# And to display Pandas's built-in documentation, you can use this:
# 
# ```ipython
# In [4]: pd?
# ```
# 
# More detailed documentation, along with tutorials and other resources, can be found at http://pandas.pydata.org/.

# <!--NAVIGATION-->
# < [Structured Data: NumPy's Structured Arrays](02.09-Structured-Data-NumPy.ipynb) | [Contents](Index.ipynb) | [Introducing Pandas Objects](03.01-Introducing-Pandas-Objects.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [Further Resources](03.13-Further-Resources.ipynb) | [Contents](Index.ipynb) | [Simple Line Plots](04.01-Simple-Line-Plots.ipynb) >
# 

# # Visualization with Matplotlib
# 

# We'll now take an in-depth look at the Matplotlib package for visualization in Python.
# Matplotlib is a multi-platform data visualization library built on NumPy arrays, and designed to work with the broader SciPy stack.
# It was conceived by John Hunter in 2002, originally as a patch to IPython for enabling interactive MATLAB-style plotting via gnuplot from the IPython command line.
# IPython's creator, Fernando Perez, was at the time scrambling to finish his PhD, and let John know he wouldn’t have time to review the patch for several months.
# John took this as a cue to set out on his own, and the Matplotlib package was born, with version 0.1 released in 2003.
# It received an early boost when it was adopted as the plotting package of choice of the Space Telescope Science Institute (the folks behind the Hubble Telescope), which financially supported Matplotlib’s development and greatly expanded its capabilities.
# 
# One of Matplotlib’s most important features is its ability to play well with many operating systems and graphics backends.
# Matplotlib supports dozens of backends and output types, which means you can count on it to work regardless of which operating system you are using or which output format you wish.
# This cross-platform, everything-to-everyone approach has been one of the great strengths of Matplotlib.
# It has led to a large user base, which in turn has led to an active developer base and Matplotlib’s powerful tools and ubiquity within the scientific Python world.
# 
# In recent years, however, the interface and style of Matplotlib have begun to show their age.
# Newer tools like ggplot and ggvis in the R language, along with web visualization toolkits based on D3js and HTML5 canvas, often make Matplotlib feel clunky and old-fashioned.
# Still, I'm of the opinion that we cannot ignore Matplotlib's strength as a well-tested, cross-platform graphics engine.
# Recent Matplotlib versions make it relatively easy to set new global plotting styles (see [Customizing Matplotlib: Configurations and Style Sheets](04.11-Settings-and-Stylesheets.ipynb)), and people have been developing new packages that build on its powerful internals to drive Matplotlib via cleaner, more modern APIs—for example, Seaborn (discussed in [Visualization With Seaborn](04.14-Visualization-With-Seaborn.ipynb)), [ggpy](http://yhat.github.io/ggpy/), [HoloViews](http://holoviews.org/), [Altair](http://altair-viz.github.io/), and even Pandas itself can be used as wrappers around Matplotlib's API.
# Even with wrappers like these, it is still often useful to dive into Matplotlib's syntax to adjust the final plot output.
# For this reason, I believe that Matplotlib itself will remain a vital piece of the data visualization stack, even if new tools mean the community gradually moves away from using the Matplotlib API directly.
# 

# ## General Matplotlib Tips
# 
# Before we dive into the details of creating visualizations with Matplotlib, there are a few useful things you should know about using the package.
# 

# ### Importing Matplotlib
# 
# Just as we use the ``np`` shorthand for NumPy and the ``pd`` shorthand for Pandas, we will use some standard shorthands for Matplotlib imports:
# 

import matplotlib as mpl
import matplotlib.pyplot as plt


# The ``plt`` interface is what we will use most often, as we shall see throughout this chapter.
# 

# ### Setting Styles
# 
# We will use the ``plt.style`` directive to choose appropriate aesthetic styles for our figures.
# Here we will set the ``classic`` style, which ensures that the plots we create use the classic Matplotlib style:
# 

plt.style.use('classic')


# Throughout this section, we will adjust this style as needed.
# Note that the stylesheets used here are supported as of Matplotlib version 1.5; if you are using an earlier version of Matplotlib, only the default style is available.
# For more information on stylesheets, see [Customizing Matplotlib: Configurations and Style Sheets](04.11-Settings-and-Stylesheets.ipynb).
# 

# ### ``show()`` or No ``show()``? How to Display Your Plots
# 

# A visualization you can't see won't be of much use, but just how you view your Matplotlib plots depends on the context.
# The best use of Matplotlib differs depending on how you are using it; roughly, the three applicable contexts are using Matplotlib in a script, in an IPython terminal, or in an IPython notebook.
# 

# #### Plotting from a script
# 
# If you are using Matplotlib from within a script, the function ``plt.show()`` is your friend.
# ``plt.show()`` starts an event loop, looks for all currently active figure objects, and opens one or more interactive windows that display your figure or figures.
# 
# So, for example, you may have a file called *myplot.py* containing the following:
# 
# ```python
# # ------- file: myplot.py ------
# import matplotlib.pyplot as plt
# import numpy as np
# 
# x = np.linspace(0, 10, 100)
# 
# plt.plot(x, np.sin(x))
# plt.plot(x, np.cos(x))
# 
# plt.show()
# ```
# 
# You can then run this script from the command-line prompt, which will result in a window opening with your figure displayed:
# 
# ```
# $ python myplot.py
# ```
# 
# The ``plt.show()`` command does a lot under the hood, as it must interact with your system's interactive graphical backend.
# The details of this operation can vary greatly from system to system and even installation to installation, but matplotlib does its best to hide all these details from you.
# 
# One thing to be aware of: the ``plt.show()`` command should be used *only once* per Python session, and is most often seen at the very end of the script.
# Multiple ``show()`` commands can lead to unpredictable backend-dependent behavior, and should mostly be avoided.
# 

# #### Plotting from an IPython shell
# 
# It can be very convenient to use Matplotlib interactively within an IPython shell (see [IPython: Beyond Normal Python](01.00-IPython-Beyond-Normal-Python.ipynb)).
# IPython is built to work well with Matplotlib if you specify Matplotlib mode.
# To enable this mode, you can use the ``%matplotlib`` magic command after starting ``ipython``:
# 
# ```ipython
# In [1]: %matplotlib
# Using matplotlib backend: TkAgg
# 
# In [2]: import matplotlib.pyplot as plt
# ```
# 
# At this point, any ``plt`` plot command will cause a figure window to open, and further commands can be run to update the plot.
# Some changes (such as modifying properties of lines that are already drawn) will not draw automatically: to force an update, use ``plt.draw()``.
# Using ``plt.show()`` in Matplotlib mode is not required.
# 

# #### Plotting from an IPython notebook
# 
# The IPython notebook is a browser-based interactive data analysis tool that can combine narrative, code, graphics, HTML elements, and much more into a single executable document (see [IPython: Beyond Normal Python](01.00-IPython-Beyond-Normal-Python.ipynb)).
# 
# Plotting interactively within an IPython notebook can be done with the ``%matplotlib`` command, and works in a similar way to the IPython shell.
# In the IPython notebook, you also have the option of embedding graphics directly in the notebook, with two possible options:
# 
# - ``%matplotlib notebook`` will lead to *interactive* plots embedded within the notebook
# - ``%matplotlib inline`` will lead to *static* images of your plot embedded in the notebook
# 
# For this book, we will generally opt for ``%matplotlib inline``:
# 

get_ipython().magic('matplotlib inline')


# After running this command (it needs to be done only once per kernel/session), any cell within the notebook that creates a plot will embed a PNG image of the resulting graphic:
# 

import numpy as np
x = np.linspace(0, 10, 100)

fig = plt.figure()
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--');


# ### Saving Figures to File
# 
# One nice feature of Matplotlib is the ability to save figures in a wide variety of formats.
# Saving a figure can be done using the ``savefig()`` command.
# For example, to save the previous figure as a PNG file, you can run this:
# 

fig.savefig('my_figure.png')


# We now have a file called ``my_figure.png`` in the current working directory:
# 

get_ipython().system('ls -lh my_figure.png')


# To confirm that it contains what we think it contains, let's use the IPython ``Image`` object to display the contents of this file:
# 

from IPython.display import Image
Image('my_figure.png')


# In ``savefig()``, the file format is inferred from the extension of the given filename.
# Depending on what backends you have installed, many different file formats are available.
# The list of supported file types can be found for your system by using the following method of the figure canvas object:
# 

fig.canvas.get_supported_filetypes()


# Note that when saving your figure, it's not necessary to use ``plt.show()`` or related commands discussed earlier.
# 

# ## Two Interfaces for the Price of One
# 
# A potentially confusing feature of Matplotlib is its dual interfaces: a convenient MATLAB-style state-based interface, and a more powerful object-oriented interface. We'll quickly highlight the differences between the two here.
# 

# #### MATLAB-style Interface
# 
# Matplotlib was originally written as a Python alternative for MATLAB users, and much of its syntax reflects that fact.
# The MATLAB-style tools are contained in the pyplot (``plt``) interface.
# For example, the following code will probably look quite familiar to MATLAB users:
# 

plt.figure()  # create a plot figure

# create the first of two panels and set current axis
plt.subplot(2, 1, 1) # (rows, columns, panel number)
plt.plot(x, np.sin(x))

# create the second panel and set current axis
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x));


# It is important to note that this interface is *stateful*: it keeps track of the "current" figure and axes, which are where all ``plt`` commands are applied.
# You can get a reference to these using the ``plt.gcf()`` (get current figure) and ``plt.gca()`` (get current axes) routines.
# 
# While this stateful interface is fast and convenient for simple plots, it is easy to run into problems.
# For example, once the second panel is created, how can we go back and add something to the first?
# This is possible within the MATLAB-style interface, but a bit clunky.
# Fortunately, there is a better way.

# #### Object-oriented interface
# 
# The object-oriented interface is available for these more complicated situations, and for when you want more control over your figure.
# Rather than depending on some notion of an "active" figure or axes, in the object-oriented interface the plotting functions are *methods* of explicit ``Figure`` and ``Axes`` objects.
# To re-create the previous plot using this style of plotting, you might do the following:
# 

# First create a grid of plots
# ax will be an array of two Axes objects
fig, ax = plt.subplots(2)

# Call plot() method on the appropriate object
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x));


# For more simple plots, the choice of which style to use is largely a matter of preference, but the object-oriented approach can become a necessity as plots become more complicated.
# Throughout this chapter, we will switch between the MATLAB-style and object-oriented interfaces, depending on what is most convenient.
# In most cases, the difference is as small as switching ``plt.plot()`` to ``ax.plot()``, but there are a few gotchas that we will highlight as they come up in the following sections.
# 

# <!--NAVIGATION-->
# < [Further Resources](03.13-Further-Resources.ipynb) | [Contents](Index.ipynb) | [Simple Line Plots](04.01-Simple-Line-Plots.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [In Depth: Gaussian Mixture Models](05.12-Gaussian-Mixtures.ipynb) | [Contents](Index.ipynb) | [Application: A Face Detection Pipeline](05.14-Image-Features.ipynb) >
# 

# # In-Depth: Kernel Density Estimation
# 
# In the previous section we covered Gaussian mixture models (GMM), which are a kind of hybrid between a clustering estimator and a density estimator.
# Recall that a density estimator is an algorithm which takes a $D$-dimensional dataset and produces an estimate of the $D$-dimensional probability distribution which that data is drawn from.
# The GMM algorithm accomplishes this by representing the density as a weighted sum of Gaussian distributions.
# *Kernel density estimation* (KDE) is in some senses an algorithm which takes the mixture-of-Gaussians idea to its logical extreme: it uses a mixture consisting of one Gaussian component *per point*, resulting in an essentially non-parametric estimator of density.
# In this section, we will explore the motivation and uses of KDE.
# 
# We begin with the standard imports:
# 

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np


# ## Motivating KDE: Histograms
# 
# As already discussed, a density estimator is an algorithm which seeks to model the probability distribution that generated a dataset.
# For one dimensional data, you are probably already familiar with one simple density estimator: the histogram.
# A histogram divides the data into discrete bins, counts the number of points that fall in each bin, and then visualizes the results in an intuitive manner.
# 
# For example, let's create some data that is drawn from two normal distributions:
# 

def make_data(N, f=0.3, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f * N):] += 5
    return x

x = make_data(1000)


# We have previously seen that the standard count-based histogram can be created with the ``plt.hist()`` function.
# By specifying the ``normed`` parameter of the histogram, we end up with a normalized histogram where the height of the bins does not reflect counts, but instead reflects probability density:
# 

hist = plt.hist(x, bins=30, normed=True)


# Notice that for equal binning, this normalization simply changes the scale on the y-axis, leaving the relative heights essentially the same as in a histogram built from counts.
# This normalization is chosen so that the total area under the histogram is equal to 1, as we can confirm by looking at the output of the histogram function:
# 

density, bins, patches = hist
widths = bins[1:] - bins[:-1]
(density * widths).sum()


# One of the issues with using a histogram as a density estimator is that the choice of bin size and location can lead to representations that have qualitatively different features.
# For example, if we look at a version of this data with only 20 points, the choice of how to draw the bins can lead to an entirely different interpretation of the data!
# Consider this example:
# 

x = make_data(20)
bins = np.linspace(-5, 10, 10)


fig, ax = plt.subplots(1, 2, figsize=(12, 4),
                       sharex=True, sharey=True,
                       subplot_kw={'xlim':(-4, 9),
                                   'ylim':(-0.02, 0.3)})
fig.subplots_adjust(wspace=0.05)
for i, offset in enumerate([0.0, 0.6]):
    ax[i].hist(x, bins=bins + offset, normed=True)
    ax[i].plot(x, np.full_like(x, -0.01), '|k',
               markeredgewidth=1)


# On the left, the histogram makes clear that this is a bimodal distribution.
# On the right, we see a unimodal distribution with a long tail.
# Without seeing the preceding code, you would probably not guess that these two histograms were built from the same data: with that in mind, how can you trust the intuition that histograms confer?
# And how might we improve on this?
# 
# Stepping back, we can think of a histogram as a stack of blocks, where we stack one block within each bin on top of each point in the dataset.
# Let's view this directly:

fig, ax = plt.subplots()
bins = np.arange(-3, 8)
ax.plot(x, np.full_like(x, -0.1), '|k',
        markeredgewidth=1)
for count, edge in zip(*np.histogram(x, bins)):
    for i in range(count):
        ax.add_patch(plt.Rectangle((edge, i), 1, 1,
                                   alpha=0.5))
ax.set_xlim(-4, 8)
ax.set_ylim(-0.2, 8)


# The problem with our two binnings stems from the fact that the height of the block stack often reflects not on the actual density of points nearby, but on coincidences of how the bins align with the data points.
# This mis-alignment between points and their blocks is a potential cause of the poor histogram results seen here.
# But what if, instead of stacking the blocks aligned with the *bins*, we were to stack the blocks aligned with the *points they represent*?
# If we do this, the blocks won't be aligned, but we can add their contributions at each location along the x-axis to find the result.
# Let's try this:

x_d = np.linspace(-4, 8, 2000)
density = sum((abs(xi - x_d) < 0.5) for xi in x)

plt.fill_between(x_d, density, alpha=0.5)
plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)

plt.axis([-4, 8, -0.2, 8]);


# The result looks a bit messy, but is a much more robust reflection of the actual data characteristics than is the standard histogram.
# Still, the rough edges are not aesthetically pleasing, nor are they reflective of any true properties of the data.
# In order to smooth them out, we might decide to replace the blocks at each location with a smooth function, like a Gaussian.
# Let's use a standard normal curve at each point instead of a block:
# 

from scipy.stats import norm
x_d = np.linspace(-4, 8, 1000)
density = sum(norm(xi).pdf(x_d) for xi in x)

plt.fill_between(x_d, density, alpha=0.5)
plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)

plt.axis([-4, 8, -0.2, 5]);


# This smoothed-out plot, with a Gaussian distribution contributed at the location of each input point, gives a much more accurate idea of the shape of the data distribution, and one which has much less variance (i.e., changes much less in response to differences in sampling).
# 
# These last two plots are examples of kernel density estimation in one dimension: the first uses a so-called "tophat" kernel and the second uses a Gaussian kernel.
# We'll now look at kernel density estimation in more detail.
# 

# ## Kernel Density Estimation in Practice
# 
# The free parameters of kernel density estimation are the *kernel*, which specifies the shape of the distribution placed at each point, and the *kernel bandwidth*, which controls the size of the kernel at each point.
# In practice, there are many kernels you might use for a kernel density estimation: in particular, the Scikit-Learn KDE implementation supports one of six kernels, which you can read about in Scikit-Learn's [Density Estimation documentation](http://scikit-learn.org/stable/modules/density.html).
# 
# While there are several versions of kernel density estimation implemented in Python (notably in the SciPy and StatsModels packages), I prefer to use Scikit-Learn's version because of its efficiency and flexibility.
# It is implemented in the ``sklearn.neighbors.KernelDensity`` estimator, which handles KDE in multiple dimensions with one of six kernels and one of a couple dozen distance metrics.
# Because KDE can be fairly computationally intensive, the Scikit-Learn estimator uses a tree-based algorithm under the hood and can trade off computation time for accuracy using the ``atol`` (absolute tolerance) and ``rtol`` (relative tolerance) parameters.
# The kernel bandwidth, which is a free parameter, can be determined using Scikit-Learn's standard cross validation tools as we will soon see.
# 
# Let's first show a simple example of replicating the above plot using the Scikit-Learn ``KernelDensity`` estimator:
# 

from sklearn.neighbors import KernelDensity

# instantiate and fit the KDE model
kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
kde.fit(x[:, None])

# score_samples returns the log of the probability density
logprob = kde.score_samples(x_d[:, None])

plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
plt.ylim(-0.02, 0.22)


# The result here is normalized such that the area under the curve is equal to 1.
# 

# ### Selecting the bandwidth via cross-validation
# 
# The choice of bandwidth within KDE is extremely important to finding a suitable density estimate, and is the knob that controls the bias–variance trade-off in the estimate of density: too narrow a bandwidth leads to a high-variance estimate (i.e., over-fitting), where the presence or absence of a single point makes a large difference. Too wide a bandwidth leads to a high-bias estimate (i.e., under-fitting) where the structure in the data is washed out by the wide kernel.
# 
# There is a long history in statistics of methods to quickly estimate the best bandwidth based on rather stringent assumptions about the data: if you look up the KDE implementations in the SciPy and StatsModels packages, for example, you will see implementations based on some of these rules.
# 
# In machine learning contexts, we've seen that such hyperparameter tuning often is done empirically via a cross-validation approach.
# With this in mind, the ``KernelDensity`` estimator in Scikit-Learn is designed such that it can be used directly within the Scikit-Learn's standard grid search tools.
# Here we will use ``GridSearchCV`` to optimize the bandwidth for the preceding dataset.
# Because we are looking at such a small dataset, we will use leave-one-out cross-validation, which minimizes the reduction in training set size for each cross-validation trial:
# 

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import LeaveOneOut

bandwidths = 10 ** np.linspace(-1, 1, 100)
grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths},
                    cv=LeaveOneOut(len(x)))
grid.fit(x[:, None]);


# Now we can find the choice of bandwidth which maximizes the score (which in this case defaults to the log-likelihood):
# 

grid.best_params_


# The optimal bandwidth happens to be very close to what we used in the example plot earlier, where the bandwidth was 1.0 (i.e., the default width of ``scipy.stats.norm``).
# 

# ## Example: KDE on a Sphere
# 
# Perhaps the most common use of KDE is in graphically representing distributions of points.
# For example, in the Seaborn visualization library (see [Visualization With Seaborn](04.14-Visualization-With-Seaborn.ipynb)), KDE is built in and automatically used to help visualize points in one and two dimensions.
# 
# Here we will look at a slightly more sophisticated use of KDE for visualization of distributions.
# We will make use of some geographic data that can be loaded with Scikit-Learn: the geographic distributions of recorded observations of two South American mammals, *Bradypus variegatus* (the Brown-throated Sloth) and *Microryzomys minutus* (the Forest Small Rice Rat).
# 
# With Scikit-Learn, we can fetch this data as follows:
# 

from sklearn.datasets import fetch_species_distributions

data = fetch_species_distributions()

# Get matrices/arrays of species IDs and locations
latlon = np.vstack([data.train['dd lat'],
                    data.train['dd long']]).T
species = np.array([d.decode('ascii').startswith('micro')
                    for d in data.train['species']], dtype='int')


# With this data loaded, we can use the Basemap toolkit (mentioned previously in [Geographic Data with Basemap](04.13-Geographic-Data-With-Basemap.ipynb)) to plot the observed locations of these two species on the map of South America.
# 

from mpl_toolkits.basemap import Basemap
from sklearn.datasets.species_distributions import construct_grids

xgrid, ygrid = construct_grids(data)

# plot coastlines with basemap
m = Basemap(projection='cyl', resolution='c',
            llcrnrlat=ygrid.min(), urcrnrlat=ygrid.max(),
            llcrnrlon=xgrid.min(), urcrnrlon=xgrid.max())
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='#FFEEDD')
m.drawcoastlines(color='gray', zorder=2)
m.drawcountries(color='gray', zorder=2)

# plot locations
m.scatter(latlon[:, 1], latlon[:, 0], zorder=3,
          c=species, cmap='rainbow', latlon=True);


# Unfortunately, this doesn't give a very good idea of the density of the species, because points in the species range may overlap one another.
# You may not realize it by looking at this plot, but there are over 1,600 points shown here!
# 
# Let's use kernel density estimation to show this distribution in a more interpretable way: as a smooth indication of density on the map.
# Because the coordinate system here lies on a spherical surface rather than a flat plane, we will use the ``haversine`` distance metric, which will correctly represent distances on a curved surface.
# 
# There is a bit of boilerplate code here (one of the disadvantages of the Basemap toolkit) but the meaning of each code block should be clear:
# 

# Set up the data grid for the contour plot
X, Y = np.meshgrid(xgrid[::5], ygrid[::5][::-1])
land_reference = data.coverages[6][::5, ::5]
land_mask = (land_reference > -9999).ravel()
xy = np.vstack([Y.ravel(), X.ravel()]).T
xy = np.radians(xy[land_mask])

# Create two side-by-side plots
fig, ax = plt.subplots(1, 2)
fig.subplots_adjust(left=0.05, right=0.95, wspace=0.05)
species_names = ['Bradypus Variegatus', 'Microryzomys Minutus']
cmaps = ['Purples', 'Reds']

for i, axi in enumerate(ax):
    axi.set_title(species_names[i])
    
    # plot coastlines with basemap
    m = Basemap(projection='cyl', llcrnrlat=Y.min(),
                urcrnrlat=Y.max(), llcrnrlon=X.min(),
                urcrnrlon=X.max(), resolution='c', ax=axi)
    m.drawmapboundary(fill_color='#DDEEFF')
    m.drawcoastlines()
    m.drawcountries()
    
    # construct a spherical kernel density estimate of the distribution
    kde = KernelDensity(bandwidth=0.03, metric='haversine')
    kde.fit(np.radians(latlon[species == i]))

    # evaluate only on the land: -9999 indicates ocean
    Z = np.full(land_mask.shape[0], -9999.0)
    Z[land_mask] = np.exp(kde.score_samples(xy))
    Z = Z.reshape(X.shape)

    # plot contours of the density
    levels = np.linspace(0, Z.max(), 25)
    axi.contourf(X, Y, Z, levels=levels, cmap=cmaps[i])


# Compared to the simple scatter plot we initially used, this visualization paints a much clearer picture of the geographical distribution of observations of these two species.
# 

# ## Example: Not-So-Naive Bayes
# 
# This example looks at Bayesian generative classification with KDE, and demonstrates how to use the Scikit-Learn architecture to create a custom estimator.
# 
# In [In Depth: Naive Bayes Classification](05.05-Naive-Bayes.ipynb), we took a look at naive Bayesian classification, in which we created a simple generative model for each class, and used these models to build a fast classifier.
# For Gaussian naive Bayes, the generative model is a simple axis-aligned Gaussian.
# With a density estimation algorithm like KDE, we can remove the "naive" element and perform the same classification with a more sophisticated generative model for each class.
# It's still Bayesian classification, but it's no longer naive.
# 
# The general approach for generative classification is this:
# 
# 1. Split the training data by label.
# 
# 2. For each set, fit a KDE to obtain a generative model of the data.
#    This allows you for any observation $x$ and label $y$ to compute a likelihood $P(x~|~y)$.
#    
# 3. From the number of examples of each class in the training set, compute the *class prior*, $P(y)$.
# 
# 4. For an unknown point $x$, the posterior probability for each class is $P(y~|~x) \propto P(x~|~y)P(y)$.
#    The class which maximizes this posterior is the label assigned to the point.
# 
# The algorithm is straightforward and intuitive to understand; the more difficult piece is couching it within the Scikit-Learn framework in order to make use of the grid search and cross-validation architecture.
# 
# This is the code that implements the algorithm within the Scikit-Learn framework; we will step through it following the code block:
# 

from sklearn.base import BaseEstimator, ClassifierMixin


class KDEClassifier(BaseEstimator, ClassifierMixin):
    """Bayesian generative classification based on KDE
    
    Parameters
    ----------
    bandwidth : float
        the kernel bandwidth within each class
    kernel : str
        the kernel name, passed to KernelDensity
    """
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        
    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth,
                                      kernel=self.kernel).fit(Xi)
                        for Xi in training_sets]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
                           for Xi in training_sets]
        return self
        
    def predict_proba(self, X):
        logprobs = np.array([model.score_samples(X)
                             for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)
        
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


# ### The anatomy of a custom estimator
# 

# Let's step through this code and discuss the essential features:
# 
# ```python
# from sklearn.base import BaseEstimator, ClassifierMixin
# 
# class KDEClassifier(BaseEstimator, ClassifierMixin):
#     """Bayesian generative classification based on KDE
#     
#     Parameters
#     ----------
#     bandwidth : float
#         the kernel bandwidth within each class
#     kernel : str
#         the kernel name, passed to KernelDensity
#     """
# ```
# 
# Each estimator in Scikit-Learn is a class, and it is most convenient for this class to inherit from the ``BaseEstimator`` class as well as the appropriate mixin, which provides standard functionality.
# For example, among other things, here the ``BaseEstimator`` contains the logic necessary to clone/copy an estimator for use in a cross-validation procedure, and ``ClassifierMixin`` defines a default ``score()`` method used by such routines.
# We also provide a doc string, which will be captured by IPython's help functionality (see [Help and Documentation in IPython](01.01-Help-And-Documentation.ipynb)).
# 

# Next comes the class initialization method:
# 
# ```python
#     def __init__(self, bandwidth=1.0, kernel='gaussian'):
#         self.bandwidth = bandwidth
#         self.kernel = kernel
# ```
# 
# This is the actual code that is executed when the object is instantiated with ``KDEClassifier()``.
# In Scikit-Learn, it is important that *initialization contains no operations* other than assigning the passed values by name to ``self``.
# This is due to the logic contained in ``BaseEstimator`` required for cloning and modifying estimators for cross-validation, grid search, and other functions.
# Similarly, all arguments to ``__init__`` should be explicit: i.e. ``*args`` or ``**kwargs`` should be avoided, as they will not be correctly handled within cross-validation routines.
# 

# Next comes the ``fit()`` method, where we handle training data:
# 
# ```python 
#     def fit(self, X, y):
#         self.classes_ = np.sort(np.unique(y))
#         training_sets = [X[y == yi] for yi in self.classes_]
#         self.models_ = [KernelDensity(bandwidth=self.bandwidth,
#                                       kernel=self.kernel).fit(Xi)
#                         for Xi in training_sets]
#         self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
#                            for Xi in training_sets]
#         return self
# ```
# 
# Here we find the unique classes in the training data, train a ``KernelDensity`` model for each class, and compute the class priors based on the number of input samples.
# Finally, ``fit()`` should always return ``self`` so that we can chain commands. For example:
# ```python
# label = model.fit(X, y).predict(X)
# ```
# Notice that each persistent result of the fit is stored with a trailing underscore (e.g., ``self.logpriors_``).
# This is a convention used in Scikit-Learn so that you can quickly scan the members of an estimator (using IPython's tab completion) and see exactly which members are fit to training data.
# 

# Finally, we have the logic for predicting labels on new data:
# ```python
#     def predict_proba(self, X):
#         logprobs = np.vstack([model.score_samples(X)
#                               for model in self.models_]).T
#         result = np.exp(logprobs + self.logpriors_)
#         return result / result.sum(1, keepdims=True)
#         
#     def predict(self, X):
#         return self.classes_[np.argmax(self.predict_proba(X), 1)]
# ```
# Because this is a probabilistic classifier, we first implement ``predict_proba()`` which returns an array of class probabilities of shape ``[n_samples, n_classes]``.
# Entry ``[i, j]`` of this array is the posterior probability that sample ``i`` is a member of class ``j``, computed by multiplying the likelihood by the class prior and normalizing.
# 
# Finally, the ``predict()`` method uses these probabilities and simply returns the class with the largest probability.
# 

# ### Using our custom estimator
# 
# Let's try this custom estimator on a problem we have seen before: the classification of hand-written digits.
# Here we will load the digits, and compute the cross-validation score for a range of candidate bandwidths using the ``GridSearchCV`` meta-estimator (refer back to [Hyperparameters and Model Validation](05.03-Hyperparameters-and-Model-Validation.ipynb)):
# 

from sklearn.datasets import load_digits
from sklearn.grid_search import GridSearchCV

digits = load_digits()

bandwidths = 10 ** np.linspace(0, 2, 100)
grid = GridSearchCV(KDEClassifier(), {'bandwidth': bandwidths})
grid.fit(digits.data, digits.target)

scores = [val.mean_validation_score for val in grid.grid_scores_]


# Next we can plot the cross-validation score as a function of bandwidth:
# 

plt.semilogx(bandwidths, scores)
plt.xlabel('bandwidth')
plt.ylabel('accuracy')
plt.title('KDE Model Performance')
print(grid.best_params_)
print('accuracy =', grid.best_score_)


# We see that this not-so-naive Bayesian classifier reaches a cross-validation accuracy of just over 96%; this is compared to around 80% for the naive Bayesian classification:
# 

from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import cross_val_score
cross_val_score(GaussianNB(), digits.data, digits.target).mean()


# One benefit of such a generative classifier is interpretability of results: for each unknown sample, we not only get a probabilistic classification, but a *full model* of the distribution of points we are comparing it to!
# If desired, this offers an intuitive window into the reasons for a particular classification that algorithms like SVMs and random forests tend to obscure.
# 
# If you would like to take this further, there are some improvements that could be made to our KDE classifier model:
# 
# - we could allow the bandwidth in each class to vary independently
# - we could optimize these bandwidths not based on their prediction score, but on the likelihood of the training data under the generative model within each class (i.e. use the scores from ``KernelDensity`` itself rather than the global prediction accuracy)
# 
# Finally, if you want some practice building your own estimator, you might tackle building a similar Bayesian classifier using Gaussian Mixture Models instead of KDE.
# 

# <!--NAVIGATION-->
# < [In Depth: Gaussian Mixture Models](05.12-Gaussian-Mixtures.ipynb) | [Contents](Index.ipynb) | [Application: A Face Detection Pipeline](05.14-Image-Features.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [Customizing Matplotlib: Configurations and Stylesheets](04.11-Settings-and-Stylesheets.ipynb) | [Contents](Index.ipynb) | [Geographic Data with Basemap](04.13-Geographic-Data-With-Basemap.ipynb) >
# 

# # Three-Dimensional Plotting in Matplotlib
# 

# Matplotlib was initially designed with only two-dimensional plotting in mind.
# Around the time of the 1.0 release, some three-dimensional plotting utilities were built on top of Matplotlib's two-dimensional display, and the result is a convenient (if somewhat limited) set of tools for three-dimensional data visualization.
# three-dimensional plots are enabled by importing the ``mplot3d`` toolkit, included with the main Matplotlib installation:
# 

from mpl_toolkits import mplot3d


# Once this submodule is imported, a three-dimensional axes can be created by passing the keyword ``projection='3d'`` to any of the normal axes creation routines:
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()
ax = plt.axes(projection='3d')


# With this three-dimensional axes enabled, we can now plot a variety of three-dimensional plot types. 
# Three-dimensional plotting is one of the functionalities that benefits immensely from viewing figures interactively rather than statically in the notebook; recall that to use interactive figures, you can use ``%matplotlib notebook`` rather than ``%matplotlib inline`` when running this code.
# 

# ## Three-dimensional Points and Lines
# 
# The most basic three-dimensional plot is a line or collection of scatter plot created from sets of (x, y, z) triples.
# In analogy with the more common two-dimensional plots discussed earlier, these can be created using the ``ax.plot3D`` and ``ax.scatter3D`` functions.
# The call signature for these is nearly identical to that of their two-dimensional counterparts, so you can refer to [Simple Line Plots](04.01-Simple-Line-Plots.ipynb) and [Simple Scatter Plots](04.02-Simple-Scatter-Plots.ipynb) for more information on controlling the output.
# Here we'll plot a trigonometric spiral, along with some points drawn randomly near the line:
# 

ax = plt.axes(projection='3d')

# Data for a three-dimensional line
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');


# Notice that by default, the scatter points have their transparency adjusted to give a sense of depth on the page.
# While the three-dimensional effect is sometimes difficult to see within a static image, an interactive view can lead to some nice intuition about the layout of the points.
# 

# ## Three-dimensional Contour Plots
# 
# Analogous to the contour plots we explored in [Density and Contour Plots](04.04-Density-and-Contour-Plots.ipynb), ``mplot3d`` contains tools to create three-dimensional relief plots using the same inputs.
# Like two-dimensional ``ax.contour`` plots, ``ax.contour3D`` requires all the input data to be in the form of two-dimensional regular grids, with the Z data evaluated at each point.
# Here we'll show a three-dimensional contour diagram of a three-dimensional sinusoidal function:
# 

def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');


# Sometimes the default viewing angle is not optimal, in which case we can use the ``view_init`` method to set the elevation and azimuthal angles. In the following example, we'll use an elevation of 60 degrees (that is, 60 degrees above the x-y plane) and an azimuth of 35 degrees (that is, rotated 35 degrees counter-clockwise about the z-axis):
# 

ax.view_init(60, 35)
fig


# Again, note that this type of rotation can be accomplished interactively by clicking and dragging when using one of Matplotlib's interactive backends.
# 

# ## Wireframes and Surface Plots
# 
# Two other types of three-dimensional plots that work on gridded data are wireframes and surface plots.
# These take a grid of values and project it onto the specified three-dimensional surface, and can make the resulting three-dimensional forms quite easy to visualize.
# Here's an example of using a wireframe:
# 

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z, color='black')
ax.set_title('wireframe');


# A surface plot is like a wireframe plot, but each face of the wireframe is a filled polygon.
# Adding a colormap to the filled polygons can aid perception of the topology of the surface being visualized:
# 

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('surface');


# Note that though the grid of values for a surface plot needs to be two-dimensional, it need not be rectilinear.
# Here is an example of creating a partial polar grid, which when used with the ``surface3D`` plot can give us a slice into the function we're visualizing:
# 

r = np.linspace(0, 6, 20)
theta = np.linspace(-0.9 * np.pi, 0.8 * np.pi, 40)
r, theta = np.meshgrid(r, theta)

X = r * np.sin(theta)
Y = r * np.cos(theta)
Z = f(X, Y)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none');


# ## Surface Triangulations
# 
# For some applications, the evenly sampled grids required by the above routines is overly restrictive and inconvenient.
# In these situations, the triangulation-based plots can be very useful.
# What if rather than an even draw from a Cartesian or a polar grid, we instead have a set of random draws?

theta = 2 * np.pi * np.random.random(1000)
r = 6 * np.random.random(1000)
x = np.ravel(r * np.sin(theta))
y = np.ravel(r * np.cos(theta))
z = f(x, y)


# We could create a scatter plot of the points to get an idea of the surface we're sampling from:
# 

ax = plt.axes(projection='3d')
ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5);


# This leaves a lot to be desired.
# The function that will help us in this case is ``ax.plot_trisurf``, which creates a surface by first finding a set of triangles formed between adjacent points (remember that x, y, and z here are one-dimensional arrays):
# 

ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, z,
                cmap='viridis', edgecolor='none');


# The result is certainly not as clean as when it is plotted with a grid, but the flexibility of such a triangulation allows for some really interesting three-dimensional plots.
# For example, it is actually possible to plot a three-dimensional Möbius strip using this, as we'll see next.
# 

# ### Example: Visualizing a Möbius strip
# 
# A Möbius strip is similar to a strip of paper glued into a loop with a half-twist.
# Topologically, it's quite interesting because despite appearances it has only a single side!
# Here we will visualize such an object using Matplotlib's three-dimensional tools.
# The key to creating the Möbius strip is to think about it's parametrization: it's a two-dimensional strip, so we need two intrinsic dimensions. Let's call them $\theta$, which ranges from $0$ to $2\pi$ around the loop, and $w$ which ranges from -1 to 1 across the width of the strip:
# 

theta = np.linspace(0, 2 * np.pi, 30)
w = np.linspace(-0.25, 0.25, 8)
w, theta = np.meshgrid(w, theta)


# Now from this parametrization, we must determine the *(x, y, z)* positions of the embedded strip.
# 
# Thinking about it, we might realize that there are two rotations happening: one is the position of the loop about its center (what we've called $\theta$), while the other is the twisting of the strip about its axis (we'll call this $\phi$). For a Möbius strip, we must have the strip makes half a twist during a full loop, or $\Delta\phi = \Delta\theta/2$.
# 

phi = 0.5 * theta


# Now we use our recollection of trigonometry to derive the three-dimensional embedding.
# We'll define $r$, the distance of each point from the center, and use this to find the embedded $(x, y, z)$ coordinates:
# 

# radius in x-y plane
r = 1 + w * np.cos(phi)

x = np.ravel(r * np.cos(theta))
y = np.ravel(r * np.sin(theta))
z = np.ravel(w * np.sin(phi))


# Finally, to plot the object, we must make sure the triangulation is correct. The best way to do this is to define the triangulation *within the underlying parametrization*, and then let Matplotlib project this triangulation into the three-dimensional space of the Möbius strip.
# This can be accomplished as follows:
# 

# triangulate in the underlying parametrization
from matplotlib.tri import Triangulation
tri = Triangulation(np.ravel(w), np.ravel(theta))

ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, z, triangles=tri.triangles,
                cmap='viridis', linewidths=0.2);

ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1);


# Combining all of these techniques, it is possible to create and display a wide variety of three-dimensional objects and patterns in Matplotlib.
# 

# <!--NAVIGATION-->
# < [Customizing Matplotlib: Configurations and Stylesheets](04.11-Settings-and-Stylesheets.ipynb) | [Contents](Index.ipynb) | [Geographic Data with Basemap](04.13-Geographic-Data-With-Basemap.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [The Basics of NumPy Arrays](02.02-The-Basics-Of-NumPy-Arrays.ipynb) | [Contents](Index.ipynb) | [Aggregations: Min, Max, and Everything In Between](02.04-Computation-on-arrays-aggregates.ipynb) >
# 

# # Computation on NumPy Arrays: Universal Functions
# 

# Up until now, we have been discussing some of the basic nuts and bolts of NumPy; in the next few sections, we will dive into the reasons that NumPy is so important in the Python data science world.
# Namely, it provides an easy and flexible interface to optimized computation with arrays of data.
# 
# Computation on NumPy arrays can be very fast, or it can be very slow.
# The key to making it fast is to use *vectorized* operations, generally implemented through NumPy's *universal functions* (ufuncs).
# This section motivates the need for NumPy's ufuncs, which can be used to make repeated calculations on array elements much more efficient.
# It then introduces many of the most common and useful arithmetic ufuncs available in the NumPy package.
# 

# ## The Slowness of Loops
# 
# Python's default implementation (known as CPython) does some operations very slowly.
# This is in part due to the dynamic, interpreted nature of the language: the fact that types are flexible, so that sequences of operations cannot be compiled down to efficient machine code as in languages like C and Fortran.
# Recently there have been various attempts to address this weakness: well-known examples are the [PyPy](http://pypy.org/) project, a just-in-time compiled implementation of Python; the [Cython](http://cython.org) project, which converts Python code to compilable C code; and the [Numba](http://numba.pydata.org/) project, which converts snippets of Python code to fast LLVM bytecode.
# Each of these has its strengths and weaknesses, but it is safe to say that none of the three approaches has yet surpassed the reach and popularity of the standard CPython engine.
# 
# The relative sluggishness of Python generally manifests itself in situations where many small operations are being repeated – for instance looping over arrays to operate on each element.
# For example, imagine we have an array of values and we'd like to compute the reciprocal of each.
# A straightforward approach might look like this:
# 

import numpy as np
np.random.seed(0)

def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output
        
values = np.random.randint(1, 10, size=5)
compute_reciprocals(values)


# This implementation probably feels fairly natural to someone from, say, a C or Java background.
# But if we measure the execution time of this code for a large input, we see that this operation is very slow, perhaps surprisingly so!
# We'll benchmark this with IPython's ``%timeit`` magic (discussed in [Profiling and Timing Code](01.07-Timing-and-Profiling.ipynb)):
# 

big_array = np.random.randint(1, 100, size=1000000)
get_ipython().magic('timeit compute_reciprocals(big_array)')


# It takes several seconds to compute these million operations and to store the result!
# When even cell phones have processing speeds measured in Giga-FLOPS (i.e., billions of numerical operations per second), this seems almost absurdly slow.
# It turns out that the bottleneck here is not the operations themselves, but the type-checking and function dispatches that CPython must do at each cycle of the loop.
# Each time the reciprocal is computed, Python first examines the object's type and does a dynamic lookup of the correct function to use for that type.
# If we were working in compiled code instead, this type specification would be known before the code executes and the result could be computed much more efficiently.
# 

# ## Introducing UFuncs
# 
# For many types of operations, NumPy provides a convenient interface into just this kind of statically typed, compiled routine. This is known as a *vectorized* operation.
# This can be accomplished by simply performing an operation on the array, which will then be applied to each element.
# This vectorized approach is designed to push the loop into the compiled layer that underlies NumPy, leading to much faster execution.
# 
# Compare the results of the following two:
# 

print(compute_reciprocals(values))
print(1.0 / values)


# Looking at the execution time for our big array, we see that it completes orders of magnitude faster than the Python loop:
# 

get_ipython().magic('timeit (1.0 / big_array)')


# Vectorized operations in NumPy are implemented via *ufuncs*, whose main purpose is to quickly execute repeated operations on values in NumPy arrays.
# Ufuncs are extremely flexible – before we saw an operation between a scalar and an array, but we can also operate between two arrays:
# 

np.arange(5) / np.arange(1, 6)


# And ufunc operations are not limited to one-dimensional arrays–they can also act on multi-dimensional arrays as well:
# 

x = np.arange(9).reshape((3, 3))
2 ** x


# Computations using vectorization through ufuncs are nearly always more efficient than their counterpart implemented using Python loops, especially as the arrays grow in size.
# Any time you see such a loop in a Python script, you should consider whether it can be replaced with a vectorized expression.
# 

# ## Exploring NumPy's UFuncs
# 
# Ufuncs exist in two flavors: *unary ufuncs*, which operate on a single input, and *binary ufuncs*, which operate on two inputs.
# We'll see examples of both these types of functions here.
# 

# ### Array arithmetic
# 
# NumPy's ufuncs feel very natural to use because they make use of Python's native arithmetic operators.
# The standard addition, subtraction, multiplication, and division can all be used:
# 

x = np.arange(4)
print("x     =", x)
print("x + 5 =", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)
print("x / 2 =", x / 2)
print("x // 2 =", x // 2)  # floor division


# There is also a unary ufunc for negation, and a ``**`` operator for exponentiation, and a ``%`` operator for modulus:
# 

print("-x     = ", -x)
print("x ** 2 = ", x ** 2)
print("x % 2  = ", x % 2)


# In addition, these can be strung together however you wish, and the standard order of operations is respected:
# 

-(0.5*x + 1) ** 2


# Each of these arithmetic operations are simply convenient wrappers around specific functions built into NumPy; for example, the ``+`` operator is a wrapper for the ``add`` function:
# 

np.add(x, 2)


# The following table lists the arithmetic operators implemented in NumPy:
# 
# | Operator	    | Equivalent ufunc    | Description                           |
# |---------------|---------------------|---------------------------------------|
# |``+``          |``np.add``           |Addition (e.g., ``1 + 1 = 2``)         |
# |``-``          |``np.subtract``      |Subtraction (e.g., ``3 - 2 = 1``)      |
# |``-``          |``np.negative``      |Unary negation (e.g., ``-2``)          |
# |``*``          |``np.multiply``      |Multiplication (e.g., ``2 * 3 = 6``)   |
# |``/``          |``np.divide``        |Division (e.g., ``3 / 2 = 1.5``)       |
# |``//``         |``np.floor_divide``  |Floor division (e.g., ``3 // 2 = 1``)  |
# |``**``         |``np.power``         |Exponentiation (e.g., ``2 ** 3 = 8``)  |
# |``%``          |``np.mod``           |Modulus/remainder (e.g., ``9 % 4 = 1``)|
# 
# Additionally there are Boolean/bitwise operators; we will explore these in [Comparisons, Masks, and Boolean Logic](02.06-Boolean-Arrays-and-Masks.ipynb).
# 

# ### Absolute value
# 
# Just as NumPy understands Python's built-in arithmetic operators, it also understands Python's built-in absolute value function:
# 

x = np.array([-2, -1, 0, 1, 2])
abs(x)


# The corresponding NumPy ufunc is ``np.absolute``, which is also available under the alias ``np.abs``:
# 

np.absolute(x)


np.abs(x)


# This ufunc can also handle complex data, in which the absolute value returns the magnitude:
# 

x = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j])
np.abs(x)


# ### Trigonometric functions
# 
# NumPy provides a large number of useful ufuncs, and some of the most useful for the data scientist are the trigonometric functions.
# We'll start by defining an array of angles:
# 

theta = np.linspace(0, np.pi, 3)


# Now we can compute some trigonometric functions on these values:
# 

print("theta      = ", theta)
print("sin(theta) = ", np.sin(theta))
print("cos(theta) = ", np.cos(theta))
print("tan(theta) = ", np.tan(theta))


# The values are computed to within machine precision, which is why values that should be zero do not always hit exactly zero.
# Inverse trigonometric functions are also available:
# 

x = [-1, 0, 1]
print("x         = ", x)
print("arcsin(x) = ", np.arcsin(x))
print("arccos(x) = ", np.arccos(x))
print("arctan(x) = ", np.arctan(x))


# ### Exponents and logarithms
# 
# Another common type of operation available in a NumPy ufunc are the exponentials:
# 

x = [1, 2, 3]
print("x     =", x)
print("e^x   =", np.exp(x))
print("2^x   =", np.exp2(x))
print("3^x   =", np.power(3, x))


# The inverse of the exponentials, the logarithms, are also available.
# The basic ``np.log`` gives the natural logarithm; if you prefer to compute the base-2 logarithm or the base-10 logarithm, these are available as well:
# 

x = [1, 2, 4, 10]
print("x        =", x)
print("ln(x)    =", np.log(x))
print("log2(x)  =", np.log2(x))
print("log10(x) =", np.log10(x))


# There are also some specialized versions that are useful for maintaining precision with very small input:
# 

x = [0, 0.001, 0.01, 0.1]
print("exp(x) - 1 =", np.expm1(x))
print("log(1 + x) =", np.log1p(x))


# When ``x`` is very small, these functions give more precise values than if the raw ``np.log`` or ``np.exp`` were to be used.
# 

# ### Specialized ufuncs
# 
# NumPy has many more ufuncs available, including hyperbolic trig functions, bitwise arithmetic, comparison operators, conversions from radians to degrees, rounding and remainders, and much more.
# A look through the NumPy documentation reveals a lot of interesting functionality.
# 
# Another excellent source for more specialized and obscure ufuncs is the submodule ``scipy.special``.
# If you want to compute some obscure mathematical function on your data, chances are it is implemented in ``scipy.special``.
# There are far too many functions to list them all, but the following snippet shows a couple that might come up in a statistics context:
# 

from scipy import special


# Gamma functions (generalized factorials) and related functions
x = [1, 5, 10]
print("gamma(x)     =", special.gamma(x))
print("ln|gamma(x)| =", special.gammaln(x))
print("beta(x, 2)   =", special.beta(x, 2))


# Error function (integral of Gaussian)
# its complement, and its inverse
x = np.array([0, 0.3, 0.7, 1.0])
print("erf(x)  =", special.erf(x))
print("erfc(x) =", special.erfc(x))
print("erfinv(x) =", special.erfinv(x))


# There are many, many more ufuncs available in both NumPy and ``scipy.special``.
# Because the documentation of these packages is available online, a web search along the lines of "gamma function python" will generally find the relevant information.
# 

# ## Advanced Ufunc Features
# 
# Many NumPy users make use of ufuncs without ever learning their full set of features.
# We'll outline a few specialized features of ufuncs here.
# 

# ### Specifying output
# 
# For large calculations, it is sometimes useful to be able to specify the array where the result of the calculation will be stored.
# Rather than creating a temporary array, this can be used to write computation results directly to the memory location where you'd like them to be.
# For all ufuncs, this can be done using the ``out`` argument of the function:
# 

x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y)
print(y)


# This can even be used with array views. For example, we can write the results of a computation to every other element of a specified array:
# 

y = np.zeros(10)
np.power(2, x, out=y[::2])
print(y)


# If we had instead written ``y[::2] = 2 ** x``, this would have resulted in the creation of a temporary array to hold the results of ``2 ** x``, followed by a second operation copying those values into the ``y`` array.
# This doesn't make much of a difference for such a small computation, but for very large arrays the memory savings from careful use of the ``out`` argument can be significant.
# 

# ### Aggregates
# 
# For binary ufuncs, there are some interesting aggregates that can be computed directly from the object.
# For example, if we'd like to *reduce* an array with a particular operation, we can use the ``reduce`` method of any ufunc.
# A reduce repeatedly applies a given operation to the elements of an array until only a single result remains.
# 
# For example, calling ``reduce`` on the ``add`` ufunc returns the sum of all elements in the array:
# 

x = np.arange(1, 6)
np.add.reduce(x)


# Similarly, calling ``reduce`` on the ``multiply`` ufunc results in the product of all array elements:
# 

np.multiply.reduce(x)


# If we'd like to store all the intermediate results of the computation, we can instead use ``accumulate``:
# 

np.add.accumulate(x)


np.multiply.accumulate(x)


# Note that for these particular cases, there are dedicated NumPy functions to compute the results (``np.sum``, ``np.prod``, ``np.cumsum``, ``np.cumprod``), which we'll explore in [Aggregations: Min, Max, and Everything In Between](02.04-Computation-on-arrays-aggregates.ipynb).
# 

# ### Outer products
# 
# Finally, any ufunc can compute the output of all pairs of two different inputs using the ``outer`` method.
# This allows you, in one line, to do things like create a multiplication table:
# 

x = np.arange(1, 6)
np.multiply.outer(x, x)


# The ``ufunc.at`` and ``ufunc.reduceat`` methods, which we'll explore in [Fancy Indexing](02.07-Fancy-Indexing.ipynb), are very helpful as well.
# 
# Another extremely useful feature of ufuncs is the ability to operate between arrays of different sizes and shapes, a set of operations known as *broadcasting*.
# This subject is important enough that we will devote a whole section to it (see [Computation on Arrays: Broadcasting](02.05-Computation-on-arrays-broadcasting.ipynb)).
# 

# ## Ufuncs: Learning More
# 

# More information on universal functions (including the full list of available functions) can be found on the [NumPy](http://www.numpy.org) and [SciPy](http://www.scipy.org) documentation websites.
# 
# Recall that you can also access information directly from within IPython by importing the packages and using IPython's tab-completion and help (``?``) functionality, as described in [Help and Documentation in IPython](01.01-Help-And-Documentation.ipynb).
# 

# <!--NAVIGATION-->
# < [The Basics of NumPy Arrays](02.02-The-Basics-Of-NumPy-Arrays.ipynb) | [Contents](Index.ipynb) | [Aggregations: Min, Max, and Everything In Between](02.04-Computation-on-arrays-aggregates.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [Hyperparameters and Model Validation](05.03-Hyperparameters-and-Model-Validation.ipynb) | [Contents](Index.ipynb) | [In Depth: Naive Bayes Classification](05.05-Naive-Bayes.ipynb) >
# 

# # Feature Engineering
# 
# The previous sections outline the fundamental ideas of machine learning, but all of the examples assume that you have numerical data in a tidy, ``[n_samples, n_features]`` format.
# In the real world, data rarely comes in such a form.
# With this in mind, one of the more important steps in using machine learning in practice is *feature engineering*: that is, taking whatever information you have about your problem and turning it into numbers that you can use to build your feature matrix.
# 
# In this section, we will cover a few common examples of feature engineering tasks: features for representing *categorical data*, features for representing *text*, and features for representing *images*.
# Additionally, we will discuss *derived features* for increasing model complexity and *imputation* of missing data.
# Often this process is known as *vectorization*, as it involves converting arbitrary data into well-behaved vectors.
# 

# ## Categorical Features
# 
# One common type of non-numerical data is *categorical* data.
# For example, imagine you are exploring some data on housing prices, and along with numerical features like "price" and "rooms", you also have "neighborhood" information.
# For example, your data might look something like this:
# 

data = [
    {'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
    {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
    {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
    {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}
]


# You might be tempted to encode this data with a straightforward numerical mapping:
# 

{'Queen Anne': 1, 'Fremont': 2, 'Wallingford': 3};


# It turns out that this is not generally a useful approach in Scikit-Learn: the package's models make the fundamental assumption that numerical features reflect algebraic quantities.
# Thus such a mapping would imply, for example, that *Queen Anne < Fremont < Wallingford*, or even that *Wallingford - Queen Anne = Fremont*, which (niche demographic jokes aside) does not make much sense.
# 
# In this case, one proven technique is to use *one-hot encoding*, which effectively creates extra columns indicating the presence or absence of a category with a value of 1 or 0, respectively.
# When your data comes as a list of dictionaries, Scikit-Learn's ``DictVectorizer`` will do this for you:
# 

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False, dtype=int)
vec.fit_transform(data)


# Notice that the 'neighborhood' column has been expanded into three separate columns, representing the three neighborhood labels, and that each row has a 1 in the column associated with its neighborhood.
# With these categorical features thus encoded, you can proceed as normal with fitting a Scikit-Learn model.
# 
# To see the meaning of each column, you can inspect the feature names:
# 

vec.get_feature_names()


# There is one clear disadvantage of this approach: if your category has many possible values, this can *greatly* increase the size of your dataset.
# However, because the encoded data contains mostly zeros, a sparse output can be a very efficient solution:
# 

vec = DictVectorizer(sparse=True, dtype=int)
vec.fit_transform(data)


# Many (though not yet all) of the Scikit-Learn estimators accept such sparse inputs when fitting and evaluating models. ``sklearn.preprocessing.OneHotEncoder`` and ``sklearn.feature_extraction.FeatureHasher`` are two additional tools that Scikit-Learn includes to support this type of encoding.
# 

# ## Text Features
# 
# Another common need in feature engineering is to convert text to a set of representative numerical values.
# For example, most automatic mining of social media data relies on some form of encoding the text as numbers.
# One of the simplest methods of encoding data is by *word counts*: you take each snippet of text, count the occurrences of each word within it, and put the results in a table.
# 
# For example, consider the following set of three phrases:
# 

sample = ['problem of evil',
          'evil queen',
          'horizon problem']


# For a vectorization of this data based on word count, we could construct a column representing the word "problem," the word "evil," the word "horizon," and so on.
# While doing this by hand would be possible, the tedium can be avoided by using Scikit-Learn's ``CountVectorizer``:
# 

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()
X = vec.fit_transform(sample)
X


# The result is a sparse matrix recording the number of times each word appears; it is easier to inspect if we convert this to a ``DataFrame`` with labeled columns:
# 

import pandas as pd
pd.DataFrame(X.toarray(), columns=vec.get_feature_names())


# There are some issues with this approach, however: the raw word counts lead to features which put too much weight on words that appear very frequently, and this can be sub-optimal in some classification algorithms.
# One approach to fix this is known as *term frequency-inverse document frequency* (*TF–IDF*) which weights the word counts by a measure of how often they appear in the documents.
# The syntax for computing these features is similar to the previous example:
# 

from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(sample)
pd.DataFrame(X.toarray(), columns=vec.get_feature_names())


# For an example of using TF-IDF in a classification problem, see [In Depth: Naive Bayes Classification](05.05-Naive-Bayes.ipynb).
# 

# ## Image Features
# 
# Another common need is to suitably encode *images* for machine learning analysis.
# The simplest approach is what we used for the digits data in [Introducing Scikit-Learn](05.02-Introducing-Scikit-Learn.ipynb): simply using the pixel values themselves.
# But depending on the application, such approaches may not be optimal.
# 
# A comprehensive summary of feature extraction techniques for images is well beyond the scope of this section, but you can find excellent implementations of many of the standard approaches in the [Scikit-Image project](http://scikit-image.org).
# For one example of using Scikit-Learn and Scikit-Image together, see [Feature Engineering: Working with Images](05.14-Image-Features.ipynb).
# 

# ## Derived Features
# 
# Another useful type of feature is one that is mathematically derived from some input features.
# We saw an example of this in [Hyperparameters and Model Validation](05.03-Hyperparameters-and-Model-Validation.ipynb) when we constructed *polynomial features* from our input data.
# We saw that we could convert a linear regression into a polynomial regression not by changing the model, but by transforming the input!
# This is sometimes known as *basis function regression*, and is explored further in [In Depth: Linear Regression](05.06-Linear-Regression.ipynb).
# 
# For example, this data clearly cannot be well described by a straight line:
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 2, 1, 3, 7])
plt.scatter(x, y);


# Still, we can fit a line to the data using ``LinearRegression`` and get the optimal result:
# 

from sklearn.linear_model import LinearRegression
X = x[:, np.newaxis]
model = LinearRegression().fit(X, y)
yfit = model.predict(X)
plt.scatter(x, y)
plt.plot(x, yfit);


# It's clear that we need a more sophisticated model to describe the relationship between $x$ and $y$.
# 
# One approach to this is to transform the data, adding extra columns of features to drive more flexibility in the model.
# For example, we can add polynomial features to the data this way:
# 

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3, include_bias=False)
X2 = poly.fit_transform(X)
print(X2)


# The derived feature matrix has one column representing $x$, and a second column representing $x^2$, and a third column representing $x^3$.
# Computing a linear regression on this expanded input gives a much closer fit to our data:
# 

model = LinearRegression().fit(X2, y)
yfit = model.predict(X2)
plt.scatter(x, y)
plt.plot(x, yfit);


# This idea of improving a model not by changing the model, but by transforming the inputs, is fundamental to many of the more powerful machine learning methods.
# We explore this idea further in [In Depth: Linear Regression](05.06-Linear-Regression.ipynb) in the context of *basis function regression*.
# More generally, this is one motivational path to the powerful set of techniques known as *kernel methods*, which we will explore in [In-Depth: Support Vector Machines](05.07-Support-Vector-Machines.ipynb).
# 

# ## Imputation of Missing Data
# 
# Another common need in feature engineering is handling of missing data.
# We discussed the handling of missing data in ``DataFrame``s in [Handling Missing Data](03.04-Missing-Values.ipynb), and saw that often the ``NaN`` value is used to mark missing values.
# For example, we might have a dataset that looks like this:
# 

from numpy import nan
X = np.array([[ nan, 0,   3  ],
              [ 3,   7,   9  ],
              [ 3,   5,   2  ],
              [ 4,   nan, 6  ],
              [ 8,   8,   1  ]])
y = np.array([14, 16, -1,  8, -5])


# When applying a typical machine learning model to such data, we will need to first replace such missing data with some appropriate fill value.
# This is known as *imputation* of missing values, and strategies range from simple (e.g., replacing missing values with the mean of the column) to sophisticated (e.g., using matrix completion or a robust model to handle such data).
# 
# The sophisticated approaches tend to be very application-specific, and we won't dive into them here.
# For a baseline imputation approach, using the mean, median, or most frequent value, Scikit-Learn provides the ``Imputer`` class:
# 

from sklearn.preprocessing import Imputer
imp = Imputer(strategy='mean')
X2 = imp.fit_transform(X)
X2


# We see that in the resulting data, the two missing values have been replaced with the mean of the remaining values in the column. This imputed data can then be fed directly into, for example, a ``LinearRegression`` estimator:
# 

model = LinearRegression().fit(X2, y)
model.predict(X2)


# ## Feature Pipelines
# 
# With any of the preceding examples, it can quickly become tedious to do the transformations by hand, especially if you wish to string together multiple steps.
# For example, we might want a processing pipeline that looks something like this:
# 
# 1. Impute missing values using the mean
# 2. Transform features to quadratic
# 3. Fit a linear regression
# 
# To streamline this type of processing pipeline, Scikit-Learn provides a ``Pipeline`` object, which can be used as follows:
# 

from sklearn.pipeline import make_pipeline

model = make_pipeline(Imputer(strategy='mean'),
                      PolynomialFeatures(degree=2),
                      LinearRegression())


# This pipeline looks and acts like a standard Scikit-Learn object, and will apply all the specified steps to any input data.
# 

model.fit(X, y)  # X with missing values, from above
print(y)
print(model.predict(X))


# All the steps of the model are applied automatically.
# Notice that for the simplicity of this demonstration, we've applied the model to the data it was trained on; this is why it was able to perfectly predict the result (refer back to [Hyperparameters and Model Validation](05.03-Hyperparameters-and-Model-Validation.ipynb) for further discussion of this).
# 
# For some examples of Scikit-Learn pipelines in action, see the following section on naive Bayes classification, as well as [In Depth: Linear Regression](05.06-Linear-Regression.ipynb), and [In-Depth: Support Vector Machines](05.07-Support-Vector-Machines.ipynb).
# 

# <!--NAVIGATION-->
# < [Hyperparameters and Model Validation](05.03-Hyperparameters-and-Model-Validation.ipynb) | [Contents](Index.ipynb) | [In Depth: Naive Bayes Classification](05.05-Naive-Bayes.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [Customizing Plot Legends](04.06-Customizing-Legends.ipynb) | [Contents](Index.ipynb) | [Multiple Subplots](04.08-Multiple-Subplots.ipynb) >
# 

# # Customizing Colorbars
# 

# Plot legends identify discrete labels of discrete points.
# For continuous labels based on the color of points, lines, or regions, a labeled colorbar can be a great tool.
# In Matplotlib, a colorbar is a separate axes that can provide a key for the meaning of colors in a plot.
# Because the book is printed in black-and-white, this section has an accompanying online supplement where you can view the figures in full color (https://github.com/jakevdp/PythonDataScienceHandbook).
# We'll start by setting up the notebook for plotting and importing the functions we will use:
# 

import matplotlib.pyplot as plt
plt.style.use('classic')


get_ipython().magic('matplotlib inline')
import numpy as np


# As we have seen several times throughout this section, the simplest colorbar can be created with the ``plt.colorbar`` function:
# 

x = np.linspace(0, 10, 1000)
I = np.sin(x) * np.cos(x[:, np.newaxis])

plt.imshow(I)
plt.colorbar();


# We'll now discuss a few ideas for customizing these colorbars and using them effectively in various situations.
# 

# ## Customizing Colorbars
# 
# The colormap can be specified using the ``cmap`` argument to the plotting function that is creating the visualization:
# 

plt.imshow(I, cmap='gray');


# All the available colormaps are in the ``plt.cm`` namespace; using IPython's tab-completion will give you a full list of built-in possibilities:
# ```
# plt.cm.<TAB>
# ```
# But being *able* to choose a colormap is just the first step: more important is how to *decide* among the possibilities!
# The choice turns out to be much more subtle than you might initially expect.
# 

# ### Choosing the Colormap
# 
# A full treatment of color choice within visualization is beyond the scope of this book, but for entertaining reading on this subject and others, see the article ["Ten Simple Rules for Better Figures"](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833).
# Matplotlib's online documentation also has an [interesting discussion](http://Matplotlib.org/1.4.1/users/colormaps.html) of colormap choice.
# 
# Broadly, you should be aware of three different categories of colormaps:
# 
# - *Sequential colormaps*: These are made up of one continuous sequence of colors (e.g., ``binary`` or ``viridis``).
# - *Divergent colormaps*: These usually contain two distinct colors, which show positive and negative deviations from a mean (e.g., ``RdBu`` or ``PuOr``).
# - *Qualitative colormaps*: these mix colors with no particular sequence (e.g., ``rainbow`` or ``jet``).
# 
# The ``jet`` colormap, which was the default in Matplotlib prior to version 2.0, is an example of a qualitative colormap.
# Its status as the default was quite unfortunate, because qualitative maps are often a poor choice for representing quantitative data.
# Among the problems is the fact that qualitative maps usually do not display any uniform progression in brightness as the scale increases.
# 
# We can see this by converting the ``jet`` colorbar into black and white:
# 

from matplotlib.colors import LinearSegmentedColormap

def grayscale_cmap(cmap):
    """Return a grayscale version of the given colormap"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    
    # convert RGBA to perceived grayscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
        
    return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)
    

def view_colormap(cmap):
    """Plot a colormap with its grayscale equivalent"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    
    cmap = grayscale_cmap(cmap)
    grayscale = cmap(np.arange(cmap.N))
    
    fig, ax = plt.subplots(2, figsize=(6, 2),
                           subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow([colors], extent=[0, 10, 0, 1])
    ax[1].imshow([grayscale], extent=[0, 10, 0, 1])


view_colormap('jet')


# Notice the bright stripes in the grayscale image.
# Even in full color, this uneven brightness means that the eye will be drawn to certain portions of the color range, which will potentially emphasize unimportant parts of the dataset.
# It's better to use a colormap such as ``viridis`` (the default as of Matplotlib 2.0), which is specifically constructed to have an even brightness variation across the range.
# Thus it not only plays well with our color perception, but also will translate well to grayscale printing:
# 

view_colormap('viridis')


# If you favor rainbow schemes, another good option for continuous data is the ``cubehelix`` colormap:
# 

view_colormap('cubehelix')


# For other situations, such as showing positive and negative deviations from some mean, dual-color colorbars such as ``RdBu`` (*Red-Blue*) can be useful. However, as you can see in the following figure, it's important to note that the positive-negative information will be lost upon translation to grayscale!
# 

view_colormap('RdBu')


# We'll see examples of using some of these color maps as we continue.
# 
# There are a large number of colormaps available in Matplotlib; to see a list of them, you can use IPython to explore the ``plt.cm`` submodule. For a more principled approach to colors in Python, you can refer to the tools and documentation within the Seaborn library (see [Visualization With Seaborn](04.14-Visualization-With-Seaborn.ipynb)).
# 

# ### Color limits and extensions
# 
# Matplotlib allows for a large range of colorbar customization.
# The colorbar itself is simply an instance of ``plt.Axes``, so all of the axes and tick formatting tricks we've learned are applicable.
# The colorbar has some interesting flexibility: for example, we can narrow the color limits and indicate the out-of-bounds values with a triangular arrow at the top and bottom by setting the ``extend`` property.
# This might come in handy, for example, if displaying an image that is subject to noise:
# 

# make noise in 1% of the image pixels
speckles = (np.random.random(I.shape) < 0.01)
I[speckles] = np.random.normal(0, 3, np.count_nonzero(speckles))

plt.figure(figsize=(10, 3.5))

plt.subplot(1, 2, 1)
plt.imshow(I, cmap='RdBu')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(I, cmap='RdBu')
plt.colorbar(extend='both')
plt.clim(-1, 1);


# Notice that in the left panel, the default color limits respond to the noisy pixels, and the range of the noise completely washes-out the pattern we are interested in.
# In the right panel, we manually set the color limits, and add extensions to indicate values which are above or below those limits.
# The result is a much more useful visualization of our data.
# 

# ### Discrete Color Bars
# 
# Colormaps are by default continuous, but sometimes you'd like to represent discrete values.
# The easiest way to do this is to use the ``plt.cm.get_cmap()`` function, and pass the name of a suitable colormap along with the number of desired bins:
# 

plt.imshow(I, cmap=plt.cm.get_cmap('Blues', 6))
plt.colorbar()
plt.clim(-1, 1);


# The discrete version of a colormap can be used just like any other colormap.
# 

# ## Example: Handwritten Digits
# 
# For an example of where this might be useful, let's look at an interesting visualization of some hand written digits data.
# This data is included in Scikit-Learn, and consists of nearly 2,000 $8 \times 8$ thumbnails showing various hand-written digits.
# 
# For now, let's start by downloading the digits data and visualizing several of the example images with ``plt.imshow()``:
# 

# load images of the digits 0 through 5 and visualize several of them
from sklearn.datasets import load_digits
digits = load_digits(n_class=6)

fig, ax = plt.subplots(8, 8, figsize=(6, 6))
for i, axi in enumerate(ax.flat):
    axi.imshow(digits.images[i], cmap='binary')
    axi.set(xticks=[], yticks=[])


# Because each digit is defined by the hue of its 64 pixels, we can consider each digit to be a point lying in 64-dimensional space: each dimension represents the brightness of one pixel.
# But visualizing relationships in such high-dimensional spaces can be extremely difficult.
# One way to approach this is to use a *dimensionality reduction* technique such as manifold learning to reduce the dimensionality of the data while maintaining the relationships of interest.
# Dimensionality reduction is an example of unsupervised machine learning, and we will discuss it in more detail in [What Is Machine Learning?](05.01-What-Is-Machine-Learning.ipynb).
# 
# Deferring the discussion of these details, let's take a look at a two-dimensional manifold learning projection of this digits data (see [In-Depth: Manifold Learning](05.10-Manifold-Learning.ipynb) for details):
# 

# project the digits into 2 dimensions using IsoMap
from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
projection = iso.fit_transform(digits.data)


# We'll use our discrete colormap to view the results, setting the ``ticks`` and ``clim`` to improve the aesthetics of the resulting colorbar:
# 

# plot the results
plt.scatter(projection[:, 0], projection[:, 1], lw=0.1,
            c=digits.target, cmap=plt.cm.get_cmap('cubehelix', 6))
plt.colorbar(ticks=range(6), label='digit value')
plt.clim(-0.5, 5.5)


# The projection also gives us some interesting insights on the relationships within the dataset: for example, the ranges of 5 and 3 nearly overlap in this projection, indicating that some hand written fives and threes are difficult to distinguish, and therefore more likely to be confused by an automated classification algorithm.
# Other values, like 0 and 1, are more distantly separated, and therefore much less likely to be confused.
# This observation agrees with our intuition, because 5 and 3 look much more similar than do 0 and 1.
# 
# We'll return to manifold learning and to digit classification in [Chapter 5](05.00-Machine-Learning.ipynb).
# 

# <!--NAVIGATION-->
# < [Customizing Plot Legends](04.06-Customizing-Legends.ipynb) | [Contents](Index.ipynb) | [Multiple Subplots](04.08-Multiple-Subplots.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [Help and Documentation in IPython](01.01-Help-And-Documentation.ipynb) | [Contents](Index.ipynb) | [IPython Magic Commands](01.03-Magic-Commands.ipynb) >
# 

# # Keyboard Shortcuts in the IPython Shell
# 
# If you spend any amount of time on the computer, you've probably found a use for keyboard shortcuts in your workflow.
# Most familiar perhaps are the Cmd-C and Cmd-V (or Ctrl-C and Ctrl-V) for copying and pasting in a wide variety of programs and systems.
# Power-users tend to go even further: popular text editors like Emacs, Vim, and others provide users an incredible range of operations through intricate combinations of keystrokes.
# 
# The IPython shell doesn't go this far, but does provide a number of keyboard shortcuts for fast navigation while typing commands.
# These shortcuts are not in fact provided by IPython itself, but through its dependency on the GNU Readline library: as such, some of the following shortcuts may differ depending on your system configuration.
# Also, while some of these shortcuts do work in the browser-based notebook, this section is primarily about shortcuts in the IPython shell.
# 
# Once you get accustomed to these, they can be very useful for quickly performing certain commands without moving your hands from the "home" keyboard position.
# If you're an Emacs user or if you have experience with Linux-style shells, the following will be very familiar.
# We'll group these shortcuts into a few categories: *navigation shortcuts*, *text entry shortcuts*, *command history shortcuts*, and *miscellaneous shortcuts*.
# 

# ## Navigation shortcuts
# 
# While the use of the left and right arrow keys to move backward and forward in the line is quite obvious, there are other options that don't require moving your hands from the "home" keyboard position:
# 
# | Keystroke                         | Action                                     |
# |-----------------------------------|--------------------------------------------|
# | ``Ctrl-a``                        | Move cursor to the beginning of the line   |
# | ``Ctrl-e``                        | Move cursor to the end of the line         |
# | ``Ctrl-b`` or the left arrow key  | Move cursor back one character             |
# | ``Ctrl-f`` or the right arrow key | Move cursor forward one character          |
# 

# ## Text Entry Shortcuts
# 
# While everyone is familiar with using the Backspace key to delete the previous character, reaching for the key often requires some minor finger gymnastics, and it only deletes a single character at a time.
# In IPython there are several shortcuts for removing some portion of the text you're typing.
# The most immediately useful of these are the commands to delete entire lines of text.
# You'll know these have become second-nature if you find yourself using a combination of Ctrl-b and Ctrl-d instead of reaching for Backspace to delete the previous character!
# 
# | Keystroke                     | Action                                           |
# |-------------------------------|--------------------------------------------------|
# | Backspace key                 | Delete previous character in line                |
# | ``Ctrl-d``                    | Delete next character in line                    |
# | ``Ctrl-k``                    | Cut text from cursor to end of line              |
# | ``Ctrl-u``                    | Cut text from beginning of line to cursor        |
# | ``Ctrl-y``                    | Yank (i.e. paste) text that was previously cut   |
# | ``Ctrl-t``                    | Transpose (i.e., switch) previous two characters |
# 

# ## Command History Shortcuts
# 
# Perhaps the most impactful shortcuts discussed here are the ones IPython provides for navigating the command history.
# This command history goes beyond your current IPython session: your entire command history is stored in a SQLite database in your IPython profile directory.
# The most straightforward way to access these is with the up and down arrow keys to step through the history, but other options exist as well:
# 
# | Keystroke                           | Action                                     |
# |-------------------------------------|--------------------------------------------|
# | ``Ctrl-p`` (or the up arrow key)    | Access previous command in history         |
# | ``Ctrl-n`` (or the down arrow key)  | Access next command in history             |
# | ``Ctrl-r``                          | Reverse-search through command history     |
# 

# The reverse-search can be particularly useful.
# Recall that in the previous section we defined a function called ``square``.
# Let's reverse-search our Python history from a new IPython shell and find this definition again.
# When you press Ctrl-r in the IPython terminal, you'll see the following prompt:
# 
# ```ipython
# In [1]:
# (reverse-i-search)`': 
# ```
# 
# If you start typing characters at this prompt, IPython will auto-fill the most recent command, if any, that matches those characters:
# 
# ```ipython
# In [1]: 
# (reverse-i-search)`sqa': square??
# ```
# 
# At any point, you can add more characters to refine the search, or press Ctrl-r again to search further for another command that matches the query. If you followed along in the previous section, pressing Ctrl-r twice more gives:
# 
# ```ipython
# In [1]: 
# (reverse-i-search)`sqa': def square(a):
#     """Return the square of a"""
#     return a ** 2
# ```
# 
# Once you have found the command you're looking for, press Return and the search will end.
# We can then use the retrieved command, and carry-on with our session:
# 
# ```ipython
# In [1]: def square(a):
#     """Return the square of a"""
#     return a ** 2
# 
# In [2]: square(2)
# Out[2]: 4
# ```
# 
# Note that Ctrl-p/Ctrl-n or the up/down arrow keys can also be used to search through history, but only by matching characters at the beginning of the line.
# That is, if you type **``def``** and then press Ctrl-p, it would find the most recent command (if any) in your history that begins with the characters ``def``.

# ## Miscellaneous Shortcuts
# 
# Finally, there are a few miscellaneous shortcuts that don't fit into any of the preceding categories, but are nevertheless useful to know:
# 
# | Keystroke                     | Action                                     |
# |-------------------------------|--------------------------------------------|
# | ``Ctrl-l``                    | Clear terminal screen                      |
# | ``Ctrl-c``                    | Interrupt current Python command           |
# | ``Ctrl-d``                    | Exit IPython session                       |
# 
# The Ctrl-c in particular can be useful when you inadvertently start a very long-running job.
# 

# While some of the shortcuts discussed here may seem a bit tedious at first, they quickly become automatic with practice.
# Once you develop that muscle memory, I suspect you will even find yourself wishing they were available in other contexts.
# 

# <!--NAVIGATION-->
# < [Help and Documentation in IPython](01.01-Help-And-Documentation.ipynb) | [Contents](Index.ipynb) | [IPython Magic Commands](01.03-Magic-Commands.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [Text and Annotation](04.09-Text-and-Annotation.ipynb) | [Contents](Index.ipynb) | [Customizing Matplotlib: Configurations and Stylesheets](04.11-Settings-and-Stylesheets.ipynb) >
# 

# # Customizing Ticks
# 

# Matplotlib's default tick locators and formatters are designed to be generally sufficient in many common situations, but are in no way optimal for every plot. This section will give several examples of adjusting the tick locations and formatting for the particular plot type you're interested in.
# 
# Before we go into examples, it will be best for us to understand further the object hierarchy of Matplotlib plots.
# Matplotlib aims to have a Python object representing everything that appears on the plot: for example, recall that the ``figure`` is the bounding box within which plot elements appear.
# Each Matplotlib object can also act as a container of sub-objects: for example, each ``figure`` can contain one or more ``axes`` objects, each of which in turn contain other objects representing plot contents.
# 
# The tick marks are no exception. Each ``axes`` has attributes ``xaxis`` and ``yaxis``, which in turn have attributes that contain all the properties of the lines, ticks, and labels that make up the axes.
# 

# ## Major and Minor Ticks
# 
# Within each axis, there is the concept of a *major* tick mark, and a *minor* tick mark. As the names would imply, major ticks are usually bigger or more pronounced, while minor ticks are usually smaller. By default, Matplotlib rarely makes use of minor ticks, but one place you can see them is within logarithmic plots:
# 

import matplotlib.pyplot as plt
plt.style.use('classic')
get_ipython().magic('matplotlib inline')
import numpy as np


ax = plt.axes(xscale='log', yscale='log')
ax.grid();


# We see here that each major tick shows a large tickmark and a label, while each minor tick shows a smaller tickmark with no label.
# 
# These tick properties—locations and labels—that is, can be customized by setting the ``formatter`` and ``locator`` objects of each axis. Let's examine these for the x axis of the just shown plot:
# 

print(ax.xaxis.get_major_locator())
print(ax.xaxis.get_minor_locator())


print(ax.xaxis.get_major_formatter())
print(ax.xaxis.get_minor_formatter())


# We see that both major and minor tick labels have their locations specified by a ``LogLocator`` (which makes sense for a logarithmic plot). Minor ticks, though, have their labels formatted by a ``NullFormatter``: this says that no labels will be shown.
# 
# We'll now show a few examples of setting these locators and formatters for various plots.
# 

# ## Hiding Ticks or Labels
# 
# Perhaps the most common tick/label formatting operation is the act of hiding ticks or labels.
# This can be done using ``plt.NullLocator()`` and ``plt.NullFormatter()``, as shown here:
# 

ax = plt.axes()
ax.plot(np.random.rand(50))

ax.yaxis.set_major_locator(plt.NullLocator())
ax.xaxis.set_major_formatter(plt.NullFormatter())


# Notice that we've removed the labels (but kept the ticks/gridlines) from the x axis, and removed the ticks (and thus the labels as well) from the y axis.
# Having no ticks at all can be useful in many situations—for example, when you want to show a grid of images.
# For instance, consider the following figure, which includes images of different faces, an example often used in supervised machine learning problems (see, for example, [In-Depth: Support Vector Machines](05.07-Support-Vector-Machines.ipynb)):
# 

fig, ax = plt.subplots(5, 5, figsize=(5, 5))
fig.subplots_adjust(hspace=0, wspace=0)

# Get some face data from scikit-learn
from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces().images

for i in range(5):
    for j in range(5):
        ax[i, j].xaxis.set_major_locator(plt.NullLocator())
        ax[i, j].yaxis.set_major_locator(plt.NullLocator())
        ax[i, j].imshow(faces[10 * i + j], cmap="bone")


# Notice that each image has its own axes, and we've set the locators to null because the tick values (pixel number in this case) do not convey relevant information for this particular visualization.
# 

# ## Reducing or Increasing the Number of Ticks
# 
# One common problem with the default settings is that smaller subplots can end up with crowded labels.
# We can see this in the plot grid shown here:
# 

fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)


# Particularly for the x ticks, the numbers nearly overlap and make them quite difficult to decipher.
# We can fix this with the ``plt.MaxNLocator()``, which allows us to specify the maximum number of ticks that will be displayed.
# Given this maximum number, Matplotlib will use internal logic to choose the particular tick locations:
# 

# For every axis, set the x and y major locator
for axi in ax.flat:
    axi.xaxis.set_major_locator(plt.MaxNLocator(3))
    axi.yaxis.set_major_locator(plt.MaxNLocator(3))
fig


# This makes things much cleaner. If you want even more control over the locations of regularly-spaced ticks, you might also use ``plt.MultipleLocator``, which we'll discuss in the following section.
# 

# ## Fancy Tick Formats
# 
# Matplotlib's default tick formatting can leave a lot to be desired: it works well as a broad default, but sometimes you'd like do do something more.
# Consider this plot of a sine and a cosine:
# 

# Plot a sine and cosine curve
fig, ax = plt.subplots()
x = np.linspace(0, 3 * np.pi, 1000)
ax.plot(x, np.sin(x), lw=3, label='Sine')
ax.plot(x, np.cos(x), lw=3, label='Cosine')

# Set up grid, legend, and limits
ax.grid(True)
ax.legend(frameon=False)
ax.axis('equal')
ax.set_xlim(0, 3 * np.pi);


# There are a couple changes we might like to make. First, it's more natural for this data to space the ticks and grid lines in multiples of $\pi$. We can do this by setting a ``MultipleLocator``, which locates ticks at a multiple of the number you provide. For good measure, we'll add both major and minor ticks in multiples of $\pi/4$:
# 

ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
fig


# But now these tick labels look a little bit silly: we can see that they are multiples of $\pi$, but the decimal representation does not immediately convey this.
# To fix this, we can change the tick formatter. There's no built-in formatter for what we want to do, so we'll instead use ``plt.FuncFormatter``, which accepts a user-defined function giving fine-grained control over the tick outputs:
# 

def format_func(value, tick_number):
    # find number of multiples of pi/2
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)

ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
fig


# This is much better! Notice that we've made use of Matplotlib's LaTeX support, specified by enclosing the string within dollar signs. This is very convenient for display of mathematical symbols and formulae: in this case, ``"$\pi$"`` is rendered as the Greek character $\pi$.
# 
# The ``plt.FuncFormatter()`` offers extremely fine-grained control over the appearance of your plot ticks, and comes in very handy when preparing plots for presentation or publication.
# 

# ## Summary of Formatters and Locators
# 
# We've mentioned a couple of the available formatters and locators.
# We'll conclude this section by briefly listing all the built-in locator and formatter options. For more information on any of these, refer to the docstrings or to the Matplotlib online documentaion.
# Each of the following is available in the ``plt`` namespace:
# 
# Locator class        | Description
# ---------------------|-------------
# ``NullLocator``      | No ticks
# ``FixedLocator``     | Tick locations are fixed
# ``IndexLocator``     | Locator for index plots (e.g., where x = range(len(y)))
# ``LinearLocator``    | Evenly spaced ticks from min to max
# ``LogLocator``       | Logarithmically ticks from min to max
# ``MultipleLocator``  | Ticks and range are a multiple of base
# ``MaxNLocator``      | Finds up to a max number of ticks at nice locations
# ``AutoLocator``      | (Default.) MaxNLocator with simple defaults.
# ``AutoMinorLocator`` | Locator for minor ticks
# 
# Formatter Class       | Description
# ----------------------|---------------
# ``NullFormatter``     | No labels on the ticks
# ``IndexFormatter``    | Set the strings from a list of labels
# ``FixedFormatter``    | Set the strings manually for the labels
# ``FuncFormatter``     | User-defined function sets the labels
# ``FormatStrFormatter``| Use a format string for each value
# ``ScalarFormatter``   | (Default.) Formatter for scalar values
# ``LogFormatter``      | Default formatter for log axes
# 
# We'll see further examples of these through the remainder of the book.
# 

# <!--NAVIGATION-->
# < [Text and Annotation](04.09-Text-and-Annotation.ipynb) | [Contents](Index.ipynb) | [Customizing Matplotlib: Configurations and Stylesheets](04.11-Settings-and-Stylesheets.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [Further Resources](04.15-Further-Resources.ipynb) | [Contents](Index.ipynb) | [What Is Machine Learning?](05.01-What-Is-Machine-Learning.ipynb) >
# 

# # Machine Learning
# 

# In many ways, machine learning is the primary means by which data science manifests itself to the broader world.
# Machine learning is where these computational and algorithmic skills of data science meet the statistical thinking of data science, and the result is a collection of approaches to inference and data exploration that are not about effective theory so much as effective computation.
# 
# The term "machine learning" is sometimes thrown around as if it is some kind of magic pill: *apply machine learning to your data, and all your problems will be solved!*
# As you might expect, the reality is rarely this simple.
# While these methods can be incredibly powerful, to be effective they must be approached with a firm grasp of the strengths and weaknesses of each method, as well as a grasp of general concepts such as bias and variance, overfitting and underfitting, and more.
# 
# This chapter will dive into practical aspects of machine learning, primarily using Python's [Scikit-Learn](http://scikit-learn.org) package.
# This is not meant to be a comprehensive introduction to the field of machine learning; that is a large subject and necessitates a more technical approach than we take here.
# Nor is it meant to be a comprehensive manual for the use of the Scikit-Learn package (for this, you can refer to the resources listed in [Further Machine Learning Resources](05.15-Learning-More.ipynb)).
# Rather, the goals of this chapter are:
# 
# - To introduce the fundamental vocabulary and concepts of machine learning.
# - To introduce the Scikit-Learn API and show some examples of its use.
# - To take a deeper dive into the details of several of the most important machine learning approaches, and develop an intuition into how they work and when and where they are applicable.
# 
# Much of this material is drawn from the Scikit-Learn tutorials and workshops I have given on several occasions at PyCon, SciPy, PyData, and other conferences.
# Any clarity in the following pages is likely due to the many workshop participants and co-instructors who have given me valuable feedback on this material over the years!
# 
# Finally, if you are seeking a more comprehensive or technical treatment of any of these subjects, I've listed several resources and references in [Further Machine Learning Resources](05.15-Learning-More.ipynb).
# 

# <!--NAVIGATION-->
# < [Further Resources](04.15-Further-Resources.ipynb) | [Contents](Index.ipynb) | [What Is Machine Learning?](05.01-What-Is-Machine-Learning.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [Hierarchical Indexing](03.05-Hierarchical-Indexing.ipynb) | [Contents](Index.ipynb) | [Combining Datasets: Merge and Join](03.07-Merge-and-Join.ipynb) >
# 

# # Combining Datasets: Concat and Append
# 

# Some of the most interesting studies of data come from combining different data sources.
# These operations can involve anything from very straightforward concatenation of two different datasets, to more complicated database-style joins and merges that correctly handle any overlaps between the datasets.
# ``Series`` and ``DataFrame``s are built with this type of operation in mind, and Pandas includes functions and methods that make this sort of data wrangling fast and straightforward.
# 
# Here we'll take a look at simple concatenation of ``Series`` and ``DataFrame``s with the ``pd.concat`` function; later we'll dive into more sophisticated in-memory merges and joins implemented in Pandas.
# 
# We begin with the standard imports:
# 

import pandas as pd
import numpy as np


# For convenience, we'll define this function which creates a ``DataFrame`` of a particular form that will be useful below:
# 

def make_df(cols, ind):
    """Quickly make a DataFrame"""
    data = {c: [str(c) + str(i) for i in ind]
            for c in cols}
    return pd.DataFrame(data, ind)

# example DataFrame
make_df('ABC', range(3))


# In addition, we'll create a quick class that allows us to display multiple ``DataFrame``s side by side. The code makes use of the special ``_repr_html_`` method, which IPython uses to implement its rich object display:
# 

class display(object):
    """Display HTML representation of multiple objects"""
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""
    def __init__(self, *args):
        self.args = args
        
    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                         for a in self.args)
    
    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a))
                           for a in self.args)
    


# The use of this will become clearer as we continue our discussion in the following section.
# 

# ## Recall: Concatenation of NumPy Arrays
# 
# Concatenation of ``Series`` and ``DataFrame`` objects is very similar to concatenation of Numpy arrays, which can be done via the ``np.concatenate`` function as discussed in [The Basics of NumPy Arrays](02.02-The-Basics-Of-NumPy-Arrays.ipynb).
# Recall that with it, you can combine the contents of two or more arrays into a single array:
# 

x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]
np.concatenate([x, y, z])


# The first argument is a list or tuple of arrays to concatenate.
# Additionally, it takes an ``axis`` keyword that allows you to specify the axis along which the result will be concatenated:
# 

x = [[1, 2],
     [3, 4]]
np.concatenate([x, x], axis=1)


# ## Simple Concatenation with ``pd.concat``
# 

# Pandas has a function, ``pd.concat()``, which has a similar syntax to ``np.concatenate`` but contains a number of options that we'll discuss momentarily:
# 
# ```python
# # Signature in Pandas v0.18
# pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False,
#           keys=None, levels=None, names=None, verify_integrity=False,
#           copy=True)
# ```
# 
# ``pd.concat()`` can be used for a simple concatenation of ``Series`` or ``DataFrame`` objects, just as ``np.concatenate()`` can be used for simple concatenations of arrays:
# 

ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])
pd.concat([ser1, ser2])


# It also works to concatenate higher-dimensional objects, such as ``DataFrame``s:
# 

df1 = make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])
display('df1', 'df2', 'pd.concat([df1, df2])')


# By default, the concatenation takes place row-wise within the ``DataFrame`` (i.e., ``axis=0``).
# Like ``np.concatenate``, ``pd.concat`` allows specification of an axis along which concatenation will take place.
# Consider the following example:
# 

df3 = make_df('AB', [0, 1])
df4 = make_df('CD', [0, 1])
display('df3', 'df4', "pd.concat([df3, df4], axis='col')")


# We could have equivalently specified ``axis=1``; here we've used the more intuitive ``axis='col'``. 
# 

# ### Duplicate indices
# 
# One important difference between ``np.concatenate`` and ``pd.concat`` is that Pandas concatenation *preserves indices*, even if the result will have duplicate indices!
# Consider this simple example:
# 

x = make_df('AB', [0, 1])
y = make_df('AB', [2, 3])
y.index = x.index  # make duplicate indices!
display('x', 'y', 'pd.concat([x, y])')


# Notice the repeated indices in the result.
# While this is valid within ``DataFrame``s, the outcome is often undesirable.
# ``pd.concat()`` gives us a few ways to handle it.
# 

# #### Catching the repeats as an error
# 
# If you'd like to simply verify that the indices in the result of ``pd.concat()`` do not overlap, you can specify the ``verify_integrity`` flag.
# With this set to True, the concatenation will raise an exception if there are duplicate indices.
# Here is an example, where for clarity we'll catch and print the error message:
# 

try:
    pd.concat([x, y], verify_integrity=True)
except ValueError as e:
    print("ValueError:", e)


# #### Ignoring the index
# 
# Sometimes the index itself does not matter, and you would prefer it to simply be ignored.
# This option can be specified using the ``ignore_index`` flag.
# With this set to true, the concatenation will create a new integer index for the resulting ``Series``:
# 

display('x', 'y', 'pd.concat([x, y], ignore_index=True)')


# #### Adding MultiIndex keys
# 
# Another option is to use the ``keys`` option to specify a label for the data sources; the result will be a hierarchically indexed series containing the data:
# 

display('x', 'y', "pd.concat([x, y], keys=['x', 'y'])")


# The result is a multiply indexed ``DataFrame``, and we can use the tools discussed in [Hierarchical Indexing](03.05-Hierarchical-Indexing.ipynb) to transform this data into the representation we're interested in.
# 

# ### Concatenation with joins
# 
# In the simple examples we just looked at, we were mainly concatenating ``DataFrame``s with shared column names.
# In practice, data from different sources might have different sets of column names, and ``pd.concat`` offers several options in this case.
# Consider the concatenation of the following two ``DataFrame``s, which have some (but not all!) columns in common:
# 

df5 = make_df('ABC', [1, 2])
df6 = make_df('BCD', [3, 4])
display('df5', 'df6', 'pd.concat([df5, df6])')


# By default, the entries for which no data is available are filled with NA values.
# To change this, we can specify one of several options for the ``join`` and ``join_axes`` parameters of the concatenate function.
# By default, the join is a union of the input columns (``join='outer'``), but we can change this to an intersection of the columns using ``join='inner'``:
# 

display('df5', 'df6',
        "pd.concat([df5, df6], join='inner')")


# Another option is to directly specify the index of the remaininig colums using the ``join_axes`` argument, which takes a list of index objects.
# Here we'll specify that the returned columns should be the same as those of the first input:
# 

display('df5', 'df6',
        "pd.concat([df5, df6], join_axes=[df5.columns])")


# The combination of options of the ``pd.concat`` function allows a wide range of possible behaviors when joining two datasets; keep these in mind as you use these tools for your own data.
# 

# ### The ``append()`` method
# 
# Because direct array concatenation is so common, ``Series`` and ``DataFrame`` objects have an ``append`` method that can accomplish the same thing in fewer keystrokes.
# For example, rather than calling ``pd.concat([df1, df2])``, you can simply call ``df1.append(df2)``:
# 

display('df1', 'df2', 'df1.append(df2)')


# Keep in mind that unlike the ``append()`` and ``extend()`` methods of Python lists, the ``append()`` method in Pandas does not modify the original object–instead it creates a new object with the combined data.
# It also is not a very efficient method, because it involves creation of a new index *and* data buffer.
# Thus, if you plan to do multiple ``append`` operations, it is generally better to build a list of ``DataFrame``s and pass them all at once to the ``concat()`` function.
# 
# In the next section, we'll look at another more powerful approach to combining data from multiple sources, the database-style merges/joins implemented in ``pd.merge``.
# For more information on ``concat()``, ``append()``, and related functionality, see the ["Merge, Join, and Concatenate" section](http://pandas.pydata.org/pandas-docs/stable/merging.html) of the Pandas documentation.
# 

# <!--NAVIGATION-->
# < [Hierarchical Indexing](03.05-Hierarchical-Indexing.ipynb) | [Contents](Index.ipynb) | [Combining Datasets: Merge and Join](03.07-Merge-and-Join.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [IPython: Beyond Normal Python](01.00-IPython-Beyond-Normal-Python.ipynb) | [Contents](Index.ipynb) | [Keyboard Shortcuts in the IPython Shell](01.02-Shell-Keyboard-Shortcuts.ipynb) >
# 

# # Help and Documentation in IPython
# 

# If you read no other section in this chapter, read this one: I find the tools discussed here to be the most transformative contributions of IPython to my daily workflow.
# 
# When a technologically-minded person is asked to help a friend, family member, or colleague with a computer problem, most of the time it's less a matter of knowing the answer as much as knowing how to quickly find an unknown answer.
# In data science it's the same: searchable web resources such as online documentation, mailing-list threads, and StackOverflow answers contain a wealth of information, even (especially?) if it is a topic you've found yourself searching before.
# Being an effective practitioner of data science is less about memorizing the tool or command you should use for every possible situation, and more about learning to effectively find the information you don't know, whether through a web search engine or another means.
# 
# One of the most useful functions of IPython/Jupyter is to shorten the gap between the user and the type of documentation and search that will help them do their work effectively.
# While web searches still play a role in answering complicated questions, an amazing amount of information can be found through IPython alone.
# Some examples of the questions IPython can help answer in a few keystrokes:
# 
# - How do I call this function? What arguments and options does it have?
# - What does the source code of this Python object look like?
# - What is in this package I imported? What attributes or methods does this object have?
# 
# Here we'll discuss IPython's tools to quickly access this information, namely the ``?`` character to explore documentation, the ``??`` characters to explore source code, and the Tab key for auto-completion.

# ## Accessing Documentation with ``?``
# 
# The Python language and its data science ecosystem is built with the user in mind, and one big part of that is access to documentation.
# Every Python object contains the reference to a string, known as a *doc string*, which in most cases will contain a concise summary of the object and how to use it.
# Python has a built-in ``help()`` function that can access this information and prints the results.
# For example, to see the documentation of the built-in ``len`` function, you can do the following:
# 
# ```ipython
# In [1]: help(len)
# Help on built-in function len in module builtins:
# 
# len(...)
#     len(object) -> integer
#     
#     Return the number of items of a sequence or mapping.
# ```
# 
# Depending on your interpreter, this information may be displayed as inline text, or in some separate pop-up window.
# 

# Because finding help on an object is so common and useful, IPython introduces the ``?`` character as a shorthand for accessing this documentation and other relevant information:
# 
# ```ipython
# In [2]: len?
# Type:        builtin_function_or_method
# String form: <built-in function len>
# Namespace:   Python builtin
# Docstring:
# len(object) -> integer
# 
# Return the number of items of a sequence or mapping.
# ```

# This notation works for just about anything, including object methods:
# 
# ```ipython
# In [3]: L = [1, 2, 3]
# In [4]: L.insert?
# Type:        builtin_function_or_method
# String form: <built-in method insert of list object at 0x1024b8ea8>
# Docstring:   L.insert(index, object) -- insert object before index
# ```
# 
# or even objects themselves, with the documentation from their type:
# 
# ```ipython
# In [5]: L?
# Type:        list
# String form: [1, 2, 3]
# Length:      3
# Docstring:
# list() -> new empty list
# list(iterable) -> new list initialized from iterable's items
# ```

# Importantly, this will even work for functions or other objects you create yourself!
# Here we'll define a small function with a docstring:
# 
# ```ipython
# In [6]: def square(a):
#   ....:     """Return the square of a."""
#   ....:     return a ** 2
#   ....:
# ```
# 
# Note that to create a docstring for our function, we simply placed a string literal in the first line.
# Because doc strings are usually multiple lines, by convention we used Python's triple-quote notation for multi-line strings.
# 

# Now we'll use the ``?`` mark to find this doc string:
# 
# ```ipython
# In [7]: square?
# Type:        function
# String form: <function square at 0x103713cb0>
# Definition:  square(a)
# Docstring:   Return the square of a.
# ```
# 
# This quick access to documentation via docstrings is one reason you should get in the habit of always adding such inline documentation to the code you write!

# ## Accessing Source Code with ``??``
# Because the Python language is so easily readable, another level of insight can usually be gained by reading the source code of the object you're curious about.
# IPython provides a shortcut to the source code with the double question mark (``??``):
# 
# ```ipython
# In [8]: square??
# Type:        function
# String form: <function square at 0x103713cb0>
# Definition:  square(a)
# Source:
# def square(a):
#     "Return the square of a"
#     return a ** 2
# ```
# 
# For simple functions like this, the double question-mark can give quick insight into the under-the-hood details.

# If you play with this much, you'll notice that sometimes the ``??`` suffix doesn't display any source code: this is generally because the object in question is not implemented in Python, but in C or some other compiled extension language.
# If this is the case, the ``??`` suffix gives the same output as the ``?`` suffix.
# You'll find this particularly with many of Python's built-in objects and types, for example ``len`` from above:
# 
# ```ipython
# In [9]: len??
# Type:        builtin_function_or_method
# String form: <built-in function len>
# Namespace:   Python builtin
# Docstring:
# len(object) -> integer
# 
# Return the number of items of a sequence or mapping.
# ```
# 
# Using ``?`` and/or ``??`` gives a powerful and quick interface for finding information about what any Python function or module does.

# ## Exploring Modules with Tab-Completion
# 
# IPython's other useful interface is the use of the tab key for auto-completion and exploration of the contents of objects, modules, and name-spaces.
# In the examples that follow, we'll use ``<TAB>`` to indicate when the Tab key should be pressed.
# 

# ### Tab-completion of object contents
# 
# Every Python object has various attributes and methods associated with it.
# Like with the ``help`` function discussed before, Python has a built-in ``dir`` function that returns a list of these, but the tab-completion interface is much easier to use in practice.
# To see a list of all available attributes of an object, you can type the name of the object followed by a period ("``.``") character and the Tab key:
# 
# ```ipython
# In [10]: L.<TAB>
# L.append   L.copy     L.extend   L.insert   L.remove   L.sort     
# L.clear    L.count    L.index    L.pop      L.reverse  
# ```
# 
# To narrow-down the list, you can type the first character or several characters of the name, and the Tab key will find the matching attributes and methods:
# 
# ```ipython
# In [10]: L.c<TAB>
# L.clear  L.copy   L.count  
# 
# In [10]: L.co<TAB>
# L.copy   L.count 
# ```
# 
# If there is only a single option, pressing the Tab key will complete the line for you.
# For example, the following will instantly be replaced with ``L.count``:
# 
# ```ipython
# In [10]: L.cou<TAB>
# 
# ```
# 
# Though Python has no strictly-enforced distinction between public/external attributes and private/internal attributes, by convention a preceding underscore is used to denote such methods.
# For clarity, these private methods and special methods are omitted from the list by default, but it's possible to list them by explicitly typing the underscore:
# 
# ```ipython
# In [10]: L._<TAB>
# L.__add__           L.__gt__            L.__reduce__
# L.__class__         L.__hash__          L.__reduce_ex__
# ```
# 
# For brevity, we've only shown the first couple lines of the output.
# Most of these are Python's special double-underscore methods (often nicknamed "dunder" methods).
# 

# ### Tab completion when importing
# 
# Tab completion is also useful when importing objects from packages.
# Here we'll use it to find all possible imports in the ``itertools`` package that start with ``co``:
# ```
# In [10]: from itertools import co<TAB>
# combinations                   compress
# combinations_with_replacement  count
# ```
# Similarly, you can use tab-completion to see which imports are available on your system (this will change depending on which third-party scripts and modules are visible to your Python session):
# ```
# In [10]: import <TAB>
# Display all 399 possibilities? (y or n)
# Crypto              dis                 py_compile
# Cython              distutils           pyclbr
# ...                 ...                 ...
# difflib             pwd                 zmq
# 
# In [10]: import h<TAB>
# hashlib             hmac                http         
# heapq               html                husl         
# ```
# (Note that for brevity, I did not print here all 399 importable packages and modules on my system.)
# 

# ### Beyond tab completion: wildcard matching
# 
# Tab completion is useful if you know the first few characters of the object or attribute you're looking for, but is little help if you'd like to match characters at the middle or end of the word.
# For this use-case, IPython provides a means of wildcard matching for names using the ``*`` character.
# 
# For example, we can use this to list every object in the namespace that ends with ``Warning``:
# 
# ```ipython
# In [10]: *Warning?
# BytesWarning                  RuntimeWarning
# DeprecationWarning            SyntaxWarning
# FutureWarning                 UnicodeWarning
# ImportWarning                 UserWarning
# PendingDeprecationWarning     Warning
# ResourceWarning
# ```
# 
# Notice that the ``*`` character matches any string, including the empty string.
# 
# Similarly, suppose we are looking for a string method that contains the word ``find`` somewhere in its name.
# We can search for it this way:
# 
# ```ipython
# In [10]: str.*find*?
# str.find
# str.rfind
# ```
# 
# I find this type of flexible wildcard search can be very useful for finding a particular command when getting to know a new package or reacquainting myself with a familiar one.

# <!--NAVIGATION-->
# < [IPython: Beyond Normal Python](01.00-IPython-Beyond-Normal-Python.ipynb) | [Contents](Index.ipynb) | [Keyboard Shortcuts in the IPython Shell](01.02-Shell-Keyboard-Shortcuts.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [Density and Contour Plots](04.04-Density-and-Contour-Plots.ipynb) | [Contents](Index.ipynb) | [Customizing Plot Legends](04.06-Customizing-Legends.ipynb) >
# 

# # Histograms, Binnings, and Density
# 

# A simple histogram can be a great first step in understanding a dataset.
# Earlier, we saw a preview of Matplotlib's histogram function (see [Comparisons, Masks, and Boolean Logic](02.06-Boolean-Arrays-and-Masks.ipynb)), which creates a basic histogram in one line, once the normal boiler-plate imports are done:
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

data = np.random.randn(1000)


plt.hist(data);


# The ``hist()`` function has many options to tune both the calculation and the display; 
# here's an example of a more customized histogram:
# 

plt.hist(data, bins=30, normed=True, alpha=0.5,
         histtype='stepfilled', color='steelblue',
         edgecolor='none');


# The ``plt.hist`` docstring has more information on other customization options available.
# I find this combination of ``histtype='stepfilled'`` along with some transparency ``alpha`` to be very useful when comparing histograms of several distributions:
# 

x1 = np.random.normal(0, 0.8, 1000)
x2 = np.random.normal(-2, 1, 1000)
x3 = np.random.normal(3, 2, 1000)

kwargs = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=40)

plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs);


# If you would like to simply compute the histogram (that is, count the number of points in a given bin) and not display it, the ``np.histogram()`` function is available:
# 

counts, bin_edges = np.histogram(data, bins=5)
print(counts)


# ## Two-Dimensional Histograms and Binnings
# 
# Just as we create histograms in one dimension by dividing the number-line into bins, we can also create histograms in two-dimensions by dividing points among two-dimensional bins.
# We'll take a brief look at several ways to do this here.
# We'll start by defining some data—an ``x`` and ``y`` array drawn from a multivariate Gaussian distribution:
# 

mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T


# ### ``plt.hist2d``: Two-dimensional histogram
# 
# One straightforward way to plot a two-dimensional histogram is to use Matplotlib's ``plt.hist2d`` function:
# 

plt.hist2d(x, y, bins=30, cmap='Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')


# Just as with ``plt.hist``, ``plt.hist2d`` has a number of extra options to fine-tune the plot and the binning, which are nicely outlined in the function docstring.
# Further, just as ``plt.hist`` has a counterpart in ``np.histogram``, ``plt.hist2d`` has a counterpart in ``np.histogram2d``, which can be used as follows:
# 

counts, xedges, yedges = np.histogram2d(x, y, bins=30)


# For the generalization of this histogram binning in dimensions higher than two, see the ``np.histogramdd`` function.
# 

# ### ``plt.hexbin``: Hexagonal binnings
# 
# The two-dimensional histogram creates a tesselation of squares across the axes.
# Another natural shape for such a tesselation is the regular hexagon.
# For this purpose, Matplotlib provides the ``plt.hexbin`` routine, which will represents a two-dimensional dataset binned within a grid of hexagons:
# 

plt.hexbin(x, y, gridsize=30, cmap='Blues')
cb = plt.colorbar(label='count in bin')


# ``plt.hexbin`` has a number of interesting options, including the ability to specify weights for each point, and to change the output in each bin to any NumPy aggregate (mean of weights, standard deviation of weights, etc.).
# 

# ### Kernel density estimation
# 
# Another common method of evaluating densities in multiple dimensions is *kernel density estimation* (KDE).
# This will be discussed more fully in [In-Depth: Kernel Density Estimation](05.13-Kernel-Density-Estimation.ipynb), but for now we'll simply mention that KDE can be thought of as a way to "smear out" the points in space and add up the result to obtain a smooth function.
# One extremely quick and simple KDE implementation exists in the ``scipy.stats`` package.
# Here is a quick example of using the KDE on this data:
# 

from scipy.stats import gaussian_kde

# fit an array of size [Ndim, Nsamples]
data = np.vstack([x, y])
kde = gaussian_kde(data)

# evaluate on a regular grid
xgrid = np.linspace(-3.5, 3.5, 40)
ygrid = np.linspace(-6, 6, 40)
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

# Plot the result as an image
plt.imshow(Z.reshape(Xgrid.shape),
           origin='lower', aspect='auto',
           extent=[-3.5, 3.5, -6, 6],
           cmap='Blues')
cb = plt.colorbar()
cb.set_label("density")


# KDE has a smoothing length that effectively slides the knob between detail and smoothness (one example of the ubiquitous bias–variance trade-off).
# The literature on choosing an appropriate smoothing length is vast: ``gaussian_kde`` uses a rule-of-thumb to attempt to find a nearly optimal smoothing length for the input data.
# 
# Other KDE implementations are available within the SciPy ecosystem, each with its own strengths and weaknesses; see, for example, ``sklearn.neighbors.KernelDensity`` and ``statsmodels.nonparametric.kernel_density.KDEMultivariate``.
# For visualizations based on KDE, using Matplotlib tends to be overly verbose.
# The Seaborn library, discussed in [Visualization With Seaborn](04.14-Visualization-With-Seaborn.ipynb), provides a much more terse API for creating KDE-based visualizations.
# 

# <!--NAVIGATION-->
# < [Density and Contour Plots](04.04-Density-and-Contour-Plots.ipynb) | [Contents](Index.ipynb) | [Customizing Plot Legends](04.06-Customizing-Legends.ipynb) >
# 

# # Python Data Science Handbook
# 
# *Jake VanderPlas*
# 
# ![Book Cover](figures/PDSH-cover.png)

# This is the Jupyter notebook version of the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!
# 

# ## Table of Contents
# 
# ### [Preface](00.00-Preface.ipynb)
# 
# ### [1. IPython: Beyond Normal Python](01.00-IPython-Beyond-Normal-Python.ipynb)
# - [Help and Documentation in IPython](01.01-Help-And-Documentation.ipynb)
# - [Keyboard Shortcuts in the IPython Shell](01.02-Shell-Keyboard-Shortcuts.ipynb)
# - [IPython Magic Commands](01.03-Magic-Commands.ipynb)
# - [Input and Output History](01.04-Input-Output-History.ipynb)
# - [IPython and Shell Commands](01.05-IPython-And-Shell-Commands.ipynb)
# - [Errors and Debugging](01.06-Errors-and-Debugging.ipynb)
# - [Profiling and Timing Code](01.07-Timing-and-Profiling.ipynb)
# - [More IPython Resources](01.08-More-IPython-Resources.ipynb)
# 
# ### [2. Introduction to NumPy](02.00-Introduction-to-NumPy.ipynb)
# - [Understanding Data Types in Python](02.01-Understanding-Data-Types.ipynb)
# - [The Basics of NumPy Arrays](02.02-The-Basics-Of-NumPy-Arrays.ipynb)
# - [Computation on NumPy Arrays: Universal Functions](02.03-Computation-on-arrays-ufuncs.ipynb)
# - [Aggregations: Min, Max, and Everything In Between](02.04-Computation-on-arrays-aggregates.ipynb)
# - [Computation on Arrays: Broadcasting](02.05-Computation-on-arrays-broadcasting.ipynb)
# - [Comparisons, Masks, and Boolean Logic](02.06-Boolean-Arrays-and-Masks.ipynb)
# - [Fancy Indexing](02.07-Fancy-Indexing.ipynb)
# - [Sorting Arrays](02.08-Sorting.ipynb)
# - [Structured Data: NumPy's Structured Arrays](02.09-Structured-Data-NumPy.ipynb)
# 
# ### [3. Data Manipulation with Pandas](03.00-Introduction-to-Pandas.ipynb)
# - [Introducing Pandas Objects](03.01-Introducing-Pandas-Objects.ipynb)
# - [Data Indexing and Selection](03.02-Data-Indexing-and-Selection.ipynb)
# - [Operating on Data in Pandas](03.03-Operations-in-Pandas.ipynb)
# - [Handling Missing Data](03.04-Missing-Values.ipynb)
# - [Hierarchical Indexing](03.05-Hierarchical-Indexing.ipynb)
# - [Combining Datasets: Concat and Append](03.06-Concat-And-Append.ipynb)
# - [Combining Datasets: Merge and Join](03.07-Merge-and-Join.ipynb)
# - [Aggregation and Grouping](03.08-Aggregation-and-Grouping.ipynb)
# - [Pivot Tables](03.09-Pivot-Tables.ipynb)
# - [Vectorized String Operations](03.10-Working-With-Strings.ipynb)
# - [Working with Time Series](03.11-Working-with-Time-Series.ipynb)
# - [High-Performance Pandas: eval() and query()](03.12-Performance-Eval-and-Query.ipynb)
# - [Further Resources](03.13-Further-Resources.ipynb)
# 
# ### [4. Visualization with Matplotlib](04.00-Introduction-To-Matplotlib.ipynb)
# - [Simple Line Plots](04.01-Simple-Line-Plots.ipynb)
# - [Simple Scatter Plots](04.02-Simple-Scatter-Plots.ipynb)
# - [Visualizing Errors](04.03-Errorbars.ipynb)
# - [Density and Contour Plots](04.04-Density-and-Contour-Plots.ipynb)
# - [Histograms, Binnings, and Density](04.05-Histograms-and-Binnings.ipynb)
# - [Customizing Plot Legends](04.06-Customizing-Legends.ipynb)
# - [Customizing Colorbars](04.07-Customizing-Colorbars.ipynb)
# - [Multiple Subplots](04.08-Multiple-Subplots.ipynb)
# - [Text and Annotation](04.09-Text-and-Annotation.ipynb)
# - [Customizing Ticks](04.10-Customizing-Ticks.ipynb)
# - [Customizing Matplotlib: Configurations and Stylesheets](04.11-Settings-and-Stylesheets.ipynb)
# - [Three-Dimensional Plotting in Matplotlib](04.12-Three-Dimensional-Plotting.ipynb)
# - [Geographic Data with Basemap](04.13-Geographic-Data-With-Basemap.ipynb)
# - [Visualization with Seaborn](04.14-Visualization-With-Seaborn.ipynb)
# - [Further Resources](04.15-Further-Resources.ipynb)
# 
# ### [5. Machine Learning](05.00-Machine-Learning.ipynb)
# - [What Is Machine Learning?](05.01-What-Is-Machine-Learning.ipynb)
# - [Introducing Scikit-Learn](05.02-Introducing-Scikit-Learn.ipynb)
# - [Hyperparameters and Model Validation](05.03-Hyperparameters-and-Model-Validation.ipynb)
# - [Feature Engineering](05.04-Feature-Engineering.ipynb)
# - [In Depth: Naive Bayes Classification](05.05-Naive-Bayes.ipynb)
# - [In Depth: Linear Regression](05.06-Linear-Regression.ipynb)
# - [In-Depth: Support Vector Machines](05.07-Support-Vector-Machines.ipynb)
# - [In-Depth: Decision Trees and Random Forests](05.08-Random-Forests.ipynb)
# - [In Depth: Principal Component Analysis](05.09-Principal-Component-Analysis.ipynb)
# - [In-Depth: Manifold Learning](05.10-Manifold-Learning.ipynb)
# - [In Depth: k-Means Clustering](05.11-K-Means.ipynb)
# - [In Depth: Gaussian Mixture Models](05.12-Gaussian-Mixtures.ipynb)
# - [In-Depth: Kernel Density Estimation](05.13-Kernel-Density-Estimation.ipynb)
# - [Application: A Face Detection Pipeline](05.14-Image-Features.ipynb)
# - [Further Machine Learning Resources](05.15-Learning-More.ipynb)
# 
# ### [Appendix: Figure Code](06.00-Figure-Code.ipynb)
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [IPython and Shell Commands](01.05-IPython-And-Shell-Commands.ipynb) | [Contents](Index.ipynb) | [Profiling and Timing Code](01.07-Timing-and-Profiling.ipynb) >
# 

# # Errors and Debugging
# 
# Code development and data analysis always require a bit of trial and error, and IPython contains tools to streamline this process.
# This section will briefly cover some options for controlling Python's exception reporting, followed by exploring tools for debugging errors in code.
# 

# ## Controlling Exceptions: ``%xmode``
# 
# Most of the time when a Python script fails, it will raise an Exception.
# When the interpreter hits one of these exceptions, information about the cause of the error can be found in the *traceback*, which can be accessed from within Python.
# With the ``%xmode`` magic function, IPython allows you to control the amount of information printed when the exception is raised.
# Consider the following code:
# 

def func1(a, b):
    return a / b

def func2(x):
    a = x
    b = x - 1
    return func1(a, b)


func2(1)


# Calling ``func2`` results in an error, and reading the printed trace lets us see exactly what happened.
# By default, this trace includes several lines showing the context of each step that led to the error.
# Using the ``%xmode`` magic function (short for *Exception mode*), we can change what information is printed.
# 
# ``%xmode`` takes a single argument, the mode, and there are three possibilities: ``Plain``, ``Context``, and ``Verbose``.
# The default is ``Context``, and gives output like that just shown before.
# ``Plain`` is more compact and gives less information:
# 

get_ipython().magic('xmode Plain')


func2(1)


# The ``Verbose`` mode adds some extra information, including the arguments to any functions that are called:
# 

get_ipython().magic('xmode Verbose')


func2(1)


# This extra information can help narrow-in on why the exception is being raised.
# So why not use the ``Verbose`` mode all the time?
# As code gets complicated, this kind of traceback can get extremely long.
# Depending on the context, sometimes the brevity of ``Default`` mode is easier to work with.

# ## Debugging: When Reading Tracebacks Is Not Enough
# 
# The standard Python tool for interactive debugging is ``pdb``, the Python debugger.
# This debugger lets the user step through the code line by line in order to see what might be causing a more difficult error.
# The IPython-enhanced version of this is ``ipdb``, the IPython debugger.
# 
# There are many ways to launch and use both these debuggers; we won't cover them fully here.
# Refer to the online documentation of these two utilities to learn more.
# 
# In IPython, perhaps the most convenient interface to debugging is the ``%debug`` magic command.
# If you call it after hitting an exception, it will automatically open an interactive debugging prompt at the point of the exception.
# The ``ipdb`` prompt lets you explore the current state of the stack, explore the available variables, and even run Python commands!
# 
# Let's look at the most recent exception, then do some basic tasks–print the values of ``a`` and ``b``, and type ``quit`` to quit the debugging session:
# 

get_ipython().magic('debug')


# The interactive debugger allows much more than this, though–we can even step up and down through the stack and explore the values of variables there:
# 

get_ipython().magic('debug')


# This allows you to quickly find out not only what caused the error, but what function calls led up to the error.
# 
# If you'd like the debugger to launch automatically whenever an exception is raised, you can use the ``%pdb`` magic function to turn on this automatic behavior:
# 

get_ipython().magic('xmode Plain')
get_ipython().magic('pdb on')
func2(1)


# Finally, if you have a script that you'd like to run from the beginning in interactive mode, you can run it with the command ``%run -d``, and use the ``next`` command to step through the lines of code interactively.
# 

# ### Partial list of debugging commands
# 
# There are many more available commands for interactive debugging than we've listed here; the following table contains a description of some of the more common and useful ones:
# 
# | Command         |  Description                                                |
# |-----------------|-------------------------------------------------------------|
# | ``list``        | Show the current location in the file                       |
# | ``h(elp)``      | Show a list of commands, or find help on a specific command |
# | ``q(uit)``      | Quit the debugger and the program                           |
# | ``c(ontinue)``  | Quit the debugger, continue in the program                  |
# | ``n(ext)``      | Go to the next step of the program                          |
# | ``<enter>``     | Repeat the previous command                                 |
# | ``p(rint)``     | Print variables                                             |
# | ``s(tep)``      | Step into a subroutine                                      |
# | ``r(eturn)``    | Return out of a subroutine                                  |
# 
# For more information, use the ``help`` command in the debugger, or take a look at ``ipdb``'s [online documentation](https://github.com/gotcha/ipdb).
# 

# <!--NAVIGATION-->
# < [IPython and Shell Commands](01.05-IPython-And-Shell-Commands.ipynb) | [Contents](Index.ipynb) | [Profiling and Timing Code](01.07-Timing-and-Profiling.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [Combining Datasets: Concat and Append](03.06-Concat-And-Append.ipynb) | [Contents](Index.ipynb) | [Aggregation and Grouping](03.08-Aggregation-and-Grouping.ipynb) >
# 

# # Combining Datasets: Merge and Join
# 

# One essential feature offered by Pandas is its high-performance, in-memory join and merge operations.
# If you have ever worked with databases, you should be familiar with this type of data interaction.
# The main interface for this is the ``pd.merge`` function, and we'll see few examples of how this can work in practice.
# 
# For convenience, we will start by redefining the ``display()`` functionality from the previous section:
# 

import pandas as pd
import numpy as np

class display(object):
    """Display HTML representation of multiple objects"""
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""
    def __init__(self, *args):
        self.args = args
        
    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                         for a in self.args)
    
    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a))
                           for a in self.args)


# ## Relational Algebra
# 
# The behavior implemented in ``pd.merge()`` is a subset of what is known as *relational algebra*, which is a formal set of rules for manipulating relational data, and forms the conceptual foundation of operations available in most databases.
# The strength of the relational algebra approach is that it proposes several primitive operations, which become the building blocks of more complicated operations on any dataset.
# With this lexicon of fundamental operations implemented efficiently in a database or other program, a wide range of fairly complicated composite operations can be performed.
# 
# Pandas implements several of these fundamental building-blocks in the ``pd.merge()`` function and the related ``join()`` method of ``Series`` and ``Dataframe``s.
# As we will see, these let you efficiently link data from different sources.
# 

# ## Categories of Joins
# 
# The ``pd.merge()`` function implements a number of types of joins: the *one-to-one*, *many-to-one*, and *many-to-many* joins.
# All three types of joins are accessed via an identical call to the ``pd.merge()`` interface; the type of join performed depends on the form of the input data.
# Here we will show simple examples of the three types of merges, and discuss detailed options further below.
# 

# ### One-to-one joins
# 
# Perhaps the simplest type of merge expresion is the one-to-one join, which is in many ways very similar to the column-wise concatenation seen in [Combining Datasets: Concat & Append](03.06-Concat-And-Append.ipynb).
# As a concrete example, consider the following two ``DataFrames`` which contain information on several employees in a company:
# 

df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})
display('df1', 'df2')


# To combine this information into a single ``DataFrame``, we can use the ``pd.merge()`` function:
# 

df3 = pd.merge(df1, df2)
df3


# The ``pd.merge()`` function recognizes that each ``DataFrame`` has an "employee" column, and automatically joins using this column as a key.
# The result of the merge is a new ``DataFrame`` that combines the information from the two inputs.
# Notice that the order of entries in each column is not necessarily maintained: in this case, the order of the "employee" column differs between ``df1`` and ``df2``, and the ``pd.merge()`` function correctly accounts for this.
# Additionally, keep in mind that the merge in general discards the index, except in the special case of merges by index (see the ``left_index`` and ``right_index`` keywords, discussed momentarily).
# 

# ### Many-to-one joins
# 

# Many-to-one joins are joins in which one of the two key columns contains duplicate entries.
# For the many-to-one case, the resulting ``DataFrame`` will preserve those duplicate entries as appropriate.
# Consider the following example of a many-to-one join:
# 

df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                    'supervisor': ['Carly', 'Guido', 'Steve']})
display('df3', 'df4', 'pd.merge(df3, df4)')


# The resulting ``DataFrame`` has an aditional column with the "supervisor" information, where the information is repeated in one or more locations as required by the inputs.
# 

# ### Many-to-many joins
# 

# Many-to-many joins are a bit confusing conceptually, but are nevertheless well defined.
# If the key column in both the left and right array contains duplicates, then the result is a many-to-many merge.
# This will be perhaps most clear with a concrete example.
# Consider the following, where we have a ``DataFrame`` showing one or more skills associated with a particular group.
# By performing a many-to-many join, we can recover the skills associated with any individual person:
# 

df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
                              'Engineering', 'Engineering', 'HR', 'HR'],
                    'skills': ['math', 'spreadsheets', 'coding', 'linux',
                               'spreadsheets', 'organization']})
display('df1', 'df5', "pd.merge(df1, df5)")


# These three types of joins can be used with other Pandas tools to implement a wide array of functionality.
# But in practice, datasets are rarely as clean as the one we're working with here.
# In the following section we'll consider some of the options provided by ``pd.merge()`` that enable you to tune how the join operations work.
# 

# ## Specification of the Merge Key
# 

# We've already seen the default behavior of ``pd.merge()``: it looks for one or more matching column names between the two inputs, and uses this as the key.
# However, often the column names will not match so nicely, and ``pd.merge()`` provides a variety of options for handling this.
# 

# ### The ``on`` keyword
# 
# Most simply, you can explicitly specify the name of the key column using the ``on`` keyword, which takes a column name or a list of column names:
# 

display('df1', 'df2', "pd.merge(df1, df2, on='employee')")


# This option works only if both the left and right ``DataFrame``s have the specified column name.
# 

# ### The ``left_on`` and ``right_on`` keywords
# 
# At times you may wish to merge two datasets with different column names; for example, we may have a dataset in which the employee name is labeled as "name" rather than "employee".
# In this case, we can use the ``left_on`` and ``right_on`` keywords to specify the two column names:
# 

df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'salary': [70000, 80000, 120000, 90000]})
display('df1', 'df3', 'pd.merge(df1, df3, left_on="employee", right_on="name")')


# The result has a redundant column that we can drop if desired–for example, by using the ``drop()`` method of ``DataFrame``s:
# 

pd.merge(df1, df3, left_on="employee", right_on="name").drop('name', axis=1)


# ### The ``left_index`` and ``right_index`` keywords
# 
# Sometimes, rather than merging on a column, you would instead like to merge on an index.
# For example, your data might look like this:
# 

df1a = df1.set_index('employee')
df2a = df2.set_index('employee')
display('df1a', 'df2a')


# You can use the index as the key for merging by specifying the ``left_index`` and/or ``right_index`` flags in ``pd.merge()``:
# 

display('df1a', 'df2a',
        "pd.merge(df1a, df2a, left_index=True, right_index=True)")


# For convenience, ``DataFrame``s implement the ``join()`` method, which performs a merge that defaults to joining on indices:
# 

display('df1a', 'df2a', 'df1a.join(df2a)')


# If you'd like to mix indices and columns, you can combine ``left_index`` with ``right_on`` or ``left_on`` with ``right_index`` to get the desired behavior:
# 

display('df1a', 'df3', "pd.merge(df1a, df3, left_index=True, right_on='name')")


# All of these options also work with multiple indices and/or multiple columns; the interface for this behavior is very intuitive.
# For more information on this, see the ["Merge, Join, and Concatenate" section](http://pandas.pydata.org/pandas-docs/stable/merging.html) of the Pandas documentation.
# 

# ## Specifying Set Arithmetic for Joins
# 

# In all the preceding examples we have glossed over one important consideration in performing a join: the type of set arithmetic used in the join.
# This comes up when a value appears in one key column but not the other. Consider this example:
# 

df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
                    'food': ['fish', 'beans', 'bread']},
                   columns=['name', 'food'])
df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
                    'drink': ['wine', 'beer']},
                   columns=['name', 'drink'])
display('df6', 'df7', 'pd.merge(df6, df7)')


# Here we have merged two datasets that have only a single "name" entry in common: Mary.
# By default, the result contains the *intersection* of the two sets of inputs; this is what is known as an *inner join*.
# We can specify this explicitly using the ``how`` keyword, which defaults to ``"inner"``:
# 

pd.merge(df6, df7, how='inner')


# Other options for the ``how`` keyword are ``'outer'``, ``'left'``, and ``'right'``.
# An *outer join* returns a join over the union of the input columns, and fills in all missing values with NAs:
# 

display('df6', 'df7', "pd.merge(df6, df7, how='outer')")


# The *left join* and *right join* return joins over the left entries and right entries, respectively.
# For example:
# 

display('df6', 'df7', "pd.merge(df6, df7, how='left')")


# The output rows now correspond to the entries in the left input. Using
# ``how='right'`` works in a similar manner.
# 
# All of these options can be applied straightforwardly to any of the preceding join types.
# 

# ## Overlapping Column Names: The ``suffixes`` Keyword
# 

# Finally, you may end up in a case where your two input ``DataFrame``s have conflicting column names.
# Consider this example:
# 

df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [1, 2, 3, 4]})
df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [3, 1, 4, 2]})
display('df8', 'df9', 'pd.merge(df8, df9, on="name")')


# Because the output would have two conflicting column names, the merge function automatically appends a suffix ``_x`` or ``_y`` to make the output columns unique.
# If these defaults are inappropriate, it is possible to specify a custom suffix using the ``suffixes`` keyword:
# 

display('df8', 'df9', 'pd.merge(df8, df9, on="name", suffixes=["_L", "_R"])')


# These suffixes work in any of the possible join patterns, and work also if there are multiple overlapping columns.
# 

# For more information on these patterns, see [Aggregation and Grouping](03.08-Aggregation-and-Grouping.ipynb) where we dive a bit deeper into relational algebra.
# Also see the [Pandas "Merge, Join and Concatenate" documentation](http://pandas.pydata.org/pandas-docs/stable/merging.html) for further discussion of these topics.
# 

# ## Example: US States Data
# 
# Merge and join operations come up most often when combining data from different sources.
# Here we will consider an example of some data about US states and their populations.
# The data files can be found at http://github.com/jakevdp/data-USstates/:
# 

# Following are shell commands to download the data
# !curl -O https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-population.csv
# !curl -O https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-areas.csv
# !curl -O https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-abbrevs.csv


# Let's take a look at the three datasets, using the Pandas ``read_csv()`` function:
# 

pop = pd.read_csv('data/state-population.csv')
areas = pd.read_csv('data/state-areas.csv')
abbrevs = pd.read_csv('data/state-abbrevs.csv')

display('pop.head()', 'areas.head()', 'abbrevs.head()')


# Given this information, say we want to compute a relatively straightforward result: rank US states and territories by their 2010 population density.
# We clearly have the data here to find this result, but we'll have to combine the datasets to find the result.
# 
# We'll start with a many-to-one merge that will give us the full state name within the population ``DataFrame``.
# We want to merge based on the ``state/region``  column of ``pop``, and the ``abbreviation`` column of ``abbrevs``.
# We'll use ``how='outer'`` to make sure no data is thrown away due to mismatched labels.
# 

merged = pd.merge(pop, abbrevs, how='outer',
                  left_on='state/region', right_on='abbreviation')
merged = merged.drop('abbreviation', 1) # drop duplicate info
merged.head()


# Let's double-check whether there were any mismatches here, which we can do by looking for rows with nulls:
# 

merged.isnull().any()


# Some of the ``population`` info is null; let's figure out which these are!
# 

merged[merged['population'].isnull()].head()


# It appears that all the null population values are from Puerto Rico prior to the year 2000; this is likely due to this data not being available from the original source.
# 
# More importantly, we see also that some of the new ``state`` entries are also null, which means that there was no corresponding entry in the ``abbrevs`` key!
# Let's figure out which regions lack this match:
# 

merged.loc[merged['state'].isnull(), 'state/region'].unique()


# We can quickly infer the issue: our population data includes entries for Puerto Rico (PR) and the United States as a whole (USA), while these entries do not appear in the state abbreviation key.
# We can fix these quickly by filling in appropriate entries:
# 

merged.loc[merged['state/region'] == 'PR', 'state'] = 'Puerto Rico'
merged.loc[merged['state/region'] == 'USA', 'state'] = 'United States'
merged.isnull().any()


# No more nulls in the ``state`` column: we're all set!
# 
# Now we can merge the result with the area data using a similar procedure.
# Examining our results, we will want to join on the ``state`` column in both:
# 

final = pd.merge(merged, areas, on='state', how='left')
final.head()


# Again, let's check for nulls to see if there were any mismatches:
# 

final.isnull().any()


# There are nulls in the ``area`` column; we can take a look to see which regions were ignored here:
# 

final['state'][final['area (sq. mi)'].isnull()].unique()


# We see that our ``areas`` ``DataFrame`` does not contain the area of the United States as a whole.
# We could insert the appropriate value (using the sum of all state areas, for instance), but in this case we'll just drop the null values because the population density of the entire United States is not relevant to our current discussion:
# 

final.dropna(inplace=True)
final.head()


# Now we have all the data we need. To answer the question of interest, let's first select the portion of the data corresponding with the year 2000, and the total population.
# We'll use the ``query()`` function to do this quickly (this requires the ``numexpr`` package to be installed; see [High-Performance Pandas: ``eval()`` and ``query()``](03.12-Performance-Eval-and-Query.ipynb)):
# 

data2010 = final.query("year == 2010 & ages == 'total'")
data2010.head()


# Now let's compute the population density and display it in order.
# We'll start by re-indexing our data on the state, and then compute the result:
# 

data2010.set_index('state', inplace=True)
density = data2010['population'] / data2010['area (sq. mi)']


density.sort_values(ascending=False, inplace=True)
density.head()


# The result is a ranking of US states plus Washington, DC, and Puerto Rico in order of their 2010 population density, in residents per square mile.
# We can see that by far the densest region in this dataset is Washington, DC (i.e., the District of Columbia); among states, the densest is New Jersey.
# 
# We can also check the end of the list:
# 

density.tail()


# We see that the least dense state, by far, is Alaska, averaging slightly over one resident per square mile.
# 
# This type of messy data merging is a common task when trying to answer questions using real-world data sources.
# I hope that this example has given you an idea of the ways you can combine tools we've covered in order to gain insight from your data!
# 

# <!--NAVIGATION-->
# < [Combining Datasets: Concat and Append](03.06-Concat-And-Append.ipynb) | [Contents](Index.ipynb) | [Aggregation and Grouping](03.08-Aggregation-and-Grouping.ipynb) >
# 

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 

# <!--NAVIGATION-->
# < [Simple Line Plots](04.01-Simple-Line-Plots.ipynb) | [Contents](Index.ipynb) | [Visualizing Errors](04.03-Errorbars.ipynb) >
# 

# # Simple Scatter Plots
# 

# Another commonly used plot type is the simple scatter plot, a close cousin of the line plot.
# Instead of points being joined by line segments, here the points are represented individually with a dot, circle, or other shape.
# We’ll start by setting up the notebook for plotting and importing the functions we will use:
# 

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np


# ## Scatter Plots with ``plt.plot``
# 
# In the previous section we looked at ``plt.plot``/``ax.plot`` to produce line plots.
# It turns out that this same function can produce scatter plots as well:
# 

x = np.linspace(0, 10, 30)
y = np.sin(x)

plt.plot(x, y, 'o', color='black');


# The third argument in the function call is a character that represents the type of symbol used for the plotting. Just as you can specify options such as ``'-'``, ``'--'`` to control the line style, the marker style has its own set of short string codes. The full list of available symbols can be seen in the documentation of ``plt.plot``, or in Matplotlib's online documentation. Most of the possibilities are fairly intuitive, and we'll show a number of the more common ones here:
# 

rng = np.random.RandomState(0)
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
    plt.plot(rng.rand(5), rng.rand(5), marker,
             label="marker='{0}'".format(marker))
plt.legend(numpoints=1)
plt.xlim(0, 1.8);


# For even more possibilities, these character codes can be used together with line and color codes to plot points along with a line connecting them:
# 

plt.plot(x, y, '-ok');


# Additional keyword arguments to ``plt.plot`` specify a wide range of properties of the lines and markers:
# 

plt.plot(x, y, '-p', color='gray',
         markersize=15, linewidth=4,
         markerfacecolor='white',
         markeredgecolor='gray',
         markeredgewidth=2)
plt.ylim(-1.2, 1.2);


# This type of flexibility in the ``plt.plot`` function allows for a wide variety of possible visualization options.
# For a full description of the options available, refer to the ``plt.plot`` documentation.
# 

# ## Scatter Plots with ``plt.scatter``
# 
# A second, more powerful method of creating scatter plots is the ``plt.scatter`` function, which can be used very similarly to the ``plt.plot`` function:
# 

plt.scatter(x, y, marker='o');


# The primary difference of ``plt.scatter`` from ``plt.plot`` is that it can be used to create scatter plots where the properties of each individual point (size, face color, edge color, etc.) can be individually controlled or mapped to data.
# 
# Let's show this by creating a random scatter plot with points of many colors and sizes.
# In order to better see the overlapping results, we'll also use the ``alpha`` keyword to adjust the transparency level:
# 

rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)

plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,
            cmap='viridis')
plt.colorbar();  # show color scale


# Notice that the color argument is automatically mapped to a color scale (shown here by the ``colorbar()`` command), and that the size argument is given in pixels.
# In this way, the color and size of points can be used to convey information in the visualization, in order to visualize multidimensional data.
# 
# For example, we might use the Iris data from Scikit-Learn, where each sample is one of three types of flowers that has had the size of its petals and sepals carefully measured:
# 

from sklearn.datasets import load_iris
iris = load_iris()
features = iris.data.T

plt.scatter(features[0], features[1], alpha=0.2,
            s=100*features[3], c=iris.target, cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1]);


# We can see that this scatter plot has given us the ability to simultaneously explore four different dimensions of the data:
# the (x, y) location of each point corresponds to the sepal length and width, the size of the point is related to the petal width, and the color is related to the particular species of flower.
# Multicolor and multifeature scatter plots like this can be useful for both exploration and presentation of data.
# 

# ## ``plot`` Versus ``scatter``: A Note on Efficiency
# 
# Aside from the different features available in ``plt.plot`` and ``plt.scatter``, why might you choose to use one over the other? While it doesn't matter as much for small amounts of data, as datasets get larger than a few thousand points, ``plt.plot`` can be noticeably more efficient than ``plt.scatter``.
# The reason is that ``plt.scatter`` has the capability to render a different size and/or color for each point, so the renderer must do the extra work of constructing each point individually.
# In ``plt.plot``, on the other hand, the points are always essentially clones of each other, so the work of determining the appearance of the points is done only once for the entire set of data.
# For large datasets, the difference between these two can lead to vastly different performance, and for this reason, ``plt.plot`` should be preferred over ``plt.scatter`` for large datasets.
# 

# <!--NAVIGATION-->
# < [Simple Line Plots](04.01-Simple-Line-Plots.ipynb) | [Contents](Index.ipynb) | [Visualizing Errors](04.03-Errorbars.ipynb) >
# 

