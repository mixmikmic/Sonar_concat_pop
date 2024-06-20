# <DIV ALIGN=CENTER>
# 
# # Introduction to Pandas
# ## Professor Robert J. Brunner
#   
# </DIV>  
# -----
# -----
# 

# ## Introduction
# 
# One of the early criticisms of many in the data science arena of the
# Python language was the lack of useful data structures for performing
# data analysis tasks. This stemmed in part from comparisons between the R
# language and Python, since R has a built-in _DataFrame_ object that
# greatly simplified many data analysis tasks. This deficiency was
# addressed in 2008 by Wes McKinney with the creation of [Pandas][1] (the
# name was originally an abbreviation of Panel datas). To quote the Pandas
# documentation:
# 
# >Python has long been great for data munging and preparation, but less
# >so for data analysis and modeling. pandas helps fill this gap, enabling
# >you to carry out your entire data analysis workflow in Python without
# >having to switch to a more domain specific language like R.
# 
# Pandas introduces several new data structures like the `Series`,
# `DataFrame`, and `Panel` that build on top of existing
# tools like `numpy` to speed-up data analysis tasks. Pandas also provides
# efficient mechanisms for moving data between in memory representations
# and different data formats including CSV and text files, JSON files, SQL
# databases, HDF5 format files, and even Excel spreadsheets. Pandas also
# provides support for dealing with missing or incomplete data and
# aggregating or grouping data.
# 
# -----
# [1]: http://pandas.pydata.org
# 

# ## Brief introduction to Pandas
# 
# Before using Pandas, we must first import the Pandas library:
# 
#     import pandas as pd
# 
# Second, we simply create and use the appropriate Pandas data structure.
# The two most important data structures for typical data science tasks
# are the `Series` and the `DataFrame`:
# 
# 1. `Series`: a one-dimensional labeled array that can hold any data type
# such as integers, floating-point numbers, strings, or Python objects. A
# `Series` has both a data column and a label column called the _index_.
# 
# 2. `DataFrame`: a two-dimensional labeled data structure with columns
# that can be hold different data types, similar to a spreadsheet or
# relational database table. 
# 
# Pandas also provides a date/time data structure sometimes refereed to as
# a `TimeSeries` and a three-dimensional data structure known as a
# `Panel`. 
# 
# ### `Series`
# 
# A `Series` is useful to hold data that should be accesible by using a
# specific label. You create a `Series` by passing in an appropriate data
# set along with an optional index:
# 
#     values = pd.Series(data, index=idx)
# 
# The index varies depending on the type of data passed in to create the
# `Series:
# 
# - if data is a NumPy array, the index should be the same length as the
# data array. If no index is provided one will be created that enables
# integer access that mirrors a traditional NumPy array (i.e., zero
# indexed). 
# 
# - if data is a Python dictionary, `idx` can contain specific labels to
# indicate which values in the dictionary should be used to create the
# `Series`. If no index is specified, an index is created from the sorted
# dictionary keys. 
# 
# - if data is a scalar value, an inde must be supplied. In this case, the
# scalar value will be repeated to ensure that each label in the index has
# a value in the `Series`.
# 
# These different options are demonstrated in the following code cells.
# 
# -----
# [df]: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe
# 

import pandas as pd
import numpy as np

# We label the random values
s1 = pd.Series(np.random.rand(6), index=['q', 'w', 'e', 'r', 't', 'y'])

print(s1)


d = {'q': 11, 'w': 21, 'e': 31, 'r': 41}

# We pick out the q, w, and r keys, but have an undefined y key.
s2 = pd.Series(d, index = ['q', 'w', 'r', 'y'])

print(s2)


# We create a Series from an integer constant with explicit labels
s3 = pd.Series(42, index = ['q', 'w', 'e', 'r', 't', 'y'])

print(s3)

print('\nThe "e" value is ', s3['e'])


# We can slice like NumPy arrays

print(s1[:-2])

# We can also perform vectorized operations
print('\nSum Series:')
print(s1 + s3)
print('\nSeries operations:')
print(s2 * 5 - 1.2)


# -----
# 
# ### `DataFrame`
# 
# The second major data structure that Pandas provdis is he `DataFrame`,
# which is a two-dimensional array, where each column is effectively a
# `Series` with a shared index. A DataFrame is a very powerful data
# structure and provides a nice mapping for a number of different data
# formats (and storage mechanisms). For example, you can easily read data
# from a CSV file, a fixed width format text file, a JSOPN file, an HTML
# file, and HDF file, and a relational database into a Pandas `DataFrame`.
# This is demonstrated in the next set of code cells, where we extract
# data from files we created in the [Introduction to Data
# Formats][df] Notebook.
# 
# -----
# [df]: dataformats.ipynb
# 

# Read data from CSV file, and display subset

dfa = pd.read_csv('data.csv', delimiter='|', index_col='iata')

# We can grab the first five rows, and only extract the 'airport' column
print(dfa[['airport', 'city', 'state']].head(5))


# Read data from our JSON file
dfb = pd.read_json('data.json')

# Grab the last five rows
print(dfb[[0, 1, 2, 3, 5, 6]].tail(5))


# -----
# 
# In the previous code cells, we read data first from a delimiter
# separated value file and second from a JSON file into a Pandas
# `DataFrame`. In each code cell, we display data contained in the new
# DataFrame, first by using the `head` method to display the first few
# rows, and second by using the `tail` method to display the last few
# lines. For the delimiter separated value file, we explicitly specified
# the delimiter, which is a vertical bar `|`, the default is to assume a
# comma as the delimiter. We also explicitly specify the `iata` column
# should be used as the index column, which is how we can refer to rows in
# the array. 
# 
# We also explicitly select columns for display in both code cells. In the
# first code cell, we explicitly name the columns, passing in a list of
# the names to the DataFrame. Alternatively, in the second code cell, we
# pass in a list of the column ids, which we must do since we did not
# create named columns when reading data from the JSON file. The list of
# integers can be used even if the columns of the array have been assigned
# names.
# 
# Pandas includes a tremendous amount of functionality, especially for
# the `DataFrame`, to learn more, view the [detailed documentation][pdd].
# Several useful functions are demonstrated below, however, including
# information summaries, slicing, and column operations on DataFrames.
# 
# -----
# 
# [pdd]: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe
# 

# Lets look at the datatypes of each column
dfa.dtypes


# We can get a summary of numerical information in the dataframe

dfa.describe()


# Notice the JSON data did not automatically specify data types
dfb.dtypes


# This affects the output of the describe method, dfb has no numerical data types.

dfb.describe()


# We can slice out rows using the indicated index for dfa

print(dfa.loc[['00V', '11R', '12C']])


# We can slice out rows using the row index for dfb
print(dfb[100:105])


# We can also select rows based on boolean tests on columns
print(dfa[(dfa.lat > 48) & (dfa.long < -170)])


# -----
# 
# We can also perform numerical operations on a `DataFrame`, just as was the
# case with NumPy arrays. To demonstrate this, we create a numerical
# DataFrame, apply different operations, and view the results.
# 
# -----
# 

df = pd.DataFrame(np.random.randn(5, 6))

print(df)


# We can incorporate operate with basic scalar values

df + 2.5


# And perform more complex scalar operations

-1.0 * df + 3.5


# We can also apply vectorized functions

np.sin(df)


# We can tranpose the dataframe

df.T


# -----
# 
# The above description merely scratches the surface of what you can do
# with a Pandas `Series` or a `DataFrame`. The best way to learn how to
# effectively use these data structures is to just do it!
# 
# -----
# 

# ### Additional References
# 
# 1. [Pandas Documentation][pdd]
# 2. A slightly dated Pandas [tutorial][pdt]
# -----
# 
# [pdd]: http://pandas.pydata.org/pandas-docs/stable/index.html
# [pdt]: http://www.gregreda.com/2013/10/26/intro-to-pandas-data-structures/
# 

# <DIV ALIGN=CENTER>
# 
# # Optimizing Python Performance
# ## Professor Robert J. Brunner
#   
# </DIV>  
# -----
# -----
# 

# ## Introduction
# 
# While writing (and maintaining) Python code is often much easier than
# writing similar code in more traditional high performance computing
# languages such as C, C++, or Fortran, Python is generally slower than
# similar programs written in higher performance languages. In those cases
# where end-to-end performance (i.e., concept to execution) is less
# important, perhaps because an application will be run many times, a
# programmer will need to consider new approaches to increase the
# performance of a Python program.
# 
# Before proceeding further, however, some strong words of caution. Many
# programmers spend an inordinate amount of time on unneeded
# optimizations. To quote [Donald Knuth][dk] (1974 Turing Award Lecture):
# 
# > Premature optimization is the root of all evil (or at least most of
# > it) in programming.
# 
# Put simply, one should not worry about optimization until it has been
# shown to be necessary. And then one needs to very carefully decide what
# should can and should be optimized. This follows [Amdahl's law][al]
# which quantifies he maximum speed-up possible by improving the execution
# speed of only part of a program. For example, if only half of a program
# can be optimized, then the maximum possible speed-up is two times the
# original version. While modern multi- and many-core systems offer new
# performance benefits, they also come at an increased cost of code
# development, maintenance, and readability.
# 
# Python, however, does provide both standard (i.e., included with the
# standard Python distribution) modules to leverage threading and
# multi-processing, as well as additional libraries and tools that can
# significantly improve the performance of specific types of applications.
# One major limitation that must be overcome when employing these
# techniques is the [_Global Interpreter Lock_][gil] or GIL. The GIL is
# used by the standard Python interpreter (known as CPython) to only allow
# one thread to execute Python code at one time. This is done to simplify
# the implementation of the Python object model and to safeguard against
# concurrent access. In other words, the entire Python interpreter is
# locked, and only one thread at a time is allowed access. 
# 
# While this simplifies the development of the Python interpreter, it
# diminishes the ability of Python programs to leverage the inherent
# parallelism that is available with multi-processor machines. Two caveats
# to the GIL are that the global lock is always released when doing IO
# operations (which might otherwise block or consume a lengthy period) and
# that either standard or third-party extension modules can explicitly
# release the global lock when doing computationally intensive tasks.
# 
# In the rest of this Notebook, we will first explore standard Python
# modules for improving program performance. Next, we will explore the use
# of the IPython parallel programming capabilities. We will then discuss
# some non-standard extensions that can be used to improve application
# performance. We will finish with a quick introduction to several Python
# high performance 
# 
# -----
# 
# [dk]: https://en.wikiquote.org/wiki/Donald_Knuth#Computer_Programming_as_an_Art_.281974.29
# [al]: https://en.wikipedia.org/wiki/Amdahl%27s_law
# [gil]: https://docs.python.org/3/glossary.html#term-global-interpreter-lock
# 

# ## Standard Python Modules
# 
# The Python interpreter comes with a number of standard modules that
# collectively form the [Python Standard Library][sl]. The Python3
# standard library contains a set of related modules for [concurrent
# execution][ce] that includes the `threading`, `multiprocessing`,
# `concurrent.futures`, `subprocess`, `sched`, and `queue` modules. In
# this section, we will quickly introduce the first two modules. Although
# the `concurrent` module looks promising as a way to employ either
# threads or processes in a similar manner.
# 
# -----
# [pl]: https://docs.python.org/3.4/library/index.html
# [ce]: https://docs.python.org/3.4/library/concurrency.html
# 

# ### Python Threads
# 
# [Threads][t] are lightweight process element that can are often used to
# improve code performance by allowing multiple threads of program
# execution to occur within a single process. In Python, however, threads
# do not in general offer the same level of performance improvement seen
# in other languages since programming languages since Python employs the
# global interpreter lock. Yet the `threading` module still can offer some
# improvement to IO intensive applications and also can provide an easier
# path to learning how to effectively employ parallel programming (which
# will subsequently be useful when using other techniques such as the
# `multiprocessing` module or HPC constructs like _MPI_ or _OpenCL_.
# 
# The `threading` module is built on the `Thread` object, which
# encapsulates the creation, interaction, and destruction of threads in a
# Python program. In this Notebook we will simply introduce the basic
# concepts; a number of [other resources][or] exist to provide additional
# details.
# 
# TO use a thread in Python, we first mus create a `Thread` object, to
# which we can assign a name, a function to execute, and parameters that
# should be used within the threading function. For example, given a
# function `my_func` that takes a single integer value, we could create a
# new thread by executing the following Python statement:
# 
#     t = threading.Thread(target=my_func, args=(10,))
# 
# We build on this simple example in the following code cell to
# demonstrate how to create and use a worker thread.
# 
# -----
# [t]: https://en.wikipedia.org/wiki/Thread_(computing)
# [or]: https://en.wikipedia.org/wiki/Thread_(computing)
# 

import threading
import time

# Generic worker thread
def worker(num):
        
    # Get this Thread's name
    name = threading.currentThread().getName()
    
    # Print Starting Message
    print('{0:s} starting.\n'.format(name), flush=True)
    
    # We sleep for two seconds
    time.sleep(2)
    
    # Print computation
    print('Computation = {0:d}\n'.format(10**num), flush=True)
    
    # Print Exiting Message
    print('{0:s} exiting.\n'.format(name), flush=True)

# We will spawn several threads.
for i in range(5):
    t = threading.Thread(name='Thread #{0:d}'.format(i), target=worker, args=(i,))
    t.start()
    
print("Threads all created", flush=True)


# ### Multiprocessing
# 
# One way to circumvent the GIL is to use multiple Python interpreters that each run in their own process. This can be accomplished by using the `multiprocessing` module. In this module, processes essentially take the place of threads, but since each process will read the same Python code file, we need to ensure that only one process (the main process) creates the other processes, or else we can create an infinite loop that quickly consumes all hardware resources. This is done by using the following statement prior to the main program body:
# 
#     if __name__ == '__main__':
# 
# Inside the main program code, we can create Processes and start them in a similar manner as we did with threads earlier.
# 
# -----
# [mp]: http://pymotw.com/2/multiprocessing/index.html#module-multiprocessing
# 

import multiprocessing 
import time

# Generic worker process
def worker(num):
        
    # Get this Process' name
    name = multiprocessing.current_process().name
    
    # Print Starting Message
    print('{0:s} starting.\n'.format(name))
    
    # We sleep for two seconds
    time.sleep(2)
    
    # Print computation
    print('Computation = {0:d}\n'.format(num**10))
    
    # Print Exiting Message
    print('{0:s} exiting.\n'.format(name))

if __name__ == '__main__':

    # We will spawn several processes.
    for i in range(3):
        p = multiprocessing.Process(name='Process #{0:d}'.format(i), target=worker, args=(i,))
        p.start()
        
    print("Processing complete", flush=True)


# ## IPython Cluster
# 
# The [Ipython Server][ipy] has built-in support for parallel processing.
# This can be initialized in an automated manner by using the `ipcluster`
# command, or in a manual approach by using the `ipcontroller` and
# `ipengine` commands. The first approach simply automated the process of
# using the controller and engines, and requires the creation of a IPYthon
# profile, which is done by using the `ipython profile create` command.
# `ipcluster` works with both MPI and batch processing clusters (ew.g.,
# via PBS), and can be made to work with other schedulers such as condor.
# 
# If necessary, you can also manually control the process by directly
# instantiating the IPython controller and engines. The controller must be
# started first, after which you can create as many engines as necessary,
# given your hardware constraints. IPython clustering works best on
# multi-processing machines or compute clusters.
# 
# -----
# 
# [ipy]: http://ipython.org/ipython-doc/dev/parallel/index.html
# 

# ## Third-Party Python Tools
# 
# There are a number of third-party Python modules or packages that can be
# used to improve the performance of a Python application.
# 
# 1. [Numba][nj] is a just in time compiler from Continuum Analytics that
# can increase the performance of certain  functions (e.g., numerical
# work).
# 
# 2. [PYPY][py] is an alternative implementation of the Python language
# that includes a just in time compiler that speeds up many Python
# programs.
# 
# 3. [Cython][cy] is a static optimizing compiler for Python and also
# provides a method for easily including C or C++ code in a Python program.
# 
# -----
# 
# [nj]: http://numba.pydata.org
# [py]: http://pypy.org
# [cy]: http://cython.org
# 

# ## Python and HPC
# 
# While Python programs can be easily used for embarrassingly parallel
# programming on high performance compute systems and Python is also used
# to glue advanced computation programs together for batch processing
# there are also projects underway that enable Python code to directly
# leverage high performance programming paradigms:
# 
# - [MPI][m] is message passing interface and is a protocol used to
# communicate messages (or data) between compute nodes in a large,
# distributed compute cluster. [mpi4py][m2] is a Python module that
# brings a significant part of the MPI specification to Python programs.
# 
# - [OpenCL][o] is a framework that enables programs to run on heterogeneous
# platforms including CPUs, GPUs, DSP, and FPGAs. The [Python OpenCL][po]
# package enables Python programs to use OpenCL to write code that runs on
# these different processor types efficiently and effectively.
# 
# -----
# [m]: https://en.wikipedia.org/wiki/Message_Passing_Interface
# [m2]: https://bitbucket.org/mpi4py/mpi4py/overview
# [o]: https://en.wikipedia.org/wiki/OpenCL
# [po]: http://mathema.tician.de/software/pyopencl/
# 

# <DIV ALIGN=CENTER>
# 
# # Introduction to JSON Data Format
# ## Professor Robert J. Brunner
#   
# </DIV>  
# -----
# -----
# 

# ## Introduction
# 
# We have already touched upon data formats in the context of data
# persistence. But one of the most important tasks when starting a data
# analysis project is understanding the format of a data file and how to
# best extract the necessary information from the data, whatever the
# format. In this notebook, we explore the JSON data format, and present
# how to read and write data in this format by using standard, built-in
# Python tools.
# 
# -----
# 
# 

# Before we begin, however, we need to read in test data to be able to
# have data that we can write and read to a JSON format.
# 
# This Notebook will only work after the [Text Data Format][tdf] notebook
# has been successfully completed.
# 
# -----
# 
# [tdf]: text-dataformat.ipynb
# 

import csv

airports = []

with open('/home/data_scientist/rppdm/data/airports.csv', 'r') as csvfile:
    
    for row in csv.reader(csvfile, delimiter=','):
        airports.append(row)

print(airports[0:3])


# ### JSON
# 
# [JavaScript Object Notation][json], or JSON, is a text-based data
# interchange format that is easy to read and write both for humans and
# programs. JSON is a [standard][st], published by the [ECMA
# International][ecma] standard organization, which was originally known
# as the European Computer Manufacturers Association, but is now a more
# global organization for the development of global computer and
# electronic standards. JSON is language independent but uses a syntax
# that is familiar to anyone who knows a C-based language, like Python.
# JSON is built on two types of constructs: a dictionary and a list, and
# the standard dictates how data are mapped into these constructs.
# 
# The JSON standard is fairly simple, as it defines an object, an array, a
# value, a string, and a number data formats, upon which most of the rest
# of the standard is based. This maes writing and reading JSON data
# formats fairly straightforward. In Python, this functionality is
# provided by the built-in [`json`][jspy] module, which simplifies the
# process of [reading or writing][jsd] Python data structures _serialize_
# a data hierarchy into a string representation via the `dump` method, and
# can _deserialize_ via the `load` module. These processes are
# demonstrated in the next few code cells.
# 
# -----
# 
# [json]: http://www.json.org
# [st]: http://www.ecma-international.org/publications/files/ECMA-ST/ECMA-404.pdf
# [ecma]: http://www.ecma-international.org
# [jspy]: https://docs.python.org/3/library/json.html
# [jsd]: https://docs.python.org/3/tutorial/inputoutput.html#saving-structured-data-with-json
# 

import json

with open('data.json', 'w') as fout:
    json.dump(airports, fout)


# ----- 
# 
# We display the contents of our new JSON file in the following code cell;
# however, since this data format doesn't automatically split data over
# different lines, the entire file is contained in a single line. This can
# complicate viewing the data, but does not affect the utility of this
# data format in programmatic instances.
# 
# -----
# 

get_ipython().system('head data.json')


# -----
# 
# The beauty of a self-describing data format like JSON is that reading
# and reconstructing data from this format is straightforward. As
# demonstrated in the next code cell, we simply open the file and load the
# JSON formatted data.
# 
# -----
# 

# First we can display the first few rows of the original data for comparison.

print(airports[:3], '\n', '-'*80)

# We use the pretty-print method to 
from pprint import pprint

# Open file and read the JSON formatted data
with open('data.json', 'r') as fin:
    data = json.load(fin)

# Pretty-print the first few rows
pprint(data[:3])


# ### Additional References
# 
# 4. [JSON Tutorial][4] by W3Schools.
# 
# -----
# 
# [4]: http://www.w3schools.com/json/default.asp
# 

# <DIV ALIGN=CENTER>
# 
# # Introduction to XML Data Format
# ## Professor Robert J. Brunner
#   
# </DIV>  
# -----
# -----
# 

# ## Introduction
# 
# We have already touched upon data formats in the context of data
# persistence. But one of the most important tasks when starting a data
# analysis project is understanding the format of a data file and how to
# best extract the necessary information from the data, whatever the
# format. In this notebook, we explore the XML data format, and present
# how to read and write data in this format by using standard, built-in
# Python tools.
# 
# -----
# 
# 

# Before we begin, however, we need to read in test data to be able to
# have data that we can write and read to an XML format.
# 
# This Notebook will only work after the [Text Data Format][tdf] notebook
# has been successfully completed.
# 
# -----
# 
# [tdf]: text-dataformat.ipynb
# 

import csv

airports = []

with open('/home/data_scientist/rppdm/data/airports.csv', 'r') as csvfile:
    
    for row in csv.reader(csvfile, delimiter=','):
        airports.append(row)

print(airports[0:3])


# -----
# 
# ## XML
# 
# [Extensible Markup Language][xml], or XML, is a simple, self-describing
# text-based data format. XML is a standard developed by the W3C, or
# World-Wide Web Consortium, originally for large scale publishing, but
# with the growth of the web it has taken on new roles. XML is based on
# the concept of element, that can have attributes and values. Elements
# can be nested, which can indicate parent-child relationships or a form of
# containerization. While you may not ever deal directly with XML files,
# you wil interact with other data formats that are based on XML, such as
# the latest version of HyperTextMarkup Language (HTML5) or Scalable
# Vector Graphics format (SVG).
# 
# Given its structured format, you don't simply read an XML document, you
# must parse the document to build up a model of the elements and their
# relationships. The [`ElementTree`][xmlpy] parsing model is implemented
# within the standard Python distribution in the `xml` library. T0 write
# an XML file, we simply need to create an instance of this, for example
# by passing a string into the class constructor, and then writing this
# XML encoded data to a file. One caveat with this entire process,
# however, is that the following five characters: `<`, `>`, `&`, `'`, and
# `"` are used by the actual markup language, they must be replaced by
# their corresponding _entity code_. For these five characters, that can
# be easily done by using the `html`.escape` method as shown in the
# following code cell.
# 
# -----
# [xml]: http://www.w3.org/XML/
# [w3c]: http://www.w3.org
# [html5]: http://www.w3.org/TR/html5/
# [svg]: http://www.w3.org/Graphics/SVG/
# [xmlpy]: https://docs.python.org/3/library/markup.html
# 

import html 
import xml.etree.ElementTree as ET

data = '<?xml version="1.0"?>\n' + '<airports>\n'

for airport in airports[1:]:
    data += '    <airport name="{0}">\n'.format(html.escape(airport[1]))
    data += '        <iata>' + str(airport[0]) + '</iata>\n'
    data += '        <city>' + str(airport[2]) + '</city>\n'
    data += '        <state>' + str(airport[3]) + '</state>\n'
    data += '        <country>' + str(airport[4]) + '</country>\n'
    data += '        <latitude>' + str(airport[5]) + '</latitude>\n'
    data += '        <longitude>' + str(airport[6]) + '</longitude>\n'

    data += '    </airport>\n'

data += '</airports>\n'

tree = ET.ElementTree(ET.fromstring(data))


with open('data.xml', 'w') as fout:
    tree.write(fout, encoding='unicode')


# -----
# 
# Since the XML format is text based, we can easily view the contents of
# our new XML file by using the `head` command, as done before. In this
# case, the XML format is our own creation, but if we were following a
# standard, additional information would be present to indicate the full
# document provenance.
# 
# -----
# 

get_ipython().system('head -9 data.xml')


# -----
# 
# As the XML document contents demonstrate above, the XML format can be
# quite verbose. However, the document's contents are clearly visible and
# are easily understood. This enables an XML document to be [parsed][ps]
# based on a rough knowledge of the document. First we need to create and
# `ElementTree` object and parse the contents of the document, which we
# can do with the `parse` method and passing in the name of our XML
# document file. 
# 
# When parsing an XML document, we have a tree model for the XML elements
# contained in the document. The base of this model is the _root_ element,
# which is returned by the `parse` method. While there are a number of
# methods that can be used to find or iterate through elements in the
# document, in our case we simply want to process each `airport` element;
# thus we use the `findall` method to find all `airport` elements. The
# child elements of each `airport` element can be accessed like a Python
# `list`. The text within an element is accessed by requesting the `text`
# attribute for that element, while an element attribute is accessed like
# a `dictionary` where the name of the attribute acts as the _key_ to
# request a particular _value_. These techniques are demonstrated in the
# next code cell, where we read in our new XML document, and extract the
# airport information.
# 
# -----
# 
# [ps]: https://docs.python.org/3/library/xml.etree.elementtree.html#parsing-xml
# 

data = [["iata", "airport", "city", "state", "country", "lat", "long"]]

tree = ET.parse('data.xml')
root = tree.getroot()

for airport in root.findall('airport'):
    row = []
    row.append(airport[0].text)
    row.append(airport.attrib['name'])
    row.append(airport[1].text)
    row.append(airport[2].text)
    row.append(airport[3].text)
    row.append(airport[4].text)
    row.append(airport[5].text)

    data.append(row)
    
print(data[:5])


# -----
# 
# The preceding data formats: fixed-width, delimiter separated value,
# JSON, and XML are the primary text-based data formats that data
# scientists need to be able to use. While easy to read and relatively
# easy to parse, they are not always the best solution, especially for
# large, numerical data. While specialized binary formats exist, which are
# often domain-specific formats, there is one widely used format that
# continues to gain ground in data science applications.
# 
# -----
# 

# ### Additional References
# 
# 1. [XML Tutorial][1] by W3Schools.
# 2. [HTML Tutorial][2], an XML specified document language, by W3Schools.
# 3. [SVG Tutorial][3], an XML specified image language, by W3Schools.
# 
# 
# -----
# 
# [1]: http://www.w3schools.com/xml/default.asp
# [2]: http://www.w3schools.com/html/default.asp
# [3]: http://www.w3schools.com/svg/default.asp
# 

# <DIV ALIGN=CENTER>
# 
# # Introduction to NumPy
# ## Professor Robert J. Brunner
#   
# </DIV>  
# -----
# -----
# 

# ## Introduction 
# 
# As we discused previously, the Python programming language provides a rich set of data
# structures such as the list, tuple, dictionary, and string, which can
# greatly simply common programming tasks. Of these, all but the string
# are heterogeneous, which means they can hold data of different types.
# This flexibility comes at a cost, however, as it is more expensive in
# both computational time and storage to maintain an arbitrary collection
# of data than to hold a pre-defined set of data.
# 
# (Advanced) For example, the Python list is implemented (in the
# [Cython](http://cython.org) implementation) as a variable length array
# that contains pointers to the objects held in the array. While flexible,
# it takes time to create, resize, and iterate over, even if the data
# contained in the list is homogenous. In addition, even though you can
# create multiple-dimensional lists, creating and working with them is not
# simple nor intuitive. Yet, many applications require multi-dimensional
# representation; for example, location on the surface of the Earth or
# pixel properties in an image.
# 
# Thus, these data structures are clearly not designed nor optimized for
# data intensive computing. Scientific and engineering computing
# applications have a long history of using optimized data structures and
# libraries, including codes written in C, Fortran, or MatLab. These
# applications expect to have vector and matrix data structures and
# optimized algorithms that can leverage these structures seamlessly. 
# Fortunately, since many data science applications, including
# statistical and machine learning, are built on this academic legacy, a
# community of open source developers have built the [Numerical Python
# (NumPy)](http://numpy.org) library to which is a fundamental numerical
# package to facilitate scientific computing in Python.
# 
# NumPy is built around a new, n-dimensional array (`ndarray`) data
# structure that provides fast support for numerical computations. This
# data type for objects stored in the array can be specified at creation
# time, but the array is homogenous. This array can be used to represent a
# vector (one-dimensional set of numerical values) or matrix
# (multiple-dimensional set of vectors). Furthermore, NumPy provides
# additional benefits built on top of the `array` object, including
# _masked arrays_, _universal functions_, _sampling from random
# distributions_, and support for _user-defined, arbitrary data-types_
# that allow the `array` to become an efficient, multi-dimensional generic
# data container.
# 
# -----
# 

# ### Is NumPy worth learning?
# 
# Despite the discussion in the previous section, you might be curious if
# the benefits of learning NumPy are worth the effort of learning to
# effectively use a new Python data structure, especially one that is not
# part of the standard Python distribution. In the end, you will have to
# make this decision, but there are several definite benefits to using
# NumPy:
# 
# 1. NumPy is much faster than using a `list`.
# 2. NumPy is generally more intuitive than using a `list`.
# 3. NumPy is used by many other libraries like SciPy, MatPlotLib, and Pandas.
# 4. NumPy is part of the standard **data science** Python distribution.
# 
# NumPy is a very powerful library, with a rich and detailed [user guide][1].
# Every time I look through the documentation, I learn new tricks. The
# time you spend learning to use this library properly will be amply
# rewarded. In the rest of this IPython notebook, we will introduce many of
# the basic NumPy features, but to fully master this library you will need
# to spend time reading the documentation and trying out the full
# capabilities of the library.
# 
# To demonstrate the first two points, consider the programming task of
# computing basic mathematical functions on a large number of data points.
# In the first code block, we first import both the `math` library and the
# `numpy` library. Second, we define two constants: size, which is the
# number of data points to process, and delta, which is a floating point
# offset we add to the array elements. You can change these two parameters
# in order to see how the performance of the different approaches varies.
# Finally, we create the `list` and the NumPy `array` that we will use in
# the next few codes blocks:
# 
# -----
# 
# [1]: http://docs.scipy.org/doc/numpy/user/
# 

import math
import numpy as np

size = 100000
delta = 1.0E-2

aList = [(x + delta) for x in range(size)]
anArray = np.arange(size) + delta

print(aList[2:6])
print(anArray[2:6])


# -----
# 
# At this point, we have created and populated both data structures, and
# you have seen that they are both indexed in the same manner, meaning it
# is probably easier to learn and use NumPy arrays than you might have
# thought. Next, we can apply several standard mathematical functions to
# our `list`, creating new `list`s in the process. To determine how long
# these operations take, we use the IPython `%timeit` magic function,
# which will, by default, run the code contained on the rest of the line
# multiple times and report the _average_ best time.
# 
# -----
# 

get_ipython().magic('timeit [math.sin(x) for x in aList]')
get_ipython().magic('timeit [math.cos(x) for x in aList]')
get_ipython().magic('timeit [math.log(x) for x in aList]')


# -----
# 
# As you can see, to create these new `list`s, we apply the mathematical
# function to every angle in the original `list`. These operations are
# fairly fast and all roughly constant in time, demonstrating the overall
# speed of _list comprehensions_ in Python. Now lets, try doing the same
# thing by using the NumPy library.
# 
# -----
# 

get_ipython().magic('timeit np.sin(anArray)')
get_ipython().magic('timeit np.cos(anArray)')
get_ipython().magic('timeit np.log10(anArray)')


# -----
# 
# First, the creation of each of these new arrays was much faster, nearly a
# factor of ten in each case! (Actual results will depend on the host computer 
# and Python version.) Second, the operations themselves were
# arguably simpler both to write and to read, as the function is applied to
# the data structure itself and not each individual element. But perhaps
# we should compare the results to ensure they are the same?
# 
# -----

l = [math.sin(x) for x in aList]
a = np.sin(anArray)

print("Python List: ", l[2:10:3])
print("NumPY array:", a[2:10:3])

print("Difference = ", a[5:7] - np.array(l[5:7]))
      
# Now create a NumPy array from a Python list
get_ipython().magic('timeit (np.sin(aList))')


# -----
# 
# As the previous code block demonstrates, the NumPy results agree with
# the standard Python results, although the NumPy results are more
# conveniently displayed. As a last test, we create a new NumPy `array`
# from the original Python `list` by applying the `np.sin` function,
# which, while not as fast as the pure NumPy version, is faster than the
# Python version and easier to read.
# 
# Now lets change gears and actually introduce the NumPy library.
# 
# 
# -----
# 

# ### Creating an Array
# 
# [NumPy arrays][i], which are instances of the `ndarray` class are
# statically-typed, homogenous data structures that can be created in a
# number of [different ways][1]. You can create an array from an existing
# Python `list` or `tuple`, or use one of the many built-in NumPy
# convenience methods:
# 
# - `empty`: Creates a new array whose elements are uninitialized.
# - `zeros`: Create a new array whose elements are initialized to zero.
# - `ones`: Create a new array whose elements are initialized to one.
# - `empty_like`: Create a new array whose size matches the input array 
# and whose values are uninitialized.
# - `zero_like`: Create a new array whose size matches the input array 
# and whose values are initialized to zero.
# - `ones_like`: Create a new array whose size matches the input array 
# and whose values are initialized to unity.
# 
# -----
# [i]: http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html
# [1]: http://docs.scipy.org/doc/numpy/user/basics.creation.html
# 

# Make and print out simple NumPy arrays

print(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

print("\n", np.empty(10))
print("\n", np.zeros(10))
print("\n", np.ones(10))
print("\n", np.ones_like(np.arange(10)))


# -----
# 
# We can also create NumPy arrays that have specific initialization
# patterns. For example, the `arange(start, end, stride)` method works
# like the normal Python `range` method, creating an array whose elements
# begin with the `start` parameter. Subsequent elements are incremented
# successively by the `stride` parameter, stopping when the `end` parameter
# would either be reached or surpassed. As was the case with the `range`
# method, the `start` and `stride` parameters are optional, defaulting to
# zero and one respectively, and the `end` value is not included in the
# array.
# 
# -----
# 

# Demonstrate the np.arange method

print(np.arange(10))
print(np.arange(3, 10, 2))


# -----
# 
# NumPy also provides two convenience methods that take a similar
# form, but assign to the elements values that are evenly
# spaced. The first method is the `linspace(start, end, num)` method,
# which creates `num` elements and assign values that are linearly spaced
# between `start` and `end`. 
# 
# The second method, `logspace(start, end, num)`, creates `num` elements
# and assigns values that are logarithmically spaced between `start` and
# `end`. The `num` parameter is optional and defaults to fifty. Unlike the
# `arange` method, these two methods are inclusive, which means both the
# `start` and `end` parameters are included as elements in the new array.
# There is an optional parameter called `base`, that you can use to
# specify the base of the logarithm used to create the intervals. By
# default this value is ten, making the intervals `log10` spaced.
# 
# -----
# 

# Demonstrate linear and logarthmic array creation.

print("Linear space bins [0, 10] = {}\n".format(np.linspace(0, 10, 4)))

print("Default linspace bins = {}\n".format(len(np.linspace(0,10))))


print("Log space bins [0, 1] = {}\n".format(np.logspace(0, 1, 4)))

print("Default linspace bins = {}\n".format(len(np.logspace(0,10))))


# -----
# 
# ### Array Attributes
# 
# Each NumPy array has several attributes that describe the general
# features of the array. These attributes include the following:
# - `ndim`: Number of dimensions of the array (previous examples were all unity).
# - `shape`: The dimensions of the array, so a matrix with n rows and m
# columns has `shape` equal to `(n, m)`.
# - `size`: The total number of elements in the array. For a matrix with n
# rows and m columns, the `size` is equal to the product of $n \times m$.
# - `dtype`: The data type of each element in the array.
# - `itemsize`: The size in bytes of each element in the array.
# -`data`: The buffer that actually holds the array data.
# 
# -----
# 

# ### Array Data Types
# 
# NumPy arrays are statically-typed, thus their [data type][1] is
# specified when they are created. The default data type is `float`, but
# this can be specified in several ways. First, if you use an existing
# `list` (as we did in the previous code block) or `array` to initialize
# the new `array`, the data type of the previous data structure will be
# used. If a heterogeneous Python `list` is used, the greatest data type
# will be used in order guarantee that all values will be safely contained
# in the new `array`. If using a NumPy function to create the new `array`,
# the data type can be specified explicitly by using the `dtype` argument,
# which can either be one of the predefined built-in data types or a user
# defined custom data type.
# 
# The full list of built-in data types can be obtained from the
# `np.sctypeDict.keys()` method; but for brevity, we list some of the more
# commonly used built-in data types below, along with their maximum size
# in bits, which constrains the maximum allowed value that may be stored
# in the new `array`:
# 
# - Integer: `int8`, `int16`, `int32`, and `int64` 
# - Unsigned Integer: `uint8`, `uint16`, `uint32`, and `uint64` 
# - Floating Point: `float16`, `float32`, `float64`, and `float128` 
# 
# Other data types include complex numbers, byte arrays, character arrays,
# and dat/time arrays. 
# 
# To check the type of an array, you can simply access the array's `dtype`
# attribute. A `ValueError` exception will be thrown if you try to assign
# an incompatible value to an element in an `array`. 
# 
# -----
# [1]: http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
# 

# Access our previously created array's data type 

a.dtype


# Try to assign a string to a floating point array element

a[0] = 'Hello!'


# -----
# 
# ### Multidimensional Arrays
# 
# NumPy supports multidimensional arrays, although for simplicity we will
# rarely discuss anything other than two- or three-dimensional arrays.
# Higher dimensional arrays can be created in the normal process, where an
# array with the correct number of elements is created, and subsequently
# reshaped into the correct dimensionality. For example, you can create a
# NumPy array with one hundred elements and reshape this new array into a
# ten by ten matrix.
# 
# -----
# 

# Make a 10 x 10 array

data = np.arange(100)

mat = data.reshape(10, 10)

print(mat)


# -----
# 
# Special convenience functions also exist to create special
# multidimensional arrays. For example, you can create an identity matrix,
# where the diagonal elements are all one and all other elements are zero
# by using the `eye` method (since the Identity matrix is often denoted by
# the capital letter 'I'). You can also specify the diagonal (or certain
# off-diagonal) elements by using the `diag` function, where the input
# array is assigned to a set of diagonal elements in the new array. If the
# `k` parameter to the `np.diag`  method is zero, the diagonal elements
# will be initialized. If the `k` parameter is a positive (negative)
# integer, the diagonal corresponding to the integer value of `k` is
# initialized. The size of the resulting array will be the minimum
# possible size to allow the input array to be properly initialized.
# 
# -----
# 

# Create special two-dimensional arrays

print("Matrix will be 4 x 4.\n", np.eye(4))
print("\nMatrix will be 4 x 4.\n", np.diag(np.arange(4), 0))
print("\nMatrix will be 5 x 5.\n", np.diag(np.arange(4), 1))
print("\nMatrix will be 5 x 5.\n", np.diag(np.arange(4), -1))


# -----
# 
# ### Indexing Arrays
# 
# NumPy supports many different ways to [access elements][1] in an array.
# Elements can be indexed or sliced in the same way a Python list or tuple
# can be indexed or sliced, as demonstrated in the following code cell.
# 
# ------
# [1]: http://docs.scipy.org/doc/numpy/user/basics.indexing.html
# 

a = np.arange(9)
print("Original Array = ", a)

a[1] = 3
a[3:5] = 4
a[0:6:2] *= -1

print("\nNew Array = ", a)


# -----
# 
# ### Slicing Multidimensional Arrays
# 
# Multi-dimensional arrays can be sliced, the only trick is to remember
# the proper ordering for the elements. Each dimension is differentiated
# by a comma in the slicing operation, so a two-dimensional array is
# sliced with `[start1:end1, start2:end2]`, while a three-dimensional
# array is sliced with `[start1:end1, start2:end2. start3:end3]`,
# continuing on with higher dimensions. If only one dimension is
# specified, it will default to the first dimension. These concepts are
# demonstrated in the following two code cells, first for a
# two-dimensional array, followed by a three-dimensional array.
# 
# Note that each of these slicing operations (i.e., `start:end`) can also
# include an optional `stride` value as well.
# 
# -----
# 

b = np.arange(9).reshape((3,3))

print("3 x 3 array = \n",b)

print("\nSlice in first dimension (row 1): ",b[0])
print("\nSlice in first dimension (row 3): ",b[2])

print("\nSlice in second dimension (col 1): ",b[:,0])
print("\nSlice in second dimension (col 3): ", b[:,2])

print("\nSlice in first and second dimension: ", b[0:1, 1:2])


print("\nDirect Element access: ", b[0,1])


c = np.arange(27).reshape((3,3, 3))

print("3 x 3 x 3 array = \n",c)
print("\nSlice in first dimension (first x axis slice):\n",c[0])

print("\nSlice in first and second dimension: ", c[0, 1])

print("\nSlice in first dimension (third x axis slice):\n", c[2])

print("\nSlice in second dimension (first y axis slice):\n", c[:,0])
print("\nSlice in second dimension (third y axis slice):\n", c[:,2])

print("\nSlice in first and second dimension: ", c[0:1, 1:2])

print("\nSlice in first and second dimension:\n", c[0,1])
print("\nSlice in first and third dimension: ", c[0,:,1])
print("\nSlice in first, second, and third dimension: ", c[0:1,1:2,2:])

print("\nDirect element access: ", c[0,1, 2])


# -----
# 
# ### Special Indexing
# 
# NumPy also provides several other _special_ indexing techniques. The
# first such technique is the use of an [index array][1], where you use an
# array to specify the elements to be selected. The second technique is a
# [Boolean mask array][2]. In this case, the Boolean array is the same
# size as the primary NumPy array, and if the element in the mask array
# is `True` the corresponding element in the primary array is selected,
# and vice-versa for a `False` mask array element. These two special
# indexing techniques are demonstrated in the following two code cells.
# 
# -----
# [1]: http://docs.scipy.org/doc/numpy/user/basics.indexing.html#index-arrays
# [2]: http://docs.scipy.org/doc/numpy/user/basics.indexing.html#boolean-or-mask-index-arrays
# 

# Demonstration of an index array

a = np.arange(10)

print("\nStarting array:\n", a)
print("\nIndex Access: ", a[np.array([1, 3, 5, 7])])

c = np.arange(10).reshape((2, 5))

print("\nStarting array:\n", c)
print("\nIndex Array access: \n", c[np.array([0, 1]) , np.array([3, 4])])


# Demonstrate Boolean mask access

# Simple case

a = np.arange(10)
print("Original Array:", a)

print("\nMask Array: ", a > 4)

# Now change the values by using the mask

a[a > 4] = -1.0
print("\nNew Array: ", a)

# Now a more complicated example.

print("\n--------------------")
c = np.arange(25).reshape((5, 5))
print("\n Starting Array: \n", c)

# Build a mask that is True for all even elements with value greater than four
mask1 = (c > 4)
mask2 = (c % 2 == 0)

print("\nMask 1:\n", mask1)
print("\nMask 2:\n", mask2)

# We use the logical_and ufunc here, but it is described later
mask = np.logical_and(mask1, mask2)

print("\nMask :\n", mask)

print("\nMasked Array :\n", c[mask])
c[mask] /= -2.

print("\nNew Array :\n", c)


# -----
# 
# ### Random Data
# 
# NumPy has a rich support for [random number][1] generation, which can be
# used to create and populate an array of a given shape and size. NumPy
# provides support for sampling random values from over thirty different
# distributions including the `binomial`, `normal`, and `poisson`, there
# are also special convenience functions to simplify the sampling of
# random data over the uniform or normal distributions. These techniques
# are demonstrated in the following code cell.
# 
# -----
# [1]: http://docs.scipy.org/doc/numpy/reference/routines.random.html
# 
# 
# 

# Create arrays of random data from the uniform distribution

print("Uniform sampling [0, 1): ", np.random.rand(5))
print("Uniform sampling, integers [0, 1): ", np.random.randint(0, 10, 5))
print("Normal sampling (0, 1) : ", np.random.randn(5))


# -----
# 
# ### Basic Operations
# 
# NumPy arrays naturally support basic mathematical operations, including
# addition, subtraction, multiplication, division, integer division, and
# remainder allowing you to easily combine a scalar (a single number) with
# a vector (a one-dimensional array), or a matrix (a multi-dimensional
# array). In the next code block, we first create a one-dimensional array,
# and subsequently operate on this array to demonstrate how to combine a
# scalar with a vector.
# 
# -----
# 

# Create and use a vector
a = np.arange(10)

print(a)
print("\n", (2.0 * a + 1)/3)
print("\n", a%2)
print("\n", a//2)


# -----
# 
# These same operations can be used to combine a scalar with a matrix. In
# the next code block we create a two-dimensional array and use that array
# to demonstrate how to operate on a scalar and a matrix.
# 
# -----
# 

# Create a two-dimensional array

b = np.arange(9).reshape((3,3))

print("Matrix = \n", b)

print("\nMatrix + 10.5 =\n", (b + 10.5))

print("\nMatrix * 0.25 =\n", (b * 0.25))

print("\nMatrix % 2 =\n", (b % 2))

print("\nMatrix / 3.0 =\n", ((b - 4.0) / 3.))


# ----- 
# 
# We also can combine arrays as long as they have the same dimensionality.
# In the next code block we create a one-dimensional and a two-dimensional
# array and demonstrate how these two arrays can be combined.
# 
# -----
# 

# Create two arrays

a = np.arange(1, 10)
b = (10. - a).reshape((3, 3))
print("Array = \n",a)
print("\nMatrix = \n",b)

print("\nArray[0:3] + Matrix Row 1 = ",a[:3] + b[0,:,])

print("\nArray[0:3] + Matrix[:0] = ", a[:3] + b[:,0])

print("\nArray[3:6] + Matrix[0:] = ", a[3:6] + b[0, :])

# Now combine scalar operations

print("\n3.0 * Array[3:6] + (10.5 + Matrix[0:]) = ", 3.0 * a[3:6] + (10.5 + b[0, :]))


# -----
# 
# ### Summary Functions
# 
# NumPy provides convenience functions that can quickly summarize the
# values of an array, which can be very useful for specific data
# processing tasks. These functions include basic [statistical
# measures][1] (`mean`, `median`, `var`, `std`, `min`, and `max`), the
# total sum or product of all elements in the array (`sum`, `prod`), as
# well as running sums or products for all elements in the array
# (`cumsum`, `cumprod`). The last two functions actually produce arrays
# that are of the same size as the input array, where each element is
# replaced by the respective running sum/product up to and including the
# current element. Another function, `trace`, calculates the trace of an
# array, which simply sums up the diagonal elements in the
# multi-dimensional array.
# 
# -----
# 
# [1]: http://docs.scipy.org/doc/numpy/reference/routines.statistics.html
# 

# Demonstrate data processing convenience functions

# Make an array = [1, 2, 3, 4, 5]
a = np.arange(1, 6)

print("Mean value = {}".format(np.mean(a)))
print("Median value = {}".format(np.median(a)))
print("Variance = {}".format(np.var(a)))
print("Std. Deviation = {}\n".format(np.std(a)))

print("Minimum value = {}".format(np.min(a)))
print("Maximum value = {}\n".format(np.max(a)))

print("Sum of all values = {}".format(np.sum(a)))
print("Running cumulative sum of all values = {}\n".format(np.cumsum(a)))

print("Product of all values = {}".format(np.prod(a)))
print("Running product of all values = {}\n".format(np.cumprod(a)))

# Now compute trace of 5 x 5 diagonal matrix (= 5)
print(np.trace(np.eye(5)))


# -----
# 
# ### Universal Functions
# 
# NumPy also includes methods that are _universal functions_ or
# [__ufuncs__][1] that are vectorized and thus operate on each element in
# the array, without the need for a loop. You have already seen examples
# of some of these functions at the start of this IPython Notebook when we
# compared the speed and simplicity of NumPy versus normal Python for
# numerical operations. These functions almost all include an optional
# `out` parameter that allows a pre-defined NumPy array to be used to hold
# the results of the calculation, which can often speed-up the processing
# by eliminating the need for the creation and destruction of temporary
# arrays. These functions will all still return the final array, even if
# the `out` parameter is used. 
# 
# NumPy includes over sixty _ufuncs_ that come in several different
# categories:
# 
# - Math operations, which can be called explicitly or simply implicitly
# when the standard math operators are used on NumPy arrays. Example
# functions in this category include `add`, `divide`, `power`, `sqrt`,
# `log`, and `exp`.
# - Trigonometric functions, which assume angles measured in radians.
# Example functions include the `sin`, `cos`, `arctan`, `sinh`, and
# `deg2rad` functions.
# - Bit-twiddling functions, which manipulate integer arrays as if they
# are bit patterns. Example functions include the `bitwise_and`,
# `bitwise_or`, `invert`, and `right_shift`.
# - Comparison functions, which can be called explicitly or implicitly
# when using standard comparison operators that compare two arrays,
# element-by-element, returning a new array of the same dimension. Example
# functions include `greater`, `equal`, `logical_and`, and `maximum`.
# - Floating functions, which compute floating point tests or operations,
# element-by-element. Example functions include `isreal`, `isnan`,
# `signbit`, and `fmod`.
# 
# Look at the official [NumPy _ufunc_][1] reference guide for more
# information on any of these functions, for example, the [isnan][2]
# function, since the user guide has a full breakdown of each function and
# sample code demonstrating how to use the function. 
# 
# In the following code blocks, we demonstrate several of these _ufuncs_.
# 
# -----
# [1]: http://docs.scipy.org/doc/numpy/reference/ufuncs.html
# [2]: http://docs.scipy.org/doc/numpy/reference/generated/numpy.isnan.html#numpy.isnan
# 

b = np.arange(1, 10).reshape(3, 3)

print('original array:\n', b)

c = np.sin(b)

print('\nnp.sin : \n', c)

print('\nnp.log and np.abs : \n', np.log10(np.abs(c)))

print('\nnp.mod : \n', np.mod(b, 2))

print('\nnp.logical_and : \n', np.logical_and(np.mod(b, 2), True))



# Demonstrate Boolean tests with operators

d = np.arange(9).reshape(3, 3)

print("Greater Than or Equal Test: \n", d >= 5)

# Now combine to form Boolean Matrix

np.logical_and(d > 3, d % 2)


# -----
# 
# ### Masked Arrays
# 
# NumPy provides support for [masked arrays][1], where certain elements
# can be _masked_ based on some criterion and ignored during subsequent
# calculations (i.e., these elements are masked). Masked arrays are in the
# `numpy.ma` package, and simply require a masked array to be created from
# a given array and a condition that indicates which elements should be
# masked. This new masked array can be used as a normal NumPy array,
# except the masked elements are ignored. NumPy provides [operations][2]
# for masked arrays, allowing them to be used in a similar manner as
# normal NumPy arrays.
# 
# You can also impute missing (or bad) values by using a masked array, and
# replacing masked elements with a different value, such as the mean
# value. Masked arrays can also be used to mask out bad values in a
# calculation such as divide-by-zero or logarithm of zero, and a masked
# array will ignore error conditions during standard operations, making
# them very useful since they operate in a graceful manner.
# 
# -----
# 
# [1]: http://docs.scipy.org/doc/numpy/reference/maskedarray.generic.html#examples
# [2]: http://docs.scipy.org/doc/numpy/reference/routines.ma.html
# 

# Create and demonstrate a masked array

import numpy.ma as ma

x = [0.,1.,-9999.,3.,4.]
print("Original array = :", x)


mx = ma.masked_values (x, -9999.)
print("\nMasked array = :", mx)


print("\nMean value of masked elements:", mx.mean())
print("\nOperate on unmaksed elements: ", mx - mx.mean())
print("\n Impute missing values (using mean): ", mx.filled(mx.mean())) # Imputation


# Create two arrays with masks
x = ma.array([1., -1., 3., 4., 5., 6.], mask=[0,0,0,0,1,0])
y = ma.array([1., 2., 0., 4., 5., 6.], mask=[0,0,0,0,0,1])

# Now take square root, ignores div by zero and masked elements.
print(np.sqrt(x/y))


# Now try some random data

d = np.random.rand(1000)

# Now mask for values within some specified range (0.1 to 0.9)
print("Masked array mean value: ", ma.masked_outside(d, 0.1, 0.9).mean())


# -----
# 
# ### NumPy File Input/Output
# 
# NumPy has support for reading or writing data to [files][1]. Of these,
# two of the most useful are the [`loadtxt` method][3] and the
# [`genfromext` method][2], each of which allow you to easily read data
# from a text file into a NumPy array. The primary difference is that the
# `genfromtxt` method can handle missing data, while the `loadtxt` can
# not. Both methods allow you to specify the column delimiter, easily
# skip header or footer rows, specify which columns should be extracted
# for each row, and allow you to _unpack_ the row so that each column goes
# into a separate array.
# 
# For example, the following code snippet demonstrates how to use the
# `loadtxt` method to pull out the second and fourth columns from the
# `fin` file handle, where the file is assumed to be in CSV format/ The
# data is persisted into the `a` and `b` NumPy arrays.
# 
# ```python
# a, b = np.loadtxt(fin, delimeter = ',', usecols=(1, 3), unpack=True)
# ```
# 
# We demonstrate the `genfromtxt` method in the following code block, where
# we first create the test data file before reading that data back into a
# NumPy array.
# 
# -----
# 
# [1]: http://docs.scipy.org/doc/numpy/user/basics.io.html
# [2]: http://docs.scipy.org/doc/numpy/user/basics.io.genfromtxt.html#defining-the-input
# [3]: http://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html#numpy.loadtxt
# 

# First write data to a file using Unix commands. 
info = "1, 2, 3, 4, 5 \n 6, 7, 8, 9, 10"
with open("test.csv", 'w') as fout:
    print(info, file=fout)

# Now we can read it back into a NumPy array.
d = np.genfromtxt("test.csv", delimiter=",")

print("New Array = \n", d)


# -----
# 
# In addition to the normal Python `help` function, NumPy provides a
# special `lookup` function that will search the NumPy library for
# classes, types, or methods that match the search string passed to the
# function. This can be useful for finding specific information given a
# general concept, or to learn more about related topics by performing a
# search.
# 
# -----
# 

np.lookfor('masked array')


# ### Additional References
# 
# 1. [Numpy Tutorial][1]
# 2. [Numpy Cheatsheet][2]
# 3. [Numpy Demonstration][3]
# 4. [NumPy Notebook Demo][4]
# -----
# 
# [1]: http://docs.scipy.org/doc/numpy/user/index.html
# [2]: http://pages.physics.cornell.edu/~myers/teaching/ComputationalMethods/python/arrays.html
# [3]: http://www.tp.umu.se/~nylen/pylect/intro/numpy/numpy.html
# [4]: http://nbviewer.ipython.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-2-Numpy.ipynb
# 

# <DIV ALIGN=CENTER>
# 
# # Introduction to Pandas & Databases
# ## Professor Robert J. Brunner
#   
# </DIV>  
# -----
# -----
# 

# ### Pandas and SQL
# 
# After a previous IPython Notebook explored how to use standard Python to work
# with a database, you have probably realized that there is a lot of
# standard code we must write to simply execute some SQL commands. While
# there are different Python libraries that exist to simplify these steps,
# we will focus on using the Pandas library, which is a standard library
# for doing _Data Analysis in Python_.
# 
# ----
# 

import sqlite3 as sl
import pandas as pd


# -----
# 
# Pandas provides built-in support for executing a SQL query and
# retrieving the result as a DataFrame. This is demonstrated in the next
# code cell, where we execute a SQL query on the airlines database. We
# select several columns, and for simplicity we restrict our query result
# to only ten rows by using the ANSI SQL `LIMIT` clause.
# 
# The Pandas method to execute a SQL statement is `read_sql`, and mimics
# in appearance other Panda methods for _reading_ data into a Pandas
# DataFrame. In this case, the method takes our SQL statement, database
# connection, and an optional parameter, `index_col` that we can use to
# specify which column in our result should be treated as an index column.
# Pandas will supply an auto-incrementing column if no column is explicitly
# supplied. To save space in the output display, we specify our own column
# in these examples.
# 
# 
# -----
# 

query = "SELECT code, airport, city, state, latitude, longitude FROM airports LIMIT 10 ;"

database = '/home/data_scientist/rppdm/database/rppds'

with sl.connect(database) as con:
    data = pd.read_sql(query, con, index_col ='code')
    
    print(data)


# -----
# 
# In the next code cell, we use the column selection feature with a Pandas
# DataFrame to select only those rows that have airports in the state of
# Mississippi. We do this by selecting the `state` attribute of the
# DataFrame, which corresponds to the _state_ column, and applying a
# Boolean condition.
# 
# -----
# 

query = "SELECT code, airport, city, state, latitude, longitude FROM airports LIMIT 100 ;"

with sl.connect(database) as con:
    data = pd.read_sql(query, con, index_col ='code')
    
    print(data[data.state == 'MS'])


# -----
# 
# Pandas also simplifies the insertion of new data into a SQL database.
# For this, we can simply take an existing Pandas DataFrame and call the
# `to_sql()` method. This method requires two parameters, the name of the
# database table, and the database connection. If the table does not
# exist, a new table will be created to match the DataFrame, including
# appropriate column names and data types. 
# 
# In the next two code blocks, we first query the airports table, and use
# Pandas to extract all airports in Illinois. We next insert this data
# back into our database as a new table called `ILAirports`. The following
# code block queries this new table and display the results for
# confirmation.
# 
# -----
# 

# Creating table automatically works better if columns are explicitly listed.

query = "SELECT code, airport, city, state, latitude, longitude FROM airports ;"
with sl.connect(database) as con:
    data = pd.read_sql(query, con)

    data[data.state == 'IL'].to_sql('ILAirports', con)


with sl.connect(database) as con:
    data = pd.read_sql('SELECT code, city, airport, latitude, longitude FROM ILAirports', 
                       con, index_col ='code')
    
    print(data[10:20])


# ### Additional References
# 
# 1. [Pandas Documentation][pdd]
# 2. A slightly dated Pandas [tutorial][pdt]
# -----
# 
# [pdd]: http://pandas.pydata.org/pandas-docs/stable/index.html
# [pdt]: http://www.gregreda.com/2013/10/26/intro-to-pandas-data-structures/
# 

# <DIV ALIGN=CENTER>
# 
# # Introduction to Data Visualization
# ## Professor Robert J. Brunner
#   
# </DIV>  
# -----
# -----
# 

# ## Introduction
# 
# In this notebook, we demonstrate a simple data science task that
# acquiring web-accessible data, processing the data, and generating a
# visual result. You should try changing the year in the unemployment data
# table to see how the result changes. You also could change the mapping
# between unemployment values and color codes. A final task would be to
# use a different type of data, rather then unemployment data to make this
# visualization.
# 
# -----
# 

# This example inspired by a post by Nathan Yau on FlowingData
# http://flowingdata.com/2009/11/12/how-to-make-a-us-county-thematic-map-using-free-tools/

import requests

resp = requests.get("http://upload.wikimedia.org/wikipedia/commons/5/5f/USA_Counties_with_FIPS_and_names.svg")
usa = resp.content


from IPython.display import display_svg

with open('usa.svg', 'wb') as fout:
    fout.write(usa)


# Display the original image. This is an SVG so it can scale seamlessly

# The data file on the website http://www.bls.gov/lau/#tables
# spans the years 1990 to 2013, so we can simply change the 13 
# in the URL to a different value to get data/plot for a different year.

display_svg(usa, raw=True)


# The data file on the website http://www.bls.gov/lau/#tables
# spans the years 1990 to 2013, so we can simply change the 13 
# in the URL to a different value to get data/plot for a different year.

# Grab and parse the Labor data by using Pandas

import pandas as pd 

# We specify our Column Names

headers = ['LAUS Code', 'StateFIPS', 'CountyFIPS', 'County Name', 'Year',            'Labor Force', 'Employed', 'Unemployed', 'Rate']

# The column widths for fixed width formatting
cs =[(0, 16), (17, 21), (22, 28), (29, 79), (80, 85), (86, 99), (100, 112), (113, 123), (124, 132)]

# The converter functions. We simply use string, we can't use dtypes with a Python engine
cvf = {0 : str, 1 : str, 2: str, 3 : str, 4 : str, 5 : str, 6 : str, 7 : str, 8 : str}

# Read in the data. We skip first five rows that are header info
ud = pd.read_fwf('http://www.bls.gov/lau/laucnty13.txt', converters = cvf, colspecs = cs,                  skiprows=5, header=0, names=headers)

# We drop last three rows that contain footer info.

ud = ud.dropna()

# Now we build a second DataFrame that has the Rate indexed by the FIPS code.
unemployment = pd.DataFrame(ud.Rate.astype(float))
unemployment['FIPS'] = ud.StateFIPS + ud.CountyFIPS
unemployment.set_index('FIPS', inplace = True)

# Test the result

print(unemployment.head(3))
print(unemployment.tail(3))


# Now we turn to parsing the SVG file to modify the styles.

from bs4 import BeautifulSoup

# The SVG file is an XML file so parse appropriately.

soup = BeautifulSoup(usa)

# Find distinct FIPS zones, which are each in a different path.
paths = soup.findAll('path')

# For Color selection, this is a fgreat site:
# http://colorbrewer2.org

# Default FlowingData Map colors
#colors = ["#F1EEF6", "#D4B9DA", "#C994C7", "#DF65B0", "#DD1C77", "#980043"]
 
# I Prefer Blues, and more variation.
colors = ['#eff3ff', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
    
# FlowingData County style (I changed opacity to 0.75 from 1)
path_style = 'font-size:12px;fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:0.75;' +     'stroke-width:0.1;stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt;' +     'marker-start:none;stroke-linejoin:bevel;fill:'

# Now iterate through the path elements to modify the path style based on the unemployment rate
for p in paths:
    
    # We need to not try to modify two special paths for State Lines or Separators
    
    if p['id'] not in ['State_lines', 'separator ']:
        try:
            # We simply access our Panda DataFrame
            rate = unemployment.Rate[p['id']]
        except:
            continue
            
        # Now we simply cascade through the unemployemtn rates. Ideally we chance this to a function

        if rate > 11.9:
            color_class = 6
        elif rate > 9.9:
            color_class = 5
        elif rate > 9.9:
            color_class = 4
        elif rate > 7.9:
            color_class = 3
        elif rate > 5.9:
            color_class = 2
        elif rate > 3.9:
            color_class = 1
        else:
            color_class = 0

        # Modify color by our scheme
        color = colors[color_class]
        
        # Now set the nw path style. Ideally again this is built by using string formatting.
        p['style'] = path_style + color
 
# Now our parse tree is done so output it appropriately.
cusa  = soup.prettify()

# And show the result.
display_svg(soup, raw=True)

with open('cusa.svg', 'w') as fout:
    fout.write(cusa)


# Alternative display method. Can use this to compare to different year's values.

from IPython.display import HTML
HTML(cusa)


# ### Additional References
# 
# 1. [XML Tutorial][1] by W3Schools.
# 3. [SVG Tutorial][3] by W3Schools.
# 
# -----
# 
# [1]: http://www.w3schools.com/xml/default.asp
# [3]: http://www.w3schools.com/svg/default.asp
# 

