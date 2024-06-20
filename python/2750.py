# %load ../standard_import.txt
import IPython as ipy
import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
import sklearn as skl
import statsmodels as sm
import scipy as sp
import pydot as pdt
import patsy as pat


print('IPython {}'.format(ipy.__version__))
print('pandas {}'.format(pd.__version__))
print('numpy {}'.format(np.__version__))
print('scikit-learn {}'.format(skl.__version__))
print('statsmodels {}'.format(sm.__version__))
print('patsy {} (For regression splines)'.format(pat.__version__))
print('matplotlib {}'.format(mpl.__version__))
print('seaborn {}'.format(sns.__version__))
print('scipy {}'.format(sp.__version__))
print('pydot {} (For visualizing tree based models)'.format(pdt.__version__))


# For pydot I used the following fork to have python-3 support: https://github.com/nlhepler/pydot
# 

