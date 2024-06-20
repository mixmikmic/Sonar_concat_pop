# # Analysing Brazil - Life Expectancy versus GDP
# 

# Getting the data
# Initiallly, three datasets of the World Bank are considered. From there we took Brazil's data for all included years.
# - One dataset, available at http://data.worldbank.org/indicator/NY.GDP.MKTP.CD, lists the GDP of the world's countries in current US dollars, for various years.
# - The other dataset, available at http://data.worldbank.org/indicator/SP.DYN.LE00.IN, lists the life expectancy of the world's countries.
# - The last one: https://data.worldbank.org/indicator/SP.POP.TOTL, lists the population.
# 

import warnings
warnings.simplefilter('ignore', FutureWarning)

from pandas import *
from pandas_datareader.wb import download

get_ipython().run_line_magic('matplotlib', 'inline')


# Function to return rounded values in 'thousants'
def roundTo1000 (value):
    return round(value/1000)


# Function to return rounded values in millions
def roundToMillions (value):
    return round(value / 1000000)


# Download World Bank Population indicator
INDICATOR = 'SP.POP.TOTL'
START_YEAR = 1960
END_YEAR = 2016
COUNTRY = 'BRA'

popBr = download(indicator=INDICATOR, country=COUNTRY, start=START_YEAR, end=END_YEAR)
popBr.shape # (57, 1)


popBr.info()


# Download World Bank GDP indicator
INDICATOR = 'NY.GDP.MKTP.CD'
START_YEAR = 1960
END_YEAR = 2016
COUNTRY = 'BRA'

gdpBr = download(indicator=INDICATOR, country=COUNTRY, start=START_YEAR, end=END_YEAR)
gdpBr.shape # (57, 1)


gdpBr.info()


# Download World Bank Life Expectance indicator
INDICATOR = 'SP.DYN.LE00.IN'
START_YEAR = 1960
END_YEAR = 2016
COUNTRY = 'BRA'

lifeBr = download(indicator=INDICATOR, country=COUNTRY, start=START_YEAR, end=END_YEAR)
lifeBr.shape # (57, 1)


lifeBr.info()


# Transforming
popBr['popTot(1000)'] = popBr['SP.POP.TOTL'].apply(roundTo1000)  
gdpBr['gdpTot(USDm)'] = gdpBr['NY.GDP.MKTP.CD'].apply(roundToMillions) 
lifeBr['lifeExp(years)'] = lifeBr['SP.DYN.LE00.IN'].apply(round) 


popBr = popBr.reset_index()
gdpBr = gdpBr.reset_index()
lifeBr = lifeBr.reset_index()


dataBr = merge(popBr, gdpBr)


dataBr = merge(dataBr, lifeBr)


dataBr.head()


dataBr = dataBr[['year', 'popTot(1000)', 'gdpTot(USDm)', 'lifeExp(years)']]


dataBr.head()


dataBr['gdpPc(USD)'] = ((dataBr['gdpTot(USDm)'] / dataBr['popTot(1000)']) *1000).apply(round)


dataBr.tail()


YEAR = 'year'
POP = 'popTot(1000)'
GDP = 'gdpTot(USDm)'
GDP_PC = 'gdpPc(USD)'
LIFE ='lifeExp(years)'


dataBr = dataBr.sort_values(YEAR)


dataBr.plot(x=YEAR, y=LIFE, kind='line', grid=True, logx=False, figsize = (10, 4))


dataBr.plot(x=YEAR, y=GDP, kind='line', grid=True, logy=False, figsize = (10, 4))


dataBr.plot(x=YEAR, y=POP, kind='line', grid=True, logy=False, figsize = (10, 4))


dataBr.plot(x=YEAR, y=GDP_PC, kind='line', grid=True, logy=False, figsize = (10, 4))


from scipy.stats import spearmanr

dataA = dataBr[GDP]
dataB = dataBr[LIFE]
(correlation, pValue) = spearmanr(dataA, dataB)
print('The correlation is', correlation)
if pValue < 0.05:
    print('It is statistically significant.')
else:
    print('It is not statistically significant.')


dataBr.plot(x=GDP, y=LIFE, kind='scatter', grid=True, logx=False, figsize = (10, 4))


# ## Conclusion
# 

# There is a total correlation between the GDP Growth and a higher poplulation Life expectancy - More money, more years to live!
# 

