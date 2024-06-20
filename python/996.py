# # Gold contango
# 
# We examine futures prices for gold, in particular, the contango situation. 
# The futures price may be either higher or lower than the spot price. 
# When the spot price is higher than the futures price, the market is said to 
# be in **backwardation**. If the spot price is lower than the futures price, 
# the market is in **contango**.
# 
# The futures or forward curve would typically be upward sloping, 
# since contracts for further dates would typically trade at even higher prices. 
# A contango is normal for a non-perishable commodity that has 
# a *cost of carry*. Such costs include warehousing fees and 
# interest forgone on money tied up, 
# less income from leasing out the commodity if possible (e.g. gold). 
# 
# Our study examines a segment of the futures curve, specifically the 
# nearby contract versus another dated six months thereafter, 
# for gold traded on the COMEX exchange. We use the expected 
# LIBOR interest rate for the identical segment to adjust the 
# cost of carry. We then compare this supply/demand indicator against spot prices. 
# 
# The *London Bullion Market Association* ceased publishing daily data 
# on their *Gold Forward Offered Rate* (**GOFO**), as of 30 January 2015 -- 
# so we develop an observable proxy called *tango*. 
# 
# During 2015 we detected strong *negative* correlation between price change and tango, 
# however, in 2016 that strong correlation became *positive* -- 
# thus we conclude the relationship is spurious. 
# The observed correlations are mere artifacts 
# which do not imply any significant economic relationships.
# 
# Tango as an indicator is **not** insightful for price changes 
# whenever it is at the extremes of its expected range. 
# This can be verified by comparing various commits over time 
# of this notebook in the repository.
# 
# Short URL: https://git.io/xau-contango
# 

# *Dependencies:*
# 
# - fecon235 repository https://github.com/rsvp/fecon235
# - Python: matplotlib, pandas
#      
# *CHANGE LOG*
# 
#     2016-12-04  Solve #2 by v5 and PREAMBLE-p6.16.0428 upgrades. 
#                    Switch from fecon to fecon235 for main import module. 
#                    Minor edits given more data and change in futures basis.
#                    Previous conclusion is negated. Correlation is artificial.
#     2015-10-11  Code review.
#     2015-09-11  First version.
# 

from fecon235.fecon235 import *


#  PREAMBLE-p6.16.0428 :: Settings and system details
from __future__ import absolute_import, print_function
system.specs()
pwd = system.getpwd()   # present working directory as variable.
print(" ::  $pwd:", pwd)
#  If a module is modified, automatically reload it:
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
#       Use 0 to disable this feature.

#  Notebook DISPLAY options:
#      Represent pandas DataFrames as text; not HTML representation:
import pandas as pd
pd.set_option( 'display.notebook_repr_html', False )
from IPython.display import HTML # useful for snippets
#  e.g. HTML('<iframe src=http://en.mobile.wikipedia.org/?useformat=mobile width=700 height=350></iframe>')
from IPython.display import Image 
#  e.g. Image(filename='holt-winters-equations.png', embed=True) # url= also works
from IPython.display import YouTubeVideo
#  e.g. YouTubeVideo('1j_HxD4iLn8', start='43', width=600, height=400)
from IPython.core import page
get_ipython().set_hook('show_in_pager', page.as_hook(page.display_page), 0)
#  Or equivalently in config file: "InteractiveShell.display_page = True", 
#  which will display results in secondary notebook pager frame in a cell.

#  Generate PLOTS inside notebook, "inline" generates static png:
get_ipython().magic('matplotlib inline')
#          "notebook" argument allows interactive zoom and resize.


#  SET UP the particular (f4) futures contracts of interest:
s_libor = 'f4libor16z'
s_xau1  = 'f4xau16z'
s_xau2  = 'f4xau17m'

#  f4libor* refers to the CME Eurodollar futures.

#  The second nearby contract for gold (xau)
#  should be 6 months after the first using the 
#  June (m) and December (z) cycle

#  RE-RUN this entire study by merely changing the string symbols.


#  Retrieve data:
libor = todf( 100 - get(s_libor) )
#             ^convert quotes to conventional % format
xau1 = get(s_xau1)
xau2 = get(s_xau2)


tail(libor)


tail(xau1)


tail(xau2)


# Usually contango is described in price unit terms, however, we prefer the scale-free annualized percentage format. This places the measure on par with the way interest rates are usually quoted.
# 

#  Compute the contango in terms of annualized percentage:
contango = todf( ((xau2 / xau1) - 1) * 200 )

#  Multiply by 200 instead of 100 since 
#  the gold contracts are stipulated to be six months apart.


tail( contango )


plot( contango )


# The largest variable component to the cost-of-carry is **interest**. We filter that out by subtracting the LIBOR rate obtained from the futures on Eurodollars. We shall call the result: **tango**.
# 

tango = todf( contango - libor )


tail( tango )


#  MAIN chart <- pay attention here !!
plot( tango )


tango.describe()


# Usually *tango* has approximate mean of zero, with somewhat 
# wide latitude: 2015-10-09 at 18 bp standard deviation (annualized), 
# 2016-12-02 at 42 bp standard deviation (annualized).
# 
# Since warehousing costs for gold are fairly constant across time, 
# changes in tango mainly reflect supply and demand. 
# A major component of tango is the **leasing rate**. 
# 
# The *London Bullion Market Association* had published daily data 
# on the *Gold Forward Offered Rate*, or **GOFO**.  These are rates 
# at which LBMA members are prepared to lend gold on a swap against 
# U.S. dollars. Historically there has been negative leasing rates, 
# meaning that the lessors were willing to actually pay you to borrow gold from 
# them [but mostly likely it is a  central bank is paying some 
# bullion bank to take their gold]. 
# Unfortunately, the GOFO dataset has been **discontinued** as of 30 January 2015.
# 

#  For historians:
Image(url='https://www.mcoscillator.com/data/charts/weekly/GOFO_1mo_1995-2014.gif', embed=False)


# ## Relationship to cash prices
# 
# We now look at the current London PM gold fix series.
# 

xau = get( d4xau )


#  This matches the futures sample size:
xau0 = tail( xau, 512 )


plot( xau0 )


#  Is there near-term correlation between price and tango?
#  stat2( xau0[Y], tango[Y] )
#  2015-09-11  correlation: 0.09, so NO.


#  Running annual percentage change in spot price:
xau0pc = tail( pcent(xau, 256), 512 )


plot ( xau0pc )


#  Is there near-term correlation between price change and tango?
stat2( xau0pc[Y], tango[Y] )
#  2015-09-11  correlation: -0.85, so YES.
#  2015-10-09  correlation: -0.83
#  2016-12-02  correlation: +0.81, but change in sign!


# ## Closing comments 2015-10-11
# 
# So roughly speaking, **increasing tango is correlated to decling prices.** 
# Thus increasing selling pressure in the near-term versus the 
# distant-term (which in effect widens tango) is correlated to 
# future declines in cash prices. 
# Equivalently, **decreasing tango is correlated to increasing prices.**
# 
# Since tango is currently near its mean value, 
# it seems equally likely to decrease or increase 
# (though its short-term trend is upwards), so the 
# future direction of gold prices seems inconclusive.
# 
# ## Closing comments 2016-12-02
# 
# The new data over the last 13 months has reversed the sign 
# of the correlation. In other words, the relationship previously 
# noted in closing is spurious. 
# (Please see the previous commit of this entire notebook 
# for specific details.)
# 
# Tango as an indicator is [**not**] insightful for price changes 
# whenever it is at the extremes of its expected range.
# 

# # Parsing SEC 13F forms
# 
# We examine 13F filings which are quarterly reports filed per SEC regulations
# by institutional investment managers containing all equity assets under management
# of at least \$ 100 million in value. **Form 13F is required to be filed
# within 45 days of the end of a calendar quarter**
# (*which should be considered as significant information latency*).
# 
# Form 13F *only reports long* positions.
# Short positions are not required to be disclosed and are not reported.
# Section 13(f) securities generally include equity securities
# that trade on an exchange (including Nasdaq), certain equity options and warrants,
# shares of closed-end investment companies, and certain convertible debt securities.
# The shares of open-end investment companies
# (i.e. mutual funds) are not Section 13(f) securities.
# See [Official List of Section 13(f) Securities](http://www.sec.gov/divisions/investment/13flists.htm) and our caveats section below.
# 
# Form 13F surprisingly excludes total portfolio value and percentage allocation
# of each stock listed. We remedy that, and also parse the report for easy reading.
# Our notebook then develops into a module **yi_secform** which will do
# all the work via one function.
# 
# As specific example, we follow Druckenmiller and Paulson as asset managers
# who have significant positions in GLD, a gold ETF.
# We show the Druckenmiller's sudden accumulation,
# and Paulson's dramatic liquidation.
# 
# *Top holdings are easily analyzed by a single Python module:* **yi_secform**
# Caveats are disclosed in the first section.
# 
# Shortcut to this notebook: https://git.io/13F
# 

# *Dependencies:*
# 
# - Repository: https://github.com/rsvp/fecon235 -- Module: yi_secform
# - Python: pandas, numpy, lxml, bs4, html5lib
#      
# *CHANGE LOG*
# 
#     2016-02-22  Fix issue #2 by v4 and p6 updates.
#                    Replace .sort(columns=...) with .sort_values(by=...)
#                    since pandas 0.17.1 gives us future deprecation warning.
#                    Paulson liquidates 37.6% of his GLD inventory.
#     2015-11-16  Update in Appendix for Druckenmiller and Paulson.
#     2015-08-28  First version.
# 

from fecon235.fecon235 import *

#  pandas will give best results
#  if it can call the Python package: lxml,
#  and as a fallback: bs4 and html5lib.
#  They parse (non-strict) XML and HTML pages.
#  Be sure those three packages are pre-installed.

from fecon235.lib import yi_secform
#  We are going to derive this module in this notebook.


#  PREAMBLE-p6.15.1223 :: Settings and system details
from __future__ import absolute_import, print_function
system.specs()
pwd = system.getpwd()   # present working directory as variable.
print(" ::  $pwd:", pwd)
#  If a module is modified, automatically reload it:
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
#       Use 0 to disable this feature.

#  Notebook DISPLAY options:
#      Represent pandas DataFrames as text; not HTML representation:
import pandas as pd
pd.set_option( 'display.notebook_repr_html', False )
#  Beware, for MATH display, use %%latex, NOT the following:
#                   from IPython.display import Math
#                   from IPython.display import Latex
from IPython.display import HTML # useful for snippets
#  e.g. HTML('<iframe src=http://en.mobile.wikipedia.org/?useformat=mobile width=700 height=350></iframe>')
from IPython.display import Image 
#  e.g. Image(filename='holt-winters-equations.png', embed=True) # url= also works
from IPython.display import YouTubeVideo
#  e.g. YouTubeVideo('1j_HxD4iLn8', start='43', width=600, height=400)
from IPython.core import page
get_ipython().set_hook('show_in_pager', page.as_hook(page.display_page), 0)
#  Or equivalently in config file: "InteractiveShell.display_page = True", 
#  which will display results in secondary notebook pager frame in a cell.

#  Generate PLOTS inside notebook, "inline" generates static png:
get_ipython().magic('matplotlib inline')
#          "notebook" argument allows interactive zoom and resize.


# ## Caveats regarding 13F
# 
# - 13F filings disclose hedge fund long positions in US equity markets, American Depositary Receipts (ADRs), both put and call options, as well as convertible notes. **They do not disclose short sales, cash positions, or any other asset class.**
# 
# - Yet performance results for the [clones](http://blog.alphaclone.com/alphaclone/2011/01/clone-vs-fund-2010.html) are somewhat in line with the actual performance of the fund. E.g. Paulson & Co (John Paulson)
# 
# - Clones track managers that normally run net long. Tracking global macro funds (Bridgewater, Tudor) or credit funds (Fortress, Cerberus) is misguided because the vast majority of their positions are in asset classes that they don't have to disclose (futures, commodities, bonds, currencies, etc).  And while quant funds (RenTec, AQR) often disclose stocks, following them is a folly because you have absolutely no idea why their algorithms bought in the first place (statistical arbitrage may entail positions in foreign countries).  
# 
# - 13F does not reveal international holdings (except for ADR's).
# 
# - Follow long-term oriented funds to reduce the effect of the delay in 13F disclosures.
# 
# - Money managers allocate the most capital to their best ideas. Pay attention to "new positions" in their disclosures as these are their most recent ideas. 
# 
# - Always remember that the **13F is not their whole portfolio and that it's a past snapshot.**
# 
# - Monitor all SEC filings, not just 13F's:  13G filings, 13D filings, as well as various Form 3 and Form 4's are filed on a more timely basis and provide a more current look at what managers are buying or selling.  They are required to file when they've purchased 5% or more of a company. 
# 
# - Caveats source: http://www.marketfolly.com/2012/10/hedge-fund-13f-filing-pros-and-cons.html
# 

# ## Analysis of the 13F format
# 
# Source: http://www.sec.gov/answers/form13f.htm
# 
# 
# ### Obtaining 13F
# 
# You can search for and retrieve Form 13F filings using the [SEC's EDGAR database](http://www.sec.gov/edgar/searchedgar/companysearch.html). To find the filings of a particular money manager, enter the money manager's name in the Company Name field. To see all recently filed 13Fs, use the ["Latest Filings"](http://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent) search function and enter "13F" in the Form Type box.
# 
# 
# ### Example: Druckenmiller, 14 August 2015 filing
# 
# Stanley Druckenmiller closed his Duquesne Capital Management hedge fund in 2010 but he still discloses the holdings of his family office (search: [“Duquesne Family Office”](http://www.sec.gov/cgi-bin/browse-edgar?company=Duquesne+Family+Office&owner=exclude&action=getcompany) picking the Information Table in html).
# 
# He made headlines because of his new acquisition in gold (the GLD ETF) which now is his largest position (about 22% of his entire portfolio)!
# 

#            https cannot be read by lxml, surprisingly.
druck150814='http://www.sec.gov/Archives/edgar/data/1536411/000153641115000006/xslForm13F_X01/form13f_20150630.xml'


# ## HOWTO parse 13F reports: deriving our module
# 
# If bugs appear because of format changes at the SEC
# this section will useful for interactive debugging.
# 
# Take note that: **lxml** *cannot read https, so use http
# when specifying URL.*
# 

#     START HERE with a particular URL:
url = druck150814


#  Let's display the web page as in the browser to understand the semantics:
HTML("<iframe src=" + url + " width=1400 height=350></iframe>")


#  Use pandas to read in the xml page...
#  See http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_html.html

#  It searches for <table> elements and only for <tr> and <th> rows and <td> elements 
#  within each <tr> or <th> element in the table.

page = pd.read_html( url )


#  Nasty output in full:

#uncomment:  page


#  page is a list of length 4:
len( page )


#  But only the last element of page interests us:
df = page[-1]
#  which turns out to be a dataframe!


#  Let's rename columns for our sanity:
df.columns = [ 'stock', 'class', 'cusip', 'usd', 'size', 'sh_prin', 'putcall', 'discret', 'manager', 'vote1', 'vote2', 'vote3'] 


#  But first three rows are SEC labels, not data, 
#  so delete them:
df = df[3:]

#  Start a new index from 0 instead of 3:
df.reset_index( drop=True )


#  Delete irrevelant columns:
dflite = df.drop( df.columns[[1, 4, 5, 7, 8, 9, 10, 11]], axis=1 )
#         inplac=True only after pandas 0.13
#uncomment: dflite


#  usd needs float type since usd was read as string:
dflite[['usd']] = dflite[['usd']].astype( float )
#                  Gotcha: int as type will fail for NaN

#  Type change allows proper sort:
dfusd = dflite.sort_values( by=['usd'], ascending=[False] )


usdsum = sum( dfusd.usd )
#  Portfolio total in USD:
usdsum


#  New column for percentage of total portfolio:
dfusd['pcent'] = np.round(( dfusd.usd / usdsum ) * 100, 2)


#  Top 20 Hits!
dfusd.head( 20 )


# ## Summary module: yi_secform
# 
# **We sum up our work above in a module for easy execution using one function.**
# 

get_ipython().magic('pinfo2 yi_secform.pcent13f')


yi_secform.pcent13f( druck150814, 20 )

#  Simply enter the Information Table html URL for a 13F filing, 
#  and bang... [verifying our output in the previous cell]:


# ## Quick look at John Paulson
# 
# For comparison, since the press has been stating Paulson has been selling the GLD ETF due to redemptions. 13F search page: http://www.sec.gov/cgi-bin/browse-edgar?company=Paulson+%26+Co.&owner=exclude&action=getcompany
# 

#  13F for Paulson & Co. filed 2015-08-14:
paulson150814 = 'http://www.sec.gov/Archives/edgar/data/1035674/000114036115032242/xslForm13F_X01/form13fInfoTable.xml'


yi_secform.pcent13f( paulson150814, 20 )


# ### Gold ETF comment, August 2015
# 
# Paulson is usually the largest stakeholder in GLD also known as "SPDR GOLD TRUST". The latest 13F shows he holds \$1.037 billion worth which is equivalent to about 886,183 troy ounces (*27.6 metric tons*). It actually only represents half of his very top equity holding: Allergan at 10% allocation.
# 
# Druckenmiller's 13F disclosing his Family Office (not a public operation like Paulson & Co) shows a \$0.324 billion position in GLD, equivalent to about 276,367 troy ounces (*8.6 metric tons*). It appears to be a new bold trade since it is his largest holding at 22% allocation. 
# 
# [Spot gold on 2015-06-30, end of second calendar quarter, was fixed at \$1171 in London.]
# 
# *We do not truly know their respective net positions because they **could be hedged in the gold futures market.** Futures positions are excluded from 13F filings, as well as cash position (which would fund any further accumulation).*
# 
# Remember that we viewing past snapshots of positions dated 30 June 2015. Such positions could have been entirely liquidated during July and August when the gold market declined severely. On the other hand, gold rallied considerably days after the market received Druckenmiller's vote of bold confidence via his 13F filed on August 14th. 
# 

# ## Appendix: Getting quotes
# 
# GLD, the ETF for gold, is designed to track spot gold prices,
# less their management fees.
# Within fecon235, one can easily retrieve stock or ETF quotes,
# for example, by this syntax: get('s4gld'),
# noting the string 's4' concatenated with the symbol
# in lower case.
# 
# Notice that the SEC 13F requires unique CUSIP identifiers,
# rather than ticker symbols for their 13F forms.
# 
# For gold, we prefer to use equivalent measures used in the spot market.
# For example, quotes here are given in USD per troy ounce,
# and the London PM fix (rather than nearby futures) is
# accepted as a benchmark.
# GLD valuation can be converted into such spot terms.
# Within fecon235: get(d4xau)
# will retrieve the appropriate dataframe for you.
# See https://git.io/gold for more details.
# 

# ## Appendix: November 2015 update
# 

druck151113 = 'http://www.sec.gov/Archives/edgar/data/1536411/000153641115000008/xslForm13F_X01/form13f_20150930.xml'
paulson151116 = 'http://www.sec.gov/Archives/edgar/data/1035674/000114036115041689/xslForm13F_X01/form13fInfoTable.xml'


# Druckenmiller 13F for 2015-11-13:
yi_secform.pcent13f( druck151113, 20 )


#  Paulson 13F for 2015-11-16:
yi_secform.pcent13f( paulson151116, 20 )


# [Spot gold on 2015-09-30, end of third calendar quarter, was fixed 
# at $1114 in London, -4.9% from previous quarter.]
# 
# 2015-11-16: GLD is no longer Druckenmiller's top holding (replaced by Facebook): 
# change from last quarter "323626 NaN 21.81" to "307757 Call 26.17" -- 
# however, its allocation has been increased, though net position is 
# unchanged (given price decrease in the spot market). 
# Curiously it appears the underlying instrument has *shifted to calls*. 
# The expiration date of the calls are not known, but it is 
# indicative of a shift to short-term trading perspective. 
# Druckenmiller's gold downside is now limited to the premiums paid.
# 
# As for Paulson: change from last quarter: "1037720 NaN 4.79" to "986836 NaN 5.12" -- 
# indicates *no change in position* since the GLD valuation 
# mirrors the decrease in the spot gold market.
# 

# ## Appendix: February 2016 update
# 

druck160216='http://www.sec.gov/Archives/edgar/data/1536411/000153641116000010/xslForm13F_X01/form13f_20151231.xml'
paulson160216='http://www.sec.gov/Archives/edgar/data/1035674/000114036116053318/xslForm13F_X01/form13fInfoTable.xml'


# Druckenmiller 13F for 2016-02-16:
yi_secform.pcent13f( druck160216, 20 )


#  Paulson 13F for 2016-02-16:
yi_secform.pcent13f( paulson160216, 20 )


# [Spot gold on 2015-12-31, end of fourth calendar quarter, was fixed at $1060 in London, -4.85% from previous quarter.]
# 
# 2016-02-21: GLD is back to Druckenmiller's top holding: change from last quarter "307757 Call 26.17" to "292205 Call 29.90". If the net position was unchanged (given price decrease in the spot market), we can attribute a loss of \$ 626,000 due to decay in the call option valuation (cf. theta).
# 
# #### 2015-Q4: Drama for Paulson
# 
# As for Paulson: change from last quarter: "986836 NaN 5.12" to "585933 NaN 3.50" -- indicates HUGE change! Market movement alone would give 938974 valuation, so the
# **realized displacement on GLD is \$ 353,041,000, i.e. -353 million USD.** 
# 
# Paulson is usually the largest stakeholder in GLD also known as "SPDR GOLD TRUST".
# The 13F parsed in August 2015 showed \$ 1.037 billion worth which was equivalent to
# about 886,183 troy ounces (27.6 metric tons).
# The 13F currently shows \$ 0.586 billion worth which is equivalent to
# about 552,767 troy ounces -- a reduction of -37.6% in gold inventory.
# 
# Thus we can estimate that *Paulson liquidated the equivalent of
# about 333,416 troy ounces at a weighted average price of \$ 1059*
# (note: recorded low during 2016-Q4 for the London PM Gold fix was \$ 1049).
# From that weighted average and gold's price history,
# we can discern that the liquidation took place largely in late December 2015.
# 

# # SymPy tutorial
# 
# **SymPy** is a Python package for performing **symbolic mathematics**
# which can perform algebra, integrate and differentiate equations, 
# find solutions to differential equations, and *numerically solve
# messy equations* -- along other uses.
# 
# CHANGE LOG
#     
#     2017-06-12  First revision since 2015-12-26.
# 
# Let's import sympy and initialize its pretty print functionality 
# which will print equations using LaTeX.
# Jupyter notebooks uses Mathjax to render equations
# so we specify that option.
# 

import sympy as sym
sym.init_printing(use_latex='mathjax')

#  If you were not in a notebook environment,
#  but working within a terminal, use:
#
#  sym.init_printing(use_unicode=True)


# ## Usage
# 
# These sections are illustrated with examples drawn from
# [rlabbe](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/Appendix-A-Installation.ipynb) from his appendix for Kalman Filters.
# 
# It is important to distinguish a Python variable
# from a **declared symbol** in sympy.
# 

phi, x = sym.symbols('\phi, x')

#  x here is a sympy symbol, and we form a list:
[ phi, x ]


# Notice how we used a LaTeX expression for the symbol `phi`.
# This is not necessary, but if you do the output will render nicely as LaTeX.
# 
# Also notice how $x$ did not have a numerical value for the list to evaluate.
# 
# So what is the **derivative** of $\sqrt{\phi}$ ?
# 

sym.diff('sqrt(phi)')


# We can **factor** equations:
# 

sym.factor( phi**3 - phi**2 + phi - 1 )


# and we can **expand** them:
# 

((phi+1)*(phi-4)).expand()


# You can also use strings for equations that use symbols that you have not defined:
# 

x = sym.expand('(t+1)*2')
x


# ## Symbolic solution
# 
# Now let's use sympy to compute the **Jacobian** of a matrix. 
# Suppose we have a function,
# 
# $$h=\sqrt{(x^2 + z^2)}$$
# 
# for which we want to find the Jacobian with respect to x, y, and z.
# 

x, y, z = sym.symbols('x y z')

H = sym.Matrix([sym.sqrt(x**2 + z**2)])

state = sym.Matrix([x, y, z])

H.jacobian(state)


# Now let's compute the discrete process noise matrix $\mathbf{Q}_k$ given the continuous process noise matrix 
# $$\mathbf{Q} = \Phi_s \begin{bmatrix}0&0&0\\0&0&0\\0&0&1\end{bmatrix}$$
# 
# and the equation
# 
# $$\mathbf{Q} = \int_0^{\Delta t} \Phi(t)\mathbf{Q}\Phi^T(t) dt$$
# 
# where 
# $$\Phi(t) = \begin{bmatrix}1 & \Delta t & {\Delta t}^2/2 \\ 0 & 1 & \Delta t\\ 0& 0& 1\end{bmatrix}$$
# 

dt = sym.symbols('\Delta{t}')

F_k = sym.Matrix([[1, dt, dt**2/2],
                  [0,  1,      dt],
                  [0,  0,      1]])

Q = sym.Matrix([[0,0,0],
                [0,0,0],
                [0,0,1]])

sym.integrate(F_k*Q*F_k.T,(dt, 0, dt))


# ## Numerical solution
# 
# You can find the *numerical value* of an equation by substituting in a value for a variable:
# 

x = sym.symbols('x')

w = (x**2) - (3*x) + 4
w.subs(x, 4)


# Typically we want a numerical solution where the analytic solution is messy,
# that is, we want a **solver**.
# This is done by specifying a sympy equation, for example:
# 

LHS = (x**2) - (8*x) + 15
RHS = 0
#  where both RHS and LHS can be complicated expressions.

solved = sym.solveset( sym.Eq(LHS, RHS), x, domain=sym.S.Reals )
#  Notice how the domain solution can be specified.

solved
#  A set of solution(s) is returned.


#  Testing whether any solution(s) were found:
if solved != sym.S.EmptySet:
    print("Solution set was not empty.")


#  sympy sets are not like the usual Python sets...
type(solved)


#  ... but can easily to converted to a Python list:
l = list(solved)
print( l, type(l) )


LHS = (x**2)
RHS = -4
#  where both RHS and LHS can be complicated expressions.

solved = sym.solveset( sym.Eq(LHS, RHS), x )
#  Leaving out the domain will include the complex domain.

solved


# ## Application to financial economics
# 
# We used sympy to deduce parameters of Gaussian mixtures
# in module `lib/ys_gauss_mix.py` and the explanatory notebook
# is rendered at https://git.io/gmix 
# 

# # fecon235: Introductory documentation
# 
# Here we discuss the usage of the *fecon235* repository 
# for the casual user (while the development of the API 
# is still in progress). 
# 
# **As of v4, all *modules* will work with Python 2.7 and 3 series.** 
# However, some of the pre-2016 *notebooks* may still have python2 
# idioms and Linux dependencies. Those are easy to fix as we update. 
# Our goal is cross-platform performance (Linux, Mac, and Windows) 
# as well as compliance with both Python kernels available to 
# Jupyter notebooks (forked from IPython).
# 
# To see examples of code, please pick out a subject of interest under 
# the **nb** directory, and view that notebook at GitHub. 
# Better yet, fork this project, and execute the notebook locally as 
# you interactively experiment. 
# For developers, the main modules are located under the **lib** directory.
# 
# ## Importing the project
# 
# For python3 conformity, we have adopted absolute_import 
# throughout this project. 
# So first be sure that your PYTHONPATH can lead up to 
# the fecon235 directory. Then the following import 
# permits *easy command access*. The top-level module is 
# customarily given the same name as the project. In our case, 
# it conveniently unifies and exposes our essential lib modules 
# (older notebooks imported yi-prefixed modules individually).
# 

#  Call the MAIN module: 
from fecon235.fecon235 import *
#  This loose import style is acceptable only within 
#  interactive environments outside of any fecon235 packages.
#  (Presence of __init__.py in a directory indicates 
#  it is a "package.") 
#
#  These directories: nb and tests, are explicitly NOT packages.


# To use fecon235 in other projects, here are some proper examples:
# 
#     from fecon235 import fecon235 as fe
#     from fecon235.lib import yi_secform
#     
# If we had used the first example in our notebooks, 
# a function would require extra typing, 
# e.g. *fe.get()* instead of plain *get()*. 
# Any lib module can be imported directly 
# if specialized procedures are required. 
# The second example is used to parse SEC forms. 
# An inventory of available procedures is 
# provided below as Appendix 1.
# 
# ### Every notebook states its dependencies and changes:
# 

# *Dependencies:*
# 
# - fecon235 repository https://github.com/rsvp/fecon235
# - Python: matplotlib, numpy, pandas
#      
# *CHANGE LOG*
# 
#     2015-12-30  Add inventory of lib procedures as Appendix 1.
#     2015-12-28  First version of README notebook in docs.
# 

#  PREAMBLE-p6.15.1223 :: Settings and system details
from __future__ import absolute_import, print_function
system.specs()
pwd = system.getpwd()   # present working directory as variable.
print(" ::  $pwd:", pwd)
#  If a module is modified, automatically reload it:
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
#       Use 0 to disable this feature.

#  Notebook DISPLAY options:
#      Represent pandas DataFrames as text; not HTML representation:
import pandas as pd
pd.set_option( 'display.notebook_repr_html', False )
#  Beware, for MATH display, use %%latex, NOT the following:
#                   from IPython.display import Math
#                   from IPython.display import Latex
from IPython.display import HTML # useful for snippets
#  e.g. HTML('<iframe src=http://en.mobile.wikipedia.org/?useformat=mobile width=700 height=350></iframe>')
from IPython.display import Image 
#  e.g. Image(filename='holt-winters-equations.png', embed=True) # url= also works
from IPython.display import YouTubeVideo
#  e.g. YouTubeVideo('1j_HxD4iLn8', start='43', width=600, height=400)
from IPython.core import page
get_ipython().set_hook('show_in_pager', page.as_hook(page.display_page), 0)
#  Or equivalently in config file: "InteractiveShell.display_page = True", 
#  which will display results in secondary notebook pager frame in a cell.

#  Generate PLOTS inside notebook, "inline" generates static png:
get_ipython().magic('matplotlib inline')
#          "notebook" argument allows interactive zoom and resize.


# ## Preamble for settings
# 
# The preamble contains the latest shortcuts for notebook commands, 
# but more importantly, it lists the specific dependencies 
# which makes research **reproducible**. The "Repository:" line 
# should indicate the annotated tag associated with the last good state 
# of repository at the time of execution. The branch is then stated, 
# which completes the analogue of *requirements.txt* for notebooks.
# 
# The "Timestamp:" will indicate the staleness of the data. 
# Notebooks have executed properly at the indicated time, 
# and when committed to the repository. 
# If notebooks are re-executed the most current data 
# will be intentionally downloaded. 
# Thus many observations in notebooks include their date. 
# Changes upstream, in the meantime, can possibly generate  
# errors in re-executed fecon235 notebooks 
# (esp. deprecated pandas functions).
# 
# Notebooks implicitly function as integration tests 
# of the underlying code, and thus reveal technical failures. 
# Another notebook will cover unit tests in the *tests* directory 
# for developers.
# 

# ## Internal queries and documentation
# 
# Notebooks have a wonderful feature: **?** and **??** 
# which give further information on variables, functions, 
# classes, etc. And where to exactly look for the source.
# 
# The second question mark gives more verbose answers. 
# All our codes have detailed docstrings and comments, 
# so we strive to be self-documenting.
# 

#  What the heck is "system" mentioned in the preamble?
get_ipython().magic('pinfo system')


# ## Getting data
# 
# Our project currently has free access to data on equities, 
# government bonds, commodities, and futures -- as well as, 
# a full range of economic statistics. The key is finding 
# the string which will retrieve the desired time-series. 
# (A detailed *docs* notebook dedicated to data retrieval is forthcoming.)
# 
# ### Sample: Unemployment rate
# 
# Let's go through an example. The function **get** is 
# designed as an overlord over specialized get functions. 
# 

#  Assign a name to a dataframe
#  that will contain monthly unemployment rates.

unem = get( m4unemp )
#           m4 implies monthly frequency.


#  But does m4unemp really represent?
get_ipython().magic('pinfo m4unemp')


# #### Variables for data
# 
# So we see that m4unemp is our variable holding a string "UNRATE". 
# That string is the internal code used by FRED, the database 
# at the Federal Reserve Bank in St. Louis. Our variables are 
# generally easier to remember, and mentions the frequency. 
# 
# If there is no special variable, one can 
# always get("string") to directly retrieve data.
# 
# Sometimes a variable for a data set may trigger a 
# subroutine which post-processes the original data 
# (e.g. see our inflation measures), or brings offline 
# data into memory (for example, our compressed CSV files may 
# contain synthetic data, e.g. the euro exchange rate 
# years prior to its official circulation).
# 

#  Illustrate slicing: 1997 <= unem <= 2007:
unem07 = unem['1997':'2007']
#  Verify below by Head and Tail.


#  Quick summary:
stat( unem07 )


#  More verbose statistical summary:
stats( unem07 )


# The correlation matrix has only one entry above. 
# This is because *stats()* is designed to take 
# a dataframe with multiple columns as argument.
# Let's see how the function is written 
# and where we can find it in the filesystem.
# 
# Indeed *stats()* calls our *cormatrix()* to compute 
# the correlation matrix. And one can go on 
# further to query that function... eventually 
# that query could reach a core numerical package 
# such as numpy.
# 

get_ipython().magic('pinfo2 stats')


# #  Uncomment to see how numpy computes something simple as absolute value:
# np.abs??


# ## Computing from the data
# 
# The analysis of data is at the heart of this project. 
# Specific computational tools will be covered in 
# other notebooks under the *docs* directory.
# 
# To follow up on unemployment example, see https://git.io/fed 
# which scores the Federal Reserve on their dual mandate. 
# Visualization is provided by our plot tools, 
# which as a by-product discredits the Phillips curve 
# as adequate causal theory.
# 

# ## Questions or bugs
# 
# - Chat with fellow users at Gitter: https://gitter.im/rsvp/fecon235
# 
# - Report an issue at https://github.com/rsvp/fecon235/issues
# 
# - Summarize your usage solution at our wiki: https://github.com/rsvp/fecon235/wiki
# 
# - Blame the lead developer: *Adriano* [rsvp.github.com](https://rsvp.github.com)
# 

# ## Appendix 1: Procedures defined in lib modules
# 
# As of 2015-12-30, many of these procedures and functions 
# are unified by the top level module **fecon235.py** 
# which also simplifies their usage, for example, 
# get() and plot():
# 
# #### yi_0sys.py
# 
#      getpwd():
#          Get present working directory (Linux command is pwd).
#      program():
#          Get name of present script; works cross-platform.
#      warn( message, stub="WARNING:", prefix=" !. "):
#          Write warning solely to standard error.
#      die( message, errcode=1, prefix=" !! "):
#          Gracefully KILL script, optionally specifying error code.
#      date( hour=True, utc=True, localstr=' Local' ):
#          Get date, and optionally time, as ISO string representation.
#      pythontup():
#          Represent invoked Python version as an integer 3-tuple.
#      versionstr( module="IPython" ):
#          Represent version as a string, or None if not installed.
#      versiontup( module="IPython" ):
#          Parse version string into some integer 3-tuple.
#      version( module="IPython" ):
#          Pretty print Python or module version info.
#      utf( immigrant, xnl=True ):
#          Convert to utf-8, and possibly delete new line character.
#      run( command, xnl=True, errf=None ):
#          RUN **quote and space insensitive** SYSTEM-LEVEL command.
#      gitinfo():
#          From git, get repo name, current branch and annotated tag.
#      specs():
#          Show ecosystem specifications, including execution timestamp.
#      ROSETTA STONE FUNCTIONS approximately bridging Python 2 and 3.
#      endmodule():
#          Procedure after __main__ conditional in modules.
# 
# #### yi_1tools.py
# 
#      nona( df ):
#           Eliminate any row in a dataframe containing NA, NaN nulls.
#      head( dfx, n=7 ):
#           Quick look at the INITIAL data point(s).
#      tail( dfx, n=7 ):
#           Quick look at the LATEST data point(s).
#      tailvalue( df, pos=0, row=1 ):
#           Seek (last) row of dataframe, then the element at position pos.
#      div( numerator, denominator, floor=False ):
#           Division via numpy for pandas, Python 2 and 3 compatibility.
#      dif( dfx, freq=1 ):
#           Lagged difference for pandas series.
#      pcent( dfx, freq=1 ):
#           PERCENTAGE CHANGE method for pandas.
#      georet( dfx, yearly=256 ):
#           Compute geometric mean return in a summary list.
#      zeroprice( rate, duration=9, yearly=2, face=100 ):
#           Compute price of zero-coupon bond given its duration.
#      ema( y, alpha=0.20 ):
#           EXPONENTIAL MOVING AVERAGE using traditional weight arg.
#      normalize( dfy ):
#           Center around mean zero and standardize deviation.
#      correlate( dfy, dfx, type='pearson' ):
#           CORRELATION FUNCTION between series using pandas method.
#      cormatrix( dataframe, type='pearson' ):
#           PAIRWISE CORRELATIONS within a dataframe using pandas method.
#      regressformula( df, formula ):
#           Helper function for statsmodel linear regression using formula.
#      regressTIME( dfy, col='Y' ):
#           Regression on time since such index cannot be an independent variable.
#      regresstime( dfy, col='Y' ):
#           Regression on time since such index cannot be an independent variable.
#      regresstimeforecast( dfy, h=24, col='Y' ):
#           Forecast h-periods ahead based on linear regression on time.
#      detrend( dfy, col='Y' ):
#           Detread using linear regression on time.
#      detrendpc( dfy, col='Y' ):
#           Detread using linear regression on time; percent deviation.
#      detrendnorm( dfy, col='Y' ):
#           Detread using linear regression on time, then normalize.
#      regress( dfy, dfx ):
#          Perform LINEAR REGRESSION, a.k.a. Ordinary Least Squares.
#      stat2( dfy, dfx ):
#           Quick STATISTICAL SUMMARY and regression on two variables
#      stat( dataframe, pctiles=[0.25, 0.50, 0.75] ):
#           QUICK summary statistics on given dataframe.
#      stats( dataframe ):
#           VERBOSE statistics on given dataframe; CORRELATIONS without regression.
#      todf( data, col='Y' ):
#           CONVERT (list, Series, or DataFrame) TO DataFrame, NAMING single column.
#      paste( df_list ):
#           Merge dataframes (not Series) across their common index values.
#      writefile( dataframe, filename='tmp-yi_1tools.csv', separator=',' ):
#          Write dataframe to disk file using UTF-8 encoding.
# 
# #### yi_fred.py
# 
#      readfile( filename, separator=',', compress=None ):
#          Read file (CSV default) as pandas dataframe.
#      makeURL( fredcode ):
#          Create http address to access FRED's CSV files.
#      getdata_fred( fredcode ):
#          Download CSV file from FRED and read it as pandas DATAFRAME.
#      plotdf( dataframe, title='tmp' ):
#          Plot dataframe where its index are dates.
#      daily( dataframe ):
#           Resample data to daily using only business days.
#      monthly( dataframe ):
#           Resample data to FRED's month start frequency.
#      quarterly( dataframe ):
#           Resample data to FRED's quarterly start frequency.
#      getm4eurusd( fredcode=d4eurusd ):
#           Make monthly EURUSD, and try to prepend 1971-2002 archive.
#      getspx( fredcode=d4spx ):
#           Make daily S&P 500 series, and try to prepend 1957-archive.
#      gethomepx( fredcode=m4homepx ):
#           Make Case-Shiller 20-city, and try to prepend 1987-2000 10-city.
#      getinflations( inflations=ml_infl ):
#           Normalize and average all inflation measures.
#      getdeflator( inflation=m4infl ):
#           Construct a de-inflation dataframe suitable as multiplier.
#      getm4infleu( ):
#           Normalize and average Eurozone Consumer Prices.
#      getfred( fredcode ):
#           Retrieve from FRED in dataframe format, INCL. SPECIAL CASES.
#      plotfred( data, title='tmp', maxi=87654321 ):
#           Plot data should be it given as dataframe or fredcode.
#      holtfred( data, h=24, alpha=ts.hw_alpha, beta=ts.hw_beta ):
#           Holt-Winters forecast h-periods ahead (fredcode aware).
# 
# #### yi_plot.py
# 
#      plotn( dataframe, title='tmp' ):
#          Plot dataframe where the index is numbered (not dates).
#      boxplot( data, title='tmp', labels=[] ):
#           Make boxplot from data which could be a dataframe.
#      scatter( dataframe, title='tmp', col=[0, 1] ):
#          Scatter plot for dataframe by zero-based column positions.
#      scats( dataframe, title='tmp' ):
#          All pair-wise scatter plots for dataframe.
#      scat( dfx, dfy, title='tmp', col=[0, 1] ):
#          Scatter plot between two pasted dataframes.
# 
# #### yi_quandl.py
# 
#      setQuandlToken( API_key ):
#           Generate authtoken.p in the local directory for API access.
#      cotr_get( futures='GC', type='FO' ):
#           Get CFTC Commitment of Traders Report COTR.
#      cotr_position( futures='GC' ):
#           Extract market position from CFTC Commitment of Traders Report.
#      cotr_position_usd():
#           Market position for USD from COTR of JY and EC.
#      cotr_position_metals():
#           Market position for precious metals from COTR of GC and SI.
#      cotr_position_bonds():
#           Market position for bonds from COTR of TY and ED.
#      cotr_position_equities():
#           Market position for equities from COTR of both SP and ES.
#      fut_decode( slang ):
#          Validate and translate slang string into vendor futures code.
#      getfut( slang, maxi=512, col='Settle' ):
#           slang string retrieves single column for one futures contract.
#      getqdl( quandlcode, maxi=87654321 ):
#           Retrieve from Quandl in dataframe format, INCL. SPECIAL CASES.
#      plotqdl( data, title='tmp', maxi=87654321 ):
#           Plot data should be it given as dataframe or quandlcode.
#      holtqdl( data, h=24, alpha=ts.hw_alpha, beta=ts.hw_beta ):
#           Holt-Winters forecast h-periods ahead (quandlcode aware).
# 
# #### yi_secform.py
# 
#      parse13f( url=druck150814 ):
#           Parse SEC form 13F into a pandas dataframe.
#      pcent13f( url=druck150814, top=7654321 ):
#           Prune, then sort SEC 13F by percentage allocation, showing top N.
# 
# #### yi_simulation.py
# 
#      GET_simu_spx_pcent():
#           Retrieve normalized SPX daily percent change 1957-2014.
#      SHAPE_simu_spx_pcent( mean=MEAN_PC_SPX, std=STD_PC_SPX ):
#           Generate SPX percent change (defaults are ACTUAL annualized numbers).
#      SHAPE_simu_spx_returns( mean=MEAN_PC_SPX, std=STD_PC_SPX ):
#           Convert percent form to return form.
#      array_spx_returns( mean=MEAN_PC_SPX, std=STD_PC_SPX ):
#           Array of SPX in return form.
#      bootstrap( N, yarray ):
#           Randomly pick out N without replacment from yarray.
#      simu_prices( N, yarray ):
#           Convert bootstrap returns to price time-series into pandas DATAFRAME.
#      simu_plots_spx( charts=1, N=N_PC_SPX, mean=MEAN_PC_SPX, std=STD_PC_SPX ):
#           Display simulated SPX price charts of N days, given mean and std.
# 
# #### yi_stocks.py
# 
#      stock_decode( slang ):
#          Validate and translate slang string into vendor stock code.
#      stock_all( slang, maxi=3650 ):
#           slang string retrieves ALL columns for single stock.
#      stock_one( slang, maxi=3650, col='Close' ):
#           slang string retrieves SINGLE column for said stock.
#      getstock( slang, maxi=3650 ):
#           Retrieve stock data from Yahoo Finance or Google Finance.
# 
# #### yi_timeseries.py
# 
#      holt_winters_growth( y, alpha=hw_alpha, beta=hw_beta ):
#           Helper for Holt-Winters growth (linear) model using numpy arrays.
#      holt( data, alpha=hw_alpha, beta=hw_beta ):
#           Holt-Winters growth (linear) model outputs workout dataframe.
#      holtlevel( data, alpha=hw_alpha, beta=hw_beta ):
#           Just smoothed Level dataframe from Holt-Winters growth model.
#      holtgrow( data, alpha=hw_alpha, beta=hw_beta ):
#           Just the Growth dataframe from Holt-Winters growth model.
#      holtpc( data, yearly=256, alpha=hw_alpha, beta=hw_beta ):
#           Annualized percentage growth dataframe from H-W growth model.
#      holtforecast( holtdf, h=12 ):
#           Given a dataframe from holt, forecast ahead h periods.
#      plotholt( holtdf, h=12 ):
#           Given a dataframe from holt, plot forecasts h periods ahead.
# 

# # US GDP vs. SPX: Holt-Winters time-series forecasting
# 
# We examine the US gross domestic product's relationship to the 
# US equity market (S&P500), in real terms. Forecasts for both are 
# demonstrated using *Holt-Winters* time-series model. 
# We derive the most likely range for real GDP growth, and 
# identify excessive equity valuations aside from inflationary pressures.
# 
# Our analysis would suggest the following
# ***back of the envelope calculation***: say, SPX nominally has an annual gain of 6%
# and inflation stands at 2%, then the real SPX gain is 4%.
# Take half of that to arrive at real GDP growth: 2%.
# *Helpful because GDP numbers are announced after months of lag.*
# 

# *Dependencies:*
# 
# - Repository: https://github.com/rsvp/fecon235
# - Python: matplotlib, pandas
# 
# *CHANGE LOG*
#      
#     2017-04-09  Fix issue #2, update data and narrative, optimize HW parameters.
#                    Closing note on Didier Sornette research.
#     2015-02-20  Code review and revision.
#     2014-08-11  First version uses major revision yi_fred.
# 

from fecon235.fecon235 import *


#  PREAMBLE-p6.15.1223 :: Settings and system details
from __future__ import absolute_import, print_function
system.specs()
pwd = system.getpwd()   # present working directory as variable.
print(" ::  $pwd:", pwd)
#  If a module is modified, automatically reload it:
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
#       Use 0 to disable this feature.

#  Notebook DISPLAY options:
#      Represent pandas DataFrames as text; not HTML representation:
import pandas as pd
pd.set_option( 'display.notebook_repr_html', False )
from IPython.display import HTML # useful for snippets
#  e.g. HTML('<iframe src=http://en.mobile.wikipedia.org/?useformat=mobile width=700 height=350></iframe>')
from IPython.display import Image 
#  e.g. Image(filename='holt-winters-equations.png', embed=True) # url= also works
from IPython.display import YouTubeVideo
#  e.g. YouTubeVideo('1j_HxD4iLn8', start='43', width=600, height=400)
from IPython.core import page
get_ipython().set_hook('show_in_pager', page.as_hook(page.display_page), 0)
#  Or equivalently in config file: "InteractiveShell.display_page = True", 
#  which will display results in secondary notebook pager frame in a cell.

#  Generate PLOTS inside notebook, "inline" generates static png:
get_ipython().magic('matplotlib inline')
#          "notebook" argument allows interactive zoom and resize.


# ## Holt-Winters equations and code
# 
# We adopt the Holt-Winters notation from Rob Hyndman's book, 
# *Forecasting with Exponential Smoothing* (2008), ignoring the seasonal aspect of the H-W model.
# 
# Our Python module `yi_timeseries.py` contains the core of model, see
# https://github.com/rsvp/fecon235/blob/master/lib/yi_timeseries.py
# 

#  Note that y represents the raw series in the equations.
#  We use Y as a generic label in dataframes.

Image(filename='holt-winters-equations.png', embed=True)


# ## Working with dataframes
# 
# We now get **data** for real US GDP into a pandas DataFrame.
# Our Holt-Winters functions generally accept dataframes as arguments
# (but computation in the background is using numpy arrays,
# and we let the *fecon235* API abstract the messy details).
# The user interface is designed to easily get the job done.
# 

#  Retrieve quarterly data for real US GDP as DataFrame:
dfy = get( q4gdpusr )


# That quarterly data comes from FRED (Federal Reserve Bank, St. Louis),
# seasonally adjusted and expressed in billions of 2009 dollars.
# 

#  So we are currently discussing a 17 trillion dollar economy...
tail( dfy )


#  DEFINE start YEAR OF ANALYSIS:
start = '1980'

#  We can easily re-run the entire notebook from different start date.


# Holt-Winters requires specification of two parameters,
# **alpha** and **beta**, which sometimes are guessed.
# Our default values have been found in practice to be 
# good *a priori* choices (see Gelper 2007, ibid.):
# 

print(hw_alpha, hw_beta, " :: DEFAULT Holt-Winters alpha and beta")


# ### Optimizing Holt-Winters parameters
# 
# We developed an algorithm to optimize those parameters,
# conditional on specific data. See
# https://github.com/rsvp/fecon235/blob/master/lib/ys_opt_holt.py
# 

#  This optimization procedure may be computationally intensive
#  depending on data size and the number of grids,
#  so uncomment the next command to run it:

ab = optimize_holt( dfy['1980':], grids=50 )

ab


# The first two elements in the *ab* list are alpha and beta respectively.
# The third element gives the median absolute loss as percent
# for an one-step ahead forecast given those parameters.
# The fourth element is the median absolute loss
# (which indicates we have used a L1 loss function
# for robust optimization).
# 
# 2017-04-09: For start='1980', median absolute loss of 0.21% is respectable. 
# 

#  Let's use the optimized values for alpha and beta
#  to compute the Holt dataframe:

holtdf = holt( dfy, alpha=ab[0], beta=ab[1] )


stats( holtdf[start:] )
#  Summary stats from custom start point:


#  Y here is the raw series, i.e. real GDP in billions of dollars:
plot( holtdf['Y'][start:] )


#  Annualized geometric mean return of the raw series:
georet( todf(holtdf['Y'][start:]), yearly=4 )

#  Since 1980, real GDP growth is about 2.6%


#  Level can be thought of as the smoothed series:

# plot( holtdf['Level'][start:] )


#  Growth is the fitted slope at each point,
#  expressed in units of the original series:

# plot( holtdf['Growth'][start:] )


# But we are most interested in the *forecasted*
# annualized growth rate in percentage terms.
# The specialized function *holtpc* handles that need.
# 

pc = holtpc( dfy, yearly=4, alpha=ab[0], beta=ab[1] )
plot( pc[start:] )


gdp_forecast = tailvalue( pc )
gdp_forecast


# That *tailvalue* of pc gives us the **latest forecast
# for real GDP growth over the next year**.
# But to understand its precision,
# we examine the big picture.
# 
# ### Big GDP Trend for discerning precision
# 

#  Here is the BIG GDP TREND since the end of World War 2:

trendpc = trend( pc )

plot( trendpc )


stat( trendpc )


# Projected annualized real GDP growth has decreased from 3.8% post-WW2 
# to 2.1% most recently over the long time scale.
# 
# Using our *detrend* function we can visualize the local fluctuations
# around the BIG GDP TREND.
# 

detpc = detrend( pc )
plot( detpc )


stat( detpc )


# Standard deviation = 1.66 for the variation around the BIG GDP TREND. 
# So at 2\*std, we are roughly looking at +/- 3.3 percentage points around the long-term trend. 
# 
# 2017-04-10:  We can see from *detpc* that the United States has recovered 
# from the Great Recession to mean GDP growth. "Doing just OK, ah, on average..." 
# 
# A band of two standard deviations from +2.1 gives us a ***forward estimated range of 
# (horrible) -1.2% to (great) +5.4% for real GDP growth.***
# 

# ## Real GDP forecast using Holt-Winters
# 
# By fitting Level and Growth, Holt-Winters essentially uses 
# the slope to make point forecasts forward in time.
# We show an alternate, more direct, way to arrive at our forecast.
# 

#  Forecast real GDP, four quarters ahead:
foregdp = holtforecast( holtdf, h=4 )
foregdp


#  Forecast real GDP rate of GROWTH, four quarters ahead:
100 * ((foregdp.iloc[4] / foregdp.iloc[0]) - 1)


# We thus have computed the **Holt-Winters forecast for real GDP rate for the year ahead:**
# 2.03% (as of 2017-04-10)
# 
# This should concur with our *holtpc* method above, i.e. *gdp_forecast*. 
# 
# Note that foregdp.iloc[0] is the last actual data point, rather than the last Level.
# 

#  We can plot the point forecasts 12 quarters ahead (i.e. 3 years):

# plotholt( holtdf, 12 )


# ## Real GDP vs. real SPX (S&P 500)
# 
# This section covers the measured economy of goods and services, 
# versus the market valuation of equity shares.
# 

#  SPX is a daily series, but we can directly retrieve its monthly version:
spx = get( m4spx )

#  ... to match the monthly periodicity of our custom deflator:
defl = get( m4defl )


#  Now we synthesize a quarterly real SPX version by resampling:
spdefl = todf( spx * defl )
spq = quarterly( spdefl )


#  Real SPX resampled quarterly in current dollars:
plot( spq )


#  Geometric mean return for real SPX:
georet( spq[start:], yearly=4 )

#  cf. volatility for real GDP = 1.5% 
#      in contrast to equities = 13.2%


# **In real terms, the geometric mean return of SPX is double that of GDP.** 
# 
# Next, we examine their correlation.
# For start='1980' ***their correlation is around +0.9 which is fairly strong.***
# 

stat2( dfy['Y'][start:], spq['Y'][start:] )


# The linear regression shows approximately that $dG / dS = 5.0$
# 
# Therefore using mean values g and s: $ (dG/g) / (s/dS) = 5.0 * (s/g) $
# 
# The right-hand side can be roughly interpreted as the ratio between percentage changes.
# 

#  2017-04-09, for start='1980':  5.0 * (s/g) = 

gsratio = 5.0 * div(spq[start:].mean(), dfy[start:].mean())
gsratio = gsratio.values[0]
gsratio


# Thus **1% rise (100 bp) in real SPX would suggest additional 49 bp in real US GDP growth** 
# (we note that regression $R^2 = 0.82$, so this seems reasonable). 
# 
# SPX is generally regarded as a leading economic indicator. 
# (The regression fit improves if *start* is moved farther back.)
# 
# ***Back of the envelope calculation***: say, SPX nominally has an annual gain of 6%
# and inflation stands at 2%, then the real SPX gain is 4%.
# Take half of that to arrive at real GDP growth: 2%.
# *Helpful because GDP numbers are announced after months of lag.*
# 

# ## Real SPX forecast using Holt-Winters
# 
# Given that the log returns of SPX approximately follow an AR(1) process with unit root,
# we will resort to using our default values for alpha and beta 
# which are ideal for smoothing a time-series with such characteristics.
# 
# Accordingly, we shall look at projected (not actual) annual rates of return, 
# rather than generating point forecasts.
# 

#       holtpc should be familiar from our GDP analysis.
pcspq = holtpc( spq, yearly=4, alpha=hw_alpha, beta=hw_beta )
plot( pcspq )

#  Note we use all data since 1957.


spx_forecast = tailvalue( pcspq )
spx_forecast


# Real SPX returns are very volatile, especially on the downside.
# 
# 2017-04-10: SPX is trading near all-time highs,
# and **Holt-Winters is forecasting +4.4% annualized over the next year**.
# Worth noting that the *real* rate is being projected, 
# thus if inflation stands at 2%, the *nominal* rate of return will be +6.4%.
# 
# 
# ### SPX over the long haul, bubbles and crashes
# 
# So let's now look at the long-term trend in projected SPX rate of returns.
# 

trend_pcspq = trend( pcspq )
plot( trend_pcspq )


# There are several major episodes of less than -20% forecasted on smoothed real SPX, but 
# over the long-term US equities are showing increasingly better returns against inflation.
# 
# (Curiously, in contrast, the long-term trend for real GDP growth was *downwards*.)
# 
# To help identify local **BUBBLES**, i.e. excessive equity valuations 
# apart from inflationary effects, we detrend the series as follows:
# 

det_pcspq = detrend( pcspq )
plot( det_pcspq )


# In the plot above, we have filtered out both inflation and the underlying 
# real rate of return on equities, leaving us with a picture of *extreme* valuations 
# in the short-term.
# 
# **The shift from overvaluation to collapse is very swift, 
# and occurs historically on a regular but unpredictable basis.** 
# ***Instability is the norm.***
# 

# ## Further research
# 
# *What are the arrival times of collapses in equity valuations?
# Are there signals which can be identified as precursors to such collapses?*
# 
# There are structural similarities to the prediction of earthquakes
# where threshold tremors can be regarded as precursors.
# 
# [Didier Sornette](https://scholar.google.com/citations?user=HGsSmMAAAAAJ&hl=en)
# has discovered log-periodic times, inspired by geophysics and self-organizing
# systems. A separate notebook will be dedicated to his findings regarding
# criticality: bubbles and anti-bubbles in financial markets.

# # Market position indicators using CFTC COTR 
# 
# We examine the CFTC **Commitment of Traders Reports (COTR)** 
# for futures and options to derive indicators of market position *among 
# Asset/Money Managers*. Our generalized formulation permits treating 
# asset classes which include: precious metals, US dollar, bonds, and equities. 
# Indicators may be post-processed to obtain further clarity, 
# for example, normalization and smoothing.
# 
# Detailed explantory notes regarding the raw data can be found here:
# http://www.cftc.gov/MarketReports/CommitmentsofTraders/ExplanatoryNotes 
# We shall disregard the Legacy format, and focus on the data after 13 June 2006. 
# *Current data is released weekly on Fridays* (for accounting 
# effective through Tuesday).
# 
# We note the absence of strong linear correlations 
# among our asset class position indicators. 
# 
# Lastly, we compute a dataframe of normalized position indicators 
# which is useful for comparative study across asset classes 
# and the identification of overcrowded trades.
# 
# *Shortcut to this notebook:* https://git.io/cotr 
# where **Appendix 1 gives an algorithmic summary in a few lines of code.**
# 
# Appendix 2 visualizes the chronological joint path of the positions indicators for 
# bonds and equities by color heat map -- which could be useful for asset allocation.
# 

# *Dependencies:*
# 
# - Repository: https://github.com/rsvp/fecon235
# - Python: matplotlib, pandas
#      
# *CHANGE LOG*
# 
#     2016-01-23  Fix issue #2 by v4 and p6 updates.
#                    Smooth metals by ema(). Use groupfun() to normalize.
#                    Add Appendix 1 and 2.
#     2015-08-31  Simply use fecon.py to generally access various modules.
#     2015-08-25  Update tpl to v4.15.0812. Add silver COTR,
#                    and the class of precious metals w4cotr_metals.
#     2015-08-09  Change of variable names for clarity.
#     2015-08-07  First version arising as test of the Quandl API.
# 

from fecon235.fecon235 import *


#  PREAMBLE-p6.15.1223 :: Settings and system details
from __future__ import absolute_import, print_function
system.specs()
pwd = system.getpwd()   # present working directory as variable.
print(" ::  $pwd:", pwd)
#  If a module is modified, automatically reload it:
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
#       Use 0 to disable this feature.

#  Notebook DISPLAY options:
#      Represent pandas DataFrames as text; not HTML representation:
import pandas as pd
pd.set_option( 'display.notebook_repr_html', False )
#  Beware, for MATH display, use %%latex, NOT the following:
#                   from IPython.display import Math
#                   from IPython.display import Latex
from IPython.display import HTML # useful for snippets
#  e.g. HTML('<iframe src=http://en.mobile.wikipedia.org/?useformat=mobile width=700 height=350></iframe>')
from IPython.display import Image 
#  e.g. Image(filename='holt-winters-equations.png', embed=True) # url= also works
from IPython.display import YouTubeVideo
#  e.g. YouTubeVideo('1j_HxD4iLn8', start='43', width=600, height=400)
from IPython.core import page
get_ipython().set_hook('show_in_pager', page.as_hook(page.display_page), 0)
#  Or equivalently in config file: "InteractiveShell.display_page = True", 
#  which will display results in secondary notebook pager frame in a cell.

#  Generate PLOTS inside notebook, "inline" generates static png:
get_ipython().magic('matplotlib inline')
#          "notebook" argument allows interactive zoom and resize.


# ## COTR example: Gold
# 
# To get an idea of what's contained in a Commitment of Traders Report, 
# we first look at an example from the commodity futures market. 
# Financial futures have a different format since the notion of a 
# "Producer" is not entirely appropriate 
# (though dealers may produce derivatives which may rely on the futures market). 
# 
# The latest data is downloaded via Quandl as a pandas dataframe. 
# The relevant functions are found in the yi_quandl module.
# 

#  First download latest GOLD reports:
cotr = cotr_get( f4xau )
#                ^ = 'GC' currently.

#  Show column labels and only the last report:
tail( cotr, 1 )


# ### Notable points
# 
# - Number of longs = Number of shorts, by necessity -- and each side equals the *Open Interest*.
# - Option positions are computed on a futures-equivalent basis using delta factors supplied by the exchanges.
# - Generally non-directional traders (usually with hedged positions): Producer/User, Swap Dealer, and Dealer.
# - Generally **directional traders**: *Money Manager* (commodities), Leveraged Funds and *Asset Manager* (financials). Leveraged Funds appear to take very choppy short-term positions, whereas Asset Manager will show a longer-term positional narrative.
# - Non-directional trading by a Money Manager, Asset Manager, and Leveraged Funds is categorized separately under Spreads.
# - "Non reportable" traders are small players trading contract sizes under CFTC reporting thresholds -- generally, noise traders categorized as **speculators**.
# - **COTR is released weekly on Friday** (for accounting effective through Tuesday).
# 

# ### Parsing out positions
# 
# To characterize *informed* market direction we focus on the directional traders 
# who are trading in large size which must be reported to the CFTC. 
# 
# Note that their *Longs* do not necessarily equal their *Shorts*" since the 
# counter-parties may be other players in the market, 
# e.g. hedged producers, spreaders, or small uninformed traders.
# 

longs  = cotr['Money Manager Longs']
shorts = cotr['Money Manager Shorts']


# difference in number of contracts:
lsdiff = todf( longs - shorts )


plot( lsdiff )


# The preceding chart shows the difference between the number contracts long versus short. 
# It is somewhat useful, but we would prefer a **scale-free [0,1] measure** which 
# reveal position: 0 for bearish, 0.50 for neutral, and 1 for bullish. 
# 
# This will also allow us later to *combine* readings to show position in a class, 
# for example, the US dollar versus various foreign currencies.
# 

#  Scale-free measure from zero to 1:
z_xau = todf( longs / (longs + shorts ))


#  Could interpret as prob( bullish Manager ):
plot( z_xau )


# How does our indicator compare to spot gold prices?
xau = get( d4xau )
plot( xau['2006-06-12':] )
# using the FRED database for Gold London PM fix.


# GOLD: We can definitely see a "raging bull" in action through 2013 -- 
# our position indicator approaches 1.0 frequently! Thereafter, there are 
# four major efforts to re-ignite the bull market through mid-2015, 
# but without much success as the price creeps downward. 
# 
# July 2016 sees a breakdown below the 0.6 positional support, 
# to under the 0.5 neutral mark -- which means more shorts than longs among managers. 
# The Fed raises rates on 2015-12-16 (first time in almost a decade), 
# and the indicator settles down below 0.5 for the start of 2016.
# 

# ## Generalization to Asset Classes
# 
# ### Following function will compute position for commodities and financials:
# 
# *It returns our scale-free measure on [0,1] such that comparable contracts 
# can easily be averaged, and thus interpreted.*
# 

#       ? or ?? is useful to investigate source.
get_ipython().magic('pinfo2 cotr_position')


# ## Precious metals
# 
# Let's first take a look at the COTR position for silver (spot symbol XAG). 
# Graphically it is quite similar to gold, so we will also 
# examine their statistical correlation.
# 

#  Silver position:
z_xag = cotr_position( f4xag )
plot( z_xag )


#  Correlation and regression:
#  stat2( z_xag['Y'], z_xau['Y'] )


# COTR positions for gold and silver are correlated (approximately 77%), 
# but a linear regression model is not satisfactory. This means they 
# each contain some information that the other lacks. But since the 
# market sentiment regarding both are similar, it is useful to 
# combine their signals as precious metals.
# 
# #### TECHNIQUE: compute the mean of indicators for an asset class
# 
# We use the futures and options COTR for contracts on both gold and silver, 
# then average their position indicators. We can run this procedure by 
# retrieval of a variable called w4cotr_metals 
# (where w4 tells us that it's weekly series). 
# See the yi_quandl module for details.
# 

#  PRECIOUS METALS position:
z_metals = get( w4cotr_metals )
plot( z_metals )


# Positions in the precious metals have recently exhibited 
# swings characteristic of momentum trading. 
# That may be also due to the fact that strategic positions 
# are taken for the long-term in ETFs, for example: 
# GLD in the case of gold and John Paulson, rather than 
# in the futures market to avoid rollover costs and slippage. 
# Counter-tactical positions may be traded against 
# the ETFs in the short-term in the futures/options market, 
# however, those will not be accounted as spreads in the COTR. 
# Thus we must be aware of bias created by trading in related 
# markets which are not under CFTC jurisdiction.
# 
# Worth noting is the August 2015 dip into net short region. 
# 
# #### TECHNIQUE: damped indicator swings can show trends in the underlying prices
# 
# We can demonstrate this by applying 
# *exponential moving average*, **ema()**.
# 

z_metal_ema = ema(z_metals, 0.05)
plot( z_metal_ema )


# ## US Dollar position
# 
# We use the futures and options COTR for contracts on both the euro and yen, 
# then average their position indicators. To invert direction due to quotation style, 
# we take the complement, i.e. (1-mean), so we still retain the [0,1] range. 
# 
# We can run this procedure by retrieval of a variable called *w4cotr_usd* 
# (where w4 tells us that it's weekly series).
# 

#  Dollar position (not price):
z_usd = get( w4cotr_usd )
plot( z_usd )


# As the U.S. subprime mortgage crisis expanded worldwide, 
# flight to safe USD makes a peak in our indicator exceeding 0.7, 
# and thereafter, we see an orderly decline through 2013 due to 
# Fed's QE quantitative easing. A bull market develops due to the 
# termination of QE by the Fed, while QE is relentlessly pursued by BoJ, 
# and finally the ECB activates its own QE. 
# The sudden acceleration at the beginning of 2013 was a 
# huge early warning sign of change in market sentiment.
# 
# Next is the dollar index (against most currencies) used by the 
# Federal Reserve Bank which considers the real trade balance RTB between countries:
# 

# Fed US Dollar index, m4 means monthly frequency:
usd = get( m4usdrtb )
plot( usd['2006-06-01':] )


# ## Bonds position
# 
# We use the futures and options COTR for contracts on both the 
# eurodollar (strips) and 10-year Treasury bond, then average their position indicators. 
# We can run this procedure by retrieval of a variable called *w4cotr_bonds* 
# (where w4 tells us that it's weekly series). 
# 
# The indicator intends to show market position across the active yield curve, 
# not in terms of rates, but rather prices of fixed-income instruments. 
# This is useful to gauge the effects of Fed policy.
# 

#  Bonds position:
z_bonds = get( w4cotr_bonds )
plot( z_bonds )


# From the beginnings of the subprime mortgage crisis to 2012, 
# there is a commonplacent bull market conviction that the 
# Fed wants lower rates across the yield curve: bond rates fall from 5% to 1.5%. 
# Thereafter, the market seeks to front-run the Fed in the event 
# ZIRP Zero Interest Rate Policy is reversed, but is denied on several occassions. 
# 
# Only after 2015 does our indicator spend time below the neutral halfway mark: 
# Fed has ended QE, and the market seeks to determine the time of the first rate hike 
# since the Great Recession. Bottoming rates in 2012 and 2015 technically 
# imply a floor around 1.8% for the 10-y Treasuries. 
# 
# Given favorable economic conditions, esp. unemployment, and the 
# expected hike in September of 2015: a bear market is developing for bond valuations. 
# But surprise: FOMC postpones the rate hike until 2015-12-16, 
# and our indicator gets less bearish. Compared globally, US bonds yields 
# are very attractive, and the USD is very strong.
# 

bondrate = get( d4bond10 )
#  10-y Treasury rate INVERTED in lieu of price:
plot( -bondrate['2006-06-01':] )


# ## Equities position
# 
# We use the futures and options COTR for contracts on both 
# the *S&P 500 and and its e-mini version*, then average their position indicators. 
# We can run this procedure by retrieval of a variable called *w4cotr_equities* 
# (where w4 tells us that it's weekly series). 
# 
# It is worth noting that our position indicator is extremely bullish 
# going into the Great Recession, however, even during the worst sell-offs 
# in equities the indicator never goes into bear territory. 
# We suspect this is because many asset managers are constrained to going net long. 
# 

#  Equities position:
z_eq = get( w4cotr_equities )
plot( z_eq  )


# #### TECHNIQUE: normalize market indicator
# 
# This basically translates the [0, 1] indicator into statistical terms. 
# A normalized series has 0 as its mean, and its units are stated 
# as standard deviations.
# 

#  So let's normalize the equities indicator:
plot(normalize( z_eq ))


# 2015-08-08: Given our data, the neutral mark is around 0.77 for equities position. 
# Normalized data shows we are currently over 1 standard deviation into bear territory, 
# even though the market seems to make advances upward every day.
# 
# 2016-01-18: Since the last Fed rate hike, equities indicator has gone under -2 std 
# which is the most bearish reading thus far -- *including the Great Recession!*
# 

#  SPX price data showing the post-2009 equities bull market:
spx = get( d4spx )
plot( spx['2006-06-12':] )


# ## Correlation among position indicators
# 
# Note that this is quite different than the usual look at price correlations -- 
# it would be more akin to seeing how informed trading correlates across asset classes.
# 

# class consists of precious metals, US dollar, bonds, equities:
z_class = [ z_metals, z_usd, z_bonds, z_eq ]

z_data = paste( z_class )
z_data.columns = ['z_metals', 'z_usd', 'z_bonds', 'z_eq']
#  Created "group" dataframe z_data with above named columns.
#  Please see fecon235 module for more details.


stats( z_data )


# ***No strong correlations among position indicators.*** 
# 
# As expected z_metals and z_usd are negatively correlated, 
# but precious metals sentiment is most correlated to bonds. 
# We have closely examined the relationship between gold and interest rates 
# in another notebook, and indeed, *real rates* are very significant. 
# Our bond indicator, however, only relates inversely to nominal rates.
# 

# ## Normalized position relative to history
# 
# We saw that our position indicator may not entirely span [0,1] as expected 
# due to institutional reasons (see the equities case above). 
# Thus a *position indicator reading of 0.50 does not strictly 
# imply neutrality with respect to positions*.
# 
# As a remedy, we can normalize an indicator, *relative to its history*, 
# and then look **comparatively across asset classes**, especially recent values.
# 
# Be aware though that as history changes, a normalized indicator 
# for a given date will generally *not* stay fixed (unlike the [0,1] indicator).
# 

#  Normalize indicators:
z_nor = groupfun( normalize, z_data )


#  Compare recent normalized indicators:
tail( z_nor, 24 )


# 2015-12-16: This is the day of first Fed rate hike in almost a decade. 
# It is very instructive to compare the readings for the previous day 
# to the most recent output.
# 
# 2016-01-18: Both z_metals and z_eq are two standard deviations 
# into a bear market -- although their position indicators 
# are in neutral territory. This is a great example of 
# the utility of normalized position indicators.
# 
# *Normalized positions can useful in the identification of "overcrowded" trades.*
# 

# ## Closing remarks
# 
# In general, we hope that the position indicators are *not* correlated 
# too closely with the underlying prices because we are seeking information 
# which is not derivable from observable prices. 
# It is most interesting when market positions *diverge* 
# from identified patterns in price charts.
# 
# **Normalized position indicators are useful to examine 
# asset classes since the units are comparable across 
# their COTR histories.**
# 

# ## Appendix 1: COTR at the command line
# 
# We can algorithmically summarize this notebook for quick computation 
# by the function **groupcotr()**.
# 

#  Asset classes are specified as a "group" in a fecon235 dictionary:
cotr4w


#  Source code for retrieval of COTR [0,1] position indicators
#  (exactly like z_data, except column names are nicer),  
#  followed by operator to normalize the position indicators:
get_ipython().magic('pinfo2 groupcotr')

#  This encapsulates the TECHNIQUES in this notebook,
#  including the option to apply smoothing.

#  group* functions are defined in the fecon235 module.


#  MAIN in action!
norpos = groupcotr( cotr4w, alpha=0 )


#  Most recent results as a test exhibit:
tail( norpos, 3 )


# ***Thus in a few lines of fecon235 code, we can comparatively observe how 
# various asset classes are positioned in the futures and options market by informed traders.***
# 

# ## Appendix 2: Joint path of Bonds and Equities
# 
# We found earlier that the linear correlation between Bonds and Equities 
# is about +28% with respect to their position indicators. 
# A scatter plot is useful to discern non-linearities, 
# but even better would be one with a ***heat map which reveals 
# their evolution chronologically.***
# 

#  To broadly visualize a decade of data, 
#  we resample the weekly normalized position indicators
#  into MONTHLY frequency (median method).
norpos_month = groupfun( monthly, norpos )


#  Plot from 2006 to present:
scatter( norpos_month, col=[0, 1] )


# By using a color heat map, from blue to green to red, 
# we can visualize the joint path of Bonds and Equities 
# chronologically since 2006.
# 
# The col argument tells us that the x-axis is for Bonds, 
# and the y-axis is for Equities. The *narrative* starts 
# just prior to the Great Recession, when they both 
# indicate bullish positions (Quandrand I in blue). 
# Due to the recovery in employment, the Great Recession 
# era is said to have concluded, but in 2016 
# we find ourselves in joint bearish territory 
# (Quandrant III in red).
# 
# From 2009 through 2015 (horizontal drift to the east from green to red), 
# Equities rose in price while its normalized position indicator fluctuated around -0.5. 
# During that epoch, the market increasing became more 
# bearish with respect to Bonds (from +1.3 to -2.2). 
# This coincides with ZIRP and numerous QE Fed policies
# (Zero Interest Rate Policy and Quantitative Easing).
# 
# The most recent all-time peak for the S&P 500 is 2130.82 
# on 2015-05-21. Months later, on 2015-12-16 the Fed had its 
# first rate hike in almost a decade (thus terminating ZIRP). 
# The scatter plot illustrates the break from the foregoing 
# horizontal epoch (dark red points, Equities <= -2.0).
# 
# It is important to note that *the narrative above is 
# not derived from price action, but rather the number of 
# contracts positioned in the futures and options markets* 
# by informed traders: Asset/Money Managers. 
# 

