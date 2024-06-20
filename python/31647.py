# ## Continuous Reactor Example 
# ### Simulation of a CSTR/PSR/WSR 
# 
# In this example we will illustrate how Cantera can be used to simulate a Continuously Stirred Tank Reactor (CSTR), also interchangeably referred to as a Perfectly Stirred Reactor or a Well Stirred Reactor, a Jet Stirred Reactor or a Longwell Reactor (there may well be more "aliases"). A cartoon of such a reactor is shown below
# 
# <img src="images/stirredReactorCartoon.png" alt="Cartoon of a Stirred Reactor" style="width: 300px;"/>
# 
# As the figure illustrates, this is an open system (unlike a Batch Reactor which is isolated). P, V and T are the reactor's pressure, volume and temperature respectively. The mass flow rate at which reactants come in is the same as that of the products which exit; and these stay in the reactor for a characteristic time $\tau$, called the *residence time*. This is a key quantity in sizing the reactor and is defined as follows:
# 
# \begin{equation*}
# \tau = \frac{m}{\dot{m}}
# \end{equation*}
# 
# where $m$ is the mass of the gas
# 

from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import time
import cantera as ct

print("Running Cantera version: {}".format(ct.__version__))


# ### Define the gas
# In this example, we will work with $nC
# _{7}H_{16}$/$O_{2}$/$He$ mixtures, for which experimental data can be found in the paper by [Zhang et al.](http://dx.doi.org/10.1016/j.combustflame.2015.08.001). We will use the same mechanism reported in the paper. It consists of 1268 species and 5336 reactions
# 

gas = ct.Solution('data/galway.cti')


# ### Define initial conditions
# #### Inlet conditions for the gas and reactor parameters
# 

# Inlet gas conditions
reactorTemperature = 925 #Kelvin
reactorPressure = 1.046138*ct.one_atm #in atm. This equals 1.06 bars
concentrations = {'NC7H16': 0.005, 'O2': 0.0275, 'HE': 0.9675}
gas.TPX = reactorTemperature, reactorPressure, concentrations 

# Reactor parameters
residenceTime = 2 #s
reactorVolume = 30.5*(1e-2)**3 #m3

# Instrument parameters

# This is the "conductance" of the pressure valve and will determine its efficiency in 
# holding the reactor pressure to the desired conditions. 
pressureValveCoefficient = 0.01

# This parameter will allow you to decide if the valve's conductance is acceptable. If there
# is a pressure rise in the reactor beyond this tolerance, you will get a warning
maxPressureRiseAllowed = 0.01


# #### Simulation parameters
# 

# Simulation termination criterion
maxSimulationTime = 50 # seconds


# ### Reactor arrangement
# 
# We showed a cartoon of the reactor in the first figure in this notebook; but to actually simulate that, we need a few peripherals. A mass-flow controller upstream of the stirred reactor will allow us to flow gases in, and in-turn, a "reservoir" which simulates a gas tank is required to supply gases to the mass flow controller. Downstream of the reactor, we install a pressure regulator which allows the reactor pressure to stay within. Downstream of the regulator we will need another reservoir which acts like a "sink" or capture tank to capture all exhaust gases (even our simulations are environmentally friendly !). This arrangment is illustrated below
# 
# <img src="images/stirredReactorCanteraSimulation.png" alt="Cartoon of a Stirred Reactor" style="width: 600px;"/>
# 

# #### Initialize the stirred reactor and connect all peripherals
# 

fuelAirMixtureTank = ct.Reservoir(gas)
exhaust = ct.Reservoir(gas)

stirredReactor = ct.IdealGasReactor(gas, energy='off', volume=reactorVolume)

massFlowController = ct.MassFlowController(upstream=fuelAirMixtureTank,
                                           downstream=stirredReactor,
                                           mdot=stirredReactor.mass/residenceTime)

pressureRegulator = ct.Valve(upstream=stirredReactor,
                             downstream=exhaust,
                             K=pressureValveCoefficient)

reactorNetwork = ct.ReactorNet([stirredReactor])


# now compile a list of all variables for which we will store data
columnNames = [stirredReactor.component_name(item) for item in range(stirredReactor.n_vars)]
columnNames = ['pressure'] + columnNames

# use the above list to create a DataFrame
timeHistory = pd.DataFrame(columns=columnNames)


# Start the stopwatch
tic = time.time()

# Set simulation start time to zero
t = 0
counter = 1
while t < maxSimulationTime:
    t = reactorNetwork.step()

    # We will store only every 10th value. Remember, we have 1200+ species, so there will be
    # 1200 columns for us to work with
    if(counter%10 == 0):
        #Extract the state of the reactor
        state = np.hstack([stirredReactor.thermo.P, stirredReactor.mass, 
                   stirredReactor.volume, stirredReactor.T, stirredReactor.thermo.X])
        
        #Update the dataframe
        timeHistory.loc[t] = state
    
    counter += 1

# Stop the stopwatch
toc = time.time()

print('Simulation Took {:3.2f}s to compute, with {} steps'.format(toc-tic, counter))

# We now check to see if the pressure rise during the simulation, a.k.a the pressure valve
# was okay
pressureDifferential = timeHistory['pressure'].max()-timeHistory['pressure'].min()
if(abs(pressureDifferential/reactorPressure) > maxPressureRiseAllowed):
    print("WARNING: Non-trivial pressure rise in the reactor. Adjust K value in valve")


# ## Plot the results
# 

# ### Import modules and set plotting defaults
# 

import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().magic('matplotlib notebook')

plt.style.use('ggplot')
plt.style.use('seaborn-pastel')

plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['figure.autolayout'] = True


# As a test, we plot the mole fraction of $CO$ and see if the simulation has converged. If not, go back and adjust max. number of steps and/or simulation time
# 

plt.figure()
plt.semilogx(timeHistory.index, timeHistory['CO'],'-o')
plt.xlabel('Time (s)')
plt.ylabel(r'Mole Fraction : $X_{CO}$');


# ## Illustration : Modeling experimental data
# ### Let us see if the reactor can reproduce actual experimental measurements
# 
# We first load the data. This is also supplied in the paper by [Zhang et al.](http://dx.doi.org/10.1016/j.combustflame.2015.08.001) as an excel sheet
# 

expData = pd.read_csv('data/zhangExpData.csv')
expData.head()


# Define all the temperatures at which we will run simulations. These should overlap
# with the values reported in the paper as much as possible
T = [650, 700, 750, 775, 825, 850, 875, 925, 950, 1075, 1100]

# Create a data frame to store values for the above points
tempDependence = pd.DataFrame(columns=timeHistory.columns)
tempDependence.index.name = 'Temperature'


# Now we simply run the reactor code we used above for each temperature
# 

inletConcentrations = {'NC7H16': 0.005, 'O2': 0.0275, 'HE': 0.9675}
concentrations = inletConcentrations

for temperature in T:
    #Re-initialize the gas
    reactorTemperature = temperature #Kelvin
    reactorPressure = 1.046138*ct.one_atm #in atm. This equals 1.06 bars
    reactorVolume = 30.5*(1e-2)**3 #m3

    gas.TPX = reactorTemperature, reactorPressure, inletConcentrations

    # Re-initialize the dataframe used to hold values
    timeHistory = pd.DataFrame(columns=columnNames)
    
    # Re-initialize all the reactors, reservoirs, etc
    fuelAirMixtureTank = ct.Reservoir(gas)
    exhaust = ct.Reservoir(gas)
    
    # We will use concentrations from the previous iteration to speed up convergence
    gas.TPX = reactorTemperature, reactorPressure, concentrations
    
    stirredReactor = ct.IdealGasReactor(gas, energy='off', volume=reactorVolume)
    massFlowController = ct.MassFlowController(upstream=fuelAirMixtureTank,
                                               downstream=stirredReactor,
                                               mdot=stirredReactor.mass/residenceTime)
    pressureRegulator = ct.Valve(upstream=stirredReactor, 
                                 downstream=exhaust, 
                                 K=pressureValveCoefficient)
    reactorNetwork = ct.ReactorNet([stirredReactor])
    
    # Re-run the isothermal simulations
    tic = time.time()
    t = 0
    while t < maxSimulationTime:
        t = reactorNetwork.step()
        
    state = np.hstack([stirredReactor.thermo.P, 
                       stirredReactor.mass, 
                       stirredReactor.volume, 
                       stirredReactor.T, 
                       stirredReactor.thermo.X])

    toc = time.time()
    print('Simulation at T={}K took {:3.2f}s to compute'.format(temperature, toc-tic))
    
    concentrations = stirredReactor.thermo.X
    
    # Store the result in the dataframe that indexes by temperature
    tempDependence.loc[temperature] = state
    


# ### Compare the model results with experimental data
# 

plt.figure()
plt.plot(tempDependence.index, tempDependence['NC7H16'], 'r-', label=r'$nC_{7}H_{16}$')
plt.plot(tempDependence.index, tempDependence['CO'], 'b-', label='CO')
plt.plot(tempDependence.index, tempDependence['O2'], 'k-', label='O$_{2}$')

plt.plot(expData['T'], expData['NC7H16'],'ro', label=r'$nC_{7}H_{16} (exp)$')
plt.plot(expData['T'], expData['CO'],'b^', label='CO (exp)')
plt.plot(expData['T'], expData['O2'],'ks', label='O$_{2}$ (exp)')

plt.xlabel('Temperature (K)')
plt.ylabel(r'Mole Fractions')

plt.xlim([650, 1100])
plt.legend(loc=1);





# # Flame Temperature
# 
# This example demonstrates calculation of the adiabatic flame temperature for a methane/air mixture, comparing calculations which assume either complete or incomplete combustion.
# 

get_ipython().magic('matplotlib notebook')
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt


# ## Complete Combustion
# 
# The stoichiometric equation for complete combustion of a lean methane/air mixture ($\phi < 1$) is:
# 
# $$\mathrm{\phi CH_4 + 2(O_2 + 3.76 N_2) \rightarrow \phi CO_2 + 2\phi H_2O + 2 (1-\phi) O_2 + 7.52 N_2}$$
# 
# For a rich mixture ($\phi > 1$), this becomes:
# 
# $$\mathrm{\phi CH_4 + 2(O_2 + 3.76 N_2) \rightarrow CO_2 + 2 H_2O + (\phi - 1) CH_4 + 7.52 N_2}$$
# 
# To find the flame temperature resulting from these reactions using Cantera, we create a gas object containing only the species in the above stoichiometric equations, and then use the `equilibrate()` function to find the resulting mixture composition and temperature, taking advantage of the fact that equilibrium will strongly favor conversion of the fuel molecule.
# 

# Get all of the Species objects defined in the GRI 3.0 mechanism
species = {S.name: S for S in ct.Species.listFromFile('gri30.cti')}

# Create an IdealGas object with species representing complete combustion
complete_species = [species[S] for S in ('CH4','O2','N2','CO2','H2O')]
gas1 = ct.Solution(thermo='IdealGas', species=complete_species)

phi = np.linspace(0.5, 2.0, 100)
T_complete = np.zeros(phi.shape)
for i in range(len(phi)):
    gas1.TP = 300, ct.one_atm
    gas1.set_equivalence_ratio(phi[i], 'CH4', 'O2:1, N2:3.76')
    gas1.equilibrate('HP')
    T_complete[i] = gas1.T    


# ## Incomplete Combustion
# 
# In the case of incomplete combustion, the resulting mixture composition is not known in advance, but must be found by calculating the equilibrium composition at constant enthalpy and temperature:
# 
# $$\mathrm{\phi CH_4 + 2(O_2 + 3.76 N_2) \rightarrow ? CO_2 + ? CO + ? H_2 + ? H_2O + ? O_2 + 7.52 N_2 + minor\ species}$$
# 
# Now, we use a gas phase object containing all 53 species defined in the GRI 3.0 mechanism, and compute the equilibrium composition as a function of equivalence ratio.
# 

# Create an IdealGas object including incomplete combustion species
gas2 = ct.Solution(thermo='IdealGas', species=species.values())
T_incomplete = np.zeros(phi.shape)
for i in range(len(phi)):
    gas2.TP = 300, ct.one_atm
    gas2.set_equivalence_ratio(phi[i], 'CH4', 'O2:1, N2:3.76')
    gas2.equilibrate('HP')
    T_incomplete[i] = gas2.T


plt.plot(phi, T_complete, label='complete combustion', lw=2)
plt.plot(phi, T_incomplete, label='incomplete combustion', lw=2)
plt.grid(True)
plt.xlabel('Equivalence ratio, $\phi$')
plt.ylabel('Temperature [K]');





# # Batch Reactor Example
# ## Ignition delay computation
# 
# In this example we will illustrate how to setup and use a constant volume batch reactor. This reactor will then be used to compute the ignition delay of a gas at any temperature and pressure
# 
# The reactor (system) is simply an insulated box.
# 

from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np

import time

import cantera as ct
print('Runnning Cantera version: ' + ct.__version__)


# ### Define the gas
# In this example we will choose n-heptane as the gas. For a representative kinetic model, we use the 160 species [mechanism](https://combustion.llnl.gov/archived-mechanisms/alkanes/heptane-reduced-mechanism) by [Seier et al. 2000, Proc. Comb. Inst](http://dx.doi.org/10.1016/S0082-0784&#40;00&#41;80610-4). 
# 

gas = ct.Solution('data/seiser.cti')


# ### Define reactor conditions : temperature, pressure, fuel, stoichiometry
# 

# Define the reactor temperature and pressure
reactorTemperature = 1000 #Kelvin
reactorPressure = 101325.0 #Pascals

gas.TP = reactorTemperature, reactorPressure

# Define the fuel, oxidizer and set the stoichiometry
gas.set_equivalence_ratio(phi=1.0, fuel='nc7h16', oxidizer={'o2':1.0, 'n2':3.76})

# Create a batch reactor object and add it to a reactor network
# In this example, the batch reactor will be the only reactor
# in the network
r = ct.IdealGasReactor(contents=gas, name='Batch Reactor')
reactorNetwork = ct.ReactorNet([r])

# now compile a list of all variables for which we will store data
stateVariableNames = [r.component_name(item) for item in range(r.n_vars)]

# use the above list to create a DataFrame
timeHistory = pd.DataFrame(columns=stateVariableNames)


# ### Define useful functions
# 

def ignitionDelay(df, species):
    """
    This function computes the ignition delay from the occurence of the
    peak in species' concentration.
    """
    return df[species].argmax()


#Tic
t0 = time.time()

# This is a starting estimate. If you do not get an ignition within this time, increase it
estimatedIgnitionDelayTime = 0.1
t = 0

counter = 1;
while(t < estimatedIgnitionDelayTime):
    t = reactorNetwork.step()
    if (counter%10 == 0):
        # We will save only every 10th value. Otherwise, this takes too long
        # Note that the species concentrations are mass fractions
        timeHistory.loc[t] = reactorNetwork.get_state()
    counter+=1

# We will use the 'oh' species to compute the ignition delay
tau = ignitionDelay(timeHistory, 'oh')

#Toc
t1 = time.time()

print('Computed Ignition Delay: {:.3e} seconds. Took {:3.2f}s to compute'.format(tau, t1-t0))

# If you want to save all the data - molefractions, temperature, pressure, etc
# uncomment the next line
# timeHistory.to_csv("time_history.csv")


# ## Plot the result
# 

# ### Import modules and set plotting defaults
# 

import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().magic('matplotlib notebook')

plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['figure.autolayout'] = True

plt.style.use('ggplot')
plt.style.use('seaborn-pastel')


# ### Figure illustrating the definition of ignition delay
# 

plt.figure()
plt.plot(timeHistory.index, timeHistory['oh'],'-o')
plt.xlabel('Time (s)')
plt.ylabel('$Y_{OH}$')

plt.xlim([0,0.05])
plt.arrow(0, 0.008, tau, 0, width=0.0001, head_width=0.0005,
          head_length=0.001, length_includes_head=True, color='r', shape='full')
plt.annotate(r'$Ignition Delay: \tau_{ign}$', xy=(0,0), xytext=(0.01, 0.0082), fontsize=16);


# ## Illustration : NTC behavior
# A common benchmark for a reaction mechanism is its ability to reproduce the **N**egative **T**emperature **C**oefficient behavior. Intuitively, as the temperature of an explosive mixture increases, it should ignite faster. But, under certain conditions, we observe the opposite. This is referred to as NTC behavior. Reproducing experimentally observed NTC behavior is thus an important test for any mechanism. We will do this now by computing and visualizing the ignition delay for a wide range of temperatures
# 

# ### Define the temperatures for which we will run the simulations
# 

# Make a list of all the temperatures we would like to run simulations at
T = [1800, 1600, 1400, 1200, 1000, 950, 925, 900, 850, 825, 800,
     750, 700, 675, 650, 625, 600, 550, 500]

estimatedIgnitionDelayTimes = np.ones(len(T))

# Make time adjustments for the highest and lowest temperatures. This we do empirically
estimatedIgnitionDelayTimes[:6] = 6*[0.1]
estimatedIgnitionDelayTimes[-2:] = 10
estimatedIgnitionDelayTimes[-1] = 100

# Now create a dataFrame out of these
ignitionDelays = pd.DataFrame(data={'T':T})
ignitionDelays['ignDelay'] = np.nan


# Now, what we will do is simply run the code above the plots for each temperature.
# 

for i, temperature in enumerate(T):
    # Setup the gas and reactor
    reactorTemperature = temperature
    reactorPressure = 101325.0
    gas.TP = reactorTemperature, reactorPressure
    gas.set_equivalence_ratio(phi=1.0, fuel='nc7h16', oxidizer={'o2':1.0, 'n2':3.76})
    r = ct.IdealGasReactor(contents=gas, name='Batch Reactor')
    reactorNetwork = ct.ReactorNet([r])

    # Create and empty data frame
    timeHistory = pd.DataFrame(columns=timeHistory.columns)

    t0 = time.time()

    t = 0
    counter = 0
    while t < estimatedIgnitionDelayTimes[i]:
        t = reactorNetwork.step()
        if not counter % 20:
            timeHistory.loc[t] = r.get_state()
        counter += 1

    tau = ignitionDelay(timeHistory, 'oh')
    t1 = time.time()

    print('Computed Ignition Delay: {:.3e} seconds for T={}K. Took {:3.2f}s to compute'.format(tau, temperature, t1-t0))

    ignitionDelays.set_value(index=i, col='ignDelay', value=tau)


# ### Figure: ignition delay ($\tau$) vs. the inverse of temperature ($\frac{1000}{T}$). 
# 

fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy(1000/ignitionDelays['T'], ignitionDelays['ignDelay'],'o-')
ax.set_ylabel('Ignition Delay (s)')
ax.set_xlabel(r'$\frac{1000}{T (K)}$', fontsize=18)

# Add a second axis on top to plot the temperature for better readability
ax2 = ax.twiny()
ticks = ax.get_xticks()
ax2.set_xticks(ticks)
ax2.set_xticklabels((1000/ticks).round(1))
ax2.set_xlim(ax.get_xlim())
ax2.set_xlabel(r'Temperature: $T(K)$');





# # Cantera Example: Heating values
# ## Heating value of Methane
# The complete reaction for heating methane is:
# 
# $\mathrm{CH_4+2O_2\rightarrow CO_2+2H_2O}$
# 
# We compute the lower heating value (LHV) as the difference in enthalpy (per kg *mixture*) between reactants and products at constant temperature and pressure, divided by the mass fraction of fuel in the reactants.
# 

import cantera as ct
gas = ct.Solution('gri30.cti')

# Set reactants state
gas.TPX = 298, 101325, 'CH4:1, O2:2'
h1 = gas.enthalpy_mass
Y_CH4 = gas['CH4'].Y[0] # returns an array, of which we only want the first element

# set state to complete combustion products without changing T or P
gas.TPX = None, None, 'CO2:1, H2O:2' 
h2 = gas.enthalpy_mass

print('LHV = {:.3f} MJ/kg'.format(-(h2-h1)/Y_CH4/1e6))


# The LHV is calculated assuming that water remains in the gas phase. However, more energy can be extracted from the mixture if this water is condensed. This value is the higher heating value (HHV).
# 
# The ideal gas mixture model used here cannot calculate this contribution directly. However, Cantera also has a non-ideal equation of state which can be used to compute this contribution.
# 

water = ct.Water()
# Set liquid water state, with vapor fraction x = 0
water.TX = 298, 0
h_liquid = water.h
# Set gaseous water state, with vapor fraction x = 1
water.TX = 298, 1
h_gas = water.h

# Calculate higher heating value
Y_H2O = gas['H2O'].Y[0]
print('HHV = {:.3f} MJ/kg'.format(-(h2-h1 + (h_liquid-h_gas) * Y_H2O)/Y_CH4/1e6))


# ## Generalizing to arbitrary species
# We can generalize this calculation by determining the composition of the products automatically rather than directly specifying the product composition. This can be done by computing the *elemental mole fractions* of the reactants mixture and noting that for complete combustion, all of the carbon ends up as CO$_2$, all of the hydrogen ends up as H$_2$O, and all of the nitrogen ends up as N$_2$. From this, we can compute the ratio of these species in the products.
# 

def heating_value(fuel):
    """ Returns the LHV and HHV for the specified fuel """
    gas.TP = 298, ct.one_atm
    gas.set_equivalence_ratio(1.0, fuel, 'O2:1.0')
    h1 = gas.enthalpy_mass
    Y_fuel = gas[fuel].Y[0]

    # complete combustion products
    Y_products = {'CO2': gas.elemental_mole_fraction('C'),
                  'H2O': 0.5 * gas.elemental_mole_fraction('H'),
                  'N2': 0.5 * gas.elemental_mole_fraction('N')}

    gas.TPX = None, None, Y_products
    Y_H2O = gas['H2O'].Y[0]
    h2 = gas.enthalpy_mass
    LHV = -(h2-h1)/Y_fuel
    HHV = -(h2-h1 + (h_liquid-h_gas) * Y_H2O)/Y_fuel
    return LHV, HHV

fuels = ['H2', 'CH4', 'C2H6', 'C3H8', 'NH3', 'CH3OH']
print('fuel   LHV (MJ/kg)   HHV (MJ/kg)')
for fuel in fuels:
    LHV, HHV = heating_value(fuel)
    print('{:8s} {:7.3f}      {:7.3f}'.format(fuel, LHV/1e6, HHV/1e6))


