# # An Introduction To `aima-python`  
#   
# The [aima-python](https://github.com/aimacode/aima-python) repository implements, in Python code, the algorithms in the textbook *[Artificial Intelligence: A Modern Approach](http://aima.cs.berkeley.edu)*. A typical module in the repository has the code for a single chapter in the book, but some modules combine several chapters. See [the index](https://github.com/aimacode/aima-python#index-of-code) if you can't find the algorithm you want. The code in this repository attempts to mirror the pseudocode in the textbook as closely as possible and to stress readability foremost; if you are looking for high-performance code with advanced features, there are other repositories for you. For each module, there are three files, for example:
# 
# - [**`logic.py`**](https://github.com/aimacode/aima-python/blob/master/logic.py): Source code with data types and algorithms for dealing with logic; functions have docstrings explaining their use.
# - [**`logic.ipynb`**](https://github.com/aimacode/aima-python/blob/master/logic.ipynb): A notebook like this one; gives more detailed examples and explanations of use.
# - [**`tests/test_logic.py`**](https://github.com/aimacode/aima-python/blob/master/tests/test_logic.py): Test cases, used to verify the code is correct, and also useful to see examples of use.
# 
# There is also an [aima-java](https://github.com/aimacode/aima-java) repository, if you prefer Java.
#   
# ## What version of Python?
#   
# The code is tested in Python [3.4](https://www.python.org/download/releases/3.4.3/) and [3.5](https://www.python.org/downloads/release/python-351/). If you try a different version of Python 3 and find a problem, please report it as an [Issue](https://github.com/aimacode/aima-python/issues). There is an incomplete [legacy branch](https://github.com/aimacode/aima-python/tree/aima3python2) for those who must run in Python 2. 
#   
# We recommend the [Anaconda](https://www.continuum.io/downloads) distribution of Python 3.5. It comes with additional tools like the powerful IPython interpreter, the Jupyter Notebook and many helpful packages for scientific computing. After installing Anaconda, you will be good to go to run all the code and all the IPython notebooks. 
# 
# ## IPython notebooks  
#   
# The IPython notebooks in this repository explain how to use the modules, and give examples of usage. 
# You can use them in three ways: 
# 
# 1. View static HTML pages. (Just browse to the [repository](https://github.com/aimacode/aima-python) and click on a `.ipynb` file link.)
# 2. Run, modify, and re-run code, live. (Download the repository (by [zip file](https://github.com/aimacode/aima-python/archive/master.zip) or by `git` commands), start a Jupyter notebook server with the shell command "`jupyter notebook`" (issued from the directory where the files are), and click on the notebook you want to interact with.)
# 3. Binder - Click on the binder badge on the [repository](https://github.com/aimacode/aima-python) main page to open the notebooks in an executable environment, online. This method does not require any extra installation. The code can be executed and modified from the browser itself.
# 
#   
# You can [read about notebooks](https://jupyter-notebook-beginner-guide.readthedocs.org/en/latest/) and then [get started](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Running%20Code.ipynb).
# 

# # Helpful Tips
# 
# Most of these notebooks start by importing all the symbols in a module:
# 

from logic import *


# From there, the notebook alternates explanations with examples of use. You can run the examples as they are, and you can modify the code cells (or add new cells) and run your own examples. If you have some really good examples to add, you can make a github pull request.
# 
# If you want to see the source code of a function, you can open a browser or editor and see it in another window, or from within the notebook you can use the IPython magic function `%psource` (for "print source"):
# 

get_ipython().magic('psource WalkSAT')


# Or see an abbreviated description of an object with a trailing question mark:
# 

get_ipython().magic('pinfo WalkSAT')


# # Authors
# 
# This notebook by [Chirag Vertak](https://github.com/chiragvartak) and [Peter Norvig](https://github.com/norvig).
# 

# # Reinforcement Learning
# 
# This IPy notebook acts as supporting material for **Chapter 21 Reinforcement Learning** of the book* Artificial Intelligence: A Modern Approach*. This notebook makes use of the implementations in rl.py module. We also make use of implementation of MDPs in the mdp.py module to test our agents. It might be helpful if you have already gone through the IPy notebook dealing with Markov decision process. Let us import everything from the rl module. It might be helpful to view the source of some of our implementations. Please refer to the Introductory IPy file for more details.
# 

from rl import *


# ## CONTENTS
# 
# * Overview
# * Passive Reinforcement Learning
# * Active Reinforcement Learning
# 

# ## OVERVIEW
# 
# Before we start playing with the actual implementations let us review a couple of things about RL.
# 
# 1. Reinforcement Learning is concerned with how software agents ought to take actions in an environment so as to maximize some notion of cumulative reward. 
# 
# 2. Reinforcement learning differs from standard supervised learning in that correct input/output pairs are never presented, nor sub-optimal actions explicitly corrected. Further, there is a focus on on-line performance, which involves finding a balance between exploration (of uncharted territory) and exploitation (of current knowledge).
# 
# -- Source: [Wikipedia](https://en.wikipedia.org/wiki/Reinforcement_learning)
# 
# In summary we have a sequence of state action transitions with rewards associated with some states. Our goal is to find the optimal policy (pi) which tells us what action to take in each state.
# 

# ## PASSIVE REINFORCEMENT LEARNING
# 
# In passive Reinforcement Learning the agent follows a fixed policy and tries to learn the Reward function and the Transition model (if it is not aware of that).
# 

# ### Passive Temporal Difference Agent
# 
# The PassiveTDAgent class in the rl module implements the Agent Program (notice the usage of word Program) described in **Fig 21.4** of the AIMA Book. PassiveTDAgent uses temporal differences to learn utility estimates. In simple terms we learn the difference between the states and backup the values to previous states while following a fixed policy.  Let us look into the source before we see some usage examples.
# 

get_ipython().magic('psource PassiveTDAgent')


# The Agent Program can be obtained by creating the instance of the class by passing the appropriate parameters. Because of the __ call __ method the object that is created behaves like a callable and returns an appropriate action as most Agent Programs do. To instantiate the object we need a policy(pi) and a mdp whose utility of states will be estimated. Let us import a GridMDP object from the mdp module. **Figure 17.1 (sequential_decision_environment)** is similar to **Figure 21.1** but has some discounting as **gamma = 0.9**.
# 

from mdp import sequential_decision_environment


# **Figure 17.1 (sequential_decision_environment)** is a GridMDP object and is similar to the grid shown in **Figure 21.1**. The rewards in the terminal states are **+1** and **-1** and **-0.04** in rest of the states. <img src="files/images/mdp.png"> Now we define a policy similar to **Fig 21.1** in the book.
# 

# Action Directions
north = (0, 1)
south = (0,-1)
west = (-1, 0)
east = (1, 0)

policy = {
    (0, 2): east,  (1, 2): east,  (2, 2): east,   (3, 2): None,
    (0, 1): north,                (2, 1): north,  (3, 1): None,
    (0, 0): north, (1, 0): west,  (2, 0): west,   (3, 0): west, 
}


# Let us create our object now. We also use the **same alpha** as given in the footnote of the book on **page 837**.
# 

our_agent = PassiveTDAgent(policy, sequential_decision_environment, alpha=lambda n: 60./(59+n))


# The rl module also has a simple implementation to simulate iterations. The function is called **run_single_trial**. Now we can try our implementation. We can also compare the utility estimates learned by our agent to those obtained via **value iteration**.
# 

from mdp import value_iteration


# The values calculated by value iteration:
# 

print(value_iteration(sequential_decision_environment))


# Now the values estimated by our agent after **200 trials**.
# 

for i in range(200):
    run_single_trial(our_agent,sequential_decision_environment)
print(our_agent.U)


# We can also explore how these estimates vary with time by using plots similar to **Fig 21.5a**. To do so we define a function to help us with the same. We will first enable matplotlib using the inline backend.
# 

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

def graph_utility_estimates(agent_program, mdp, no_of_iterations, states_to_graph):
    graphs = {state:[] for state in states_to_graph}
    for iteration in range(1,no_of_iterations+1):
        run_single_trial(agent_program, mdp)
        for state in states_to_graph:
            graphs[state].append((iteration, agent_program.U[state]))
    for state, value in graphs.items():
        state_x, state_y = zip(*value)
        plt.plot(state_x, state_y, label=str(state))
    plt.ylim([0,1.2])
    plt.legend(loc='lower right')
    plt.xlabel('Iterations')
    plt.ylabel('U')


# Here is a plot of state (2,2).
# 

agent = PassiveTDAgent(policy, sequential_decision_environment, alpha=lambda n: 60./(59+n))
graph_utility_estimates(agent, sequential_decision_environment, 500, [(2,2)])


# It is also possible to plot multiple states on the same plot.
# 

graph_utility_estimates(agent, sequential_decision_environment, 500, [(2,2), (3,2)])


# ## ACTIVE REINFORCEMENT LEARNING
# 
# Unlike Passive Reinforcement Learning in Active Reinforcement Learning we are not bound by a policy pi and we need to select our actions. In other words the agent needs to learn an optimal policy. The fundamental tradeoff the agent needs to face is that of exploration vs. exploitation. 
# 

# ### QLearning Agent
# 
# The QLearningAgent class in the rl module implements the Agent Program described in **Fig 21.8** of the AIMA Book. In Q-Learning the agent learns an action-value function Q which gives the utility of taking a given action in a particular state. Q-Learning does not required a transition model and hence is a model free method. Let us look into the source before we see some usage examples.
# 

get_ipython().magic('psource QLearningAgent')


# The Agent Program can be obtained by creating the instance of the class by passing the appropriate parameters. Because of the __ call __ method the object that is created behaves like a callable and returns an appropriate action as most Agent Programs do. To instantiate the object we need a mdp similar to the PassiveTDAgent.
# 
#  Let us use the same GridMDP object we used above. **Figure 17.1 (sequential_decision_environment)** is similar to **Figure 21.1** but has some discounting as **gamma = 0.9**. The class also implements an exploration function **f** which returns fixed **Rplus** untill agent has visited state, action **Ne** number of times. This is the same as the one defined on page **842** of the book. The method **actions_in_state** returns actions possible in given state. It is useful when applying max and argmax operations.
# 

# Let us create our object now. We also use the **same alpha** as given in the footnote of the book on **page 837**. We use **Rplus = 2** and **Ne = 5** as defined on page 843. **Fig 21.7**  
# 

q_agent = QLearningAgent(sequential_decision_environment, Ne=5, Rplus=2, 
                         alpha=lambda n: 60./(59+n))


# Now to try out the q_agent we make use of the **run_single_trial** function in rl.py (which was also used above). Let us use **200** iterations.
# 

for i in range(200):
    run_single_trial(q_agent,sequential_decision_environment)


# Now let us see the Q Values. The keys are state-action pairs. Where differnt actions correspond according to:
# 
# north = (0, 1)
# south = (0,-1)
# west = (-1, 0)
# east = (1, 0)
# 

q_agent.Q


# The Utility **U** of each state is related to **Q** by the following equation.
# 
# **U (s) = max <sub>a</sub> Q(s, a)**
# 
# Let us convert the Q Values above into U estimates.
# 
# 

U = defaultdict(lambda: -1000.) # Very Large Negative Value for Comparison see below.
for state_action, value in q_agent.Q.items():
    state, action = state_action
    if U[state] < value:
                U[state] = value


U


# Let us finally compare these estimates to value_iteration results.
# 

print(value_iteration(sequential_decision_environment))





# # AIMA Python Binder Index
# 
# Welcome to the AIMA Python Code Repository. You should be seeing this index notebook if you clicked on the **Launch Binder** button on the [repository](https://github.com/aimacode/aima-python). If you are viewing this notebook directly on Github we suggest that you use the **Launch Binder** button instead. Binder allows you to experiment with all the code in the browser itself without the need of installing anything on your local machine. Below is the list of notebooks that should assist you in navigating the different notebooks available. 
# 
# If you are completely new to AIMA Python or Jupyter Notebooks we suggest that you start with the Introduction Notebook.
# 
# # List of Notebooks
# 
# 1. [**Introduction**](./intro.ipynb)
# 
# 2. [**Agents**](./agents.ipynb)
# 
# 3. [**Search**](./search.ipynb)
# 
# 4. [**Search - 4th edition**](./search-4e.ipynb)
# 
# 4. [**Games**](./games.ipynb)
# 
# 5. [**Constraint Satisfaction Problems**](./csp.ipynb)
# 
# 6. [**Logic**](./logic.ipynb)
# 
# 7. [**Planning**](./planning.ipynb)
# 
# 8. [**Probability**](./probability.ipynb)
# 
# 9. [**Markov Decision Processes**](./mdp.ipynb)
# 
# 10. [**Learning**](./learning.ipynb)
# 
# 11. [**Reinforcement Learning**](./rl.ipynb)
# 
# 12. [**Statistical Language Processing Tools**](./text.ipynb)
# 
# 13. [**Natural Language Processing**](./nlp.ipynb)
# 
# Besides the notebooks it is also possible to make direct modifications to the Python/JS code. To view/modify the complete set of files [click here](.) to view the Directory structure.
# 

# # Probability and Bayesian Networks
# 
# Probability theory allows us to compute the likelihood of certain events, given assumptioons about the components of the event. A Bayesian network, or Bayes net for short, is a data structure to represent a joint probability distribution over several random variables, and do inference on it. 
# 
# As an example, here is a network with five random variables, each with its conditional probability table, and with arrows from parent to child variables. The story, from Judea Pearl, is that there is a house burglar alarm, which can be triggered by either a burglary or an earthquake. If the alarm sounds, one or both of the neighbors, John and Mary, might call the owwner to say the alarm is sounding.
# 
# <p><img src="http://norvig.com/ipython/burglary2.jpg">
# 
# We implement this with the help of seven Python classes:
# 
# 
# ## `BayesNet()`
# 
# A `BayesNet` is a graph (as in the diagram above) where each node represents a random variable, and the edges are parent&rarr;child links. You can construct an empty graph with `BayesNet()`, then add variables one at a time with the method call `.add(`*variable_name, parent_names, cpt*`)`,  where the names are strings, and each of the  `parent_names` must already have been `.add`ed.
# 
# ## `Variable(`*name, cpt, parents*`)`
# 
# A random variable; the ovals in the diagram above. The value of a variable depends on the value of the parents, in a probabilistic way specified by the variable's conditional probability table (CPT). Given the parents, the variable is independent of all the other variables. For example, if I know whether *Alarm* is true or false, then I know the probability of *JohnCalls*, and evidence about the other variables won't give me any more information about *JohnCalls*. Each row of the CPT uses the same order of variables as the list of parents.
# We will only allow variables with a finite discrete domain; not continuous values. 
# 
# ## `ProbDist(`*mapping*`)`<br>`Factor(`*mapping*`)`
# 
# A probability distribution is a mapping of `{outcome: probability}` for every outcome of a random variable. 
# You can give `ProbDist` the same arguments that you would give to the `dict` initializer, for example
# `ProbDist(sun=0.6, rain=0.1, cloudy=0.3)`.
# As a shortcut for Boolean Variables, you can say `ProbDist(0.95)` instead of `ProbDist({T: 0.95, F: 0.05})`. 
# In a probability distribution, every value is between 0 and 1, and the values sum to 1.
# A `Factor` is similar to a probability distribution, except that the values need not sum to 1. Factors
# are used in the variable elimination inference method.
# 
# ## `Evidence(`*mapping*`)`
# 
# A mapping of `{Variable: value, ...}` pairs, describing the exact values for a set of variables&mdash;the things we know for sure.
# 
# ## `CPTable(`*rows, parents*`)`
# 
# A conditional probability table (or *CPT*) describes the probability of each possible outcome value of a random variable, given the values of the parent variables. A `CPTable` is a a mapping, `{tuple: probdist, ...}`, where each tuple lists the values of each of the parent variables, in order, and each probability distribution says what the possible outcomes are, given those values of the parents. The `CPTable` for *Alarm* in the diagram above would be represented as follows:
# 
#     CPTable({(T, T): .95,
#              (T, F): .94,
#              (F, T): .29,
#              (F, F): .001},
#             [Burglary, Earthquake])
#             
# How do you read this? Take the second row, "`(T, F): .94`". This means that when the first parent (`Burglary`) is true, and the second parent (`Earthquake`) is fale, then the probability of `Alarm` being true is .94. Note that the .94 is an abbreviation for `ProbDist({T: .94, F: .06})`.
#             
# ## `T = Bool(True); F = Bool(False)`
# 
# When I used `bool` values (`True` and `False`), it became hard to read rows in CPTables, because  the columns didn't line up:
# 
#      (True, True, False, False, False)
#      (False, False, False, False, True)
#      (True, False, False, True, True)
#      
# Therefore, I created the `Bool` class, with constants `T` and `F` such that  `T == True` and `F == False`, and now rows are easier to read:
# 
#      (T, T, F, F, F)
#      (F, F, F, F, T)
#      (T, F, F, T, T)
#      
# Here is the code for these classes:
# 

from collections import defaultdict, Counter
import itertools
import math
import random

class BayesNet(object):
    "Bayesian network: a graph of variables connected by parent links."
     
    def __init__(self): 
        self.variables = [] # List of variables, in parent-first topological sort order
        self.lookup = {}    # Mapping of {variable_name: variable} pairs
            
    def add(self, name, parentnames, cpt):
        "Add a new Variable to the BayesNet. Parentnames must have been added previously."
        parents = [self.lookup[name] for name in parentnames]
        var = Variable(name, cpt, parents)
        self.variables.append(var)
        self.lookup[name] = var
        return self
    
class Variable(object):
    "A discrete random variable; conditional on zero or more parent Variables."
    
    def __init__(self, name, cpt, parents=()):
        "A variable has a name, list of parent variables, and a Conditional Probability Table."
        self.__name__ = name
        self.parents  = parents
        self.cpt      = CPTable(cpt, parents)
        self.domain   = set(itertools.chain(*self.cpt.values())) # All the outcomes in the CPT
                
    def __repr__(self): return self.__name__
    
class Factor(dict): "An {outcome: frequency} mapping."

class ProbDist(Factor):
    """A Probability Distribution is an {outcome: probability} mapping. 
    The values are normalized to sum to 1.
    ProbDist(0.75) is an abbreviation for ProbDist({T: 0.75, F: 0.25})."""
    def __init__(self, mapping=(), **kwargs):
        if isinstance(mapping, float):
            mapping = {T: mapping, F: 1 - mapping}
        self.update(mapping, **kwargs)
        normalize(self)
        
class Evidence(dict): 
    "A {variable: value} mapping, describing what we know for sure."
        
class CPTable(dict):
    "A mapping of {row: ProbDist, ...} where each row is a tuple of values of the parent variables."
    
    def __init__(self, mapping, parents=()):
        """Provides two shortcuts for writing a Conditional Probability Table. 
        With no parents, CPTable(dist) means CPTable({(): dist}).
        With one parent, CPTable({val: dist,...}) means CPTable({(val,): dist,...})."""
        if len(parents) == 0 and not (isinstance(mapping, dict) and set(mapping.keys()) == {()}):
            mapping = {(): mapping}
        for (row, dist) in mapping.items():
            if len(parents) == 1 and not isinstance(row, tuple): 
                row = (row,)
            self[row] = ProbDist(dist)

class Bool(int):
    "Just like `bool`, except values display as 'T' and 'F' instead of 'True' and 'False'"
    __str__ = __repr__ = lambda self: 'T' if self else 'F'
        
T = Bool(True)
F = Bool(False)


# And here are some associated functions:
# 

def P(var, evidence={}):
    "The probability distribution for P(variable | evidence), when all parent variables are known (in evidence)."
    row = tuple(evidence[parent] for parent in var.parents)
    return var.cpt[row]

def normalize(dist):
    "Normalize a {key: value} distribution so values sum to 1.0. Mutates dist and returns it."
    total = sum(dist.values())
    for key in dist:
        dist[key] = dist[key] / total
        assert 0 <= dist[key] <= 1, "Probabilities must be between 0 and 1."
    return dist

def sample(probdist):
    "Randomly sample an outcome from a probability distribution."
    r = random.random() # r is a random point in the probability distribution
    c = 0.0             # c is the cumulative probability of outcomes seen so far
    for outcome in probdist:
        c += probdist[outcome]
        if r <= c:
            return outcome
        
def globalize(mapping):
    "Given a {name: value} mapping, export all the names to the `globals()` namespace."
    globals().update(mapping)


# # Sample Usage
# 
# Here are some examples of using the classes:
# 

# Example random variable: Earthquake:
# An earthquake occurs on 0.002 of days, independent of any other variables.
Earthquake = Variable('Earthquake', 0.002)


# The probability distribution for Earthquake
P(Earthquake)


# Get the probability of a specific outcome by subscripting the probability distribution
P(Earthquake)[T]


# Randomly sample from the distribution:
sample(P(Earthquake))


# Randomly sample 100,000 times, and count up the results:
Counter(sample(P(Earthquake)) for i in range(100000))


# Two equivalent ways of specifying the same Boolean probability distribution:
assert ProbDist(0.75) == ProbDist({T: 0.75, F: 0.25})


# Two equivalent ways of specifying the same non-Boolean probability distribution:
assert ProbDist(win=15, lose=3, tie=2) == ProbDist({'win': 15, 'lose': 3, 'tie': 2})
ProbDist(win=15, lose=3, tie=2)


# The difference between a Factor and a ProbDist--the ProbDist is normalized:
Factor(a=1, b=2, c=3, d=4)


ProbDist(a=1, b=2, c=3, d=4)


# # Example: Alarm Bayes Net
# 
# Here is how we define the Bayes net from the diagram above:
# 

alarm_net = (BayesNet()
    .add('Burglary', [], 0.001)
    .add('Earthquake', [], 0.002)
    .add('Alarm', ['Burglary', 'Earthquake'], {(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001})
    .add('JohnCalls', ['Alarm'], {T: 0.90, F: 0.05})
    .add('MaryCalls', ['Alarm'], {T: 0.70, F: 0.01}))  


# Make Burglary, Earthquake, etc. be global variables
globalize(alarm_net.lookup) 
alarm_net.variables


# Probability distribution of a Burglary
P(Burglary)


# Probability of Alarm going off, given a Burglary and not an Earthquake:
P(Alarm, {Burglary: T, Earthquake: F})


# Where that came from: the (T, F) row of Alarm's CPT:
Alarm.cpt


# # Bayes Nets as Joint Probability Distributions
# 
# A Bayes net is a compact way of specifying a full joint distribution over all the variables in the network.  Given a set of variables {*X*<sub>1</sub>, ..., *X*<sub>*n*</sub>}, the full joint distribution is:
# 
# P(*X*<sub>1</sub>=*x*<sub>1</sub>, ..., *X*<sub>*n*</sub>=*x*<sub>*n*</sub>) = <font size=large>&Pi;</font><sub>*i*</sub> P(*X*<sub>*i*</sub> = *x*<sub>*i*</sub> | parents(*X*<sub>*i*</sub>))
# 
# For a network with *n* variables, each of which has *b* values, there are *b<sup>n</sup>* rows in the joint distribution (for example, a billion rows for 30 Boolean variables), making it impractical to explicitly create the joint distribution for large networks.  But for small networks, the function `joint_distribution` creates the distribution, which can be instructive to look at, and can be used to do inference. 
# 

def joint_distribution(net):
    "Given a Bayes net, create the joint distribution over all variables."
    return ProbDist({row: prod(P_xi_given_parents(var, row, net)
                               for var in net.variables)
                     for row in all_rows(net)})

def all_rows(net): return itertools.product(*[var.domain for var in net.variables])

def P_xi_given_parents(var, row, net):
    "The probability that var = xi, given the values in this row."
    dist = P(var, Evidence(zip(net.variables, row)))
    xi = row[net.variables.index(var)]
    return dist[xi]

def prod(numbers):
    "The product of numbers: prod([2, 3, 5]) == 30. Analogous to `sum([2, 3, 5]) == 10`."
    result = 1
    for x in numbers:
        result *= x
    return result


# All rows in the joint distribution (2**5 == 32 rows)
set(all_rows(alarm_net))


# Let's work through just one row of the table:
row = (F, F, F, F, F)


# This is the probability distribution for Alarm
P(Alarm, {Burglary: F, Earthquake: F})


# Here's the probability that Alarm is false, given the parent values in this row:
P_xi_given_parents(Alarm, row, alarm_net)


# The full joint distribution:
joint_distribution(alarm_net)


# Probability that "the alarm has sounded, but neither a burglary nor an earthquake has occurred, 
# and both John and Mary call" (page 514 says it should be 0.000628)

print(alarm_net.variables)
joint_distribution(alarm_net)[F, F, T, T, T]


# # Inference by Querying the Joint Distribution
# 
# We can use `P(variable, evidence)` to get the probability of aa variable, if we know the vaues of all the parent variables. But what if we don't know? Bayes nets allow us to calculate the probability, but the calculation is not just a lookup in the CPT; it is a global calculation across the whole net. One inefficient but straightforward way of doing the calculation is to create the joint probability distribution, then pick out just the rows that
# match the evidence variables, and for each row check what the value of the query variable is, and increment the probability for that value accordningly:
# 

def enumeration_ask(X, evidence, net):
    "The probability distribution for query variable X in a belief net, given evidence."
    i    = net.variables.index(X) # The index of the query variable X in the row
    dist = defaultdict(float)     # The resulting probability distribution over X
    for (row, p) in joint_distribution(net).items():
        if matches_evidence(row, evidence, net):
            dist[row[i]] += p
    return ProbDist(dist)

def matches_evidence(row, evidence, net):
    "Does the tuple of values for this row agree with the evidence?"
    return all(evidence[v] == row[net.variables.index(v)]
               for v in evidence)


# The probability of a Burgalry, given that John calls but Mary does not: 
enumeration_ask(Burglary, {JohnCalls: F, MaryCalls: T}, alarm_net)


# The probability of an Alarm, given that there is an Earthquake and Mary calls:
enumeration_ask(Alarm, {MaryCalls: T, Earthquake: T}, alarm_net)


# # Variable Elimination
# 
# The `enumeration_ask` algorithm takes time and space that is exponential in the number of variables. That is, first it creates the joint distribution, of size *b<sup>n</sup>*, and then it sums out the values for the rows that match the evidence.  We can do better than that if we interleave the joining of variables with the summing out of values.
# This approach is called *variable elimination*. The key insight is that
# when we compute
# 
# P(*X*<sub>1</sub>=*x*<sub>1</sub>, ..., *X*<sub>*n*</sub>=*x*<sub>*n*</sub>) = <font size=large>&Pi;</font><sub>*i*</sub> P(*X*<sub>*i*</sub> = *x*<sub>*i*</sub> | parents(*X*<sub>*i*</sub>))
# 
# we are repeating the calculation of, say, P(*X*<sub>*3*</sub> = *x*<sub>*4*</sub> | parents(*X*<sub>*3*</sub>))
# multiple times, across multiple rows of the joint distribution.
# 
# 
# 

# TODO: Copy over and update Variable Elimination algorithm. Also, sampling algorithms.


# # Example: Flu Net
# 
# In this net, whether a patient gets the flu is dependent on whether they were vaccinated, and having the flu influences whether they get a fever or headache. Here `Fever` is a non-Boolean variable, with three values, `no`, `mild`, and `high`.
# 

flu_net = (BayesNet()
           .add('Vaccinated', [], 0.60)
           .add('Flu', ['Vaccinated'], {T: 0.002, F: 0.02})
           .add('Fever', ['Flu'], {T: ProbDist(no=25, mild=25, high=50),
                                   F: ProbDist(no=97, mild=2, high=1)})
           .add('Headache', ['Flu'], {T: 0.5,   F: 0.03}))

globalize(flu_net.lookup)


# If you just have a headache, you probably don't have the Flu.
enumeration_ask(Flu, {Headache: T, Fever: 'no'}, flu_net)


# Even more so if you were vaccinated.
enumeration_ask(Flu, {Headache: T, Fever: 'no', Vaccinated: T}, flu_net)


# But if you were not vaccinated, there is a higher chance you have the flu.
enumeration_ask(Flu, {Headache: T, Fever: 'no', Vaccinated: F}, flu_net)


# And if you have both headache and fever, and were not vaccinated, 
# then the flu is very likely, especially if it is a high fever.
enumeration_ask(Flu, {Headache: T, Fever: 'mild', Vaccinated: F}, flu_net)


enumeration_ask(Flu, {Headache: T, Fever: 'high', Vaccinated: F}, flu_net)


# # Entropy
# 
# We can compute the entropy of a probability distribution:
# 

def entropy(probdist):
    "The entropy of a probability distribution."
    return - sum(p * math.log(p, 2)
                 for p in probdist.values())


entropy(ProbDist(heads=0.5, tails=0.5))


entropy(ProbDist(yes=1000, no=1))


entropy(P(Alarm, {Earthquake: T, Burglary: F}))


entropy(P(Alarm, {Earthquake: F, Burglary: F}))


# For non-Boolean variables, the entropy can be greater than 1 bit:
# 

entropy(P(Fever, {Flu: T}))


# # Unknown Outcomes: Smoothing
# 
# So far we have dealt with discrete distributions where we know all the possible outcomes in advance. For Boolean variables, the only outcomes are `T` and `F`. For `Fever`, we modeled exactly three outcomes.  However, in some applications we will encounter new, previously unknown outcomes over time. For example, we could train a model on the distribution of words in English, and then somebody could coin a brand new word. To deal with this, we introduce
# the `DefaultProbDist` distribution, which uses the key `None` to stand as a placeholder for any unknown outcome(s).
# 

class DefaultProbDist(ProbDist):
    """A Probability Distribution that supports smoothing for unknown outcomes (keys).
    The default_value represents the probability of an unknown (previously unseen) key. 
    The key `None` stands for unknown outcomes."""
    def __init__(self, default_value, mapping=(), **kwargs):
        self[None] = default_value
        self.update(mapping, **kwargs)
        normalize(self)
        
    def __missing__(self, key): return self[None]        


import re

def words(text): return re.findall(r'\w+', text.lower())

english = words('''This is a sample corpus of English prose. To get a better model, we would train on much
more text. But this should give you an idea of the process. So far we have dealt with discrete 
distributions where we  know all the possible outcomes in advance. For Boolean variables, the only 
outcomes are T and F. For Fever, we modeled exactly three outcomes. However, in some applications we 
will encounter new, previously unknown outcomes over time. For example, when we could train a model on the 
words in this text, we get a distribution, but somebody could coin a brand new word. To deal with this, 
we introduce the DefaultProbDist distribution, which uses the key `None` to stand as a placeholder for any 
unknown outcomes. Probability theory allows us to compute the likelihood of certain events, given 
assumptions about the components of the event. A Bayesian network, or Bayes net for short, is a data 
structure to represent a joint probability distribution over several random variables, and do inference on it.''')

E = DefaultProbDist(0.1, Counter(english))


# 'the' is a common word:
E['the']


# 'possible' is a less-common word:
E['possible']


# 'impossible' was not seen in the training data, but still gets a non-zero probability ...
E['impossible']


# ... as do other rare, previously unseen words:
E['llanfairpwllgwyngyll']


# Note that this does not mean that 'impossible' and 'llanfairpwllgwyngyll' and all the other unknown words
# *each* have probability 0.004.
# Rather, it means that together, all the unknown words total probability 0.004. With that
# interpretation, the sum of all the probabilities is still 1, as it should be. In the `DefaultProbDist`, the
# unknown words are all represented by the key `None`:
# 

E[None]


# # Markov decision processes (MDPs)
# 
# This IPy notebook acts as supporting material for topics covered in **Chapter 17 Making Complex Decisions** of the book* Artificial Intelligence: A Modern Approach*. We makes use of the implementations in mdp.py module. This notebook also includes a brief summary of the main topics as a review. Let us import everything from the mdp module to get started.
# 

from mdp import *


# ## CONTENTS
# 
# * Overview
# * MDP
# * Grid MDP
# * Value Iteration Visualization
# 

# ## OVERVIEW
# 
# Before we start playing with the actual implementations let us review a couple of things about MDPs.
# 
# - A stochastic process has the **Markov property** if the conditional probability distribution of future states of the process (conditional on both past and present states) depends only upon the present state, not on the sequence of events that preceded it.
# 
#     -- Source: [Wikipedia](https://en.wikipedia.org/wiki/Markov_property)
# 
# Often it is possible to model many different phenomena as a Markov process by being flexible with our definition of state.
#    
# 
# - MDPs help us deal with fully-observable and non-deterministic/stochastic environments. For dealing with partially-observable and stochastic cases we make use of generalization of MDPs named POMDPs (partially observable Markov decision process).
# 
# Our overall goal to solve a MDP is to come up with a policy which guides us to select the best action in each state so as to maximize the expected sum of future rewards.
# 

# ## MDP
# 
# To begin with let us look at the implementation of MDP class defined in mdp.py The docstring tells us what all is required to define a MDP namely - set of states,actions, initial state, transition model, and a reward function. Each of these are implemented as methods. Do not close the popup so that you can follow along the description of code below.
# 

get_ipython().magic('psource MDP')


# The **_ _init_ _** method takes in the following parameters:
# 
# - init: the initial state.
# - actlist: List of actions possible in each state.
# - terminals: List of terminal states where only possible action is exit
# - gamma: Discounting factor. This makes sure that delayed rewards have less value compared to immediate ones.
# 
# **R** method returns the reward for each state by using the self.reward dict.
# 
# **T** method is not implemented and is somewhat different from the text. Here we return (probability, s') pairs where s' belongs to list of possible state by taking action a in state s.
# 
# **actions** method returns list of actions possible in each state. By default it returns all actions for states other than terminal states.
# 

# Now let us implement the simple MDP in the image below. States A, B have actions X, Y available in them. Their probabilities are shown just above the arrows. We start with using MDP as base class for our CustomMDP. Obviously we need to make a few changes to suit our case. We make use of a transition matrix as our transitions are not very simple.
# <img src="files/images/mdp-a.png">
# 

# Transition Matrix as nested dict. State -> Actions in state -> States by each action -> Probabilty
t = {
    "A": {
            "X": {"A":0.3, "B":0.7},
            "Y": {"A":1.0}
         },
    "B": {
            "X": {"End":0.8, "B":0.2},
            "Y": {"A":1.0}
         },
    "End": {}
}

init = "A"

terminals = ["End"]

rewards = {
    "A": 5,
    "B": -10,
    "End": 100
}


class CustomMDP(MDP):

    def __init__(self, transition_matrix, rewards, terminals, init, gamma=.9):
        # All possible actions.
        actlist = []
        for state in transition_matrix.keys():
            actlist.extend(transition_matrix.keys())
        actlist = list(set(actlist))

        MDP.__init__(self, init, actlist, terminals=terminals, gamma=gamma)
        self.t = transition_matrix
        self.reward = rewards
        for state in self.t:
            self.states.add(state)

    def T(self, state, action):
        return [(new_state, prob) for new_state, prob in self.t[state][action].items()]


# Finally we instantize the class with the parameters for our MDP in the picture.
# 

our_mdp = CustomMDP(t, rewards, terminals, init, gamma=.9)


# With this we have sucessfully represented our MDP. Later we will look at ways to solve this MDP.
# 

# ## GRID MDP
# 
# Now we look at a concrete implementation that makes use of the MDP as base class. The GridMDP class in the mdp module is used to represent a grid world MDP like the one shown in  in **Fig 17.1** of the AIMA Book. The code should be easy to understand if you have gone through the CustomMDP example.
# 

get_ipython().magic('psource GridMDP')


# The **_ _init_ _** method takes **grid** as an extra parameter compared to the MDP class. The grid is a nested list of rewards in states.
# 
# **go** method returns the state by going in particular direction by using vector_add.
# 
# **T** method is not implemented and is somewhat different from the text. Here we return (probability, s') pairs where s' belongs to list of possible state by taking action a in state s.
# 
# **actions** method returns list of actions possible in each state. By default it returns all actions for states other than terminal states.
# 
# **to_arrows** are used for representing the policy in a grid like format.
# 

# We can create a GridMDP like the one in **Fig 17.1** as follows: 
# 
#     GridMDP([[-0.04, -0.04, -0.04, +1],
#             [-0.04, None,  -0.04, -1],
#             [-0.04, -0.04, -0.04, -0.04]],
#             terminals=[(3, 2), (3, 1)])
#             
# In fact the **sequential_decision_environment** in mdp module has been instantized using the exact same code.
# 

sequential_decision_environment


# # Value Iteration
# 
# Now that we have looked how to represent MDPs. Let's aim at solving them. Our ultimate goal is to obtain an optimal policy. We start with looking at Value Iteration and a visualisation that should help us understanding it better.
# 
# We start by calculating Value/Utility for each of the states. The Value of each state is the expected sum of discounted future rewards given we start in that state and follow a particular policy pi.The algorithm Value Iteration (**Fig. 17.4** in the book) relies on finding solutions of the Bellman's Equation. The intuition Value Iteration works is because values propagate. This point will we more clear after we encounter the visualisation. For more information you can refer to **Section 17.2** of the book. 
# 

get_ipython().magic('psource value_iteration')


# It takes as inputs two parameters an MDP to solve and epsilon the maximum error allowed in the utility of any state. It returns a dictionary containing utilities where the keys are the states and values represent utilities. Let us solve the **sequencial_decision_enviornment** GridMDP.
# 

value_iteration(sequential_decision_environment)


# ## VALUE ITERATION VISUALIZATION
# 
# To illustrate that values propagate out of states let us create a simple visualisation. We will be using a modified version of the value_iteration function which will store U over time. We will also remove the parameter epsilon and instead add the number of iterations we want.
# 

def value_iteration_instru(mdp, iterations=20):
    U_over_time = []
    U1 = {s: 0 for s in mdp.states}
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    for _ in range(iterations):
        U = U1.copy()
        for s in mdp.states:
            U1[s] = R(s) + gamma * max([sum([p * U[s1] for (p, s1) in T(s, a)])
                                        for a in mdp.actions(s)])
        U_over_time.append(U)
    return U_over_time


# Next, we define a function to create the visualisation from the utilities returned by **value_iteration_instru**. The reader need not concern himself with the code that immediately follows as it is the usage of Matplotib with IPython Widgets. If you are interested in reading more about these visit [ipywidgets.readthedocs.io](http://ipywidgets.readthedocs.io)
# 

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from collections import defaultdict
import time

def make_plot_grid_step_function(columns, row, U_over_time):
    '''ipywidgets interactive function supports
       single parameter as input. This function
       creates and return such a function by taking
       in input other parameters
    '''
    def plot_grid_step(iteration):
        data = U_over_time[iteration]
        data = defaultdict(lambda: 0, data)
        grid = []
        for row in range(rows):
            current_row = []
            for column in range(columns):
                current_row.append(data[(column, row)])
            grid.append(current_row)
        grid.reverse() # output like book
        fig = plt.imshow(grid, cmap=plt.cm.bwr, interpolation='nearest')

        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        for col in range(len(grid)):
            for row in range(len(grid[0])):
                magic = grid[col][row]
                fig.axes.text(row, col, "{0:.2f}".format(magic), va='center', ha='center')

        plt.show()
    
    return plot_grid_step

def make_visualize(slider):
    ''' Takes an input a slider and returns 
        callback function for timer and animation
    '''
    
    def visualize_callback(Visualize, time_step):
        if Visualize is True:
            for i in range(slider.min, slider.max + 1):
                slider.value = i
                time.sleep(float(time_step))
    
    return visualize_callback


columns = 4
rows = 3
U_over_time = value_iteration_instru(sequential_decision_environment)


plot_grid_step = make_plot_grid_step_function(columns, rows, U_over_time)


import ipywidgets as widgets
from IPython.display import display

iteration_slider = widgets.IntSlider(min=1, max=15, step=1, value=0)
w=widgets.interactive(plot_grid_step,iteration=iteration_slider)
display(w)

visualize_callback = make_visualize(iteration_slider)

visualize_button = widgets.ToggleButton(desctiption = "Visualize", value = False)
time_select = widgets.ToggleButtons(description='Extra Delay:',options=['0', '0.1', '0.2', '0.5', '0.7', '1.0'])
a = widgets.interactive(visualize_callback, Visualize = visualize_button, time_step=time_select)
display(a)


# Move the slider above to observe how the utility changes across iterations. It is also possible to move the slider using arrow keys or to jump to the value by directly editing the number with a double click. The **Visualize Button** will automatically animate the slider for you. The **Extra Delay Box** allows you to set time delay in seconds upto one second for each time step.
# 

# # Planning: planning.py; chapters 10-11
# 

# This notebook describes the [planning.py](https://github.com/aimacode/aima-python/blob/master/planning.py) module, which covers Chapters 10 (Classical Planning) and  11 (Planning and Acting in the Real World) of *[Artificial Intelligence: A Modern Approach](http://aima.cs.berkeley.edu)*. See the [intro notebook](https://github.com/aimacode/aima-python/blob/master/intro.ipynb) for instructions.
# 
# We'll start by looking at `PDDL` and `Action` data types for defining problems and actions. Then, we will see how to use them by trying to plan a trip from *Sibiu* to *Bucharest* across the familiar map of Romania, from [search.ipynb](https://github.com/aimacode/aima-python/blob/master/search.ipynb). Finally, we will look at the implementation of the GraphPlan algorithm.
# 
# The first step is to load the code:
# 

from planning import *


# To be able to model a planning problem properly, it is essential to be able to represent an Action. Each action we model requires at least three things:
# * preconditions that the action must meet
# * the effects of executing the action
# * some expression that represents the action
# 

# Planning actions have been modelled using the `Action` class. Let's look at the source to see how the internal details of an action are implemented in Python.
# 

get_ipython().magic('psource Action')


# It is interesting to see the way preconditions and effects are represented here. Instead of just being a list of expressions each, they consist of two lists - `precond_pos` and `precond_neg`. This is to work around the fact that PDDL doesn't allow for negations. Thus, for each precondition, we maintain a seperate list of those preconditions that must hold true, and those whose negations must hold true. Similarly, instead of having a single list of expressions that are the result of executing an action, we have two. The first (`effect_add`) contains all the expressions that will evaluate to true if the action is executed, and the the second (`effect_neg`) contains all those expressions that would be false if the action is executed (ie. their negations would be true).
# 
# The constructor parameters, however combine the two precondition lists into a single `precond` parameter, and the effect lists into a single `effect` parameter.
# 

# The `PDDL` class is used to represent planning problems in this module. The following attributes are essential to be able to define a problem:
# * a goal test
# * an initial state
# * a set of viable actions that can be executed in the search space of the problem
# 
# View the source to see how the Python code tries to realise these.
# 

get_ipython().magic('psource PDDL')


# The `initial_state` attribute is a list of `Expr` expressions that forms the initial knowledge base for the problem. Next, `actions` contains a list of `Action` objects that may be executed in the search space of the problem. Lastly, we pass a `goal_test` function as a parameter - this typically takes a knowledge base as a parameter, and returns whether or not the goal has been reached.
# 

# Now lets try to define a planning problem using these tools. Since we already know about the map of Romania, lets see if we can plan a trip across a simplified map of Romania.
# 
# Here is our simplified map definition:
# 

from utils import *
# this imports the required expr so we can create our knowledge base

knowledge_base = [
    expr("Connected(Bucharest,Pitesti)"),
    expr("Connected(Pitesti,Rimnicu)"),
    expr("Connected(Rimnicu,Sibiu)"),
    expr("Connected(Sibiu,Fagaras)"),
    expr("Connected(Fagaras,Bucharest)"),
    expr("Connected(Pitesti,Craiova)"),
    expr("Connected(Craiova,Rimnicu)")
    ]


# Let us add some logic propositions to complete our knowledge about travelling around the map. These are the typical symmetry and transitivity properties of connections on a map. We can now be sure that our `knowledge_base` understands what it truly means for two locations to be connected in the sense usually meant by humans when we use the term.
# 
# Let's also add our starting location - *Sibiu* to the map.
# 

knowledge_base.extend([
     expr("Connected(x,y) ==> Connected(y,x)"),
     expr("Connected(x,y) & Connected(y,z) ==> Connected(x,z)"),
     expr("At(Sibiu)")
    ])


# We now have a complete knowledge base, which can be seen like this:
# 

knowledge_base


# We now define possible actions to our problem. We know that we can drive between any connected places. But, as is evident from [this](https://en.wikipedia.org/wiki/List_of_airports_in_Romania) list of Romanian airports, we can also fly directly between Sibiu, Bucharest, and Craiova.
# 
# We can define these flight actions like this:
# 

#Sibiu to Bucharest
precond_pos = [expr('At(Sibiu)')]
precond_neg = []
effect_add = [expr('At(Bucharest)')]
effect_rem = [expr('At(Sibiu)')]
fly_s_b = Action(expr('Fly(Sibiu, Bucharest)'), [precond_pos, precond_neg], [effect_add, effect_rem])

#Bucharest to Sibiu
precond_pos = [expr('At(Bucharest)')]
precond_neg = []
effect_add = [expr('At(Sibiu)')]
effect_rem = [expr('At(Bucharest)')]
fly_b_s = Action(expr('Fly(Bucharest, Sibiu)'), [precond_pos, precond_neg], [effect_add, effect_rem])

#Sibiu to Craiova
precond_pos = [expr('At(Sibiu)')]
precond_neg = []
effect_add = [expr('At(Craiova)')]
effect_rem = [expr('At(Sibiu)')]
fly_s_c = Action(expr('Fly(Sibiu, Craiova)'), [precond_pos, precond_neg], [effect_add, effect_rem])

#Craiova to Sibiu
precond_pos = [expr('At(Craiova)')]
precond_neg = []
effect_add = [expr('At(Sibiu)')]
effect_rem = [expr('At(Craiova)')]
fly_c_s = Action(expr('Fly(Craiova, Sibiu)'), [precond_pos, precond_neg], [effect_add, effect_rem])

#Bucharest to Craiova
precond_pos = [expr('At(Bucharest)')]
precond_neg = []
effect_add = [expr('At(Craiova)')]
effect_rem = [expr('At(Bucharest)')]
fly_b_c = Action(expr('Fly(Bucharest, Craiova)'), [precond_pos, precond_neg], [effect_add, effect_rem])

#Craiova to Bucharest
precond_pos = [expr('At(Craiova)')]
precond_neg = []
effect_add = [expr('At(Bucharest)')]
effect_rem = [expr('At(Craiova)')]
fly_c_b = Action(expr('Fly(Craiova, Bucharest)'), [precond_pos, precond_neg], [effect_add, effect_rem])


# And the drive actions like this.
# 

#Drive
precond_pos = [expr('At(x)')]
precond_neg = []
effect_add = [expr('At(y)')]
effect_rem = [expr('At(x)')]
drive = Action(expr('Drive(x, y)'), [precond_pos, precond_neg], [effect_add, effect_rem])


# Finally, we can define a a function that will tell us when we have reached our destination, Bucharest.
# 

def goal_test(kb):
    return kb.ask(expr("At(Bucharest)"))


# Thus, with all the components in place, we can define the planning problem.
# 

prob = PDDL(knowledge_base, [fly_s_b, fly_b_s, fly_s_c, fly_c_s, fly_b_c, fly_c_b, drive], goal_test)








# # Probability 
# 
# This IPy notebook acts as supporting material for **Chapter 13 Quantifying Uncertainty**, **Chapter 14 Probabilistic Reasoning** and **Chapter 15 Probabilistic Reasoning over Time** of the book* Artificial Intelligence: A Modern Approach*. This notebook makes use of the implementations in probability.py module. Let us import everything from the probability module. It might be helpful to view the source of some of our implementations. Please refer to the Introductory IPy file for more details on how to do so.
# 

from probability import *


# ## Probability Distribution
# 
# Let us begin by specifying discrete probability distributions. The class **ProbDist** defines a discrete probability distribution. We name our random variable and then assign probabilities to the different values of the random variable. Assigning probabilities to the values works similar to that of using a dictionary with keys being the Value and we assign to it the probability. This is possible because of the magic methods **_ _getitem_ _**  and **_ _setitem_ _** which store the probabilities in the prob dict of the object. You can keep the source window open alongside while playing with the rest of the code to get a better understanding.
# 

get_ipython().magic('psource ProbDist')


p = ProbDist('Flip')
p['H'], p['T'] = 0.25, 0.75
p['T']


# The first parameter of the constructor **varname** has a default value of '?'. So if the name is not passed it defaults to ?. The keyword argument **freqs** can be a dictionary of values of random variable:probability. These are then normalized such that the probability values sum upto 1 using the **normalize** method.
# 

p = ProbDist(freqs={'low': 125, 'medium': 375, 'high': 500})
p.varname


(p['low'], p['medium'], p['high'])


# Besides the **prob** and **varname** the object also separately keeps track of all the values of the distribution in a list called **values**. Every time a new value is assigned a probability it is appended to this list, This is done inside the **_ _setitem_ _** method.
# 

p.values


# The distribution by default is not normalized if values are added incremently. We can still force normalization by invoking the **normalize** method.
# 

p = ProbDist('Y')
p['Cat'] = 50
p['Dog'] = 114
p['Mice'] = 64
(p['Cat'], p['Dog'], p['Mice'])


p.normalize()
(p['Cat'], p['Dog'], p['Mice'])


# It is also possible to display the approximate values upto decimals using the **show_approx** method.
# 

p.show_approx()


# ## Joint Probability Distribution
# 
# The helper function **event_values** returns a tuple of the values of variables in event. An event is specified by a dict where the keys are the names of variables and the corresponding values are the value of the variable. Variables are specified with a list. The ordering of the returned tuple is same as those of the variables.
# 
# 
# Alternatively if the event is specified by a list or tuple of equal length of the variables. Then the events tuple is returned as it is.
# 

event = {'A': 10, 'B': 9, 'C': 8}
variables = ['C', 'A']
event_values (event, variables)


# _A probability model is completely determined by the joint distribution for all of the random variables._ (**Section 13.3**) The probability module implements these as the class **JointProbDist** which inherits from the **ProbDist** class. This class specifies a discrete probability distribute over a set of variables. 
# 

get_ipython().magic('psource JointProbDist')


# Values for a Joint Distribution is a an ordered tuple in which each item corresponds to the value associate with a particular variable. For Joint Distribution of X, Y where X, Y take integer values this can be something like (18, 19).
# 
# To specify a Joint distribution we first need an ordered list of variables.
# 

variables = ['X', 'Y']
j = JointProbDist(variables)
j


# Like the **ProbDist** class **JointProbDist** also employes magic methods to assign probability to different values.
# The probability can be assigned in either of the two formats for all possible values of the distribution. The **event_values** call inside  **_ _getitem_ _**  and **_ _setitem_ _** does the required processing to make this work.
# 

j[1,1] = 0.2
j[dict(X=0, Y=1)] = 0.5

(j[1,1], j[0,1])


# It is also possible to list all the values for a particular variable using the **values** method.
# 

j.values('X')


# ## Inference Using Full Joint Distributions
# 
# In this section we use Full Joint Distributions to calculate the posterior distribution given some evidence. We represent evidence by using a python dictionary with variables as dict keys and dict values representing the values.
# 
# This is illustrated in **Section 13.3** of the book. The functions **enumerate_joint** and **enumerate_joint_ask** implement this functionality. Under the hood they implement **Equation 13.9** from the book.
# 
# $$\textbf{P}(X | \textbf{e}) = α \textbf{P}(X, \textbf{e}) = α \sum_{y} \textbf{P}(X, \textbf{e}, \textbf{y})$$
# 
# Here **α** is the normalizing factor. **X** is our query variable and **e** is the evidence. According to the equation we enumerate on the remaining variables **y** (not in evidence or query variable) i.e. all possible combinations of **y**
# 
# We will be using the same example as the book. Let us create the full joint distribution from **Figure 13.3**.  
# 

full_joint = JointProbDist(['Cavity', 'Toothache', 'Catch'])
full_joint[dict(Cavity=True, Toothache=True, Catch=True)] = 0.108
full_joint[dict(Cavity=True, Toothache=True, Catch=False)] = 0.012
full_joint[dict(Cavity=True, Toothache=False, Catch=True)] = 0.016
full_joint[dict(Cavity=True, Toothache=False, Catch=False)] = 0.064
full_joint[dict(Cavity=False, Toothache=True, Catch=True)] = 0.072
full_joint[dict(Cavity=False, Toothache=False, Catch=True)] = 0.144
full_joint[dict(Cavity=False, Toothache=True, Catch=False)] = 0.008
full_joint[dict(Cavity=False, Toothache=False, Catch=False)] = 0.576


# Let us now look at the **enumerate_joint** function returns the sum of those entries in P consistent with e,provided variables is P's remaining variables (the ones not in e). Here, P refers to the full joint distribution. The function uses a recursive call in its implementation. The first parameter **variables** refers to remaining variables. The function in each recursive call keeps on variable constant while varying others.
# 

get_ipython().magic('psource enumerate_joint')


# Let us assume we want to find **P(Toothache=True)**. This can be obtained by marginalization (**Equation 13.6**). We can use **enumerate_joint** to solve for this by taking Toothache=True as our evidence. **enumerate_joint** will return the sum of probabilities consistent with evidence i.e. Marginal Probability.
# 

evidence = dict(Toothache=True)
variables = ['Cavity', 'Catch'] # variables not part of evidence
ans1 = enumerate_joint(variables, evidence, full_joint)
ans1


# You can verify the result from our definition of the full joint distribution. We can use the same function to find more complex probabilities like **P(Cavity=True and Toothache=True)** 
# 

evidence = dict(Cavity=True, Toothache=True)
variables = ['Catch'] # variables not part of evidence
ans2 = enumerate_joint(variables, evidence, full_joint)
ans2


# Being able to find sum of probabilities satisfying given evidence allows us to compute conditional probabilities like **P(Cavity=True | Toothache=True)** as we can rewrite this as $$P(Cavity=True | Toothache = True) = \frac{P(Cavity=True \ and \ Toothache=True)}{P(Toothache=True)}$$
# 
# We have already calculated both the numerator and denominator.
# 

ans2/ans1


# We might be interested in the probability distribution of a particular variable conditioned on some evidence. This can involve doing calculations like above for each possible value of the variable. This has been implemented slightly differently  using normalization in the function **enumerate_joint_ask** which returns a probability distribution over the values of the variable **X**, given the {var:val} observations **e**, in the **JointProbDist P**. The implementation of this function calls **enumerate_joint** for each value of the query variable and passes **extended evidence** with the new evidence having **X = x<sub>i</sub>**. This is followed by normalization of the obtained distribution.
# 

get_ipython().magic('psource enumerate_joint_ask')


# Let us find **P(Cavity | Toothache=True)** using **enumerate_joint_ask**.
# 

query_variable = 'Cavity'
evidence = dict(Toothache=True)
ans = enumerate_joint_ask(query_variable, evidence, full_joint)
(ans[True], ans[False])


# You can verify that the first value is the same as we obtained earlier by manual calculation.
# 

# ## Bayesian Networks
# 
# A Bayesian network is a representation of the joint probability distribution encoding a collection of conditional independence statements.
# 
# A Bayes Network is implemented as the class **BayesNet**. It consisits of a collection of nodes implemented by the class **BayesNode**. The implementation in the above mentioned classes focuses only on boolean variables. Each node is associated with a variable and it contains a **conditional probabilty table (cpt)**. The **cpt** represents the probability distribution of the variable conditioned on its parents **P(X | parents)**.
# 
# Let us dive into the **BayesNode** implementation.
# 

get_ipython().magic('psource BayesNode')


# The constructor takes in the name of **variable**, **parents** and **cpt**. Here **variable** is a the name of the variable like 'Earthquake'. **parents** should a list or space separate string with variable names of parents. The conditional probability table is a dict {(v1, v2, ...): p, ...}, the distribution P(X=true | parent1=v1, parent2=v2, ...) = p. Here the keys are combination of boolean values that the parents take. The length and order of the values in keys should be same as the supplied **parent** list/string. In all cases the probability of X being false is left implicit, since it follows from P(X=true).
# 
# The example below where we implement the network shown in **Figure 14.3** of the book will make this more clear.
# 
# <img src="files/images/bayesnet.png">
# 
# The alarm node can be made as follows: 
# 

alarm_node = BayesNode('Alarm', ['Burglary', 'Earthquake'], 
                       {(True, True): 0.95,(True, False): 0.94, (False, True): 0.29, (False, False): 0.001})


# It is possible to avoid using a tuple when there is only a single parent. So an alternative format for the **cpt** is
# 

john_node = BayesNode('JohnCalls', ['Alarm'], {True: 0.90, False: 0.05})
mary_node = BayesNode('MaryCalls', 'Alarm', {(True, ): 0.70, (False, ): 0.01}) # Using string for parents.
# Equvivalant to john_node definition. 


# The general format used for the alarm node always holds. For nodes with no parents we can also use. 
# 

burglary_node = BayesNode('Burglary', '', 0.001)
earthquake_node = BayesNode('Earthquake', '', 0.002)


# It is possible to use the node for lookup function using the **p** method. The method takes in two arguments **value** and **event**. Event must be a dict of the type {variable:values, ..} The value corresponds to the value of the variable we are interested in (False or True).The method returns the conditional probability **P(X=value | parents=parent_values)**, where parent_values are the values of parents in event. (event must assign each parent a value.)
# 

john_node.p(False, {'Alarm': True, 'Burglary': True}) # P(JohnCalls=False | Alarm=True)


# With all the information about nodes present it is possible to construct a Bayes Network using **BayesNet**. The **BayesNet** class does not take in nodes as input but instead takes a list of **node_specs**. An entry in **node_specs** is a tuple of the parameters we use to construct a **BayesNode** namely **(X, parents, cpt)**. **node_specs** must be ordered with parents before children.
# 

get_ipython().magic('psource BayesNet')


# The constructor of **BayesNet** takes each item in **node_specs** and adds a **BayesNode** to its **nodes** object variable by calling the **add** method. **add** in turn adds  node to the net. Its parents must already be in the net, and its variable must not. Thus add allows us to grow a **BayesNet** given its parents are already present.
# 
# **burglary** global is an instance of **BayesNet** corresponding to the above example.
# 
#     T, F = True, False
# 
#     burglary = BayesNet([
#         ('Burglary', '', 0.001),
#         ('Earthquake', '', 0.002),
#         ('Alarm', 'Burglary Earthquake',
#          {(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001}),
#         ('JohnCalls', 'Alarm', {T: 0.90, F: 0.05}),
#         ('MaryCalls', 'Alarm', {T: 0.70, F: 0.01})
#     ])
# 

burglary


# **BayesNet** method **variable_node** allows to reach **BayesNode** instances inside a Bayes Net. It is possible to modify the **cpt** of the nodes directly using this method.
# 

type(burglary.variable_node('Alarm'))


burglary.variable_node('Alarm').cpt


# ## Exact Inference in Bayesian Networks
# 
# A Bayes Network is a more compact representation of the full joint distribution and like full joint distributions allows us to do inference i.e. answer questions about probability distributions of random variables given some evidence.
# 
# Exact algorithms don't scale well for larger networks. Approximate algorithms are explained in the next section.
# 
# ### Inference by Enumeration
# 
# We apply techniques similar to those used for **enumerate_joint_ask** and **enumerate_joint** to draw inference from Bayesian Networks. **enumeration_ask** and **enumerate_all** implement the algorithm described in **Figure 14.9** of the book.
# 

get_ipython().magic('psource enumerate_all')


# **enumerate__all** recursively evaluates a general form of the **Equation 14.4** in the book.
# 
# $$\textbf{P}(X | \textbf{e}) = α \textbf{P}(X, \textbf{e}) = α \sum_{y} \textbf{P}(X, \textbf{e}, \textbf{y})$$ 
# 
# such that **P(X, e, y)** is written in the form of product of conditional probabilities **P(variable | parents(variable))** from the Bayesian Network.
# 
# **enumeration_ask** calls **enumerate_all** on each value of query variable **X** and finally normalizes them. 
# 

get_ipython().magic('psource enumeration_ask')


# Let us solve the problem of finding out **P(Burglary=True | JohnCalls=True, MaryCalls=True)** using the **burglary** network.**enumeration_ask** takes three arguments **X** = variable name, **e** = Evidence (in form a dict like previously explained), **bn** = The Bayes Net to do inference on.
# 

ans_dist = enumeration_ask('Burglary', {'JohnCalls': True, 'MaryCalls': True}, burglary)
ans_dist[True]


# ### Variable Elimination
# 
# The enumeration algorithm can be improved substantially by eliminating repeated calculations. In enumeration we join the joint of all hidden variables. This is of exponential size for the number of hidden variables. Variable elimination employes interleaving join and marginalization.
# 
# Before we look into the implementation of Variable Elimination we must first familiarize ourselves with Factors. 
# 
# In general we call a multidimensional array of type P(Y1 ... Yn | X1 ... Xm) a factor where some of Xs and Ys maybe assigned values. Factors are implemented in the probability module as the class **Factor**. They take as input **variables** and **cpt**. 
# 
# 
# #### Helper Functions
# 
# There are certain helper functions that help creating the **cpt** for the Factor given the evidence. Let us explore them one by one.
# 

get_ipython().magic('psource make_factor')


# **make_factor** is used to create the **cpt** and **variables** that will be passed to the constructor of **Factor**. We use **make_factor** for each variable. It takes in the arguments **var** the particular variable, **e** the evidence we want to do inference on, **bn** the bayes network.
# 
# Here **variables** for each node refers to a list consisting of the variable itself and the parents minus any variables that are part of the evidence. This is created by finding the **node.parents** and filtering out those that are not part of the evidence.
# 
# The **cpt** created is the one similar to the original **cpt** of the node with only rows that agree with the evidence.
# 

get_ipython().magic('psource all_events')


# The **all_events** function is a recursive generator function which yields a key for the orignal **cpt** which is part of the node. This works by extending evidence related to the node, thus all the output from **all_events** only includes events that support the evidence. Given **all_events** is a generator function one such event is returned on every call. 
# 
# We can try this out using the example on **Page 524** of the book. We will make **f**<sub>5</sub>(A) = P(m | A)
# 

f5 = make_factor('MaryCalls', {'JohnCalls': True, 'MaryCalls': True}, burglary)


f5


f5.cpt


f5.variables


# Here **f5.cpt** False key gives probability for **P(MaryCalls=True | Alarm = False)**. Due to our representation where we only store probabilities for only in cases where the node variable is True this is the same as the **cpt** of the BayesNode. Let us try a somewhat different example from the book where evidence is that the Alarm = True
# 

new_factor = make_factor('MaryCalls', {'Alarm': True}, burglary)


new_factor.cpt


# Here the **cpt** is for **P(MaryCalls | Alarm = True)**. Therefore the probabilities for True and False sum up to one. Note the difference between both the cases. Again the only rows included are those consistent with the evidence.
# 
# #### Operations on Factors
# 
# We are interested in two kinds of operations on factors. **Pointwise Product** which is used to created joint distributions and **Summing Out** which is used for marginalization.
# 

get_ipython().magic('psource Factor.pointwise_product')


# **Factor.pointwise_product** implements a method of creating a joint via combining two factors. We take the union of **variables** of both the factors and then generate the **cpt** for the new factor using **all_events** function. Note that the given we have eliminated rows that are not consistent with the evidence. Pointwise product assigns new probabilities by multiplying rows similar to that in a database join.
# 

get_ipython().magic('psource pointwise_product')


# **pointwise_product** extends this operation to more than two operands where it is done sequentially in pairs of two.
# 

get_ipython().magic('psource Factor.sum_out')


# **Factor.sum_out** makes a factor eliminating a variable by summing over its values. Again **events_all** is used to generate combinations for the rest of the variables.
# 

get_ipython().magic('psource sum_out')


# **sum_out** uses both **Factor.sum_out** and **pointwise_product** to finally eliminate a particular variable from all factors by summing over its values.
# 

# #### Elimination Ask
# 
# The algorithm described in **Figure 14.11** of the book is implemented by the function **elimination_ask**. We use this for inference. The key idea is that we eliminate the hidden variables by interleaving joining and marginalization. It takes in 3 arguments **X** the query variable, **e** the evidence variable and **bn** the Bayes network. 
# 
# The algorithm creates factors out of Bayes Nodes in reverse order and eliminates hidden variables using **sum_out**. Finally it takes a point wise product of all factors and normalizes. Let us finally solve the problem of inferring 
# 
# **P(Burglary=True | JohnCalls=True, MaryCalls=True)** using variable elimination.
# 

get_ipython().magic('psource elimination_ask')


elimination_ask('Burglary', dict(JohnCalls=True, MaryCalls=True), burglary).show_approx()


# ## Approximate Inference in Bayesian Networks
# 
# Exact inference fails to scale for very large and complex Bayesian Networks. This section covers implementation of randomized sampling algorithms, also called Monte Carlo algorithms.
# 

get_ipython().magic('psource BayesNode.sample')


# Before we consider the different algorithms in this section let us look at the **BayesNode.sample** method. It samples from the distribution for this variable conditioned on event's values for parent_variables. That is, return True/False at random according to with the conditional probability given the parents. The **probability** function is a simple helper from **utils** module which returns True with the probability passed to it.
# 
# ### Prior Sampling
# 
# The idea of Prior Sampling is to sample from the Bayesian Network in a topological order. We start at the top of the network and sample as per **P(X<sub>i</sub> | parents(X<sub>i</sub>)** i.e. the probability distribution from which the value is sampled is conditioned on the values already assigned to the variable's parents. This can be thought of as a simulation.
# 

get_ipython().magic('psource prior_sample')


# The function **prior_sample** implements the algorithm described in **Figure 14.13** of the book. Nodes are sampled in the topological order. The old value of the event is passed as evidence for parent values. We will use the Bayesian Network in **Figure 14.12** to try out the **prior_sample**
# 
# <img src="files/images/sprinklernet.jpg" height="500" width="500">
# 
# We store the samples on the observations. Let us find **P(Rain=True)**
# 

N = 1000
all_observations = [prior_sample(sprinkler) for x in range(N)]


# Now we filter to get the observations where Rain = True
# 

rain_true = [observation for observation in all_observations if observation['Rain'] == True]


# Finally, we can find **P(Rain=True)**
# 

answer = len(rain_true) / N
print(answer)


# To evaluate a conditional distribution. We can use a two-step filtering process. We first separate out the variables that are consistent with the evidence. Then for each value of query variable, we can find probabilities. For example to find **P(Cloudy=True | Rain=True)**. We have already filtered out the values consistent with our evidence in **rain_true**. Now we apply a second filtering step on **rain_true** to find **P(Rain=True and Cloudy=True)**
# 

rain_and_cloudy = [observation for observation in rain_true if observation['Cloudy'] == True]
answer = len(rain_and_cloudy) / len(rain_true)
print(answer)


# ### Rejection Sampling
# 
# Rejection Sampling is based on an idea similar to what we did just now. First, it generates samples from the prior distribution specified by the network. Then, it rejects all those that do not match the evidence. The function **rejection_sampling** implements the algorithm described by **Figure 14.14**
# 

get_ipython().magic('psource rejection_sampling')


# The function keeps counts of each of the possible values of the Query variable and increases the count when we see an observation consistent with the evidence. It takes in input parameters **X** - The Query Variable, **e** - evidence, **bn** - Bayes net and **N** - number of prior samples to generate.
# 
# **consistent_with** is used to check consistency.
# 

get_ipython().magic('psource consistent_with')


# To answer **P(Cloudy=True | Rain=True)**
# 

p = rejection_sampling('Cloudy', dict(Rain=True), sprinkler, 1000)
p[True]


# ### Likelihood Weighting
# 
# Rejection sampling tends to reject a lot of samples if our evidence consists of a large number of variables. Likelihood Weighting solves this by fixing the evidence (i.e. not sampling it) and then using weights to make sure that our overall sampling is still consistent.
# 
# The pseudocode in **Figure 14.15** is implemented as **likelihood_weighting** and **weighted_sample**.
# 

get_ipython().magic('psource weighted_sample')


# 
# **weighted_sample** samples an event from Bayesian Network that's consistent with the evidence **e** and returns the event and its weight, the likelihood that the event accords to the evidence. It takes in two parameters **bn** the Bayesian Network and **e** the evidence.
# 
# The weight is obtained by multiplying **P(x<sub>i</sub> | parents(x<sub>i</sub>))** for each node in evidence. We set the values of **event = evidence** at the start of the function.
# 

weighted_sample(sprinkler, dict(Rain=True))


get_ipython().magic('psource likelihood_weighting')


# **likelihood_weighting** implements the algorithm to solve our inference problem. The code is similar to **rejection_sampling** but instead of adding one for each sample we add the weight obtained from **weighted_sampling**.
# 

likelihood_weighting('Cloudy', dict(Rain=True), sprinkler, 200).show_approx()


# ### Gibbs Sampling
# 
# In likelihood sampling, it is possible to obtain low weights in cases where the evidence variables reside at the bottom of the Bayesian Network. This can happen because influence only propagates downwards in likelihood sampling.
# 
# Gibbs Sampling solves this. The implementation of **Figure 14.16** is provided in the function **gibbs_ask** 
# 

get_ipython().magic('psource gibbs_ask')


# In **gibbs_ask** we initialize the non-evidence variables to random values. And then select non-evidence variables and sample it from **P(Variable | value in the current state of all remaining vars) ** repeatedly sample. In practice, we speed this up by using **markov_blanket_sample** instead. This works because terms not involving the variable get canceled in the calculation. The arguments for **gibbs_ask** are similar to **likelihood_weighting**
# 

gibbs_ask('Cloudy', dict(Rain=True), sprinkler, 200).show_approx()


