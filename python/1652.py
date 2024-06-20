# # Record, Save and Play Moves on Poppy Humanoid
# 

# This notebook intends to show you how to:
# 
# * **record** movements using physical demonstrations,
# * **replay** those movements and sequence them,
# * **save** them to the hard drive and reload them.
# 

# You will need a working and correctly setup Poppy Humanoid to follow this notebook. 
# 
# As we will use physical demonstration, **you can not use a simulated robot.** Yet the same methods could be used on simulation if you are using other kind of demonstration (e.g. programatically).
# 

# Import some matplolib shortcuts for Jupyter notebook
get_ipython().magic('matplotlib inline')
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


# ## Connect to your Poppy Humanoid
# 

# Using the usual Python code:
# 

from pypot.creatures import PoppyHumanoid

poppy = PoppyHumanoid()


# We then put it in its init position. We will use it as our working base:
# 

poppy.init_position.start()


# ## Record movements
# 

# First, we need to import the *MoveRecorder* object that can be used to record any movements on a Poppy robot. It works with all robots including the Poppy Humanoid.
# 

from pypot.primitive.move import MoveRecorder


# To create this recorder we need some extra information:
# * on which **robot** to record the movements
# * at which **frequency** the positions need to be retrieved (50Hz is a good default values)
# * which **motors** are we actually recording (you can record a movement on a subpart of your robot)
# 

recorder = MoveRecorder(poppy, 50, poppy.l_arm)


# As you can see we used the motor group *poppy.l_arm* as the list of motors to be recorded. It is a shortcut to the list of all motors in the left arm. It is equivalent to use the list of motors *[poppy.l_shoulder_y, poppy.l_shoulder_x, poppy.l_arm_z, poppy.l_elbow_y]*.
# 
# Note that you can get all existing motors groups (aliases) with thf following command:
# 

poppy.alias


# Before actually recording a movement, we need to set the used motors compliant so they can be freely moved:
# 

for m in poppy.l_arm:
    m.compliant = True


# Now the tip of the Poppy Humanoid should be free while the base is still stiff.
# 

# **Even if the motors are free, they can still be used as sensor**. This means that you can record their present position even if you make them move by hand.
# 

# The recorder can be start and stop when you want. In the interval the positions of every selected motors will be recorded at the predefined frequency.
# 

# So, prepare yourself to record your cool move! To start the record:
# 

recorder.start()


# Now move the robot! When you are done, you can stop the record:
# 

recorder.stop()


# The move you just recorded can be accessed via:
# 

recorder.move


# You can get the number of key frames that have been recorded:
# 

print('{} key frames have been recorded.'.format(len(recorder.move.positions())))


# Let's store this move in a variable (we copy it to not erase it when we will do another record):
# 

from copy import deepcopy

my_first_move = deepcopy(recorder.move)


# You can also plot a move to see what it looks like:
# 

ax = axes()
my_first_move.plot(ax)


# Now, let's record another move. This time we will record for 10 seconds and on the whole robot. You can easily do that using the following code:
# 

# First we recreate a recorder object with all motors. We also turn all motors compliant:
# 

recorder = MoveRecorder(poppy, 50, poppy.motors)

for m in poppy.motors:
    m.compliant = True


# And then record for 10 seconds:
# 

import time

recorder.start()
time.sleep(10)
recorder.stop()


# Now that the record is done, we also store it:
# 

my_second_move = deepcopy(recorder.move)


# ## Play recorded moves
# 

# First, we put back the robot in its rest position to avoid sudden movement:
# 

poppy.init_position.start()


# Replaying move is really similar. First you need to create the *MovePlayer* object:
# 

from pypot.primitive.move import MovePlayer


# It requires:
# * the **robot** which will play the move
# * the **move** to play
# 

# For instance, if we want to replay the first move we recorded:
# 

player = MovePlayer(poppy, my_first_move)


# And to play it:
# 

player.start()


# Once it's done, you can use the same code for the other move:
# 

player = MovePlayer(poppy, my_second_move)
player.start()


# You can sequence moves by using the *wait_to_stop* method that will wait for the end of a move: 
# 

for move in [my_first_move, my_second_move]:
    player = MovePlayer(poppy, move)
    
    player.start()
    player.wait_to_stop()


# Those movements can also be played in parallel. You will have to make sure that the movements can be combined otherwise pypot will simply add the different motor positions, possibly resulting in some unexpected moves. To avoid this problem make sure the moves you record are working on different sub sets of motors.
# 

# ## Save/Load moves
# 

# To keep the moves you have recorded from one session to the other, the best solution is to store them on the hard drive of your robot. This can be done using the *save* method of a move:
# 

with open('my-first-demo.move', 'w') as f:
    my_first_move.save(f)


# If you look at the file, you will see a list (possibly quite long) of "positions". These positions are basically:
# * a **timestamp** (time in seconds since the beginning of the move)
# * the list of motors name with:
#     * the **present position**
#     * the **present speed**
# 

# Here are the first 20 lines of the first move we recorded:
# 

get_ipython().system('head -n 20 my-first-demo.move')


# This representation can be really heavy and quite cumbersome to work with. We plan to use a better representation in a future, such as one based on parametrized curve. **Contributions on this topic are welcomed!**
# 

# A move can be loaded from the disk using the opposite *load* method. It requires to import the *Move* object:
# 

from pypot.primitive.move import Move

with open('my-first-demo.move') as f:
    my_loaded_move = Move.load(f)


# ## Using demonstration in artistic context
# 

# Now you have all the tools needed to create choregraphies with Poppy Humanoid. To get inspiration, you can have a look at the amazing work of Thomas Peyruse with Poppy Humanoid: https://forum.poppy-project.org/t/danse-contemporaine-school-of-moon/1567
# 

from IPython.display import YouTubeVideo

YouTubeVideo('https://youtu.be/Hy56H2AZ_XI?list=PLdX8RO6QsgB6YCzezJHoYuRToFOhYk3Sf')


# Or the *ENTRE NOUS* project of Emmanuelle Grangier: https://forum.poppy-project.org/t/projet-entre-nous-performance-choregraphique-et-sonore/1714
# 

from IPython.display import YouTubeVideo

YouTubeVideo('hEBdz97FhS8')





# # Discover your Poppy Humanoid
# 

# This notebook will guide you in your very first steps with Poppy Humanoid in Python. 
# 
# What you will see in this notebook:
# 
# 1. Instantiate your robot
# 2. Access motors, send motor commands
# 3. Start high level behaviors
# 

# *We assume here that you are connected to a physical Poppy Humanoid. It also need to be assembled and configured (you can referer to the [documentation](http://docs.poppy-project.org/en) if you haven't done in yet).*
# <img  src="image/poppy_humanoid.jpg"  width="500"/>
# 

# Import some matplolib shortcuts for Jupyter notebook
get_ipython().magic('matplotlib inline')
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


# ## Instantiate your robot
# 

# To start using your robot in Python, you first need to instantiate it. You can do that by running the following code:
# 

from pypot.creatures import PoppyHumanoid

poppy = PoppyHumanoid()


# **If you are using V-REP simulator**, you have to run this code instead. If you have troubles, look at the [documentation to install and run V-REP](http://docs.poppy-project.org/en/installation/install-vrep.html#test-your-installation).
# 

from pypot.creatures import PoppyHumanoid

poppy = PoppyHumanoid(simulator='vrep')


# This creates a [Robot](http://poppy-project.github.io/pypot/pypot.robot.html#pypot.robot.robot.Robot) object that can be used to access the motors and sensors. It handles all the low-level communication for you, so you do not need to know anything about the serial protocol used to make a motor turn. The *motors* and *sensors* fields of the Robot are automatically synced to match the state of their hardware equivalent.
# 

# Before doing anything else, we will initalize everything by asking the robot to go to its init position (the code below will be described in more detailed later):
# 

poppy.stand_position.start()


# ## Access motors
# 

# In a Poppy Humanoid, motors are defined as illustrated below:
# 
# <img  src="image/poppy_humanoid_motors.png" width="500"/>
# 

# From the [Robot](http://poppy-project.github.io/pypot/pypot.robot.html#pypot.robot.robot.Robot) object, you can directly retrieve the list of motors connected:
# 

poppy.motors


# As you can see *poppy.motors* holds a list of all motors.
# 

# You can retrieve all motors names:
# 

for m in poppy.motors:
    print(m.name)


# Each of them can be access directly from its name. For instance:
# 
# 
# 

poppy.l_elbow_y


# ### Read values from the motors
# 

# From the motor object you can access its registers. The main ones are:
# 
# * **present_position**: the current position of the motor in degrees
# * **present_speed**: the current speed of the motor in degrees per second 
# * **present_load**: the current workload of the motor (in percentage of max load)
# * **present_temperature**: the temperature of the motor in celsius
# * **angle_limit**: the reachable limits of the motor (in degrees)
# 
# They can be accessed directly:
# 

poppy.l_elbow_y.present_temperature


# Or, to get the present position for all motors:
# 

[m.present_position for m in poppy.motors]


# It's important to understand the *poppy.m1.present_position* is automatically updated with the real motor position (at 50Hz). Similarly for other registers, the update frequency may vary depending on its importance. For instance, the temperature is only refreshed at 1Hz as it is not fluctuating that quickly.
# 

# ### Send motor commands
# 

# On top of the registers presented above, they are additional ones used to send commands. For instance, the position of the motor is split in two different registers: 
# 
# * the read-only **present_position** of the motor
# * the read-write **goal_position** which sends to the motor a target position that it will try to reach.
# 
# If you want to set a new position for a motor, you write:
# 

poppy.l_arm_z.goal_position = 50


# You should see the robot turn of 20 degrees. Sending motor command is as simple as that. To make it turn to the other side:
# 

poppy.l_arm_z.goal_position = -50


# In the examples above, the motor turned as fast as possible (its default mode). You can change its *moving_speed* (i.e. its maximum possible speed):
# 

poppy.l_arm_z.moving_speed = 50


# Now the motor *l_arm_z* can not move faster than 50 degrees per second. If we ask to move again, you should see the difference:
# 

poppy.l_arm_z.goal_position = 90


# The main write registers are:
# 
# * **goal_position**: target position in degrees
# * **moving_speed**: maximum reachable speed in degrees per second
# * **compliant** (explained below) 
# 

# The dynamixel servo motors have two modes:
# 
# * **stiff**: the normal mode for motors where they can be controlled
# * **compliant**: a mode where the motors can be freely moved by hand. This is particularly useful for phyisical human-robot interaction
# 
# You can make them switch from one mode to the other using the *compliant* register. For instance, you can turn the motor *m6* compliant via:
# 

poppy.l_arm_z.compliant = True


# You should now be able to move this motors by hand. This is particularly useful for programming your robot by demonstration (see the dedicated notebook).
# 

# And to turn it stiff again:
# 

poppy.l_arm_z.compliant = False


# ## High level behaviors
# 

# The Poppy Humanoid robot comes with a set of pre-defined behaviors. They can be specific postures - such as the init_position used at the beginning - or a *breathing* motion, ... 
# 

# You can find the exhaustive list using the *primitives* accessor:
# 

[p.name for p in poppy.primitives]


# Those behaviors (or primitives in "poppy terms") can be started, stopped, paused, etc...
# 

poppy.upper_body_idle_motion.start()


# You can make the Poppy Humanoid *breathe* for 10 seconds:
# 

import time

poppy.upper_body_idle_motion.start()
time.sleep(10)
poppy.upper_body_idle_motion.stop()


# ## Going further
# 
# Now that you have learnt the basis of what you can do with a Poppy Humanoid, there is much more to discover:
# * how to record/replay move by demonstration
# * how to define your own high-level behavior (e.g. a visual servoing of the tip of the robot using blob detection)
# * used your Poppy Humanoid as a connected device and make it communaticate with the rest of the world using HTTP requests
# * ...
# 
# You can find other examples in the [docs](http://docs.poppy-project.org) or in the notebook folder next to this one.
# 




