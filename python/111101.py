get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

import sys
if "../" not in sys.path:
  sys.path.append("../")

import gym
# import plotting
from lib.envs.cliff_walking import CliffWalkingEnv
import itertools
matplotlib.style.use('ggplot')





env = CliffWalkingEnv()





env.render()


state = env.reset()
print env.render()
env.step(0)
print env.render()


# ## Q-Learning
# 

# Create a random epsilon-greedy policy

def random_epsilon_greedy_policy(Q, epsilon, state, nA):
    A = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[state])
    A[best_action] += (1.0 - epsilon)
    return A


def q_learning(env, num_episodes, discount=1.0, alpha=0.5, epsilon=0.1, debug=False):
    
    Q = defaultdict(lambda: np.zeros(env.action_space.n, dtype=float))
    
    episode_lengths = defaultdict(float)
    episode_rewards = defaultdict(float)
    

    for i_episode in range(1, num_episodes+1):
        
        if debug:
            if i_episode % 100 == 0:
                print("\rEpisode {}/{}.".format(i_episode, num_episodes))
                
        state = env.reset()
        for t in itertools.count():
            action_probs = random_epsilon_greedy_policy(Q, epsilon, state, env.action_space.n)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            
            next_state, reward, end, _ = env.step(action)
            
            Q[state][action] += alpha * (reward + discount*np.max(Q[next_state][:]) - Q[state][action])
            
            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] = t
            
            if end:
                break
            state = next_state
            
    return Q, episode_lengths, episode_rewards


Q, episode_lengths, episode_rewards = q_learning(env, 500, debug=True)


plt.figure(1, figsize=(12,10))
plt.plot(episode_lengths.keys(), episode_lengths.values())
plt.xlabel('Episode Length')
plt.ylabel('Epdisode')

plt.figure(2, figsize=(12,10))
plt.plot(episode_rewards.keys(), episode_rewards.values())
plt.xlabel('Episode Reward')
plt.ylabel('Epdisode')





get_ipython().magic('matplotlib inline')

import gym
import numpy as np
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import psutil
import tensorflow as tf

if "../" not in sys.path:
  sys.path.append("../")

from lib import plotting
from collections import deque, namedtuple


# ## Create and test Environment
# 

env = gym.make('Breakout-v0')


print("Action space size: {}".format(env.action_space.n))
print(env.unwrapped.get_action_meanings())

observation = env.reset()
print("Observation space shape: {}".format(observation.shape))


plt.figure()
plt.imshow(env.render(mode='rgb_array'))

[env.step(2) for x in range(1)]
plt.figure()
plt.imshow(env.render(mode='rgb_array'))

env.render(close=True)

env.step(1)
env.render(close=True)


# Cropped image
plt.imshow(observation[34:-16,:,:])


# ## Build Network
# 

# Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
VALID_ACTIONS = [0,1,2,3]


class StateProcessor():
    
    def __init__(self):
        
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(dtype=tf.uint8, shape=[210, 160, 3])
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)
            
    def process(self, sess, state):
        return sess.run(self.output, feed_dict={ self.input_state:state })


class Estimator():
  
    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
      
        # Placeholders for our input
        # Our input are 4 RGB frames of shape 160, 160 each
        self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]

        # Three convolutional layers
        conv1 = tf.contrib.layers.conv2d(
            X, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])

    def predict(self, sess, s):
        
        return sess.run(self.predictions, { self.X_pl: s })

    def update(self, sess, s, a, y):
        
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss


tf.reset_default_graph()
global_step = tf.Variable(0, name="global_step", trainable=False)

e = Estimator(scope="test")
sp = StateProcessor()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    observation = env.reset()
    
    observation_p = sp.process(sess, observation)
    observation = np.stack([observation_p] * 4, axis=2)
    observations = np.array([observation] * 2)
    
    y = np.array([10.0, 10.0])
    a = np.array([1, 3])
    print(e.update(sess, observations, a, y))


class ModelParametersCopier():
    
    def __init__(self, estimator1, estimator2):
        
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)
        
        self.update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            self.update_ops.append(op)
            
    def make(self, sess):
        return sess.run(self.update_ops)


def make_epsilon_greedy_policy(estimator, nA):
    
    
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon/nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1 - epsilon)
        return A
    return policy_fn


def generator_fn(stats):
    
    yield total_t, plotting.EpisodeStats(
            episode_lengths=stats.episode_lengths[:i_episode+1],
            episode_rewards=stats.episode_rewards[:i_episode+1])


def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    state_processor,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=500000,
                    replay_memory_init_size=50000,
                    update_target_estimator_every=10000,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    batch_size=32,
                    record_video_every=50):
    
    
    Transition = namedtuple("Transition", ['state', 'action', 'reward', 'next_state', 'done'])
    
    # The replay memory
    replay_memory = []
    
    # Make model copier object
    estimator_copy = ModelParametersCopier(q_estimator, target_estimator)

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    
    # For 'system/' summaries, usefull to check if currrent process looks healthy
    current_process = psutil.Process()

    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)
        
    saver = tf.train.Saver()
    
    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
    
    # Get the current time step
    total_t = sess.run(tf.contrib.framework.get_global_step())
    
    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    
    policy = make_epsilon_greedy_policy(q_estimator, len(VALID_ACTIONS))
    
    #Initializing the replay memory
    print("populating replay memory")
    state = env.reset()
    state = state_processor.process(sess, state)
    state = np.stack([state]*4, axis=2)
    for i in range(replay_memory_init_size):
        action_probs = policy(sess, state, epsilons[min(total_t, epsilon_decay_steps-1)])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
        next_state = state_processor.process(sess, next_state)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
        replay_memory.append(Transition(state, action, reward, next_state, done))
        if done:
            state = env.reset()
            state = state_processor.process(sess, state)
            state = np.stack([state]*4, axis=2)
        else:
            state = next_state
            
            
    env = Monitor(env, directory=monitor_path, video_callable=lambda count: count % record_video_every == 0, resume=True)
    
    for i_episode in range(num_episodes):
        
        saver.save(tf.get_default_session(), checkpoint_path)
        
        state = env.reset()
        state = state_processor.process(state)
        state = np.stack([state]*4, axis=2)
        loss = None
        
        for t in itertools.count():
            
            # Maybe update the target estimator
            if total_t % update_target_estimator_every == 0:
                estimator_copy.make(sess)
                print("\nCopied model parameters to target network.")

            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                    t, total_t, i_episode + 1, num_episodes, loss))
            sys.stdout.flush()
            
            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
            next_state = state_processor.process(sess, next_state)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
            
            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # Save transition to replay memory
            replay_memory.append(Transition(state, action, reward, next_state, done))   

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t 
            
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
            
            q_values_next = target_estimator.predict(sess, next_states_batch)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * np.amax(q_values_next, axis=1)
            
            loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)
            
            if done:
                break

            state = next_state
            total_t += 1
            
        # Add summaries to tensorboard
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=epsilon, tag="episode/epsilon")
        episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], tag="episode/reward")
        episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], tag="episode/length")
        episode_summary.value.add(simple_value=current_process.cpu_percent(), tag="system/cpu_usage_percent")
        episode_summary.value.add(simple_value=current_process.memory_percent(memtype="vms"), tag="system/v_memeory_usage_percent")
        q_estimator.summary_writer.add_summary(episode_summary, i_episode)
        q_estimator.summary_writer.flush()
        
        
        generator_fn(stats)
        
    return stats


tf.reset_default_graph()

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)
    
# Create estimators
q_estimator = Estimator(scope="q_estimator", summaries_dir=experiment_dir)
target_estimator = Estimator(scope="target_q")

# State processor
state_processor = StateProcessor()

# Run it!
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t, stats in deep_q_learning(sess,
                                    env,
                                    q_estimator=q_estimator,
                                    target_estimator=target_estimator,
                                    state_processor=state_processor,
                                    experiment_dir=experiment_dir,
                                    num_episodes=10000,
                                    replay_memory_size=500000,
                                    replay_memory_init_size=50000,
                                    update_target_estimator_every=10000,
                                    epsilon_start=1.0,
                                    epsilon_end=0.1,
                                    epsilon_decay_steps=500000,
                                    discount_factor=0.99,
                                    batch_size=32):

        print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))





get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

import gym
import plotting
from lib.envs import blackjack
matplotlib.style.use('ggplot')


# ## Create Environment
# 

env=blackjack.BlackjackEnv()


# Actions:
# Hit: 1
# Sticks: 0
# 

env.action_space.sample()


env.observation_space.spaces


observation=env.reset()
observation


action = 0
observation, reward, done, info = env.step(action)
observation, reward, done, info


def observation_clean(observation):
    return (observation[0], observation[1], observation[2])

def print_observation(observation):
    player_score, dealer_score, usable_ace = observation
    print("Player Score: {} (Usable Ace: {}), Dealer Score: {}".format(
          player_score, usable_ace, dealer_score))
    
def policy(observation):
    player_score, dealer_score, usable_ace = observation
    
    # Stick (action 0) if the score is > 20, hit (action 1) otherwise
    return 0 if player_score >= 20 else 1


def play_blackjack():
    for episode in range(20):
        observation = env.reset()
        for i in range(100):
            print_observation(observation)
            
            action = policy(observation)
            print("Taking action: {}".format( ["Stick", "Hit"][action]))
            
            observation, reward, done, _ = env.step(action)
            if done:
                print('FINAL SCORE:')
                print_observation(observation)
                print("Game end. Reward: {}\n".format(float(reward)))
                break
                
play_blackjack()


# ## Monte Carlo State Value Prediction
# 

def find_state_value_function(policy, env, num_episodes, discount=1.0, debug=False):
    
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    V = defaultdict(float)
    
    for episode in range(1, num_episodes+1):
        
        if debug:
            if episode % 1000 == 0:
                print("\rEpisode {}/{}.".format(episode, num_episodes))

            
        episodes = []
        state = env.reset()
        for t in range(100):
            action = policy(state)
            next_state, reward, end, _ = env.step(action)
            episodes.append((state, action, reward))
            if end:
                break
            state = next_state
            
        states_in_episode = [tup[0] for tup in episodes]
        for state in states_in_episode:
            first_occurence_idx = next(i for i,x in enumerate(episodes) if x[0] == state)
            
            G = sum([x[2]*(discount**i) for i,x in enumerate(episodes[first_occurence_idx:])])
            
            # Calculate average return for this state over all sampled episodes
            returns_sum[state] += G
            returns_count[state] += 1.0
            V[state] = returns_sum[state] / returns_count[state]

    return V

            
            
final_V_10k = find_state_value_function(policy, env, 10000)
final_V_50k = find_state_value_function(policy, env, 500000)


plotting.plot_value_function(final_V_10k, title="10,000 Steps")


plotting.plot_value_function(final_V_50k, title="5,00,000 Steps")





get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

import sys
if "../" not in sys.path:
  sys.path.append("../")

import gym
import plotting
from lib.envs.blackjack import BlackjackEnv
matplotlib.style.use('ggplot')


env = BlackjackEnv()


# ## TD(0)
# 

def random_policy(observation):
    player_score, dealer_score, usable_ace = observation
    
    # Stick (action 0) if the score is > 20, hit (action 1) otherwise
    return 0 if player_score >= 20 else 1


def td_0_prediction(policy, env, num_episodes, alpha=1.0, discount=1.0, debug=False):
    
    V = defaultdict(float)
    
    for i_episode in range(1, num_episodes+1):
        
        if debug:
            if i_episode % 100000 == 0:
                print("\rEpisode {}/{}.".format(i_episode, num_episodes))
                
        state=env.reset()
        while(True):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            V[state] += alpha * (reward + discount*V[next_state] - V[state])
            if done:
                break
            state = next_state
            
                
    return V


V = td_0_prediction(random_policy, env, num_episodes=500000, debug=True)
plotting.plot_value_function(V, title="Optimal Value Function")





get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

import sys
if "../" not in sys.path:
  sys.path.append("../")

import gym
import plotting
from lib.envs.blackjack import BlackjackEnv
matplotlib.style.use('ggplot')


env = BlackjackEnv()


# ## On-Policy MC Control
# 

#Create an initial epsilon soft policy
def epsilon_greedy_policy(Q, epsilon, state, nA):
    A = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[state])
    A[best_action] += (1.0 - epsilon)
    return A


def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1, debug=False):
  
    
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    for i_episode in range(1, num_episodes + 1):
        
        if debug:
            if i_episode % 100000 == 0:
                print("\rEpisode {}/{}.".format(i_episode, num_episodes))
                
        state = env.reset()
        start = True
        state_action_pairs_in_episode = []
        states_in_episode = []
        while(True):
            probs=epsilon_greedy_policy(Q, epsilon, state, env.action_space.n)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, end, _ = env.step(action)
            state_action_pairs_in_episode.append(((state, action),reward))
            states_in_episode.append(state)
            if end:
                break
            state = next_state
        
        for ((st,act), reward) in state_action_pairs_in_episode:
            first_occurence_idx = next(i for i,(s_a,r) in enumerate(state_action_pairs_in_episode) if s_a==(st,act))
            G = sum([r for ((s,a),r) in state_action_pairs_in_episode[first_occurence_idx:]])
            
            #Calculate average of the returns
            returns_sum[(st,act)] += G
            returns_count[(st,act)] += 1.0
            Q[st][act] = returns_sum[(st,act)] / returns_count[(st,act)]
        
    
    return Q, epsilon_greedy_policy

Q, optimal_policy = mc_control_epsilon_greedy(env, num_episodes=100000, debug=True)


# Create value function from action-value function by picking the best action at each state
V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value

plotting.plot_value_function(V, title="Optimal Value Function")





get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

import sys
if "../" not in sys.path:
  sys.path.append("../")

import gym
import plotting
from lib.envs.blackjack import BlackjackEnv
matplotlib.style.use('ggplot')


env = BlackjackEnv()


# ## Off Policy Monte Carlo Incremental Implementation with Weighted Importance Sampling
# 

def create_behaviour_policy(nA):
    A = np.ones(nA, dtype=float) / nA
    def policy_fn(observation):
        
        return A
    return policy_fn


def create_target_policy(Q):
    
    def policy_fn(state):
        A = np.zeros_like(Q[state], dtype=float)
        best_action = np.argmax(Q[state])
        A[best_action] = 1.0
        return A
    return policy_fn


def mc_prediction_off_policy_importance_sampling(behaviour_policy, env, num_episodes, discount=1.0, debug=False):
    
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    
    target_policy = create_target_policy(Q)
    
    for i_episode in range(1, num_episodes+1):
        
        if debug:
            if i_episode % 100000 == 0:
                print("\rEpisode {}/{}.".format(i_episode, num_episodes))
                
        state = env.reset()
        episode = []
        while(True):
            probs = behaviour_policy(state)
            action = np.random.choice(len(probs), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
            
        G = 0.0
        W = 1.0
        
        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]
            G = discount*G + reward
            C[state][action] += W
            Q[state][action] += (W/C[state][action]) * (G - Q[state][action])
            W = W * (target_policy(state)[action]/behaviour_policy(state)[action])
            if W == 0:
                break
    return Q, target_policy


behaviour_policy = create_behaviour_policy(env.action_space.n)
Q, policy = mc_prediction_off_policy_importance_sampling(behaviour_policy, env, num_episodes=500000, debug=True)


V = defaultdict(float)
for state, action_values in Q.items():
    action_value = np.max(action_values)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")


get_ipython().magic('matplotlib inline')

import gym
import itertools
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing

if "../" not in sys.path:
  sys.path.append("../") 

from lib import plotting
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

matplotlib.style.use('ggplot')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ## Create Environment
# 

env = gym.envs.make("MountainCar-v0")


env.observation_space.sample()


observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()


featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.train(scaler.fit_transform(observation_examples))


class Function_Approximator():
    
    def __init__(self):
        
        self.models = []
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)
            
    
    def featurize_state(self, state):
        
        scaled = scaler.transform([state])
        features = featurizer.transform(scaled)
        return features[0]
    
    
    def predict(self, s, a=None):
        
        state_features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([state_features])[0] for m in self.models])
        else:
            return self.models[a].predict([state_features])[0]
        
    def update(self, s, a, y):
       
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])


def make_epsilon_greedy_policy(estimator, epsilon, nA):
    
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def sarsa(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
    
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    
    for i_episode in range(num_episodes):
        
        policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        for t in itertools.count():
            
            next_state, reward, end, _ = env.step(action)
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            q_values_next = estimator.predict(next_state)
            td_target = reward + discount_factor * q_values_next[next_action]
            
            estimator.update(state, action, td_target)
            
            if i_episode % 10 == 0:
                print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, reward))
                
            if end:
                break
                
            state = next_state
            action = next_action
    
    return stats


estimator = Function_Approximator()
stats = sarsa(env, estimator, 200, epsilon=0.0)


plotting.plot_cost_to_go_mountain_car(env, estimator)
plotting.plot_episode_stats(stats, smoothing_window=25)


state = env.reset()
plt.figure()
plt.imshow(env.render(mode='rgb_array'))
while(True):
    q_values = estimator.predict(state)
    best_action = np.argmax(q_values)
    plt.figure()
    plt.imshow(env.render(mode='rgb_array'))
    
    next_state, reward, end, _ = env.step(best_action)
    if end:
        break
        
    state = next_state
    env.render(close=True)
env.render(close=True)





get_ipython().magic('matplotlib inline')

import gym
import itertools
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing

if "../" not in sys.path:
  sys.path.append("../") 

from lib import plotting
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

matplotlib.style.use('ggplot')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ## Create Environment
# 

env = gym.envs.make("MountainCar-v0")


env.observation_space.sample()


observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()


featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.train(scaler.fit_transform(observation_examples))


class Function_Approximator():
    
    def __init__(self):
        
        self.models = []
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)
            
    
    def featurize_state(self, state):
        
        scaled = scaler.transform([state])
        features = featurizer.transform(scaled)
        return features[0]
    
    
    def predict(self, s, a=None):
        
        state_features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([state_features])[0] for m in self.models])
        else:
            return self.models[a].predict([state_features])[0]
        
    def update(self, s, a, y):
       
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])


def make_epsilon_greedy_policy(estimator, epsilon, nA):
    
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
    
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    
    for i_episode in range(num_episodes):
        
        policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
        state = env.reset()
        
        for t in itertools.count():
            
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            
            next_state, reward, end, _ = env.step(action)
            
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            q_values_next = estimator.predict(next_state)
            td_target = reward + discount_factor * np.max(q_values_next)
            
            estimator.update(state, action, td_target)
            
            if i_episode % 10 == 0:
                print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, reward))
                
            if end:
                break
                
            state = next_state
    
    return stats


estimator = Function_Approximator()
stats = q_learning(env, estimator, 100, epsilon=0.0)


plotting.plot_cost_to_go_mountain_car(env, estimator)
plotting.plot_episode_stats(stats, smoothing_window=25)


# ## Run the Optimal Policy
# 

state = env.reset()
plt.figure()
plt.imshow(env.render(mode='rgb_array'))
while(True):
    q_values = estimator.predict(state)
    best_action = np.argmax(q_values)
    plt.figure()
    plt.imshow(env.render(mode='rgb_array'))
    
    next_state, reward, end, _ = env.step(best_action)
    if end:
        break
        
    state = next_state
    env.render(close=True)
env.render(close=True)





import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import matplotlib
import tensorflow as tf
import pydot
from IPython.display import Image
from IPython.display import SVG
import timeit
from sklearn.cross_validation import train_test_split

# Display plots inline and change default figure size
get_ipython().magic('matplotlib inline')
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)


np.random.seed(0)
train_X, train_y = sklearn.datasets.make_moons(10000, noise=0.20)

x_train, x_val, y_train, y_val = train_test_split(train_X, train_y, train_size=0.9)
plt.scatter(train_X[:,0], train_X[:,1], s=40, c=train_y, cmap=plt.cm.Spectral)


y_train_new = []
y_val_new = []
for num, i in enumerate(y_train):
    if i==0:
        y_train_new.append([1,0])
    else:
        y_train_new.append([0,1])
        
for num, i in enumerate(y_val):
    if i==0:
        y_val_new.append([1,0])
    else:
        y_val_new.append([0,1])


# ## Initialize neural network architecture
# 

num_examples = len(X)
nn_input_dim = train_X.shape[1]
nn_output_dim = 2
nn_hdim = 100

epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01

# Define variables
def initialize_variables():
    x = tf.placeholder(tf.float32, shape=[None, nn_input_dim])
    y = tf.placeholder(tf.float32, shape=[None, nn_output_dim])

    return x, y

#Define weights and biases
def initialize_weights_biases():
    np.random.seed(0)
    W1 = tf.Variable(tf.random_normal(shape=[nn_input_dim, nn_hdim]))
    b1 = tf.Variable(tf.zeros(shape=[1, nn_hdim]))
    W2 = tf.Variable(tf.random_normal(shape=[nn_hdim, nn_output_dim]))
    b2 = tf.Variable(tf.zeros(shape=[1, nn_output_dim]))
    
    return W1, b1, W2, b2


def neural_network_model(train_X, train_y, num_rounds=10000):
    
    X, y = initialize_variables()
    W1, b1, W2, b2 = initialize_weights_biases()
    
    #Forward Propogation
    z1 = tf.matmul(X, W1) + b1
    a1 = tf.nn.sigmoid(z1)
    yhat = tf.matmul(a1, W2) + b2
    predict = tf.argmax(yhat, axis=1)
    
    #Back-Propogation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    
    #Intialize Session
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    for epoch in range(num_rounds):
        sess.run(updates, feed_dict={X:x_train, y:y_train_new})

        train_accuracy = np.mean(np.argmax(y_train_new, axis=1) ==
                             sess.run(predict, feed_dict={X: x_train, y: y_train_new}))
        
        
        test_accuracy = np.mean(np.argmax(y_val_new, axis=1) ==
                             sess.run(predict, feed_dict={X: x_val, y: y_val_new}))
        
        
        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
                  % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
    
    sess.close()
    


neural_network_model(train_X, train_Y)





get_ipython().magic('matplotlib inline')
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mode,skew,skewtest

from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


data=pd.read_csv('/Users/adityavyas/Desktop/Machine Learning and Big Data/Datasets/Breast Cancer/data.csv')


data.drop('Unnamed: 32',1,inplace=True)


data.head(4)


data.columns


data.dtypes


data.isnull().sum()


#Lets break the data into training and test sets

train,test=train_test_split(data,train_size=0.95,test_size=0.05)
train.size,test.size


sns.heatmap(train.corr(),xticklabels=False,yticklabels=False)


# We observe that a lot of features are related to each other. 

train.head(1)


train_labels=train['diagnosis']
test_labels=test['diagnosis']

#We drop the id and diagnosis columns in train and test

train.drop(['id','diagnosis'],1,inplace=True)
test.drop(['id','diagnosis'],1,inplace=True)


#We run a basic random forest on the data to know the feature importances.

forest=ensemble.RandomForestClassifier(n_estimators=100,n_jobs=-1)
forest.fit(train,train_labels)

importances=forest.feature_importances_
indices=np.argsort(importances)[::-1]
feature=train.columns
plt.yticks(range(len(indices)),feature[indices],fontsize=10)
plt.barh(range(len(indices)),importances[indices])
plt.tight_layout()


# Ok so we see that the concave_points_worst has the highest importance. Infact area,radius worst also have high importances. Similarly features pertaining to smoothness,symmetry and fractal dimensions are not so important.
# 

# We will rearrange the columns based on the feature importances

train2=train.ix[:,feature[indices]]
test2=test.ix[:,feature[indices]]


train2.head(2)


#Lets look at the overall distribution among features

train2.describe()


#We will need to normalize values because there is huge variation among the values. The area_mean has value 661 while
#the fractional_dimension_worst has mean value 0.083

scaler=StandardScaler(with_mean=True,with_std=True)
scaled_features=scaler.fit_transform(train2)
scaled_train_df=pd.DataFrame(scaled_features,index=train2.index,columns=train2.columns)

scaled_features2=scaler.fit_transform(test2)
scaled_test_df=pd.DataFrame(scaled_features2,index=test2.index,columns=test2.columns)


#We will join the labels

train3=scaled_train_df.join(train_labels)
test3=scaled_test_df.join(test_labels)


# We create training,validation and testing datasets

TRAIN,VAL=train_test_split(train3,train_size=0.8)
TEST=test3
x_TRAIN,y_TRAIN=TRAIN.drop('diagnosis',1),TRAIN['diagnosis']
x_VAL,y_VAL=VAL.drop('diagnosis',1),VAL['diagnosis']
x_TEST,y_TEST=TEST.drop('diagnosis',1),TEST['diagnosis']


#Logistic Regression

from sklearn.linear_model import LogisticRegression


logreg=LogisticRegression()
logreg.fit(x_TRAIN,y_TRAIN)
y_pred_logreg=logreg.predict(x_VAL)
val_accuracy_logreg=accuracy_score(y_pred_logreg,y_VAL)
val_accuracy_logreg

y_pred_logreg_=logreg.predict(x_TEST)
test_accuracy_logreg=accuracy_score(y_pred_logreg_,y_TEST)
test_accuracy_logreg


#Random Forest Classification

from sklearn.ensemble import RandomForestClassifier

forest=RandomForestClassifier(n_estimators=100)
forest.fit(x_TRAIN,y_TRAIN)
y_pred_forest=forest.predict(x_VAL)
val_accuracy_forest=accuracy_score(y_pred_forest,y_VAL)

y_pred_forest_=forest.predict(x_TEST)
test_accuracy_forest=accuracy_score(y_pred_forest_,y_TEST)
'validation accuracy= '+str(val_accuracy_forest)+'   '+'final accuracy= '+str(test_accuracy_forest)





# ## What is Doc2Vec?
# 
# Doc2Vec is the straightforward extension of the word2vec model taking into account the overall vectorized form of the paragraph or the document. Word2vec creates vectors of only the words but this has a disadvantage where the model looses the overall meaning of the words. This is where Doc2Vec is important.
# It is based on the 2 word2vec architectures - skipgram(SG) and continuous bag of words(CBOW). The only difference is that apart from the word vectors, we also feed a paragraph-id as input to the neural network. Given below are the architectures defined in the original paper:
# 
# **1. The first architecture is the Distributed Memory Model of Paragraph Vectors (PV-DM). This is similar to CBOW architecture of Word2Vec where, given a context we need to predict the word which would follow the sequence. However, we take a paragraph-token as a word and feed it into the neural net.**
# 
# <img src="1.png" alt="Drawing" style="width: 600px;"/>
# 
# **2. In the second architecture the only the paragraph token is fed as input to the neural network which then learns/predicts the words in a fixed context/window. This architecture is similar to the Skip-Gram model in Word2Vec and is known as Distributed Bag of Words version of Paragraph Vector (PV-DBOW)**
# 
# <img src="2.png" alt="Drawing" style="width: 600px;"/>
# 

# ## Import the libraries
# 

import numpy as np
import keras

#Import necessary nlp tools from gensim
from gensim import utils
from gensim.models.doc2vec import LabeledSentence, TaggedDocument
from gensim.models import Doc2Vec

from random import shuffle


# ## Load the text data
# 
# We will use Cornell's IMDB Movie Review dataset for our sentiment analysis. We have four text files:
# - `test-neg.txt`: 12500 negative movie reviews from the test data
# - `test-pos.txt`: 12500 positive movie reviews from the test data
# - `train-neg.txt`: 12500 negative movie reviews from the training data
# - `train-pos.txt`: 12500 positive movie reviews from the training data
# 
# I have already downloaded the preprocessed datasets but for someone collecting the dataset, one needs to carry out the required preprocessing such as removing punctuations and converting everything to lowercase.
# 
# Each of the reviews should be formatted as such:
# 
# ```
# once again mr costner has dragged out a movie for far longer than necessary aside from the terrific sea rescue sequences of which there are very few i just did not care about any of the characters most of us have ghosts in the closet and costner s character are realized early on and then forgotten until much later by which time i did not care the character we should really care about is a very cocky overconfident ashton kutcher the problem is he comes off as kid who thinks he s better than anyone else around him and shows no signs of a cluttered closet his only obstacle appears to be winning over costner finally when we are well past the half way point of this stinker costner tells us all about kutcher s ghosts we are told why kutcher is driven to be the best with no prior inkling or foreshadowing no magic here it was all i could do to keep from turning it off an hour in
# this is an example of why the majority of action films are the same generic and boring there s really nothing worth watching here a complete waste of the then barely tapped talents of ice t and ice cube who ve each proven many times over that they are capable of acting and acting well don t bother with this one go see new jack city ricochet or watch new york undercover for ice t or boyz n the hood higher learning or friday for ice cube and see the real deal ice t s horribly cliched dialogue alone makes this film grate at the teeth and i m still wondering what the heck bill paxton was doing in this film and why the heck does he always play the exact same character from aliens onward every film i ve seen with bill paxton has him playing the exact same irritating character and at least in aliens his character died which made it somewhat gratifying overall this is second rate action trash there are countless better films to see and if you really want to see this one watch judgement night which is practically a carbon copy but has better acting and a better script the only thing that made this at all worth watching was a decent hand on the camera the cinematography was almost refreshing which comes close to making up for the horrible film itself but not quite
# ```
# 

# Define source files for input data
source_dict = {'test-neg.txt':'TEST_NEG',
                'test-pos.txt':'TEST_POS',
                'train-neg.txt':'TRAIN_NEG',
                'train-pos.txt':'TRAIN_POS'
               }



# Define a LabeledDocSentence class to process multiple documents. This is an extension of the gensim's 
# LabeledLine class. Gensim's LabeledLine class does not process multiple documents, hence we need to define our own
# implementation.
class LabeledDocSentence():
    
    # Initialize the source dict
    def __init__(self, source_dict):
        self.sources = source_dict
    
    # This creates sentences as a list of words and assigns each sentence a tag 
    # e.g. [['word1', 'word2', 'word3', 'lastword'], ['label1']]
    def create_sentences(self):
        self.sentences = []
        for source_file, prefix in self.sources.items():
            with utils.smart_open(source_file) as f:
                for line_id, line in enumerate(f):
                    sentence_label = prefix + '_' + str(line_id)
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [sentence_label]))
        
        return self.sentences
             
    # Return a permutation of the sentences in each epoch. I read that this leads to the best results and 
    # helps the model to train properly.
    def get_permuted_sentences(self):
        sentences = self.create_sentences()
        shuffled = list(sentences)
        shuffle(shuffled)
        return shuffled


# ## Model Training
# 
# Now we use Gensim's Doc2Vec function to train our model on the sentences. There are various hyperparameters used in the function. Some of them are:
# - `min_count`: ignore all words with total frequency lower than this. You have to set this to 1, since the sentence labels only appear once. Setting it any higher than 1 will miss out on the sentences.
# - `window`: the maximum distance between the current and predicted word within a sentence. Word2Vec uses a skip-gram model, and this is simply the window size of the skip-gram model.
# - `size`: dimensionality of the feature vectors in output. 100 is a good number. If you're extreme, you can go up to around 400.
# - `sample`: threshold for configuring which higher-frequency words are randomly downsampled
# - `workers`: use this many worker threads to train the model 
# 
# I train the model for 10 epochs. It takes around 10 mins. We can use higher epochs for better results.
# 

labeled_doc = LabeledDocSentence(source_dict) 
model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)

# Let the model learn the vocabulary - all the words in the paragraph
model.build_vocab(labeled_doc.get_permuted_sentences())


# Train the model on the entire set of sentences/reviews for 10 epochs. At each epoch sample a different
# permutation of the sentences to make the model learn better.
for epoch in range(10):
    print epoch
    model.train(labeled_doc.get_permuted_sentences(), total_examples=model.corpus_count, epochs=10)


# To avoid retraining, we save the model
model.save('imdb.d2v')


# Load the saved model
model_saved = Doc2Vec.load('imdb.d2v')


# Check what the model learned. It will show 10 most similar words to the input word. Since we kept the window size
# to be 10, it will show the 10 most recent.
model_saved.most_similar('good')


# Our model is a Doc2Vec model, hence it also learnt the sentence vectors apart from the word embeddings. Hence we
# can see the vector of any sentence by passing the tag for the sentence.
model_saved.docvecs['TRAIN_NEG_0']


# Create a labelled training and testing set

x_train = np.zeros((25000, 100))
y_train = np.zeros(25000)
x_test = np.zeros((25000, 100))
y_test = np.zeros(25000)

for i in range(12500):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    x_train[i] = model_saved.docvecs[prefix_train_pos]
    x_train[12500 + i] = model_saved.docvecs[prefix_train_neg]
    
    y_train[i] = 1
    y_train[12500 + i] = 0
    
    
for i in range(12500):
    prefix_test_pos = 'TRAIN_POS_' + str(i)
    prefix_test_neg = 'TRAIN_NEG_' + str(i)
    x_test[i] = model_saved.docvecs[prefix_test_pos]
    x_test[12500 + i] = model_saved.docvecs[prefix_test_neg]
    
    y_test[i] = 1
    y_test[12500 + i] = 0


print x_train


# Convert the output to a categorical variable to be used for the 2 neuron output layer in the neural network.

from keras.utils import to_categorical

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


# Create a neural network with a single hidden layer and a softmax output layer with 2 neurons.

from keras.models import Sequential
from keras.layers import Dense

nnet = Sequential()
nnet.add(Dense(32, input_dim=100, activation='relu'))
nnet.add(Dense(2, input_dim=32, activation='softmax'))
nnet.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Visualize the neural net's layer
nnet.summary()


# Train the net on the training data
nnet.fit(x_train, y_train_cat, epochs=5, batch_size=32)


# Predict on the test set
score = nnet.evaluate(x_test, y_test_cat, batch_size=32)
score[1]


get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

import sys
if "../" not in sys.path:
  sys.path.append("../")

import gym
import plotting
from lib.envs.windy_gridworld import WindyGridworldEnv
import itertools
matplotlib.style.use('ggplot')


# ## Create Environment
# 

env = WindyGridworldEnv()


env.observation_space.n


print env.reset()
env.render()

print env.step(3)
env.render()


# ## SARSA On Policy TD Control
# 

#Create an initial epsilon soft policy
def epsilon_greedy_policy(Q, epsilon, state, nA):
    A = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[state])
    A[best_action] += (1.0 - epsilon)
    return A



def sarsa(env, num_episodes, epsilon=0.1, alpha=0.5, discount=1.0, debug=False):
    
    Q = defaultdict(lambda: np.zeros(env.action_space.n, dtype=float))
    
    episode_lengths = defaultdict(float)
    episode_rewards = defaultdict(float)
    

    for i_episode in range(1, num_episodes+1):
        
        if debug:
            if i_episode % 100 == 0:
                print("\rEpisode {}/{}.".format(i_episode, num_episodes))
                
        state = env.reset()
        action_probs = epsilon_greedy_policy(Q, epsilon, state, env.action_space.n)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        for t in itertools.count():
            next_state, reward, end, _ = env.step(action)
            
            next_action_probs = epsilon_greedy_policy(Q, epsilon, next_state, env.action_space.n)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            
            Q[state][action] += alpha * (reward + discount*Q[next_state][next_action] - Q[state][action])
            
            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] = t
            
            if end:
                break
            state = next_state
            action = next_action
            
    return Q, episode_lengths, episode_rewards


Q, episode_lengths, episode_rewards = sarsa(env, 200, debug=True)


plt.figure(1, figsize=(12,10))
plt.plot(episode_lengths.keys(), episode_lengths.values())
plt.xlabel('Episode Length')
plt.ylabel('Epdisode')

plt.figure(2, figsize=(12,10))
plt.plot(episode_rewards.keys(), episode_rewards.values())
plt.xlabel('Episode Reward')
plt.ylabel('Epdisode')





get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

import sys
if "../" not in sys.path:
  sys.path.append("../")

import gym
import plotting
from lib.envs.blackjack import BlackjackEnv
matplotlib.style.use('ggplot')


env = BlackjackEnv()


# ## Off Policy Monte Carlo Control with Weighted Importance Sampling
# 

def create_behaviour_policy(nA):
    
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) / nA
        return A
    return policy_fn


def create_target_policy(Q):
    
    def policy_fn(state):
        A = np.zeros_like(Q[state], dtype=float)
        best_action = np.argmax(Q[state])
        A[best_action] = 1.0
        return A
    return policy_fn


def mc_control_off_policy_importance_sampling(behaviour_policy, env, num_episodes, discount=1.0, debug=False):
    
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    
    target_policy = create_target_policy(Q)
    
    for i_episode in range(1, num_episodes+1):
        
        if debug:
            if i_episode % 100000 == 0:
                print("\rEpisode {}/{}.".format(i_episode, num_episodes))
                
        state = env.reset()
        episode = []
        while(True):
            probs = behaviour_policy(state)
            action = np.random.choice(len(probs), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
            
        G = 0.0
        W = 1.0
        
        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]
            G = discount*G + reward
            C[state][action] += W
            Q[state][action] += (W/C[state][action]) * (G - Q[state][action])
            
            if action != np.argmax(target_policy(state)):
                break
                
            W = W * (target_policy(state)[action]/behaviour_policy(state)[action])
            
    return Q, target_policy


behaviour_policy = create_behaviour_policy(env.action_space.n)
optimal_Q, optimal_policy = mc_control_off_policy_importance_sampling(behaviour_policy, env, num_episodes=500000, debug=True)


V = defaultdict(float)
for state, action_values in optimal_Q.items():
    action_value = np.max(action_values)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")





get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

import sys
if "../" not in sys.path:
  sys.path.append("../")

import gym
import plotting
from lib.envs.gridworld import GridworldEnv
import itertools
matplotlib.style.use('ggplot')


env = GridworldEnv()


env.render()


#Create an initial epsilon soft policy
def epsilon_greedy_policy(Q, epsilon, state, nA):
    A = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[state])
    A[best_action] += (1.0 - epsilon)
    return A


def online_sarsa_lambda(env, num_episodes, discount=1.0, epsilon=0.1, alpha=0.5, lbda=0.9, debug=False):
    
    Q = defaultdict(lambda: np.zeros(env.action_space.n, dtype=float))
    
    for i_episode in range(1, num_episodes+1):
        
        if debug:
            if i_episode % 1000 == 0:
                print("\rEpisode {}/{}.".format(i_episode, num_episodes))
                
        E = {key:np.zeros(4, dtype=int) for key in np.arange(16)}
        state = env.reset()
        action_probs = epsilon_greedy_policy(Q, epsilon, state, env.nA)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        for t in itertools.count():
            next_state, reward, end, _ = env.step(action)
            
            next_action_probs = epsilon_greedy_policy(Q, epsilon, next_state, env.nA)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            
            delta = reward + discount*Q[next_state][next_action] - Q[state][action]
            E[state][action] += 1
            
            for s in E.keys():
                for a in E[s]:
                    Q[s][a] += alpha*delta*E[s][a]
                    E[s][a] = discount*lbda*E[s][a]
                    
            if end:
                break
                
            state = next_state
            action = next_action
            
    return Q    


Q = online_sarsa_lambda(env, num_episodes=10000, debug=True)


Q


# ## Run the Optimal Policy
# 

state = env.reset()
print env.render()
print '#################################'
while(True):
    action = np.argmax(Q[state])
    next_state, reward, done, _ = env.step(action)
    state = next_state
    
    print env.render()
    print '#################################'
    
    if done:
        break
    





