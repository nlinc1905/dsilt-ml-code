import numpy as np
import pandas as pd
import gym
from itertools import product
import random
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


config_params = {
    'penalty_reward': -100,           # otherwise all rewards are 1 by default
    'episodes': 5000,
    'learning_rate': 0.8,             # alpha
    'max_steps_per_episode': 200,
    'discounting_rate': 0.95,         # gamma
    'exploration_rate': 1.0,
    'initial_exploration_prob': 1.0,
    'min_exploration_prob': 0.01,
    'exploration_prob_exp_decay': 0.0005   # epsilon
    }


# Create the environment
# Environment will provide states and rewards
env = gym.make("CartPole-v0")


# Utility functions for building discretized states

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]


# Create the Q-table
nbr_possible_actions = env.action_space.n
n_bins = 12
cart_position_bins = pd.cut([-4.8, 4.8], bins=n_bins, retbins=True)[1][1:-1]
cart_velocity_bins = pd.cut([-1e6, 1e6], bins=n_bins, retbins=True)[1][1:-1]
pole_angle_bins = pd.cut([-0.418, 0.418], bins=n_bins, retbins=True)[1][1:-1]
pole_tip_velocity_bins = pd.cut([-1e20, 1e20], bins=n_bins, retbins=True)[1][1:-1]
qtable_indices = [(to_bin(x[0], cart_position_bins),
                   to_bin(x[1], cart_velocity_bins),
                   to_bin(x[2], pole_angle_bins),
                   to_bin(x[3], pole_tip_velocity_bins)) \
                  for x in product(cart_position_bins,
                                   cart_velocity_bins,
                                   pole_angle_bins,
                                   pole_tip_velocity_bins)]  # All combinations of observation space
nbr_possible_states = len(qtable_indices)
qtable = np.zeros((nbr_possible_states, nbr_possible_actions))
# Sense checks
# print(qtable)
print(qtable.shape)
# print(cart_position_bins)


rewards = []

for episode in range(config_params['episodes']):
    state = env.reset()
    step = 0
    terminal_state = False
    total_rewards = 0
    
    for step in range(config_params['max_steps_per_episode']):
        # Discretize state according to pre-defined bins
        state_index = qtable_indices.index((to_bin(state[0], cart_position_bins),
                             to_bin(state[1], cart_velocity_bins),
                             to_bin(state[2], pole_angle_bins),
                             to_bin(state[3], pole_tip_velocity_bins)))

        random_nbr = random.uniform(0, 1)
        # If random_nbr > exploration_rate then follow exploitation (take the biggest Q value for this state)
        # else explore by taking a random action
        if random_nbr > config_params['exploration_rate']:
            action = np.argmax(qtable[state_index,:])
        else:
            action = env.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, terminal_state, info = env.step(action)

        # Discretize new state according to pre-defined bins
        new_state_index = qtable_indices.index((to_bin(new_state[0], cart_position_bins),
                                 to_bin(new_state[1], cart_velocity_bins),
                                 to_bin(new_state[2], pole_angle_bins),
                                 to_bin(new_state[3], pole_tip_velocity_bins)))

        if not terminal_state:
            # Update the Q-table based on Q-learning equation
            qtable[state_index, action] = qtable[state_index, action] + config_params['learning_rate'] * (reward + config_params['discounting_rate'] * np.max(qtable[new_state_index, :]) - qtable[state_index, action])
            total_rewards += reward
            state = new_state
        
        if terminal_state:
            # Implement penalty for ending the game (by default, there is no penalty, so must add a penalty reward)
            qtable[state_index, action] = qtable[state_index, action] + config_params['learning_rate'] * (reward + (1/config_params['penalty_reward']) + config_params['discounting_rate'] * np.max(qtable[new_state_index, :]) - qtable[state_index, action])
            break
        
    # Reduce epsilon (because we need less and less exploration)
    config_params['exploration_rate'] = config_params['min_exploration_prob'] + (config_params['initial_exploration_prob'] - config_params['min_exploration_prob'])*np.exp(-config_params['exploration_prob_exp_decay']*episode)
    rewards.append(total_rewards)


scaler = MinMaxScaler(feature_range=(0,200))
rewards_scaled = list(scaler.fit_transform(np.array(rewards).reshape(-1,1)))
print ("Mean Score: " +  str(sum(rewards_scaled)/config_params['episodes']))
plt.plot([x for x in range(config_params['episodes'])], rewards_scaled)
plt.title("Scores by Episode for Q-Learner")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.show()
sns.distplot(rewards_scaled)
plt.title("Score Distribution for Q-Learner")
plt.xlabel("Score")
plt.show()
print(qtable)


env.reset()

env = gym.wrappers.Monitor(env, 'C:/Users/Nick/Downloads/cartpole_video/',
                           #video_callable=lambda episode_id: True,
                           force=True)

for episode in range(5):
    state = env.reset()
    step = 0
    terminal_state = False

    for step in range(config_params['max_steps_per_episode']):
        # Discretize state according to pre-defined bins
        state_index = qtable_indices.index((to_bin(state[0], cart_position_bins),
                             to_bin(state[1], cart_velocity_bins),
                             to_bin(state[2], pole_angle_bins),
                             to_bin(state[3], pole_tip_velocity_bins)))
        
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state_index,:])
        new_state, reward, terminal_state, info = env.step(action)

        # Discretize new state according to pre-defined bins
        new_state_index = qtable_indices.index((to_bin(new_state[0], cart_position_bins),
                                 to_bin(new_state[1], cart_velocity_bins),
                                 to_bin(new_state[2], pole_angle_bins),
                                 to_bin(new_state[3], pole_tip_velocity_bins)))

        # Render the last state to view agent's progress
        if terminal_state:
            env.render()
            print("Number of steps", step)
            break
        state = new_state

env.close()

