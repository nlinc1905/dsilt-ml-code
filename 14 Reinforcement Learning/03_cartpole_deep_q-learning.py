import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import gym
import matplotlib.pyplot as plt


config_params = {
    'nnet_learning_rate': 0.001,
    'batch_size': 20,  # re-train after this many steps in an episode
    'gamma': 0.95,  # discount factor
    'initial_exploration_rate': 1.0,
    'min_exploration_rate': 0.01,
    'exploration_rate_decay': 0.99,
    'max_memory_len': 100,
    'avg_score_required_to_end_training': 150
    }


class NnetEnvRepresentation:
    def __init__(self, input_shape_tuple, nbr_outputs):
        self.model = None
        self.input_shape_tuple = input_shape_tuple
        self.nbr_outputs = nbr_outputs

    def compile_model(self):
        self.model = Sequential()
        self.model.add(Dense(units=24, input_shape=self.input_shape_tuple,
                             activation='relu', use_bias=True,
                             kernel_initializer='glorot_uniform',
                             bias_initializer='zeros',
                             kernel_regularizer=None,
                             bias_regularizer=None))
        self.model.add(Dense(units=24,
                             activation='relu', use_bias=True,
                             kernel_initializer='glorot_uniform',
                             bias_initializer='zeros',
                             kernel_regularizer=None,
                             bias_regularizer=None))
        self.model.add(Dense(units=self.nbr_outputs,
                             activation='linear', use_bias=True,
                             kernel_initializer='glorot_uniform',
                             bias_initializer='zeros',
                             kernel_regularizer=None,
                             bias_regularizer=None))
        self.model.compile(loss="mse", optimizer=Adam(lr=config_params['nnet_learning_rate']))


class DQLearner:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.nnet_env_representation = NnetEnvRepresentation(input_shape_tuple=(self.observation_space,), nbr_outputs=self.action_space)
        self.nnet_env_representation.compile_model()
        self.exploration_rate = config_params['initial_exploration_rate']
        self.memory = deque(maxlen=config_params['max_memory_len'])

    def remember(self, state, action, reward, next_state, terminal_condition):
        self.memory.append((state, action, reward, next_state, terminal_condition))

    def determine_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_table = self.nnet_env_representation.model.predict(state)
        return np.argmax(q_table[0])

    def learn_from_experience(self):
        # Do not learn until enough memories exist (sample batch must not be larger than population)
        if len(self.memory) < config_params['batch_size']:
            return
        batch = random.sample(self.memory, config_params['batch_size'])
        # print("batch:", batch) # uncomment to inspect the input to neural net
        for state, action, reward, next_state, terminal_condition in batch:
            q_update_val = reward
            if not terminal_condition:
                q_update_val = (reward + config_params['gamma'] + np.max(self.nnet_env_representation.model.predict(next_state)[0]))
            q_table = self.nnet_env_representation.model.predict(state)
            q_table[0][action] = q_update_val
            # print("q_update_val:", q_update_val) # uncomment to view the update to the value of this state-action pair in q-table
            self.nnet_env_representation.model.fit(state, q_table, verbose=0)
        self.exploration_rate *= config_params['exploration_rate_decay']
        self.exploration_rate = max(config_params['min_exploration_rate'], self.exploration_rate)

    def get_q_table(self):
        return self.nnet_env_representation.model

  
def cartpole():
    env = gym.make("CartPole-v0")
    observation_space = env.observation_space.shape[0]
    nbr_possible_actions = env.action_space.n
    dql = DQLearner(observation_space, nbr_possible_actions)
    scores = [0] # 1st score will be dropped later, but is needed to help break training loop
    episode = 0
    while np.mean(scores[-config_params['batch_size']:]) < config_params['avg_score_required_to_end_training']:
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:  # run until terminal condition: loop is broken when condition reached
            action = dql.determine_action(state)
            next_state, reward, terminal_condition, info = env.step(action)
            # print(step, "\n", next_state, "\n", reward, "\n", terminal_condition)
            if terminal_condition:
                reward = -1*reward
            next_state = np.reshape(next_state, [1, observation_space])
            dql.remember(state, action, reward, next_state, terminal_condition)
            state = next_state
            # If terminal condition reached, log the score (so log 1 total score per episode)
            if terminal_condition:
                scores.append(step)
                break # end episode by kicking out of steps loop
            dql.learn_from_experience()
            step += 1
        episode += 1
    return (dql.get_q_table(), scores[1:]) # drop the first score (0) that was used to help break training loop


def play_one(model, render=False):
    env = gym.make("CartPole-v0")
    if render:
        env = gym.wrappers.Monitor(env,
                                   'cartpole_video_deep/',
                                   force=True)
    state = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    score = 0
    while True:
        q_values_for_state = model.predict(state)
        action = np.argmax(q_values_for_state[0])
        next_state, reward, terminal_condition, info = env.step(action)
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
        state = next_state
        if terminal_condition:
            break
        score += reward
    env.close()
    return score


model, scores = cartpole()
plt.plot(range(len(scores)), scores)
plt.title("Cartpole Scores by Episode for Deep Q-Learner")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.show()

score_from_best_model = play_one(model, render=True)
print("Score from 1 Play of Trained Deep Q-Learner", score_from_best_model)


