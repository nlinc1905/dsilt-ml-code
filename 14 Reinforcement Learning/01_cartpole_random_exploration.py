import gym
import time
import matplotlib.pyplot as plt
import seaborn as sns

env = gym.make("CartPole-v1")
observation = env.reset()

# Exploration https://github.com/openai/gym/wiki/CartPole-v0
print(env.observation_space)      # array of shape (4,)
print(env.observation_space.low)  # min box coordinates
print(env.observation_space.high) # max box coordinates
print(env.action_space)           # 2 discrete actions (0 or 1)
# observation space = Box(4), where
# 0 = cart position (-4.8, 4.8)
# 1 = cart velocity (-Inf, Inf)
# 2 = pole angle (-41.8 degrees or -.418, 41.8 degrees or .418)
# 3 = pole tip velocity (-Inf, Inf) 
print("\n")
print("Initial obs:", observation)
observation, reward, done, info = env.step(0)
print("New obs:", observation)
print(reward)
print(done)
observation = env.reset()
print("Initial obs:", observation)
observation, reward, done, info = env.step(0)
print("New obs:", observation)
print(reward)
print(done)
observation = env.reset()
env.render()
time.sleep(5)
print("Initial obs:", observation)
observation, reward, done, info = env.step(0)
observation, reward, done, info = env.step(0)
observation, reward, done, info = env.step(0)
print("New obs:", observation)
print(reward)
print(done)
env.render()
time.sleep(5)
observation = env.reset()
# Action 0 decreases first 2 coordinates, increases last 2
# Action 0 moves cart left
env.render()
time.sleep(5)
print("Initial obs:", observation)
observation, reward, done, info = env.step(1)
observation, reward, done, info = env.step(1)
observation, reward, done, info = env.step(1)
print("New obs:", observation)
print(reward)
print(done)
env.render()
time.sleep(5)
observation = env.reset()
# Action 1 increases first 2 coordinates, decreases last 2
# Action 1 moves cart right
print("\n")
done = False
rewards = []
while not done:
    observation, reward, done, info = env.step(1)
    rewards.append(reward)
print("Final obs:", observation)
print("Rewards list:", rewards)
env.render()
time.sleep(5)
# Game ends before pole falls off the cart, but it has tipped far enough to fall
# Rewards are always 1, even after the done condition has been met
# A reward measures the number of steps, so the more steps before the pole falls, the better
# There is never a done state that is not a failure (pole falls)
print("Score:", sum(rewards))
print("\n")


scores_from_random_actions = []
nbr_trials = 1000
for _ in range(nbr_trials):
    observation = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample() # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        total_reward += reward
    scores_from_random_actions.append(total_reward)
        
print("Best score from {} trials of random actions:".format(nbr_trials), max(scores_from_random_actions))

env.close()

sns.distplot(scores_from_random_actions)
plt.title("Distribution of Cartpole Scores from {} Trials of Random Actions".format(nbr_trials))
plt.xlabel("Score")
plt.show()

plt.plot([x for x in range(nbr_trials)], scores_from_random_actions)
plt.title("Scores Over Time for Random Actions")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.show()
