# https://en.wikipedia.org/wiki/Q-learning

import gymnasium as gym
import numpy as np 
import random
import matplotlib.pyplot as plt

gym.make('Taxi-v3')
env = gym.make('Taxi-v3')
observation, info = env.reset()

action = env.action_space.sample()

obs, reward, terminated, truncated, info = env.step(action)

print(f'Info after reset: {info}')
print(f'Starting observation: {observation}')
episode_over = False

Q_values = np.zeros((env.observation_space.n, env.action_space.n))
alpha = 0.1
gamma = 0.99
n_episodes = 5000
eps_decay = 0.99
epsilon = 1
rewards_hist = []
for episode in range(n_episodes):
    obs, info = env.reset()
    rewards = 0
    action = np.argmax(Q_values[obs])

    obs, reward, terminated, truncated, info = env.step(action)
    rewards += reward
    epsilon *= eps_decay
    done = False
    while not done:
        if random.random() < max(epsilon, 0.05):
            new_action = env.action_space.sample()
        else:
            new_action = np.argmax(Q_values[obs])
        new_obs, reward, terminated, truncated, info = env.step(new_action)
        rewards += reward
        Q_values[obs][action] = (1-alpha)*Q_values[obs][action] + alpha*(reward + gamma*np.max(Q_values[new_obs]))

        obs = new_obs
        action = new_action
        done = terminated or truncated
    rewards_hist.append(rewards)
    print(rewards)
    
env.close()
print(rewards_hist)
plt.plot(rewards_hist)
plt.show()
render_env = gym.make("Taxi-v3", render_mode="human")
state, info = render_env.reset()
render_done = False
while not render_done:
    action = np.argmax(Q_values[state])
    render_env.render()
    state, reward, terminated, truncated, info = render_env.step(action)
    render_done = terminated or truncated
    print("dd")
render_env.close()

