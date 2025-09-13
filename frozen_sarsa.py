import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1") 
n_states = env.observation_space.n
n_actions = env.action_space.n

Q = np.zeros((n_states, n_actions), dtype=np.float32)

alpha = 0.1
gamma = 0.99
n_episodes = 5000

epsilon = 1.0
epsilon_min = 0.05
eps_decay = 0.99

rewards_hist = []


for ep in range(n_episodes):
    obs, info = env.reset()
    done = False
    ep_reward = 0

    while not done:
        # epsilon-greedy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(Q[obs]))

        new_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward

        # if done, no future reward
        if done:
            target = reward
        else:
            # SARSA difference: use action from policy (argmax) for next state
            target = reward + gamma * Q[new_obs][np.argmax(Q[new_obs])]

        # update Q-value
        Q[obs, action] = (1 - alpha) * Q[obs, action] + alpha * target

        obs = new_obs

    rewards_hist.append(ep_reward)
    # decay epsilon after each episode
    epsilon = max(epsilon_min, epsilon * eps_decay)

env.close()

# Display learning curve
plt.plot(rewards_hist)
plt.xlabel("Episode")
plt.ylabel("Cumulative reward")
plt.title("Q-learning on FrozenLake-v1")
plt.show()

# Evaluations 
N_eval = 100
eval_rewards = []
for _ in range(N_eval):
    obs, info = env.reset()
    done = False
    ep_reward = 0
    while not done:
        action = int(np.argmax(Q[obs]))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        ep_reward += reward
    eval_rewards.append(ep_reward)
print(f"Average reward over {N_eval} evaluation episodes: {np.mean(eval_rewards)}")
render_env = gym.make("FrozenLake-v1", render_mode="human")
state, info = render_env.reset()
render_done = False
while not render_done:
    action = int(np.argmax(Q[state]))
    state, reward, terminated, truncated, info = render_env.step(action)
    render_done = terminated or truncated
render_env.close()
