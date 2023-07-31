[TOC]
- [Lesson 1](#lesson-1)
- [Lesson 2](#lesson-2)
- [Lesson 3](#lesson-3)
- [Lesson 4](#lesson-4)

# Lesson 1

### Question 1.1

>So my *current state* is at the (1, 1), aka 1st row, 1st column. But I might move either right or down, and those *2* locations will be my the potential *next state*

### Question 1.2

> So, if I found the mouse, I will get **+100** reward. However, for every step I move, I get **-5** reward and if I fall into the red trap, I get **-10** reward, so sad....

### Question 1.3

> There are 4 actions I can take: *left, right, down, up*

### Question 1.4

> Mario !!

### Question 1.5

> The game world that Mario in.

### Question 1.6

> **Episodic**: Super Mario, since this game is divided into rounds. 
**Contiuning**: Stock Market, since it is always happening (at least at the time of this question was written).

### Question 1.7

> **Exploration**: I choose to take the risk of getting bad-tasting food by going to the restaurant that I have never try before. (Maybe the food will be even better than the others, who knows?)
**Exploitation**: I choose the restaurant that I always go though the food is not that great. (Playing safe here)

## Challenge

### Question 1.8

```py
import gymnasium as gym
# Create the CartPole environment
env = gym.make("CartPole-v1")
```

### Question 1.9

```py
action_space, observation_space, reward_range = env.action_space, env.observation_space, env.reward_range
action_space, observation_space, reward_range
```
> - **Action Space**: Discrete(2) -> only 2 actions to take (left, right)
> - **Observation Space**: min([cart position, cart velocity, pole angle, pole angular velocity])
max([cart position, cart velocity, pole angle, pole angular velocity])
> - **Rewards**: can adds up to this range -> (-infinity, infinity)
> - Reference: https://gymnasium.farama.org/environments/classic_control/cart_pole/

### Question 1.10

```py
observation, info = env.reset()
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
observation, reward, terminated, truncated, info
```
### Question 1.11

```py
# Create the CartPole environment
env = gym.make("CartPole-v1")

# Reset the environemnt
observation, info = env.reset()

# Take a random action
action = env.action_space.sample()
print("Action taken:", action)

observation, reward, terminated, truncated, info = env.step(action)

while(not terminated or not truncated):
  action = env.action_space.sample()
  print("Action taken:", action)
  observation, reward, terminated, truncated, info = env.step(action)

env.close()
```
### Question 1.12
```py
from stable_baselines3 import PPO
# Create the CartPole environment
env = env = gym.make("CartPole-v1")

# Instantiate the agent
model = PPO('MlpPolicy', env, verbose=1)

# Train the agent for 5 timesteps
model.learn(total_timesteps=5)

# Save the trained agent
model.save("ppo_cartpole")

# delete the model from memory
del model
```

### Question 1.13
```py
from IPython import display
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

# load the model
model = PPO.load("ppo_cartpole")

# Create the CartPole environment
env = gym.make("CartPole-v1")

# reset the environment
obs = env.reset()

img = plt.imshow(env.render(mode='rgb_array')) # only call this once

while True:

  img.set_data(env.render(mode='rgb_array')) # just update the data
  display.display(plt.gcf())
  display.clear_output(wait=True)

  action, _states = model.predict(obs)  # predict the action and state using the model
  obs, rewards, terminated, info = env.step(action) # take the predicted action

  if terminated :
    break

env.close()
```
---
---

# Lesson 2

### Question 2.1
```py
P = {
    "Sleep": {"Sleep": 0.50 , "Find": 0.25, "Eat": 0.25},
    "Find" : {"Sleep": 0.25 , "Find": 0.25, "Eat": 0.50},
    "Eat"  : {"Sleep": 1.00 , "Find": 0.00, "Eat": 0.00}
}

P
```
### Question 2.2
```py
R = {
    "Sleep": {"Sleep": -1 , "Find": 10, "Eat": 2},
    "Find" : {"Sleep": -1 , "Find": 5, "Eat":1},
    "Eat"  : {"Sleep": -5 , "Find": 0, "Eat": 0}
}

R
```
### Question 2.3
> The **Sleepy** policy is kind of..... bad :(
```py
print(calculate_expected_reward("Eat", P, R, sleepy_pi))
print(calculate_expected_reward("Find", P, R, sleepy_pi))
```

### Question 2.4
> This is only one of the way you can set it up!
```py
find_pi = {
    "Sleep": {"Sleep": 0 , "Find": 1, "Eat": 0},
    "Find" : {"Sleep": 0 , "Find": 1, "Eat": 0},
    "Eat"  : {"Sleep": 0 , "Find": 1, "Eat": 0}
}

print(calculate_total_expected_reward("Sleep", P, R, find_pi, 100))
print(calculate_total_expected_reward("Find", P, R, find_pi, 100))
print(calculate_total_expected_reward("Eat", P, R, find_pi, 100))
```
## Challenge

### Question 2.5
> Playing, see the value on "Sleep" row and ”Play" column for the transition matrix

### Question 2.6
> - State: Sleep
>   - Action: Play
> - State: Play  
>   - Action: Eat
> - State: Eat
>   - Action: Sleep

### Question 2.7
```py
# Calculate the expected reward for each state
states = ["Sleep", "Play", "Eat"]
for state in states:
    print(f"Expected reward for {state}: {calculate_expected_reward(state, P, R, policy)}")
```
### Question 2.8
```py
total_expected_reward = 0
expected_reward = 0
# Calculate the expected reward for each state
for state in states:
    expected_reward = calculate_expected_reward(state, P, R, new_policy)
    total_expected_reward += expected_reward
    print(f"Expected reward for {state}: {expected_reward}")

print(f"Total Expected reward is: {total_expected_reward}")
```
### Question 2.9
:sunglasses:

---
---
# Lesson 3

### Question 3.1
> More points lead to a better estimation (according to the "Law of Large Numbers", something you will learn in probability theory).

### Question 3.2
> It allows us to discover potential outcomes to get a better understanding of the environment.

### Question 3.3
> *Monte Carlo Method* can try out different strategies and evaluate the performance of them. And eventually approximating a optimal decision through playing the game for more time.



### Question 3.4
> For each state, we will take a random action (More, specifically, based on the probability distribution of the transition probabilities at that state, don't worry if you don't know what this mean haha).

### Question 3.5
> A high discount factor means that future rewards are considered almost as valuable as immediate rewards, while a low discount factor means that immediate rewards are much more valuable.

### Question 3.6
> The state of Room 9 has the highest value and the the only path to solve the maze has an increasing reward!

### Question 3.7
> The state values represent the expected future reward from each state, taking into account the probability of transitioning between states and the discount factor.

### Question 3.8
> The exploration rate determines how often the agent chooses a random action. A high exploration rate means the agent explores the environment more often, while a low exploration rate means the agent exploits its current knowledge of the environment more often.

## Challenge

### Question 3.9
```py
rewards = np.array([0, -10, -10, 5, -10, -10, 10, 20, 100])
Q_table = run_Q_learning(rewards)
best_path(Q_table)
```

### Question 3.10
```py
rewards = np.array([0, -1, -1, 20, 40, 80, -1, -1, 1000])
Q_table = run_Q_learning(rewards)
best_path(Q_table)
```
### Question 3.11
```py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("ALE/Breakout-v5", render_mode='rgb_array')
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
```

### Question 3.12
```py
model = DQN("CnnPolicy", env, buffer_size=10000, learning_rate=0.001, verbose=1)
model.learn(total_timesteps=20000)

```

---
---
# Lesson 4



