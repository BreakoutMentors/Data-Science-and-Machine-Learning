# Intro to Reinforcement Learning (RL)
> ⚠️🛑 ML Mentors: Please connect with Kai (via email) to discuss whether jumping into the reinforcement learning content makes sense for your student.

**Learning Objective:** Learn the foundations of reinforcement learning (RL) and understand when and how to implement RL algorithms in your projects

Learning RL can be difficult as most content is not accessible to younger students. That's why we built our own RL curriculum to enable our students to learn the foundations of RL and leverage RL algorithms in their projects! The curriculum is hands-on and interactive, just like all the other Breakout Mentors machine learning curriculums! Students interested in game development, robotics, recommendation systems (e.g., Spotify, Netflix, etc.), dynamic decision-making (finance, economics, etc.), and/or machine learning with human feedback may benefit from our RL curriculum. Talk to your ML Mentor to see if learning RL makes sense for your goals and timeline (ML Mentors, see note at the beginning of this section).  

**About:** Reinforcement Learning (RL) is a machine learning paradigm designed to enable agents (e.g., computers) to learn optimal decision-making strategies by interacting with an environment. The primary goal of RL is to train agents to make a sequence of actions that achieve the desired outcome over time. RL is invaluable in situations where explicit instructions or labeled data are scarce or impractical, as it allows agents to autonomously discover optimal behaviors through trial and error. It operates on the principle of feedback: an agent takes actions, observes the environment's response, and uses this feedback to adjust its decision-making process over successive interactions. Through this iterative process of exploration and exploitation, RL algorithms learn to make intelligent decisions, making them particularly suitable for applications in robotics, game playing, recommendation systems, and other domains where dynamic decision-making is essential.

![MOBA GAME using RL](https://miro.medium.com/v2/resize:fit:750/1*sKn2mb_gj-3MZ7bsgPw5IQ.gif)


## Section 1: Intro to Reinforcement Learning
**Learning Objective:** In these lessons, you will embark on your journey of Reinforcement Learning! You will learn basic RL terminologies and how these terminologies relate to a Markov Decision Process.

- [Lesson 1: Intro to RL](Lesson-1-Intro-to-RL.ipynb)
- [Lesson 2: Markov Decision Process](Lesson-2-Markov-Decision-Process.ipynb)
- _Challenges:_
    - [Lesson 1 Challenge](Lesson-1-challenge.ipynb)
    - [Lesson 2 Challenge](Lesson-2-challenge.ipynb)

## Section 2: Value-Based Methods
**Learning Objective:** In this lesson, we will talk about
    - Monte Carlo Method (learn by sampling)
    - Temporal Difference Method (a different way of value-update than Monte Carlo)
    - Q-Learning (using a table to store useful value)
    - Deep Q-Learning (Q-Learning using a neural network)

- [Lesson 3: Value-Based Learning Methods](Lesson-3-Value-Based-Learning-Methods.ipynb)
- _Challenge:_
    - [Lesson 3 Challenge](Lesson-3-challenge.ipynb)

## Section 3: Policy-Based Methods
**Learning Objective:** In this lesson, we will compare value-based and policy-based methods. Also, we will discuss the high-level process of policy gradient and introduce one of the policy gradient algorithms: REINFORCE. 

- [Lesson 4: Policy-Based Learning Methods](Lesson-4-Policy-Based-Learning-Methods.ipynb)
- _Challenge:_
    - [Lesson 4 Challenge](Lesson-4-challenge.ipynb)

## Section 4: Additional Topics
**Learning Objective:** In these lessons, we will discuss some additional topics. First, we will talk about two advanced reinforcement learning algorithms: A2C and PPO, and how to use them in practice. Next. we will discuss how to train more than one agent to play a game.

- [Lesson 5: Advantage Actor Critic](Lesson-5-A2C.ipynb)
- [Lesson 6: Proximal Policy Optimization](Lesson-6-PPO.ipynb)
- [Lesson 7: Multi-Agent Reinforcement Learning](Lesson-7-MARL.ipynb)

- _Challenges:_
    - [Lesson 5 Challenge](Lesson-5-A2C-challenge.ipynb)
    - [Lesson 6 Challenge](Lesson-6-PPO-challenge.ipynb)
    - [Lesson 7 Challenge](Lesson-7-MARL-challenge.ipynb)


### Additional Resources

- Learning Resources
    1. [HuggingFace RL Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction)
    2. [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)
    3. [DeepMind x UCL RL Lectures](https://youtu.be/TCCjZe0y4Qc)
    4. [MIT RL Lecture](https://youtu.be/AhyznRSDjw8)

- RL Tools
    1. [Stable-Baseline3](https://stable-baselines3.readthedocs.io/en/master/)
        > PyTorch version of Stable Baselines, reliable implementations of reinforcement learning algorithms.
    2. [RL Baselines3 Zoo](https://stable-baselines3.readthedocs.io/en/master/guide/rl_zoo.html)
        > A training framework for Stable Baselines3 reinforcement learning agents, with hyperparameter optimization and pre-trained agents included.
    3. [CleanRL](https://docs.cleanrl.dev/)
        > High-quality single file implementation of Deep Reinforcement Learning .algorithms
    4. [Gymnasium](https://gymnasium.farama.org/)
        > An API standard for reinforcement learning with a diverse collection of reference environments.
    5. [PettingZoo](https://pettingzoo.farama.org/index.html)
        > An API standard for multi-agent reinforcement learning.
    6. [Optuna](https://optuna.org/)
        > An open source hyperparameter optimization framework to automate hyperparameter search.
