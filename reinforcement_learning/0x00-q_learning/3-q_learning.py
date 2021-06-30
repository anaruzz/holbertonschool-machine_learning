#!/usr/bin/env python3
"""
Script that performs Q-learning
"""
import gym
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy

def train(env, Q, episodes=5000, max_steps=100,
          alpha=0.1, gamma=0.99, epsilon=1,
          min_epsilon=0.1, epsilon_decay=0.05):
    """
    Returns: Q, total_rewards
    Q is the updated Q-table
    total_rewards is a list containing the rewards per episode
    """
    R = []
    max_epsilon = epsilon

    for ep in range(episodes):
        state = env.reset()
        rewards = 0
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            observation, reward, done, info = env.step(action)
            if done is True and reward == 0:
                reward = -1
            Q[state, action] = (Q[state, action] * (1 - alpha)) + \
                (reward + gamma * np.max(Q[observation, :])) * alpha
            state = observation
            rewards += reward
            if done is True:
                break
        R.append(rewards)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * ep)
    return Q, R
