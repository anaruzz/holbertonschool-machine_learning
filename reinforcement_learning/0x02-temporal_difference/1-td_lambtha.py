#!/usr/bin/env python3
"""
Script that performs the TD(λ) algorithm
"""
import gym
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    Returns: V, the updated value estimate
    """
    state_d = env.observation_space.n
    Et = np.zeros(state_d)
    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            Et *= lambtha * gamma
            Et[state] += 1.0
            a = policy(state)
            new_state, reward, done, info = env.step(a)
            delta = reward + gamma * V[new_state] - V[state]
            V[state] = V[state] + alpha * delta * Et[state]
            if done:
                break
            state = new_state
    return V
