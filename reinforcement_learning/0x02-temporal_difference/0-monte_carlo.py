#!/usr/bin/env python3
"""
Script that performs the Monte carlo algorithm
"""
import numpy as np


def play(env, policy, max_steps, first):
    """
    Play an episode
    """
    state = env.reset()
    ep_rewards = []
    states = [state]
    success = 0
    for step in range(max_steps):
        action = policy(state)
        state, reward, done, info = env.step(action)
        if first and state in states:
            continue
        ep_rewards.append(reward)
        states.append(state)
        if reward > 0:
            success += 1
        if done:
            break
    return ep_rewards, states

def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=.99, first=False):
    """
    Returns: V, the updated value estimate
    """

    for _ in range(episodes):
        ep_rewards, states = play(env, policy, max_steps, first)
        total_return = 0

        for state, reward in zip(states[:-1][::-1], ep_rewards[::-1]):
            total_return = total_return * gamma + reward
            V[state] += alpha * (total_return - V[state])
    return V
