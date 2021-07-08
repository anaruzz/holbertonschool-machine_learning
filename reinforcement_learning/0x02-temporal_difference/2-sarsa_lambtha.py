#!/usr/bin/env python3
"""
Script that performs the TD(λ) algorithm
"""
import gym
import numpy as np


def epsilon_greedy(state, Q, epsilon):
    """
    Eplison greedy function
    """
    p = np.random.uniform(0, 1)
    if p >= epsilon:
        action = np.argmax(Q[state])
    else:
        action = np.random.randint(0, int(Q.shape[1]))
    return(action)


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Returns : Q, the updated Q table
    """
    state_d = env.observation_space.n
    init_e = epsilon
    Et = np.zeros((Q.shape))
    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy(state, Q, epsilon=epsilon)
        for _ in range(max_steps):
            Et = Et * lambtha * gamma
            Et[state, action] += 1.0
            new_state, reward, done, info = env.step(action)
            new_a = epsilon_greedy(new_state, Q, epsilon=epsilon)
            delta_t = reward + gamma * Q[new_state, new_a] - Q[state, action]
            Q[state, action] = Q[state, action] + \
                alpha * delta_t * Et[state, action]
            if done:
                break
            state = new_state
            action = new_a
        epsilon = min_epsilon + (init_e - min_epsilon) *\
            np.exp(-epsilon_decay * episode)
    return Q
