#!/usr/bin/python3
"""
Script that trains a model
"""
import gym
import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Returns the sum of all rewards during one episode loop
    """
    weights = np.random.rand(4, 2)
    ep_rewards = []
    for ep in range(nb_episodes):
        state = env.reset()[None, :]
        total_score = 0
        grads = []
        rewards = []

        while True:
            action, grad = policy_gradient(state, weights)
            new_state, reward, done, _ = env.step(action)
            grads.append(grad)
            rewards.append(reward)
            total_score += reward
            state = new_state[None, :]
            if done:
                break

        for i in range(len(grads)):
            weights += (alpha * grads[i] *
                        sum([r * gamma**r for t, r in enumerate(rewards[i:])]))
        ep_rewards.append(total_score)
        print("{}: {}".format(ep, total_score), end="\r", flush=False)

    return ep_rewards
