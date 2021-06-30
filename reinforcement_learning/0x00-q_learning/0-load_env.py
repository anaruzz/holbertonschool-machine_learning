#!/usr/bin/env python3
"""
Script that loads the premade FrozenLakeEnv environment
"""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    create the environment
    """
    env = gym.make("FrozenLake-v0",
                    map_name=map_name,
                    is_slippery=is_slippery,
                    desc = desc
                    )
    env.reset()
    return env
