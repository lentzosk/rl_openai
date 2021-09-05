#!/usr/bin/env python3
import gym


def print_env_details(name):
    e = gym.make("FrozenLake-v1")

    print("Observation space:")
    print(e.observation_space)

    print("Action space:")
    print(e.action_space)

    e.reset()

    print("\nRendering the environment:")
    e.render()

if __name__ == '__main__':
    print_env_details()


