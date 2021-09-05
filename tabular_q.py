#!/usr/bin/env python3
import gym
import collections
from tensorboardX import SummaryWriter

'''
Q - Learning
1.Start with an empty table, mapping states to values of actions.
2.By interacting with the environment, obtain the tuple (s, a, r, s′).
3.Update the Q(s, a) value using the Bellman update
4.Check convergence conditions. If not met, repeat from step 2.
'''

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
ALPHA = 0.1
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)

    # We obtain the next transition from the environment
    def sample_env(self):
        # We sample a random action from the action space
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        # The returned tuple will be used in the training loop
        return (old_state, action, reward, new_state)


    # This method receives a state and finds the best action
    # to take from this state by taking the action with the largest value in the table

    # This method will be used two times: first, in the test method that plays one episode using the current values table,
    # and the second, in the method that performs the value update to get the value of the next state.
    def best_value_and_action(self, state):
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, s, a, r, next_s):
        best_v, _ = self.best_value_and_action(next_s)
        new_val = r + GAMMA * best_v
        old_val = self.values[(s, a)]
        self.values[(s, a)] = old_val * (1-ALPHA) + new_val * ALPHA

    # This method plays one full episode using the test environment.
    # The action on every step is taken using the value table.
    # It evaluates our current policy to check the progress of learning.
    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, is_done, _ = env.step(action)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0

    # In the training loop we do one step in the environment and perform a value update
    while True:
        iter_no += 1
        s, a, r, next_s = agent.sample_env()
        agent.value_update(s, a, r, next_s)

        # We test our current policy by playing several test episodes.
        # Here we don't touch Q-values during the test (as we did in value_iter and q_iter)
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)

        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break

    writer.close()