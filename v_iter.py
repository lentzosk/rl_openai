#!/usr/bin/env python3
import gym
import collections
from tensorboardX import SummaryWriter

'''
Data structures.

Reward table: A dictionary with the composite key "source state" + "action" + "target state".
The value is obtained from the immediate reward.

Transitions table: A dictionary keeping counters of the experienced transitions.
The key is the composite "state" + "action" and the value is another dictionary
that maps the target state into a count of times that we've seen it. 

For example, if in state 0 we execute action 1 ten times, after three
times it leads us to state 4 and after seven times to state 5. 
state 0, action 1 -> state 4 (three times)
state 0, action 1 -> state 5 (seven times)
Entry with the key (0, 1) in this table will be a dict {4: 3, 5: 7}.
We use this table to estimate the probabilities of our transitions.

Value table: A dictionary that maps a state into the calculated value of this state.
'''

'''
Method.

We play 100 random steps from the environment, populating the reward and transition tables.
After those 100 steps, we perform a value iteration loop over all states, updating the value table.

Then we play several full episodes to check our improvements using the updated value table.
If the average reward for those test episodes is above 0.8, then we stop training.
 
During test episodes, we also update our reward and transition tables.
'''

# To run Tensorboard type in terminal:
# tensorboard --logdir=runs
# runs is the directory name


ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)   # reward table
        self.transits = collections.defaultdict(collections.Counter) # transition table
        self.values = collections.defaultdict(float)   # value table

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            #update reward and transition tables
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    # Q, we use it for:
    # - To select the best action to perform from the state
    # - Calculate the new value of the state on value iteration
    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)]

        # We sum all counters to obtain the total count of times we've executed this action from that state.
        # We will use this total value later to go from an individual counter to probability.
        total = sum(target_counts.values())

        action_value = 0.0

        # We iterate every target state and calculate its contribution into the total action value using Bellman
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            # immediate reward plus discounted value for the target state.
            # we multiply with the probability of this transition
            action_value += (count / total) * (reward + GAMMA * self.values[tgt_state])
        return action_value

    # Now we make a decision about the best action to take from a given state.
    # We iterate over all possible actions in the environment and calculate the value for every action.
    # The action with the largest value is returned as the action to take.
    # This action selection process is deterministic, as the play_n_random_steps() introduces enough exploration.
    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    # The play_episode uses select_action to find the best action to take and plays one full episode
    # This function is used to play test episodes
    # We don't want to mess up with the current state of the main environment used to gather random data.
    # So, we're using the second environment passed as an argument.
    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    # We loop over all states in the environment,
    # then for every state we calculate the values for the states reachable from it,
    # obtaining candidates for the value of the state.
    # Then we update the value of our current state with the maximum value of the action available from the state
    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            state_values = [self.calc_action_value(state, action)
                            for action in range(self.env.action_space.n)]
            self.values[state] = max(state_values)


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-v-iteration")

    iter_no = 0
    best_reward = 0.0

    while True:
        iter_no += 1
        # Here we do the training
        # First, we perform 100 random steps to fill our reward and transition tables with fresh data
        # then we run value iteration over all states.
        agent.play_n_random_steps(100)
        agent.value_iteration()

        # The rest of the code plays test episodes using the value table as our policy.
        # We track the best average reward, and check for the training loop stop condition.
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