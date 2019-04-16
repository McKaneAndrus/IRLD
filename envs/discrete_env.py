import numpy as np

from gym import Env, spaces
from gym.utils import seeding


def categorical_sample(distribution, np_rng):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    distribution = np.asarray(distribution)
    cumulative_distribution = np.cumsum(distribution)
    random_num = np_rng.rand()
    return (cumulative_distribution > random_num).argmax()


class DiscreteEnv(Env):

    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)

    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS


    """
    def __init__(self, num_states, num_actions, transitions, initial_state_distribution, max_steps=None, seed=0):
        self.transitions = transitions
        self.initial_state_distribution = initial_state_distribution
        self.last_action = None  # for rendering
        self.num_states = num_states
        self.num_actions = num_actions
        self.step_num = 0
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(self.num_states)

        self._seed(seed)
        self._reset()

    def _seed(self, seed=None):
        self.np_rng, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.state = categorical_sample(self.initial_state_distribution, self.np_rng)
        self.last_action = None
        self.step_num = 0
        return self.state

    def _step(self, action):
        self.step_num += 1

        transition_distribution = self.transitions[self.state][action]
        sampled_transition_index = categorical_sample([t[0] for t in transition_distribution], self.np_rng)

        probability, state, reward, done = transition_distribution[sampled_transition_index]

        if self.max_steps is not None:
            done = self.step_num > self.max_steps

        self.state = state
        self.last_action = action
        return state, reward, done, {"prob": probability}

    def get_reward(self, state, action, next_state=None):
        """
        Calculates the reward of a state, action, next_state, tuple.  If next_state is not given it will calculate
        expected reward of the next transition given state and action.
        """
        reward = 0
        if next_state is None:
            for transition in self.transitions[state][action]:
                reward += transition[0] * transition[2]
        else:
            found_transitions = 0
            for transition in self.transitions[state][action]:
                if transition[1] == next_state:
                    found_transitions += 1
                    reward = transition[2]

            # separated asserts for ease of debugging purposes
            assert(found_transitions > 0)
            assert(found_transitions < 2)

        return reward

