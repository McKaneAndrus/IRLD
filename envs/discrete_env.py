import numpy as np

from gym import Env, spaces
from gym.utils import seeding

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    r = np_random.rand()
    return (csprob_n > r).argmax()


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
    def __init__(self, nS, nA, P, isd,max_steps=None):
        self.P = P
        self.isd = isd
        self.lastaction=None # for rendering
        self.nS = nS
        self.nA = nA
        if max_steps != None:
            self.step_num = 0
            self.max_steps = max_steps

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction=None
        if hasattr(self, "step_num"):
            self.step_num = 0
        return self.s

    def _step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        if hasattr(self, "step_num"):
            self.step_num += 1
            p, s, r = transitions[i]
            d = self.step_num > self.max_steps
        else:
            p, s, r, d= transitions[i]
        self.s = s
        self.lastaction=a
        return (s, r, d, {"prob" : p})

    def get_reward(self,s,a,sprime=None):
        r = 0
        if sprime is None:
            for t in self.P[s][a]:
                r += t[0] * t[2]
        else:
            for t in self.P[s][a]:
                if t[1] == sprime:
                    r = t[2]
        return r

