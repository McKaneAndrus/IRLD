import numpy as np
from .discrete_env import categorical_sample, DiscreteEnv

import gym

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
NOOP = 4

# (x,y)
DIRDICT = {(-1, 0): 0,
           (0, 1): 1,
           (1, 0): 2,
           (0, -1): 3,
           (0, 0): 4}

# (x, y)
DIRARRAY = np.array([[-1, 0],
                     [0, 1],
                     [1, 0],
                     [0, -1],
                     [0, 0]])


MAPS = {

}


class MarsExplorerEnv(DiscreteEnv):
    """
    NASA's most recent Mars Rover has finally landed. In the surrounding area there
    are locations that are to unique interest to either the Astrobiology or Geology.
    In order
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, tile_map, reward_map, texture_map, transition_dict, time_penalty=-0.001, seed=None):
        self.tile_map = desc = np.asarray(tile_map, dtype='c')
        self.reward_map = reward_map
        self.pure_texture_map = texture_map
        self.texture_map = texture_map.flatten()
        # self.rewards = np.asarray(reward_map).flatten()
        self.rewards = reward_map.flatten()
        self.max_reward = np.max(self.rewards)
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.penalty = time_penalty
        self.potential_goal_states = []
        self.t_mat = None

        # TODO figure out if making this change breaks downstream
        # self.transition_dict = transition_dict
        # self.tile_transition_map = {}
        #
        # tt = 0
        # for key in transition_dict.keys():
        #     if transition_dict[key] not in self.tile_transition_map:
        #         self.tile_transition_map[transition_dict[key]] = tt
        #         tt += 1
        #
        # self.tile_types = len(self.tile_transition_map.keys())

        self.tile_types = 2


        # Rotational dir_dict
        self.dir_dict = {(self.ncol - 1, 0): 0,
           (0, 1): 1,
           (1, 0): 2,
           (0, self.nrow-1): 3,
           (0, 0): 4}

        num_actions = 5
        num_movement_actions = 4
        num_states = nrow * ncol
        self.num_directions = 5


        # Featurization of Used for when reward is unknown
        flat_desc = desc.flatten()
        goal_chars = b'123'
        non_goal_chars = b'BFUS'

        n_goal_types = len(goal_chars)
        self.feature_map = np.zeros((num_states, n_goal_types))
        for s, char in enumerate(flat_desc):
            if char not in non_goal_chars:
                feat_idx = int(char)
                self.feature_map[s, feat_idx-1] = 1.0


        # TODO include tile map such that this is a feasible thing to construct
        # self.thetas = np.zeros((nA, nA, self.tile_types))
        # for t in range(self.tile_types):
        #     self.thetas[0,:,t] =



        def to_s(row, col):
            return row*ncol + col

        if seed is not None:
            self._seed(seed)

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(num_actions)} for s in range(num_states)}

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(num_actions):
                    li = P[s][a]
                    letter = desc[row, col]
                    rew = self.penalty + self.rewards[s] #if a == NOOP else self.penalty
                    if a == NOOP:
                        li.append((1.0, s, rew, False))
                    else:
                        transitions = transition_dict[letter]
                        probs = [(transitions[0], a),
                                 (transitions[1], (a-1) % num_movement_actions),
                                 (transitions[1], (a+1) % num_movement_actions),
                                 (transitions[2], (a+2) % num_movement_actions),
                                 (transitions[3], NOOP)]
                        li_dict = {}
                        for p, b in probs:
                            newrow, newcol = self._inc(row, col, b)
                            newletter = desc[newrow, newcol]
                            newstate = to_s(newrow, newcol) if newletter not in b'B' else s
                            if newstate in li_dict:
                                li_dict[newstate] += p
                            else:
                                li_dict[newstate] = p
                        li += [(li_dict[newstate], newstate, rew, False) for newstate in li_dict.keys()]

        self.adt_map = np.array(np.empty((num_states, num_actions, num_states)), dtype='object')
        for i in range(num_states):
            for j in range(num_actions):
                for k in range(num_states):
                    self.adt_map[i, j, k] = self.sas_to_adt(i, j, k)

        self.adt_mask = np.where(self.adt_map == None)
        self.adt_map[self.adt_mask] = [(0, 0, 0) for _ in range(len(self.adt_mask[0]))]

        super(MarsExplorerEnv, self).__init__(num_states, num_actions, P, isd)

        #TODO Doesn't need to be so hacky about letter types, can instead uncomment above section and take tile_transition_map
        self.adt_mat = self._get_adt_transition_matrix(transition_dict, [b'F', b'U'])



    def _step(self, a):
        transitions = self.transitions[self.state][a]
        i = categorical_sample([t[0] for t in transitions], self.np_rng)
        self.step_num += 1
        p, s, r, d = transitions[i]
        adt = self.sas_to_adt(self.state, a, s)
        self.state = s
        self.last_action = a
        return s, r, d, {"prob": p, "adt": adt, "trans": transitions}

    def s_to_grid(self, s):
        return s % self.ncol, s//self.ncol

    def get_direction_moved(self, s, sprime):
        c1, r1 = self.s_to_grid(s)
        c2, r2 = self.s_to_grid(sprime)
        # ROTATIONAL
        dx, dy = (c2 - c1) % self.ncol, (r2 - r1) % self.nrow
        return self.dir_dict[(dx, dy)] if (dx, dy) in self.dir_dict else None

    # Pretty hacky way of just getting all possible next primes, designed specifically for [Batch] or [Batch, 2]
    def get_possible_sprimes(self, s, grid=False):

        def to_s(grid_points):
            grid_points[:, :, 1] *= self.ncol
            return np.sum(grid_points, axis=2)

        s = np.array(s)
        if len(s.shape) == 1:
            s = np.transpose(np.vectorize(self.s_to_grid)(s))
        sprime = np.moveaxis(np.tile(s, (self.num_directions, 1, 1)), 0, 2)
        for i in range(DIRARRAY.shape[0]):
            sprime[:, :, i] += DIRARRAY[i]
        sprime[:, 0, :] %= self.ncol
        sprime[:, 1, :] %= self.nrow
        sprime = sprime.transpose((0, 2, 1))
        ret = to_s(sprime) if not grid else sprime

        return ret

    def get_reward(self, s, a, sprime=None):
        r = 0
        if sprime is None:
            for t in self.transitions[s][a]:
                r += t[0] * t[2]
        else:
            for t in self.transitions[s][a]:
                r = t[2] if t[1] == sprime else r
        return r

    def get_reward_matrix(self):
        R = np.zeros((self.num_states, self.num_actions))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                r = 0
                for t in self.transitions[s][a]:
                    r += t[0] * t[2]
                R[s, a] = r

    def get_tile_type(self, s):
        l = self.tile_map.flatten()[s]
        return 0 if l in b'SF123' else 1

    def sas_to_adt(self, s, a, sprime):
        d = self.get_direction_moved(s, sprime)
        t = self.get_tile_type(s)
        return (a, d, t) if d is not None else None

    def adt_trans_to_sas_trans(self, adt_trans):
        T = np.zeros(self.adt_map.shape)
        for i in range(self.adt_map.shape[0]):
            for j in range(self.adt_map.shape[1]):
                for k in range(self.adt_map.shape[2]):
                    T[i, j, k] = adt_trans[self.adt_map[i, j, k]]
        T[self.adt_mask] = 0.0
        return T

    def sd_to_sprime(self, s, d):
        # Make rotational instead to make dynamics consistent
        c, r = self.s_to_grid(s)
        row, col = self._inc(r, c, d)
        return row*self.ncol + col

    def sim_step(self, s, a):
        transitions = self.transitions[s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_rng)
        p, s, r, d = transitions[i]
        return s, r, d

    # Now rotational to make dynamics consistent
    def _inc(self, row, col, a):
        if a == 0:  # left
            col = (col - 1) % self.ncol
        elif a == 1:  # down
            row = (row + 1) % self.nrow
        elif a == 2:  # right
            col = (col + 1) % self.ncol
        elif a == 3:  # up
            row = (row - 1) % self.nrow
        return row, col

    def get_transition_matrix(self):
        """Return a matrix with index S,A,S' -> P(S'|S,A)"""
        if self.t_mat is not None:
            return self.t_mat

        transition_matrix = np.zeros([self.num_states, self.num_actions, self.num_states])

        for state in range(self.num_states):
            for action in range(self.num_actions):
                transition_distribution = self.transitions[state][action]

                probability_next_state_fixed_action_current_state = {transition[1]: transition[0]
                                                                     for transition in transition_distribution}

                for next_state in range(self.num_states):
                    if next_state in probability_next_state_fixed_action_current_state:
                        transition_matrix[state, action, next_state] = \
                            probability_next_state_fixed_action_current_state[next_state]

        self.t_mat = transition_matrix
        return self.t_mat

    def _get_adt_transition_matrix(self, transition_dict, ex_tiles):
        """Return a matrix with index T,A,D -> P(D|A,T)
            This disgusts me and we should just be constructing this matrix in the init fn
            Ideally this would be modular for any number of tile_types"""

        transition_matrix = np.zeros([self.tile_types, self.num_actions, self.num_directions])

        noop_trans = np.zeros(self.num_directions)
        noop_trans[-1] = 1.0
        for tt, letter in enumerate(ex_tiles):
            transitions = transition_dict[letter]
            for a in range(self.num_actions):
                if a == NOOP:
                    transition_matrix[tt, a, :] = noop_trans
                else:
                    transition_matrix[tt,a,a] = transitions[0]
                    transition_matrix[tt, a, (a - 1) % (self.num_actions-1)] = transitions[1]
                    transition_matrix[tt, a, (a + 1) % (self.num_actions-1)] = transitions[1]
                    transition_matrix[tt, a, (a + 2) % (self.num_actions-1)] = transitions[2]
                    transition_matrix[tt, a, NOOP] = transitions[3]

        return transition_matrix

class GridTwoHotEncoding(gym.Space):
    """
    {0,...,1,...,0,...,1,...,0}

    Example usage:
    self.observation_space = GridTwoHotEncoding(width=4, height=4)
    """

    def __init__(self, width=None, height=None):
        assert isinstance(width, int) and width > 0
        assert isinstance(height, int) and height > 0
        self.width = width
        self.height = height
        gym.Space.__init__(self, (), np.int64)

    def sample(self):
        one_hot_vector = np.zeros(self.width + self.height)
        one_hot_vector[np.random.randint(self.width)] = 1
        one_hot_vector[np.random.randint(self.height)+self.width] = 1
        return one_hot_vector

    def contains(self, x):
        if isinstance(x, (list, tuple, np.ndarray)):
            number_of_zeros = list(x).contains(0)
            number_of_ones = list(x).contains(1)
            return (number_of_zeros == (self.size - 2)) and (number_of_ones == 2)
        else:
            return False

    def __repr__(self):
        return "GridTwoHotEncoding(%d, %d)" % self.width, self.height

    def __eq__(self, other):
        return self.width == other.width and self.height == other.height



# class TwoHotMarsExplorerEnv(MarsExplorerEnv):
#     """
#     Todo
#     """
#
#     metadata = {'render.modes': ['human', 'ansi']}
#
#     def __init__(self, tile_map, reward_map, texture_map, transition_dict={}, time_penalty=-0.001, seed=None):
#
#         super(TwoHotMarsExplorerEnv, self).__init__(tile_map, reward_map, texture_map, transition_dict, time_penalty, seed)
#
#         self.observation_space = GridTwoHotEncoding(self.n)
