import numpy as np
from .discrete_env import categorical_sample, DiscreteEnv



LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
NOOP = 4
EXIT = 5

# (x,y)
DIRDICT = {(-1,0):0,
           (0,1):1,
           (1,0):2,
           (0,-1):3,
           (0,0):4}

# (x, y)
DIRARRAY = np.array([[-1,0],
		   [0,1],
           [1,0],
           [0,-1],
           [0,0]])


MAPS = {
    "9x9_multigoal": [
        "F2FFUFF1F",
        "FFFBFBFFF",
        "FFFFFFFFF",
        "BFBBBBBFB",
        "FFFFSFFFF",
        "FFFFFFFFF",
        "FFBBBBBFF",
        "FUFB3BFUF",
        "1FFFFFFF2"
    ] ,
    "5x5_multigoal": [
        "1FFF2",
        "FBFUF",
        "1B2BF",
        "FFFUF",
        "SFFB3"
    ],
    "5x5_contested": [
        "1UUU2",
        "UBUUU",
        "1B2BU",
        "UUUUU",
        "SUUB3"
    ],
    "4x4_multigoal": [
        "FFFF",
        "2B1U",
        "FUFF",
        "SFB3"
    ],
    "6x6_v1": [
        "1FUUU3",
        "FFFFFF",
        "BFBUBF",
        "1FUBF2",
        "FUFUUU",
        "FSUBF2"
    ],
    "6x6_v2": [
        "1FUUUF",
        "FUFB3F",
        "FFFSBF",
        "UFFBF1",
        "2FUFFF",
        "FFUBF2"
    ],
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

    def __init__(self, tile_map, reward_dict, texture_map, transition_dict={}, time_penalty=-0.001, seed=None):
        """

        """

        self.tile_map = desc = np.asarray(tile_map, dtype='c')
        self.texture_map = texture_map.flatten()
        # self.rewards = np.asarray(reward_map).flatten()
        self.rewards = reward_dict
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.penalty = time_penalty
        self.potential_goal_states = []
        self.step_num = 0
        self.t_mat, self.theta_mat = None, None
        self.tile_types = 2 # Hard coded but shouldn't necessarily be. Should include a "Tile map"


        # Rotational dir_dict
        self.dir_dict = {(self.ncol - 1,0):0,
           (0,1):1,
           (1,0):2,
           (0,self.nrow-1):3,
           (0,0):4}


        nA = 6
        self.nmA = nmA = 4
        self.end_state = nrow * ncol
        nS = self.end_state + 1
        self.nD = 6

        # flat_desc = desc.flatten()
        # nonGoalChars = b'BFUS'
        # # goalChars = [key for key in transition_dict.keys() if key not in nonGoalChars]
        # goalIndexes = []
        # for i, char in enumerate(flat_desc):
        #     if char not in nonGoalChars:
        #         goalIndexes += [i]

        # nGoalTypes = len(goalChars)
        # self.feature_map = np.zeros((nS, nGoalTypes + 1))
        # self.feature_map[:,0] = texture_map.flatten()
        # self.feature_map[goalIndexes, 0] = 0.0

        # TODO include tile map such that this is a feasible thing to construct
        # self.thetas = np.zeros((nA, nA, self.tile_types))
        # for t in range(self.tile_types):
        #     self.thetas[0,:,t] =

        # for s in goalIndexes:
        #     char = flat_desc[s]
        #     feat_idx = int(char)
        #     self.feature_map[s, feat_idx] = 1



        def to_s(row, col):
            return row*ncol + col


        if seed is not None:
            self._seed(seed)

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}


        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(nA):
                    li = P[s][a]
                    letter = desc[row, col]
                    rew = self.penalty + self.texture_map[s]
                    if a == EXIT:
                        li.append((1.0, self.end_state, rew + reward_dict[letter], True))
                    elif a == NOOP:
                        li.append((1.0, s, rew, False))
                    else:
                        transitions = transition_dict[letter]
                        probs = [(transitions[0], a),
                                 (transitions[1],(a-1)%nmA),
                                 (transitions[1],(a+1)%nmA),
                                 (transitions[2], (a + 2) % nmA),
                                 (transitions[3], NOOP)]
                        li_dict = {}
                        for p,b in probs:
                            newrow, newcol = self.inc(row, col, b)
                            newletter = desc[newrow, newcol]
                            newstate = to_s(newrow, newcol) if newletter not in b'B' else s
                            if newstate in li_dict:
                                li_dict[newstate] += p
                            else:
                                li_dict[newstate] = p
                        li += [(li_dict[newstate], newstate, rew, False) for newstate in li_dict.keys()]

        s = self.end_state
        for a in range(nA):
            li = P[s][a]
            li.append((1.0, s, 0, True))

        self.adt_map = np.array(np.empty((nS,nA,nS)), dtype='object')
        for i in range(nS):
            for j in range(nA):
                for k in range(nS):
                    self.adt_map[i,j,k] = self.sas_to_adt(i,j,k)

        self.adt_mask = np.where(self.adt_map == None)
        self.adt_map[self.adt_mask] = [(0,0,0) for _ in range(len(self.adt_mask[0]))]


        super(MarsExplorerEnv, self).__init__(nS, nA, P, isd)


    def _step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        self.step_num += 1
        p, s, r, d = transitions[i]
        adt = self.sas_to_adt(self.s,a,s)
        self.s = s
        self.lastaction = a
        return (s, r, d, {"prob": p, "adt":adt, "trans":transitions})

    def s_to_grid(self, s):
        if s == self.end_state:
            return self.ncol * 100, self.ncol * 100
        return s%self.ncol, s//self.ncol

    def get_direction_moved(self, s, sprime):
        if sprime == self.end_state:
            return self.nD - 1
        c1,r1 = self.s_to_grid(s)
        c2,r2 = self.s_to_grid(sprime)
        # ROTATIONAL
        dx,dy = (c2 - c1) % self.ncol, (r2 - r1) % self.nrow
        return self.dir_dict[(dx,dy)] if (dx,dy) in self.dir_dict else None

    def get_possible_sprimes(self, s):
        s = np.array(s)
        if s.shape[1] == 1:
            s = np.vectorize(self.s_to_grid)(s)
        sprime = np.moveaxis(np.tile(s, (self.nD,1,1)), 0, 2)

        for i in range(DIRARRAY.shape[0]):
            sprime[:,:,i] += DIRARRAY[i]
        sprime[:,0,:] %= self.ncol
        sprime[:,1,:] %= self.nrow

        end_s = self.s_to_grid(self.end_state)
        end_s_indexes = np.where((s == np.array(end_s)).all(axis=1))
        if len(end_s_indexes) > 0:
            sprime[end_s_indexes, 0, :] = end_s[0]
            sprime[end_s_indexes, 1, :] = end_s[1]
        sprime[:,0,self.nD-1] = end_s[0]
        sprime[:,1,self.nD-1] = end_s[1]
        return sprime.transpose((0,2,1))



    def get_reward(self, s, a, sprime=None):
        r = 0
        if sprime is None:
            for t in self.P[s][a]:
                r += t[0] * t[2]
        else:
            for t in self.P[s][a]:
                r = t[2] if t[1] == sprime else r
        return r

    def get_reward_matrix(self):
        R = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            for a in range(self.nA):
                r = 0
                for t in self.P[s][a]:
                    r += t[0] * t[2]
                R[s,a] = r
        return R


    def get_tile_type(self, s):
        if s == self.end_state:
            return 2
        l = self.tile_map.flatten()[s]
        return 0 if l in b'SF123' else 1

    def sas_to_adt(self,s,a,sprime):
        d = self.get_direction_moved(s,sprime)
        t = self.get_tile_type(s)
        return (a,d,t) if d is not None else None

    def adt_trans_to_sas_trans(self, adt_trans):
        T = np.zeros(self.adt_map.shape)
        for i in range(self.adt_map.shape[0]):
            for j in range(self.adt_map.shape[1]):
                for k in range(self.adt_map.shape[2]):
                    T[i,j,k] = adt_trans[self.adt_map[i,j,k]]
        T[self.adt_mask] = 0.0
        return T

    def sd_to_sprime(self,s, d):
        # Make rotational instead to make dynamics consistent
        c,r = self.s_to_grid(s)
        row,col = self.inc(r,c,d)
        return row*self.ncol + col


    def sim_step(self, s, a):
        transitions = self.P[s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        return (s, r, d)

    # Now rotational to make dynamics consistent
    def inc(self, row, col, a):
        if a == 0:  # left
            col = (col - 1) % self.ncol
        elif a == 1:  # down
            row = (row + 1) % self.nrow
        elif a == 2:  # right
            col = (col + 1) % self.ncol
        elif a == 3:  # up
            row = (row - 1) % self.nrow
        return (row, col)

    def get_transition_matrix(self):
        """Return a matrix with index S,A,S' -> P(S'|S,A)"""
        if self.t_mat is None:
            T = np.zeros([self.nS, self.nA, self.nS])
            for s in range(self.nS):
                for a in range(self.nA):
                    transitions = self.P[s][a]
                    s_a_s = {t[1]:t[0] for t in transitions}
                    for s_prime in range(self.nS):
                        if s_prime in s_a_s:
                            T[s, a, s_prime] = s_a_s[s_prime]
            self.t_mat = T
        return self.t_mat


    



