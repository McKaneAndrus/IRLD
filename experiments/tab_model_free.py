import numpy as np
from envs import MarsExplorerEnv
import matplotlib.pyplot as plt
import time
import matplotlib.cm as cm
from datetime import datetime
from scipy.special import logsumexp
from copy import deepcopy as copy
import random
np.set_printoptions(precision=8, suppress=True,threshold=np.nan)
# from sacred import Experiment
# from sacred.observers import FileStorageObserver
import os
import pickle as pkl
import json
from shutil import copy2
import sys


#
# tab_model_free_ex = Experiment("tab_model_free")
# tab_model_free_ex.observers.append(FileStorageObserver.create('logs/sacred'))

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
NOOP = 4

cdict = {'red':   ((0.0,  0.173, 0.173),
                   (1.0,  0.925, 0.925)),

         'green': ((0.0,  0.067, 0.067),
                   (1.0, 0.384, 0.384)),

         'blue':  ((0.0,  0.027, 0.027),
                   (1.0,  0.196, 0.196))}
plt.register_cmap(name='RustPlanet', data=cdict)
REWARD_COLORS = cm.get_cmap('RustPlanet')
AGENT_COLORS = cm.get_cmap('gray')
MAP_COLORS = {b'B':"#3a0e00",
              b'F':"#933111",
              b'S':"#933111",
              b'U':"#d65b33",
              b'1':"#956F52",
              b'2':"#3C2F34",
              b'3':"#644C42"}

ROVER_PNGS = {LEFT:"resources/rover_left.png",
        RIGHT:"resources/rover_right.png",
        DOWN:"resources/rover_down.png",
        UP:"resources/rover_up.png",
        NOOP:"resources/rover_sample.png"}

# tile_map = [
#         "F3FFUFU2U",
#         "FUFUSUFUF",
#         "FFFUSUFFU",
#         "USUUUUUSU",
#         "FFUSSSUFF",
#         "FSFUSUFSF",
#         "UUUUUUUFU",
#         "UFFU1UFUU",
#         "3UUFUFFU1"
#     ]

tile_map = [
        "F22222FF1",
        "U31122FU1",
        "UF1112FU1",
        "UFUU1SFU1",
        "UFFU1FFU3",
        "3222UFS1U",
        "UUU22F1FU",
        "FSSFS12SF",
        "FSFF121FF",
        "S1S2SFFSF",
        "FUUUFFSFU",
        "U3UUUFSFU",
        "FFFUUFUUU"
    ]


# tile_map = [
#         "2FFFUFFF1",
#         "FFFUUUFFF",
#         "FFFFUFFFF",
#         "FSFUUUFSF",
#         "FSUU3UUSF",
#         "FSFUUUFSF",
#         "FFFFUFFFF",
#         "FFFSSSFFF",
#         "1FFFFFFF2"
#     ]

# tile_map = [
#   "SFFFUFFFSFUUFFU",
#   "F1F2FSF2SFFFFFF",
#   "FFFUFF1FFF2FUUF",
#   "UUFFFFSFFFUSFFF",
#   "FFSFFSFUFFUSFFF",
#   "SFU2UUFFFFUFFFU",
#   "FSFFFFSFF2FF1FU",
#   "FFFFFFSUFSFFFFF",
#   "FFUFFSFFFFFUFFS",
#   "2FFFFSFFF1SFFUF",
#   "1UFU3UFFFUFFFFF",
#   "F3FUSFFFFFFFFFF",
#   "F1FFFFFFUFFF3UF",
#   "F1FSSFFFFUSFUUF",
#   "FFFFFFFUFFFSF2F",
# ]
#
# tile_map = [
#         "3FFFFFFF1",
#         "FFFFFFFFF",
#         "FFFFUFFFF",
#         "FSFUUUFSF",
#         "FSUU3UUSF",
#         "FSFUUUFSF",
#         "FFFFUFFFF",
#         "FFFSSSFFF",
#         "1FFFFFFF2"
#     ]

tile_rewards = {'F':0.0,
              '1':0.0,
              '2':0.0,
              '3':0.0,
              'S':0.0,
              'U':0.0}


goal_tile_rewards = {'F':0.0,
              '1':0.125,
              '2':0.25,
              '3':1.0,
              'S':0.0,
              'U':0.0}

tile_reward_modifier = lambda r,x,y,mx,my: r #* 0.1 * ((x-(mx/2 + np.random.normal(scale=0.5)))**2 + (y - (mx/2 + np.random.normal(scale=0.5)))**2)

def build_reward_map(tile_map, tile_rewards, goal_tile_rewards, tile_reward_modifier):
    reward_map = np.zeros((len(tile_map),len(tile_map[0])))
    texture_map = np.zeros((len(tile_map),len(tile_map[0])))
    for y,row in enumerate(tile_map):
        for x,c in enumerate(row):
            reward_map[y,x] = texture_map[y,x] = tile_reward_modifier(tile_rewards[c],x,y,len(tile_map[0]),len(tile_map))
            reward_map[y,x] +=  goal_tile_rewards[c]
    return reward_map, texture_map

reward_map, texture_map = build_reward_map(tile_map, tile_rewards, goal_tile_rewards, tile_reward_modifier)


t0 = (0.6,0.2,0.0,0.0)
t1 = (0.0,0.0,0.0,1.0)  #(0.1,0.15,0.5,0.1)

trans_dict = {b'F':t0,
              b'1':t0,
              b'2':t0,
              b'3':t0,
              b'S':t0,
              b'U':t1}

#TODO implement theta_mat in mars_explorer so do not need to hard code, use np.roll
true_tps = np.array([[[0.8, 0.],
  [0.1, 0.],
  [0.0, 0.],
  [0.1, 0.],
  [0.0, 1.]],

 [[0.1, 0.],
  [0.8, 0.],
  [0.1, 0.],
  [0.0, 0.],
  [0.0, 1.]],

 [[0.0, 0.],
  [0.1, 0.],
  [0.8, 0.],
  [0.1, 0.],
  [0.0, 1.]],

 [[0.1, 0.],
  [0.0, 0.],
  [0.1, 0.],
  [0.8, 0.],
  [0.0, 1.]],

 [[0., 0.],
  [0., 0.],
  [0., 0.],
  [0., 0.],
  [1., 1.]]])

gamma = 0.99

alpha = 0.0001

T_theta_shape = (5,5,2)

time_penalty = 0.0

tabsoftq_iter_ftol = 1e-32
tabsoftq_iter_maxiter = 5000
tabsoftq_iter_verbose = False
tabsoftq_grad_iter_ftol = 1e-9
tabsoftq_grad_iter_maxiter = 5000
tabsoftq_grad_iter_verbose = False
batch_size = 200
n_demos = 80
demo_time_steps = 40

mdp = MarsExplorerEnv(tile_map, reward_map, texture_map, trans_dict, time_penalty)
nA = mdp.num_actions
nS = mdp.num_states

def softmax(x, axis=1):
    """Compute softmax values for each sets of scores in x."""
    axis = 0 if len(x.shape) == 1 else axis
    e_x = np.exp(x - np.max(x, axis=axis,keepdims=True))
    return (e_x / e_x.sum(axis=axis, keepdims = True))

def tabsoftq_iter(R, T, maxiter=10000, verbose=False, Q_init=None, learning_rate=0.5, ftol=1e-32):
    Q = np.zeros((nS, nA)) if Q_init is None else copy(Q_init)
    prevQ = copy(Q)
    if verbose:
        diffs = []
        meanVs = []
    for iter_idx in range(maxiter):
        V = alpha * logsumexp(prevQ / alpha, axis=1)
        V_broad = V.reshape((1, 1, nS))
        Q = np.sum(T * (R + gamma * V_broad), axis=2)
        Q = (1 - learning_rate) * prevQ + learning_rate * Q
        diff = np.mean((Q - prevQ)**2)/(np.std(Q)**2)
        if verbose:
            diffs.append(diff)
            meanVs.append(np.mean(V))
        if diff < ftol:
            break
        prevQ = copy(Q)
    if verbose:
        plt.xlabel('Number of Iterations')
        plt.ylabel('Avg. Squared Bellman Error')
        plt.title('Soft Q Iteration')
        plt.plot(diffs)
        plt.yscale('log')
        plt.show()
        plt.xlabel('Number of Iterations')
        plt.ylabel('Avg. Value of All Tles')
        plt.title('Soft Q Iteration')
        plt.plot(meanVs)
        plt.show()
    return Q

def tabsoftq_learn_Qs(mdp):
    R = mdp.rewards
    T = mdp.get_transition_matrix()
    Qs = tabsoftq_iter(R, T)
    return Qs

def tabsoftq_gen_pol(Qs, beta=50):
    softQs = softmax(Qs * beta)
    return lambda s: np.random.choice(np.asarray(range(len(Qs[s]))),p=softQs[s])

def tabsoftq_gen_pol_probs(Qs, beta=50):
    softQs = softmax(Qs * beta)
    return softQs


def generate_demonstrations(mdp, pol, n, term):
    hists = []
    for i in range(n):
        s, d, t = mdp._reset(), False, 0
        hist = []
        while not d and t < term:
            a = pol[s] if type(pol) == np.ndarray else pol(s)
            sprime, rt, _, ob_dict = mdp._step(a)
            hist += [((s,a,sprime),ob_dict['adt'])]
            t += 1
            s = sprime
        hists += [hist]
    return hists


def tabsoftq_T_grad_iter(T_thetas, Q, R, T=None, maxiter=1000, verbose=True,
                         learning_rate=1, G_init=None, ftol=0):
    T_theta_dim = T_thetas.shape[0] * T_thetas.shape[1] * T_thetas.shape[2]
    D = T_thetas.shape[1]

    P_broad = tabsoftq_gen_pol_probs(Q).reshape((nS, nA, 1))
    Tps = softmax(T_thetas, axis=1)

    if T is None:
        T = mdp.adt_trans_to_sas_trans(Tps)

    V = (alpha) * logsumexp(Q * (1 / alpha), axis=1)

    T_grad = np.zeros((nS, nA, T_theta_dim))
    for s in range(nS):
        t = mdp.get_tile_type(s)
        for a in range(nA):
            P_at = Tps[a, :, t]
            V_theta = np.array([V[mdp.sd_to_sprime(s, d)] for d in range(D)])
            R_theta = np.array([R[mdp.sd_to_sprime(s, d)] for d in range(D)])
            VR_theta = R_theta + gamma * V_theta
            D_probs = np.stack([P_at for _ in range(D)])
            grad_at = np.dot((np.eye(D) - D_probs), VR_theta)
            grad_at = np.dot(np.diag(P_at), grad_at)
            filler = np.zeros(T_thetas.shape)
            filler[a, :, t] = grad_at
            filler = filler.flatten()
            T_grad[s, a] = filler
    G = T_grad if G_init is None else G_init
    T_broad = T.reshape((nS, nA, nS, 1))
    prevG = copy(G)
    diffs = []
    for iter_idx in range(maxiter):
        expG = np.sum(P_broad * G, axis=1)
        expG_broad = expG.reshape((1, 1, nS, T_theta_dim))
        G = gamma * T_grad
        t_expG = np.sum(T_broad * expG_broad, axis=2)
        G += gamma * t_expG
        G = (1 - learning_rate) * prevG + learning_rate * G

        diff = np.mean((G - prevG) ** 2) / (np.std(G) ** 2)
        diffs.append(diff)
        if diff < ftol:
            break
        prevG = copy(G)

    if verbose:
        plt.xlabel('Number of Iterations')
        plt.ylabel('Avg. Squared Bellman Error')
        plt.title('Soft Q Gradient Iteration')
        plt.plot(diffs)
        plt.yscale('log')
        plt.show()
    expG = np.sum(P_broad * G, axis=1)
    expG_broad = expG.reshape((nS, 1, T_theta_dim))
    return (G - expG_broad)

def vectorize_rollouts(rollouts):
    sas_obs = []
    adt_obs = []
    task_idxes = []
    for rollout in rollouts:
        more_sas, more_adt = list(zip(*rollout))
        sas_obs.extend(more_sas)
        adt_obs.extend(more_adt)
    return np.array(sas_obs), np.array(adt_obs)


def transition_grad(adt, tps):
    a,d,t = adt
    grad_theta_t = np.zeros(tps.shape)
    grad_theta_t[a,d,t] += 1
    grad_theta_t[a,:,t] -= tps[a,:,t]
    return grad_theta_t


def eval_pol_likelihood(Q, sas_obs, verbose=False):
    ll = 0.0
    for obs in sas_obs:
        s, a, sprime = obs
        l = np.log(softmax(Q[s])[a])
        ll += l
    return ll


def eval_trans_likelihood(Tps, adt_obs, verbose=False):
    ll = 0.0
    for obs in adt_obs:
        a, d, t, = obs
        l = np.log(Tps[a, d, t])
        ll += l
    return ll


def eval_trans_likelihood_and_grad(T_thetas, adt_obs):
    Tps = softmax(T_thetas, axis=1)
    dT = sum([transition_grad(adt, Tps) for adt in adt_obs]).reshape(T_thetas.shape) / len(adt_obs)
    ll = eval_trans_likelihood(Tps, adt_obs)
    return ll, dT


def eval_demo_log_likelihood(sas_obs, adt_obs, T_thetas, Q):
    Tps = softmax(T_thetas, axis=1)
    p_ll = eval_pol_likelihood(Q, sas_obs)
    t_ll = eval_trans_likelihood(Tps, adt_obs)
    return p_ll, t_ll


def eval_T_pol_likelihood_and_grad(T_thetas, R, sas_obs, Q_inits=None, verbose=False):
    s = [obs[0] for obs in sas_obs]
    a = [obs[1] for obs in sas_obs]
    Tps = softmax(T_thetas,axis=1)
    T = mdp.adt_trans_to_sas_trans(Tps)
    Q = tabsoftq_iter(R, T, Q_init=Q_inits, maxiter=tabsoftq_iter_maxiter, verbose=tabsoftq_iter_verbose,
                      ftol=tabsoftq_iter_ftol)
    print(np.max(Q), np.min(Q))
    dT = tabsoftq_T_grad_iter(T_thetas, Q, R, T=T, maxiter=tabsoftq_grad_iter_maxiter,
        verbose=tabsoftq_grad_iter_verbose, ftol=tabsoftq_grad_iter_ftol)
    # Sum instead of mean because sparse results
    if verbose:
        for obs in sas_obs:
            print(obs)
            print(dT[obs[0],obs[1]].reshape(T_thetas.shape))
    dT = np.sum(dT[s,a], axis=0).reshape(T_thetas.shape) / len(sas_obs)
    ll = eval_pol_likelihood(Q, sas_obs)
    return ll, dT, Q


def tabsoftq_TR_grad_iter(T_thetas, feat_map, R, Q, T=None, maxiter=1000, verbose=True,
                          learning_rate=1, G_init=None, ftol=0.0):
    T_theta_dim = T_thetas.shape[0] * T_thetas.shape[1] * T_thetas.shape[2]
    D = T_thetas.shape[1]

    R_theta_dim = feat_map.shape[1]

    P_broad = tabsoftq_gen_pol_probs(Q).reshape((nS, nA, 1))
    Tps = softmax(T_thetas, axis=1)

    if T is None:
        T = mdp.adt_trans_to_sas_trans(Tps)

    V = (alpha) * logsumexp(Q * (1 / alpha), axis=1)

    R_grad = T.dot(feat_map)

    GR = np.zeros((nS, nA, R_theta_dim)) if G_init is None else G_init[0]
    prevGR = copy(GR)

    T_grad = np.zeros((nS, nA, T_theta_dim))
    for s in range(nS):
        t = mdp.get_tile_type(s)
        for a in range(nA):
            P_at = Tps[a, :, t]
            V_t = np.array([V[mdp.sd_to_sprime(s, d)] for d in range(D)])
            R_t = np.array([R[mdp.sd_to_sprime(s, d)] for d in range(D)])
            VR_t = R_t + gamma * V_t
            D_probs = np.stack([P_at for _ in range(D)])
            grad_at = np.dot((np.eye(D) - D_probs), VR_t)
            grad_at = np.dot(np.diag(P_at), grad_at)
            filler = np.zeros(T_thetas.shape)
            filler[a, :, t] = grad_at
            filler = filler.flatten()
            T_grad[s, a] = filler

    GT = T_grad if G_init is None else G_init[1]
    T_broad = T.reshape((nS, nA, nS, 1))
    prevGT = copy(GT)

    if verbose:
        diffs = []

    for iter_idx in range(maxiter):
        # Reward Param gradient iteration
        expGR = np.sum(P_broad * GR, axis=1)
        expGR_broad = expGR.reshape((1, 1, nS, R_theta_dim))
        GR = R_grad + gamma * np.sum(T_broad * expGR_broad, axis=2)
        GR = (1 - learning_rate) * prevGR + learning_rate * GR

        # Transition Param grad iter
        expGT = np.sum(P_broad * GT, axis=1)
        expGT_broad = expGT.reshape((1, 1, nS, T_theta_dim))
        GT = gamma * T_grad
        t_expGT = np.sum(T_broad * expGT_broad, axis=2)
        GT += gamma * t_expGT
        GT = (1 - learning_rate) * prevGT + learning_rate * GT

        diff = np.mean((GR - prevGR) ** 2) / (np.std(GR) ** 2) + np.mean((GT - prevGT) ** 2) / (np.std(GT) ** 2)
        if verbose:
            diffs.append(diff)
        if diff < ftol:
            break
        prevGR = copy(GR)
        prevGT = copy(GT)

    if verbose:
        plt.xlabel('Number of Iterations')
        plt.ylabel('Avg. Squared Bellman Error')
        plt.title('Soft Q Gradient Iteration')
        plt.plot(diffs)
        plt.yscale('log')
        plt.show()

    expGR = np.sum(P_broad * GR, axis=1)
    expGR_broad = expGR.reshape((nS, 1, R_theta_dim))
    expGT = np.sum(P_broad * GT, axis=1)
    expGT_broad = expGT.reshape((nS, 1, T_theta_dim))
    return (GR - expGR_broad), (GT - expGT_broad)

def eval_TR_pol_likelihood_and_grad(T_thetas, R, feat_map, sas_obs, Q_inits=None, verbose=False):
    s = [obs[0] for obs in sas_obs]
    a = [obs[1] for obs in sas_obs]
    Tps = softmax(T_thetas,axis=1)
    T = mdp.adt_trans_to_sas_trans(Tps)
    Q = tabsoftq_iter(R, T, Q_init=Q_inits if Q_inits is not None else None,
        maxiter=tabsoftq_iter_maxiter, verbose=tabsoftq_iter_verbose, ftol=tabsoftq_iter_ftol)
    dR, dT = tabsoftq_TR_grad_iter(T_thetas, feat_map, R, Q, T=T,
        maxiter=tabsoftq_grad_iter_maxiter, verbose=tabsoftq_grad_iter_verbose, ftol=tabsoftq_grad_iter_ftol)
    if verbose:
        for obs in sas_obs:
            print(obs)
            print(dT[obs[0],obs[1]].reshape(T_thetas.shape))
    dR = np.sum(dR[s,a], axis=0).reshape(feat_map.shape[1]) / len(sas_obs)
    dT = np.sum(dT[s,a], axis=0).reshape(T_thetas.shape) / len(sas_obs)
    ll = eval_pol_likelihood(Q, sas_obs)
    return ll, dT, dR, Q

def clean_demos(sas_obs, adt_obs, max_noops=50):
    demo_example_idxes = list(range(len(sas_obs)))
    unique_stays, stay_count = set([]), 0
    for i,(sas,adt) in enumerate(zip(sas_obs,adt_obs)):
        sas = tuple(sas)
        if adt[2] == 1:
            demo_example_idxes.remove(i)
            continue
        if sas[1] == 4:
            stay_count += 1
            if sas in unique_stays:
                if max_noops and stay_count > max_noops:
                    demo_example_idxes.remove(i)
            else:
                unique_stays.add(sas)
        else:
            stay_count = 0

    return demo_example_idxes

def sample_batch(size, ids, sas_obs, adt_obs):
    idxes = random.sample(ids, size)
    return sas_obs[idxes], adt_obs[idxes]

def T_estimate(adt_obs):
    stability = 1e-5
    T_thetas = np.zeros(T_theta_shape)
    T_counts = np.zeros(T_theta_shape) + stability
    for a,d,t in adt_obs:
        T_counts[a,d,t] += 1
    for a in range(T_theta_shape[0]):
        for t in range(T_theta_shape[2]):
            z = np.sum(T_counts[a,:,t])
            for d in range(T_theta_shape[1]):
                if z == T_counts.shape[1] * stability:
                    T_thetas[a,d,t] = 0
                else:
                    T_thetas[a,d,t] = np.log(T_counts[a,d,t]) - np.log(z)
    return T_thetas

def test_T_likelihood(Tps, sas_obs, adt_obs):
    T = mdp.adt_trans_to_sas_trans(Tps)
    R = mdp.rewards
    Q = tabsoftq_iter(R, T, Q_init=None, maxiter=tabsoftq_iter_maxiter, verbose=tabsoftq_iter_verbose, ftol=tabsoftq_iter_ftol)
    pl = eval_pol_likelihood(Q, sas_obs, verbose=True)
    tl = eval_trans_likelihood(Tps, adt_obs, verbose=True)
    print([s for s in range(nS) if mdp.get_tile_type(s)==1])
    print(Q[[s for s in range(nS) if mdp.get_tile_type(s)==1]])
    print(pl, tl)
    return pl + tl

def true_trans_loss(tps):
    tps = mdp.adt_trans_to_sas_trans(tps)
    true_tps = mdp.get_transition_matrix()
    return np.linalg.norm(true_tps-tps)/nA



learning_rate = 5.0

#DEMO Config
n_demos = 200
demo_time_steps = 40

#tab Config
batch_size = 256
n_training_iters = 500

tab_save_freq = 5

transition_likelihood_weight = 1.0

serd=True
verbose = True


if __name__ == "__main__":


    seed = int(sys.argv[1])
    print(seed)

    random.seed(seed)
    np.random.seed(seed)
    # q_scope, dyn_scope = load_scopes(data_dir)

    # now= datetime.now()
    label = "tab_" + str(round(time.time()))

    print("Starting Experiment " + label)

    out_dir = os.path.join("logs", "models", label)
    sacred_dir = os.path.join("logs", "sacred", label)

    metric_names = ["val_likelihoods", "val_nall", "val_ntll"]
    metrics = {key: {"steps" : [], "values" : []} for key in metric_names}

    exQs = tabsoftq_learn_Qs(mdp)
    demos = generate_demonstrations(mdp, tabsoftq_gen_pol(exQs), n_demos, demo_time_steps)
    sas_obs, adt_obs = vectorize_rollouts(demos)


    tab_model_out_dir = os.path.join(out_dir, "tab")
    if not os.path.exists(tab_model_out_dir):
        os.makedirs(tab_model_out_dir)
    pkl.dump(exQs, open(os.path.join(tab_model_out_dir, 'true_q_vals.pkl'), 'wb'))
    pkl.dump(mdp.adt_mat, open(os.path.join(tab_model_out_dir, 'true_adt_probs.pkl'), 'wb'))


    # Clean training set to have greater density of interesting transitions
    print(len(sas_obs))
    demo_example_idxes = clean_demos(sas_obs, adt_obs)
    print(len(demo_example_idxes))
    # demo_example_idxes = list(range(len(sas_obs)))


    random.shuffle(demo_example_idxes)
    n_train_demo_examples = int(0.9 * len(demo_example_idxes))
    train_demo_example_idxes = demo_example_idxes[:n_train_demo_examples]
    val_demo_example_idxes = demo_example_idxes[n_train_demo_examples:]
    val_sas_obs = sas_obs[val_demo_example_idxes]
    val_adt_obs = adt_obs[val_demo_example_idxes]

    train_time = 0
    T_thetas = T_estimate(adt_obs[train_demo_example_idxes])
    Q = None

    if serd:
        feats = mdp.feature_map
        R_thetas = np.random.normal(loc=1, scale=0.1, size=feats.shape[1])
    else:
        R = mdp.rewards

    try:
        for train_time in range(n_training_iters):
            batch_demo_sas, batch_demo_adt = sample_batch(batch_size, train_demo_example_idxes, sas_obs, adt_obs)

            if serd:
                R = feats.dot(R_thetas)
                tp_ll, dT_pol, dR_pol, Q = eval_TR_pol_likelihood_and_grad(T_thetas, R, feats, batch_demo_sas,Q_inits=Q)
                R_thetas += learning_rate * dR_pol
            else:
                # Should we initialize Q or nah?
                tp_ll, dT_pol, Q = eval_T_pol_likelihood_and_grad(T_thetas, R, batch_demo_sas, Q_inits=Q, verbose=False)

            # Should we initialize Qs or nah?
            tt_ll, dT_trans = eval_trans_likelihood_and_grad(T_thetas, batch_demo_adt)
            vp_ll, vt_ll = eval_demo_log_likelihood(val_sas_obs, val_adt_obs, T_thetas, Q)
            val_likelihood = vp_ll + vt_ll
            T_thetas += learning_rate * (transition_likelihood_weight * dT_trans + dT_pol)

            #         print(np.max(dT_pol),np.min(dT_pol))


            metrics["val_likelihoods"]["steps"] += [train_time]
            metrics["val_likelihoods"]["values"] += [val_likelihood]
            metrics["val_nall"]["steps"] += [train_time]
            metrics["val_nall"]["values"] += [-vp_ll]
            metrics["val_ntll"]["steps"] += [train_time]
            metrics["val_ntll"]["values"] += [-vt_ll]

            if train_time % tab_save_freq == 0:
                if verbose:
                    print(str(train_time) + "\t" + "\t".join(['action ll' + ": " + str(round(-vp_ll, 7)),
                                                              'transition ll' + ": " + str(round(-vt_ll, 7)),
                                                              'total ll' + ": " + str(round(-val_likelihood, 7))]))

                print(train_time, np.max(Q), np.min(Q))
                adt_probs = softmax(T_thetas).transpose((2, 0, 1))
                print(adt_probs)
                pkl.dump(Q, open(os.path.join(tab_model_out_dir, 'q_vals_{}.pkl'.format(train_time)), 'wb'))
                pkl.dump(adt_probs,
                         open(os.path.join(tab_model_out_dir, 'adt_probs_{}.pkl'.format(train_time)), 'wb'))

    except KeyboardInterrupt:
        print("Experiment Interrupted at timestep {}".format(train_time))
        pass

            # Tabular logging setup

    # Save as file
    adt_probs = softmax(T_thetas).transpose((2, 0, 1))
    pkl.dump(Q, open(os.path.join(tab_model_out_dir, 'final_q_vals.pkl'), 'wb'))
    pkl.dump(adt_probs, open(os.path.join(tab_model_out_dir, 'final_adt_probs.pkl'), 'wb'))

    pkl.dump(mdp, open(os.path.join(out_dir, 'mdp.pkl'), 'wb'))

    if not os.path.exists(sacred_dir):
        os.makedirs(sacred_dir)

    with open(sacred_dir + '/metrics.json', 'w') as outfile:
        json.dump(metrics, outfile)


    copy2("experiments/tab_model_free.py", sacred_dir)

    print("Ending Experiment " + label)




