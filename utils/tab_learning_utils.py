from copy import deepcopy as copy
from scipy.special import logsumexp
import numpy as np
from utils.learning_utils import softmax
import random



#### Soft Q-Learning

def tabsoftq_iter(R, T, gamma, Q_init, boltzman=10000, maxiter=10000, learning_rate=0.5, ftol=1e-32):
    Q = copy(Q_init)
    if len(R.shape) == 1:
        R = np.repeat(R[np.newaxis].T, Q.shape[1], axis=1)
    prevQ = copy(Q)
    for iter_idx in range(maxiter):
        V = logsumexp(prevQ * boltzman, axis=1) / boltzman
        V_broad = V.reshape((1, 1, V.shape[0]))
        Q = R + gamma * np.sum(T * V_broad, axis=2)
        Q = (1 - learning_rate) * prevQ + learning_rate * Q
        diff = np.mean((Q - prevQ)**2)/(np.std(Q)**2)
        if diff < ftol:
            break
        prevQ = copy(Q)
    return Q

def tabsoftq_learn_Qs(mdp, gamma=0.95):
    # R = np.repeat(mdp.rewards[np.newaxis].T, mdp.num_actions, axis=1)
    R = mdp.rewards
    T = mdp.get_transition_matrix()
    Q_init = np.zeros((mdp.num_states, mdp.num_actions))
    Qs = tabsoftq_iter(R, T, gamma, Q_init)
    return Qs

def tabsoftq_gen_pol(Qs, beta=1):
    softQs = softmax(Qs * beta)
    return lambda s: np.random.choice(np.asarray(range(len(Qs[s]))), p=softQs[s])


def tabsoftq_gen_pol_probs(Qs, beta=1):
    softQs = softmax(Qs * beta)
    return softQs

#### Generating demos with Soft-Q learner


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


def vectorize_rollouts(rollouts):
    sas_obs = []
    adt_obs = []
    for rollout in rollouts:
        more_sas, more_adt = list(zip(*rollout))
        sas_obs.extend(more_sas)
        adt_obs.extend(more_adt)
    return np.array(sas_obs), np.array(adt_obs)

def sample_tab_batch(size, sas_obs, adt_obs):
    idxes = random.sample(list(range(len(sas_obs))), size)
    return sas_obs[idxes], adt_obs[idxes]


#### Soft Q Gradient Imitation Learning


def transition_grad(adt, tps):
    a,d,t = adt
    grad_theta_t = np.zeros(tps.shape)
    grad_theta_t[a,d,t] += 1
    grad_theta_t[a,:,t] -= tps[a,:,t]
    return grad_theta_t


def tabsoftq_T_grad_iter(mdp, T_thetas, Q, R, gamma, boltzmann=10000.0, maxiter=5000, learning_rate=1.0,
                         G_init=None, ftol=1e-10):

    nS, nA = Q.shape

    T_theta_dim = T_thetas.shape[0] * T_thetas.shape[1] * T_thetas.shape[2]
    D = T_thetas.shape[1]

    P_broad = tabsoftq_gen_pol_probs(Q).reshape((nS, nA, 1))
    Tps = softmax(T_thetas, axis=1)

    T = mdp.adt_trans_to_sas_trans(Tps)

    V = logsumexp(Q * boltzmann, axis=1) / boltzmann
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
        diffs += [diff]
        if diff < ftol:
            break
        prevG = copy(G)

    expG = np.sum(P_broad * G, axis=1)
    expG_broad = expG.reshape((nS, 1, T_theta_dim))
    return G - expG_broad


def tabsoftq_TR_grad_iter(mdp, T_thetas, Q, R, gamma, boltzmann=10000, maxiter=5000, learning_rate=1,
                          G_init=None, ftol=1e-10):

    nS, nA = Q.shape

    T_theta_dim = T_thetas.shape[0] * T_thetas.shape[1] * T_thetas.shape[2]
    D = T_thetas.shape[1]
    R_theta_dim = mdp.feature_map.shape[1]


    P_broad = tabsoftq_gen_pol_probs(Q).reshape((nS, nA, 1))
    Tps = softmax(T_thetas, axis=1)

    T = mdp.adt_trans_to_sas_trans(Tps)

    V = logsumexp(Q * boltzmann, axis=1) / boltzmann

    R_grad = T.dot(mdp.feature_map)
    T_grad = np.zeros((nS, nA, T_theta_dim))
    GR = np.zeros((nS, nA, R_theta_dim)) if G_init is None else G_init[0]
    prevGR = copy(GR)

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

        if diff < ftol:
            break
        prevGR = copy(GR)
        prevGT = copy(GT)

    expGR = np.sum(P_broad * GR, axis=1)
    expGR_broad = expGR.reshape((nS, 1, R_theta_dim))
    expGT = np.sum(P_broad * GT, axis=1)
    expGT_broad = expGT.reshape((nS, 1, T_theta_dim))
    return (GR - expGR_broad), (GT - expGT_broad)


def eval_pol_likelihood(Q, sas_obs):
    ll = 0.0
    for obs in sas_obs:
        s, a, sprime = obs
        l = np.log(softmax(Q[s])[a])
        ll += l
    return ll


def eval_trans_likelihood(Tps, adt_obs):
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


def eval_T_pol_likelihood_and_grad(mdp, T_thetas, R, sas_obs, gamma, Q_inits=None):
    s = [obs[0] for obs in sas_obs]
    a = [obs[1] for obs in sas_obs]
    Tps = softmax(T_thetas, axis=1)
    T = mdp.adt_trans_to_sas_trans(Tps)
    if Q_inits is None:
        Q_inits = np.zeros((mdp.num_states, mdp.num_actions))
    Q = tabsoftq_iter(R, T, gamma, Q_init=Q_inits)
    dT = tabsoftq_T_grad_iter(mdp, T_thetas, Q, R, gamma)
    # Sum instead of mean because sparse results
    dT = np.sum(dT[s,a], axis=0).reshape(T_thetas.shape) / len(sas_obs)
    ll = eval_pol_likelihood(Q, sas_obs)
    return ll, dT, Q

def eval_TR_pol_likelihood_and_grad(mdp, T_thetas, R, sas_obs, gamma, Q_inits=None):
    s = [obs[0] for obs in sas_obs]
    a = [obs[1] for obs in sas_obs]
    Tps = softmax(T_thetas,axis=1)
    T = mdp.adt_trans_to_sas_trans(Tps)
    if Q_inits is None:
        Q_inits = np.zeros((mdp.num_states, mdp.num_actions))
    Q = tabsoftq_iter(R, T, gamma, Q_init=Q_inits)
    dR, dT = tabsoftq_TR_grad_iter(mdp, T_thetas, Q, R, gamma)
    dR = np.sum(dR[s,a], axis=0).reshape(mdp.feature_map.shape[1]) / len(sas_obs)
    dT = np.sum(dT[s,a], axis=0).reshape(T_thetas.shape) / len(sas_obs)
    ll = eval_pol_likelihood(Q, sas_obs)
    return ll, dT, dR, Q


def get_T_tab_shape(mdp):
    return (mdp.num_actions, mdp.num_directions, mdp.tile_types)


def T_estimate(mdp, adt_obs, epsilon=1e-20):
    T_theta_shape = get_T_tab_shape(mdp)
    T_thetas = np.zeros(T_theta_shape)
    T_counts = np.zeros(T_theta_shape)
    for a,d,t in adt_obs:
        T_counts[a,d,t] += 1
    for a in range(T_theta_shape[0]):
        for t in range(T_theta_shape[2]):
            z = np.sum(T_counts[a,:,t])
            for d in range(T_theta_shape[1]):
                if z == 0:
                    T_thetas[a,d,t] = 0
                else:
                    T_thetas[a,d,t] = np.log(T_counts[a,d,t] + epsilon) - np.log(z + epsilon)
    return T_thetas


def test_T_likelihood(mdp, Tps, sas_obs, adt_obs):
    T = mdp.adt_trans_to_sas_trans(Tps)
    R = mdp.rewards
    Q = tabsoftq_iter(R, T, Q_init=None)
    pl = eval_pol_likelihood(Q, sas_obs)
    tl = eval_trans_likelihood(Tps, adt_obs)
    # print([s for s in range(mdp.num_states) if mdp.get_tile_type(s)==1])
    # print(Q[[s for s in range(mdp.num_states) if mdp.get_tile_type(s)==1]])
    # print(pl, tl)
    return pl + tl
