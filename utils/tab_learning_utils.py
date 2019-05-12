from copy import deepcopy as copy
from scipy.special import logsumexp
import numpy as np
from utils.learning_utils import softmax



def tabsoftq_iter(R, T, gamma, Q_init, boltzman=10000, maxiter=10000, learning_rate=0.5, ftol=1e-32):
    Q = copy(Q_init)
    prevQ = copy(Q)
    for iter_idx in range(maxiter):
        V = logsumexp(prevQ * boltzman, axis=1) / boltzman
        V_broad = V.reshape((1, 1, V.shape[0]))
        Q = np.sum(T * (R + gamma * V_broad), axis=2)
        Q = (1 - learning_rate) * prevQ + learning_rate * Q
        diff = np.mean((Q - prevQ)**2)/(np.std(Q)**2)
        if diff < ftol:
            break
        prevQ = copy(Q)
    return Q

def tabsoftq_learn_Qs(mdp, gamma=0.95):
    R = np.repeat(mdp.rewards[np.newaxis].T, mdp.num_actions, axis=1)
    T = mdp.get_transition_matrix()
    Q_init = np.zeros_like((mdp.num_states, mdp.num_actions))
    Qs = tabsoftq_iter(R, T, gamma, Q_init)
    return Qs

def tabsoftq_gen_pol(Qs, beta=1):
    softQs = softmax(Qs * beta)
    return lambda s: np.random.choice(np.asarray(range(len(Qs[s]))), p=softQs[s])


def tabsoftq_gen_pol_probs(Qs, beta=1):
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


def vectorize_rollouts(rollouts):
    sas_obs = []
    adt_obs = []

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


def tabsoftq_T_grad_iter(mdp, T_thetas, Q, R, gamma, boltzmann=10000, maxiter=5000, learning_rate=1,
                         G_init=None, ftol=1e-10):

    nS, nA = Q.shape

    T_theta_dim = T_thetas.shape[0] * T_thetas.shape[1] * T_thetas.shape[2]
    D = T_thetas.shape[1]

    P_broad = tabsoftq_gen_pol_probs(Q).reshape((nS, nA, 1))
    Tps = softmax(T_thetas, axis=1)

    T = mdp.adt_trans_to_sas_trans(Tps)

    V = logsumexp(Q * boltzmann, axis=1) * boltzmann
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
    for iter_idx in range(maxiter):
        expG = np.sum(P_broad * G, axis=1)
        expG_broad = expG.reshape((1, 1, nS, T_theta_dim))
        G = gamma * T_grad
        t_expG = np.sum(T_broad * expG_broad, axis=2)
        G += gamma * t_expG
        G = (1 - learning_rate) * prevG + learning_rate * G

        diff = np.mean((G - prevG) ** 2) / (np.std(G) ** 2)
        if diff < ftol:
            break
        prevG = copy(G)

    expG = np.sum(P_broad * G, axis=1)
    expG_broad = expG.reshape((nS, 1, T_theta_dim))
    return G - expG_broad


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
    Q = tabsoftq_iter(R, T, gamma, Q_init=Q_inits)
    dT = tabsoftq_T_grad_iter(mdp, T_thetas, Q, R, gamma)
    # Sum instead of mean because sparse results
    dT = np.sum(dT[s,a], axis=0).reshape(T_thetas.shape) / len(sas_obs)
    ll = eval_pol_likelihood(Q, sas_obs)
    return ll, dT, Q
