from utils.learning_utils import softmax
import numpy as np
from utils.tab_learning_utils import T_estimate, sample_tab_batch, eval_demo_log_likelihood,\
    eval_T_pol_likelihood_and_grad, eval_TR_pol_likelihood_and_grad, eval_trans_likelihood_and_grad
import os
import pickle as pkl

class TabularInverseDynamicsLearner():

    def __init__(self, mdp, serd, gamma, boltz_beta, alpha=0.5, seed=0):


        self.mdp = mdp
        self.serd = serd
        self.alpha = alpha
        self.seed = seed
        self.gamma = gamma
        self.boltz_beta = boltz_beta
        self.regime = None

    def train(self, n_training_iters, sas_obs, adt_obs, batch_size, val_sas, val_adt, out_dir,  _run=None, true_qs=None,
              tab_save_freq=25, verbose=True):

        # Tabular logging setup
        tab_model_out_dir = os.path.join(out_dir, "tab")
        if not os.path.exists(tab_model_out_dir):
            os.makedirs(tab_model_out_dir)
        if true_qs is not None:
            pkl.dump(true_qs, open(os.path.join(tab_model_out_dir, 'true_q_vals.pkl'), 'wb'))
        pkl.dump(self.mdp.adt_mat, open(os.path.join(tab_model_out_dir, 'true_adt_probs.pkl'), 'wb'))

        train_time = 0
        feats = self.mdp.feature_map

        t_thetas = T_estimate(self.mdp, adt_obs)

        if self.serd:
            r_thetas = np.random.normal(loc=1, scale=0.1, size=feats.shape[1])
        else:
            R = self.mdp.rewards

        Q = None

        try:

            for train_time in range(n_training_iters):

                batch_demo_sas, batch_demo_adt = sample_tab_batch(batch_size, sas_obs, adt_obs)

                if self.serd:
                    R = feats.dot(r_thetas)
                    tp_ll, dT_pol, dR_pol, Q = eval_TR_pol_likelihood_and_grad(self.mdp, t_thetas, R, batch_demo_sas,
                                                                       self.gamma, Q_inits=Q)
                    r_thetas += self.alpha * dR_pol
                else:
                    # Should we initialize Q or nah?
                    tp_ll, dT_pol, Q = eval_T_pol_likelihood_and_grad(self.mdp, t_thetas, R, batch_demo_sas,
                                                                       self.gamma, Q_inits=Q)

                tt_ll, dT_trans = eval_trans_likelihood_and_grad(t_thetas, batch_demo_adt)

                vp_ll, vt_ll = eval_demo_log_likelihood(val_sas, val_adt, t_thetas, Q)
                val_likelihood = vp_ll + vt_ll
                t_thetas += self.alpha * (dT_trans + dT_pol)

                _run.log_scalar('val_likelihoods', val_likelihood, train_time)
                _run.log_scalar('val_nalls', vp_ll, train_time)
                _run.log_scalar('val_ntlls', vt_ll, train_time)

                if train_time % tab_save_freq == 0:
                    if verbose:
                        print(str(train_time) + "\t" + "\t".join(['action ll' + ": " + str(round(vp_ll, 7)),
                                                                  'transition ll' + ": " + str(round(vt_ll, 7)),
                                                                  'total ll' + ": " + str(round(val_likelihood, 7))]))

                    print(train_time, np.max(Q), np.min(Q))
                    adt_probs = softmax(t_thetas).transpose((2,0,1))
                    pkl.dump(Q, open(os.path.join(tab_model_out_dir, 'q_vals_{}.pkl'.format(train_time)), 'wb'))
                    pkl.dump(adt_probs, open(os.path.join(tab_model_out_dir, 'adt_probs_{}.pkl'.format(train_time)), 'wb'))

        except KeyboardInterrupt:
            print("Experiment Interrupted at timestep {}".format(train_time))
            pass

        # Save as file
        adt_probs = softmax(t_thetas).transpose((2, 0, 1))
        pkl.dump(Q, open(os.path.join(tab_model_out_dir, 'final_q_vals.pkl'), 'wb'))
        pkl.dump(adt_probs, open(os.path.join(tab_model_out_dir, 'final_adt_probs.pkl'), 'wb'))

        pkl.dump(self.mdp, open(os.path.join(out_dir, 'mdp.pkl'), 'wb'))

        return tab_model_out_dir
