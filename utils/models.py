import tensorflow as tf
from utils.tf_utils import build_mlp
from utils.learning_utils import sample_batch, softmax
import numpy as np
from utils.frank_wolfe_utils import find_min_norm_element
from utils.tf_utils import save_tf_vars, load_tf_vars
import os
import pickle as pkl


class InverseDynamicsLearner():

    def __init__(self, mdp, sess, gamma=0.99, adt=True, observed_policy=False, mellowmax=None, lse_softmax=None, boltz_beta=50, mlp_params=None,
                    alpha=1e-4, beta1=0.9, beta2=0.999999, seed=0, dyn_scope="Dynamics", q_scope="Qs"):

        mlp_params = {} if mlp_params is None else mlp_params

        assert(mellowmax is None or lse_softmax is None)

        self.mdp = mdp
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.seed = seed
        self.use_observed_policy = observed_policy

        tf.set_random_seed(seed)

        # TODO Make this retrievable by env observation space
        n_obs_feats = mdp.nrow + mdp.ncol
        n_act_dim = mdp.num_actions
        n_dirs = mdp.num_directions

        self.dyn_scope = dyn_scope
        self.q_scope = q_scope
        self.sess = sess

        dyn_n_layers = mlp_params.get('dyn_n_layers', 1)
        dyn_layer_size = mlp_params.get('dyn_layer_size', 256)
        dyn_layer_activation = mlp_params.get('dyn_layer_activation', tf.nn.relu)
        dyn_output_activation = mlp_params.get('dyn_output_activation', None)
        dyn_layer_norm = mlp_params.get('dyn_layer_norm', False)

        q_n_layers = mlp_params.get('q_n_layers', 2)
        q_layer_size = mlp_params.get('q_layer_size', 2048)
        q_layer_activation = mlp_params.get('q_layer_activation', tf.nn.tanh)
        q_output_activation = mlp_params.get('q_output_activation', None)
        q_layer_norm = mlp_params.get('q_layer_norm', False)

        weight_norm = mlp_params.get('weight_norm', True)


        self.demo_tile_t_ph = tf.placeholder(tf.int32, [None, 1], name="dtt")
        self.demo_act_t_ph = tf.placeholder(tf.int32, [None, 1], name="dat")
        self.demo_obs_t_feats_ph = tf.placeholder(tf.int32, [None, n_obs_feats], name="dotf")
        self.demo_dir_t_ph = tf.placeholder(tf.int32, [None, 1], name="ddt")
        self.demo_batch_size_ph = tf.placeholder(tf.int32, name="dbs")

        self.constraint_tile_t_ph = tf.placeholder(tf.int32, [None, 1], name="ctt")
        self.constraint_obs_t_feats_ph = tf.placeholder(tf.int32, [None, n_obs_feats], name="cotf")
        self.constraint_act_t_ph = tf.placeholder(tf.int32, [None, 1], name="cat")
        self.constraint_rew_t_ph = tf.placeholder(tf.float32, [None], name="crt")
        self.constraint_next_obs_t_feats_ph = tf.placeholder(tf.int32, [None, n_dirs, n_obs_feats], name="cnotf")
        self.constraint_batch_size_ph = tf.placeholder(tf.int32, name="cbs")

        # True Qs are only used for testing architectures and learning capabilities
        self.true_qs_ph = tf.placeholder(tf.float32, [None, n_act_dim], name="tqs")

        demo_tile_t_one_hot = self.demo_tile_t_ph #tf.one_hot(tf.squeeze(self.demo_tile_t_ph, axis=1), mdp.tile_types, dtype=tf.int32)
        constraint_tile_t_one_hot = self.constraint_tile_t_ph #tf.one_hot(tf.squeeze(self.constraint_tile_t_ph, axis=1), mdp.tile_types, dtype=tf.int32)


        demo_q_t = build_mlp(self.demo_obs_t_feats_ph,
                             n_act_dim, q_scope,
                             n_layers=q_n_layers, size=q_layer_size,
                             activation=q_layer_activation, output_activation=q_output_activation,
                             layer_norm=q_layer_norm, weight_norm=weight_norm)

        demo_v_t = tf.reduce_logsumexp(demo_q_t * boltz_beta, axis=1)

        action_indexes = tf.concat([tf.expand_dims(tf.range(self.demo_batch_size_ph), 1), self.demo_act_t_ph], axis=1)

        act_log_likelihoods = tf.gather_nd(demo_q_t * boltz_beta, action_indexes) - demo_v_t

        self.neg_avg_act_log_likelihood = -tf.reduce_mean(act_log_likelihoods)

        print(tf.shape(tf.concat((self.demo_act_t_ph, demo_tile_t_one_hot), axis=1)))

        dir_indexes = tf.concat([tf.expand_dims(tf.range(self.demo_batch_size_ph), 1), self.demo_dir_t_ph], axis=1)

        # Encoded the potential to use "sas" transitions, which are simply state, action, state observations,
        # and "adt" transitions -- post-processed to determine (a)ction taken, (d)irection moved, and starting (t)ile-type
        # Learning an "adt" transition model should be much easier

        if adt:
            self.pred_obs = build_mlp(
                tf.concat((self.demo_act_t_ph, demo_tile_t_one_hot), axis=1),
                n_dirs, dyn_scope,
                n_layers=dyn_n_layers, size=dyn_layer_size,
                activation=dyn_layer_activation, output_activation=dyn_output_activation,
                layer_norm=dyn_layer_norm, weight_norm=weight_norm)

            self.baseline_pred_obs = build_mlp(
                tf.concat((self.demo_act_t_ph, demo_tile_t_one_hot), axis=1),
                n_dirs, "dyn_baseline",
                n_layers=dyn_n_layers, size=dyn_layer_size,
                activation=dyn_layer_activation, output_activation=dyn_output_activation,
                layer_norm=dyn_layer_norm, weight_norm=weight_norm, reuse=False)
        else:
            self.pred_obs = build_mlp(
                tf.concat((demo_tile_t_one_hot, self.demo_act_t_ph), axis=1),
                n_dirs, dyn_scope,
                n_layers=dyn_n_layers, size=dyn_layer_size,
                activation=dyn_layer_activation, output_activation=dyn_output_activation,
                layer_norm=dyn_layer_norm, weight_norm=weight_norm)

            self.pred_obs = build_mlp(
                tf.concat((demo_tile_t_one_hot, self.demo_act_t_ph), axis=1),
                n_dirs, "dyn_baseline",
                n_layers=dyn_n_layers, size=dyn_layer_size,
                activation=dyn_layer_activation, output_activation=dyn_output_activation,
                layer_norm=dyn_layer_norm, weight_norm=weight_norm, reuse=False)


        # trans_log_likelihoods = tf.gather_nd(self.pred_obs, dir_indexes) - tf.reduce_logsumexp(self.pred_obs, axis=1)
        # trans_probs = tf.exp(trans_log_likelihoods)
        #
        # baseline_trans_log_likelihoods = tf.gather_nd(self.baseline_pred_obs, dir_indexes) - tf.reduce_logsumexp(self.baseline_pred_obs, axis=1)
        # baseline_trans_probs = tf.exp(baseline_trans_log_likelihoods)

        trans_probs = tf.gather_nd(tf.nn.softmax(self.pred_obs, axis=1), dir_indexes)
        trans_log_likelihoods = tf.log(trans_probs + 1e-40)

        baseline_trans_probs = tf.gather_nd(tf.nn.softmax(tf.stop_gradient(self.baseline_pred_obs), axis=1), dir_indexes)
        baseline_trans_log_likelihoods = tf.log(baseline_trans_probs)

        self.neg_avg_trans_log_likelihood = -tf.reduce_mean(trans_log_likelihoods)

        # Set up KL-Divergence over demo observations
        base_learned_KL_components = baseline_trans_probs * (baseline_trans_log_likelihoods - trans_log_likelihoods)
        learned_base_KL_components = trans_probs * (trans_log_likelihoods - baseline_trans_log_likelihoods)
        self.trans_baseline_KL = tf.reduce_mean(base_learned_KL_components)
        self.bidirectional_KL = tf.reduce_mean(base_learned_KL_components + learned_base_KL_components)
        # self.trans_baseline_KL_distance = tf.reduce_mean(tf.abs(trans_KL_components))


        ca_indexes = tf.concat([tf.expand_dims(tf.range(self.constraint_batch_size_ph), 1), self.constraint_act_t_ph], axis=1)

        self.constraint_qs_t = build_mlp(self.constraint_obs_t_feats_ph,
                                    n_act_dim, q_scope,
                                    n_layers=q_n_layers, size=q_layer_size,
                                    activation=q_layer_activation, output_activation=q_output_activation,
                                    layer_norm=q_layer_norm, weight_norm=weight_norm, reuse=True)

        constraint_q_t = tf.gather_nd(self.constraint_qs_t, ca_indexes)

        # Predicted constraint next state given inv dyns
        if adt:
            constraint_pred_obs = build_mlp(
                tf.concat((self.constraint_act_t_ph, constraint_tile_t_one_hot), axis=1),
                n_dirs, dyn_scope,
                n_layers=dyn_n_layers, size=dyn_layer_size,
                activation=dyn_layer_activation, output_activation=dyn_output_activation,
                layer_norm=dyn_layer_norm, weight_norm=weight_norm, reuse=True)
        else:
            constraint_pred_obs = build_mlp(
                tf.concat((self.constraint_obs_t_feats_ph, self.constraint_act_t_ph), axis=1),
                n_dirs, dyn_scope,
                n_layers=dyn_n_layers, size=dyn_layer_size,
                activation=dyn_layer_activation, output_activation=dyn_output_activation,
                layer_norm=dyn_layer_norm, weight_norm=weight_norm, reuse=True)

        constraint_sprimes_reshaped = tf.reshape(self.constraint_next_obs_t_feats_ph,
                                                 (self.constraint_batch_size_ph * n_dirs, n_obs_feats))

        # Q-values used to calculate 'V' in the bellman-residual
        cqtp1_misshaped = build_mlp(constraint_sprimes_reshaped,
                                    n_act_dim, scope="target",
                                    n_layers=q_n_layers, size=q_layer_size,
                                    activation=q_layer_activation, output_activation=q_output_activation,
                                    layer_norm=q_layer_norm, weight_norm=weight_norm, reuse=False)

        constraint_q_tp1 = tf.reshape(tf.stop_gradient(cqtp1_misshaped), (self.constraint_batch_size_ph, n_dirs, n_act_dim))

        if mellowmax is not None:
            constraint_v_tp1 = (tf.reduce_logsumexp(constraint_q_tp1 * mellowmax, axis=2) - np.log(n_act_dim)) / mellowmax
        elif lse_softmax is not None:
            constraint_v_tp1 = tf.reduce_logsumexp(lse_softmax * constraint_q_tp1, axis=2) / lse_softmax
        else:
            constraint_pi_tp1 = tf.nn.softmax(constraint_q_tp1 * boltz_beta, axis=2)
            constraint_v_tp1 = tf.reduce_sum(constraint_q_tp1 * constraint_pi_tp1, axis=2)


        # bellman residual penalty error
        constraint_pred_probs = tf.nn.softmax(constraint_pred_obs, axis=1)
        constraint_next_vs = tf.multiply(constraint_v_tp1, constraint_pred_probs)
        target_t = self.constraint_rew_t_ph + gamma * tf.reduce_sum(constraint_next_vs, axis=1)
        self.td_err = tf.reduce_mean((constraint_q_t - target_t) ** 2)

        # bellman residual penalty error with a stop gradient on transitions to prevent the bellman update from
        # 'hacking' the transition function to overly explain the demonstrations. This happens a lot early on before the
        # Q-values have obtained much meaning
        constraint_next_vs_sgt = tf.multiply(constraint_v_tp1, tf.stop_gradient(constraint_pred_probs))
        target_t_sgt = self.constraint_rew_t_ph + gamma * tf.reduce_sum(constraint_next_vs_sgt, axis=1)
        self.td_err_sgt = tf.reduce_mean((constraint_q_t - target_t_sgt) ** 2)

        # bellman residual penalty error with a stop gradient on q-fn to allow the bellman residual loss to update
        # differences in learned q-vals w.r.t. the dynamics model
        constraint_next_vs_sgq = tf.multiply(tf.stop_gradient(constraint_v_tp1), constraint_pred_probs)
        target_t_sgq = self.constraint_rew_t_ph + gamma * tf.reduce_sum(constraint_next_vs_sgq, axis=1)
        self.td_err_sgq = tf.reduce_mean((tf.stop_gradient(constraint_q_t) - target_t_sgq) ** 2)

        # Set up target network
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=q_scope)
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target")
        update_target_fn_vars = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_fn_vars.append(var_target.assign(var))
        self.update_target_fn = tf.group(*update_target_fn_vars)

        # Set up dynamics baseline network
        dyn_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=dyn_scope)
        baseline_dyn_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="dyn_baseline")
        update_baseline_fn_vars = []
        for var, var_target in zip(sorted(dyn_func_vars, key=lambda v: v.name),
                                   sorted(baseline_dyn_func_vars, key=lambda v: v.name)):
            update_baseline_fn_vars.append(var_target.assign(var))
        self.update_baseline_fn = tf.group(*update_baseline_fn_vars)

        #KL-divergence for observed transitions


        #Experimental
        self.lqsafer = self.neg_avg_act_log_likelihood / (1/self.td_err + 1/self.neg_avg_trans_log_likelihood)
        self.br_ball_ph = tf.placeholder(tf.float32, (), name='br_ball')
        self.llt_weighted_td_err = self.td_err_sgq * self.neg_avg_trans_log_likelihood
        self.lla_weighted_td_err = tf.maximum(self.td_err_sgt, self.br_ball_ph) * self.neg_avg_act_log_likelihood
        self.kl_ball_ph = tf.placeholder(tf.float32, (), name='kl_ball')
        self.kl_weighted_td_err = self.td_err * tf.maximum(self.bidirectional_KL, self.kl_ball_ph)
        self.kl_combo = tf.maximum(self.td_err, self.br_ball_ph) * tf.maximum(self.bidirectional_KL, self.kl_ball_ph) * self.neg_avg_act_log_likelihood
        self.kl_hinge = tf.maximum((self.bidirectional_KL / self.kl_ball_ph) - 1, 0)


        if self.use_observed_policy:

            pi_demo_q_t = build_mlp(self.demo_obs_t_feats_ph,
                                    n_act_dim, "pi",
                                    n_layers=q_n_layers, size=q_layer_size,
                                    activation=q_layer_activation, output_activation=q_output_activation,
                                    layer_norm=q_layer_norm, weight_norm=weight_norm)

            self.pi_constraint_qs_t = build_mlp(self.constraint_obs_t_feats_ph,
                                                n_act_dim, "pi",
                                                n_layers=q_n_layers, size=q_layer_size,
                                                activation=q_layer_activation, output_activation=q_output_activation,
                                                layer_norm=q_layer_norm, weight_norm=weight_norm, reuse=True)

            pi_constraint_q_t = tf.gather_nd(self.pi_constraint_qs_t, ca_indexes)

            # Q-values used to calculate 'V' in the bellman-residual for the observed Q-fn
            pi_cqtp1_misshaped = build_mlp(constraint_sprimes_reshaped,
                                           n_act_dim, scope="pi_target",
                                           n_layers=q_n_layers, size=q_layer_size,
                                           activation=q_layer_activation, output_activation=q_output_activation,
                                           layer_norm=q_layer_norm, weight_norm=weight_norm, reuse=False)

            pi_constraint_q_tp1 = tf.reshape(tf.stop_gradient(pi_cqtp1_misshaped),
                                             (self.constraint_batch_size_ph, n_dirs, n_act_dim))

            # Q-values used to calculate \pi_O, the policy learned from the demonstrations
            observed_pol_cqtp1_misshaped = build_mlp(constraint_sprimes_reshaped,
                                                     n_act_dim, scope="observed",
                                                     n_layers=q_n_layers, size=q_layer_size,
                                                     activation=q_layer_activation,
                                                     output_activation=q_output_activation,
                                                     layer_norm=q_layer_norm, weight_norm=weight_norm, reuse=False)

            observed_pol_q_tp1 = tf.reshape(tf.stop_gradient(observed_pol_cqtp1_misshaped),
                                            (self.constraint_batch_size_ph, n_dirs, n_act_dim))

            observed_pol_tp1 = tf.nn.softmax(observed_pol_q_tp1 * boltz_beta, axis=2)
            constraint_pi_v_tp1 = tf.reduce_sum(pi_constraint_q_tp1 * observed_pol_tp1, axis=2)

            # bellman residual for observed policy penalty error
            constraint_pi_next_vs = tf.multiply(constraint_pi_v_tp1, constraint_pred_probs)
            pi_target_t = self.constraint_rew_t_ph + gamma * tf.reduce_sum(constraint_pi_next_vs, axis=1)
            self.pi_td_err = tf.reduce_mean((pi_constraint_q_t - pi_target_t) ** 2)

            # bellman residual for observed policy penalty error
            constraint_pi_next_vs_sgt = tf.multiply(constraint_pi_v_tp1, tf.stop_gradient(constraint_pred_probs))
            pi_target_t_sgt = self.constraint_rew_t_ph + gamma * tf.reduce_sum(constraint_pi_next_vs_sgt, axis=1)
            self.pi_td_err_sgt = tf.reduce_mean((pi_constraint_q_t - pi_target_t_sgt) ** 2)
            # bellman residual for observed policy penalty error
            constraint_pi_next_vs_sgq = tf.multiply(tf.stop_gradient(constraint_pi_v_tp1), constraint_pred_probs)
            pi_target_t_sgq = self.constraint_rew_t_ph + gamma * tf.reduce_sum(constraint_pi_next_vs_sgq, axis=1)
            self.pi_td_err_sgq = tf.reduce_mean((tf.stop_gradient(pi_constraint_q_t) - pi_target_t_sgq) ** 2)

            # Set up pi target network
            pi_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="pi")
            pi_target_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="pi_target")
            update_pi_target_fn_vars = []
            for var, var_target in zip(sorted(pi_func_vars, key=lambda v: v.name),
                                       sorted(pi_target_func_vars, key=lambda v: v.name)):
                update_pi_target_fn_vars.append(var_target.assign(var))
            self.update_pi_target_fn = tf.group(*update_pi_target_fn_vars)

            # Set up observed policy network
            q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=q_scope)
            observed_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="observed")
            update_observed_fn_vars = []
            for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                       sorted(observed_q_func_vars, key=lambda v: v.name)):
                update_observed_fn_vars.append(var_target.assign(var))
            self.update_observed_policy_fn = tf.group(*update_observed_fn_vars)

            # Might want to compare learned Qs to target network of observed pi for better stability
            target_pi_demo_q_t = build_mlp(self.demo_obs_t_feats_ph,
                                           n_act_dim, "pi_target",
                                           n_layers=q_n_layers, size=q_layer_size,
                                           activation=q_layer_activation, output_activation=q_output_activation,
                                           layer_norm=q_layer_norm, weight_norm=weight_norm, reuse=True)

            self.learned_observed_q_diff = tf.norm(demo_q_t - tf.stop_gradient(target_pi_demo_q_t), ord=2)

            self.loss_fns = np.array([self.neg_avg_act_log_likelihood, self.neg_avg_trans_log_likelihood,
                                      self.td_err, self.td_err_sgq, self.td_err_sgt, self.llt_weighted_td_err,
                                      self.lqsafer,
                                      self.lla_weighted_td_err, self.bidirectional_KL, self.kl_weighted_td_err,
                                      self.kl_hinge, self.pi_td_err, self.pi_td_err_sgq, self.pi_td_err_sgt,
                                      self.learned_observed_q_diff])  # , self.true_q_err])
            self.loss_fns_titles = np.array(['nall', 'ntll', 'tde', 'tde_t', 'tde_q', 'llt_tde', 'lqsafe', 'lla_tde',
                                             'trans_kl_dist', 'kl_tde', 'kl', 'pi_tde', 'pi_tde_t', 'pi_tde_q',
                                             'loqd'])  # , 'true_q_err'])

            self.log_losses = self.loss_fns[[0, 1, 2, 11, 14, 8]]
            self.log_loss_titles = self.loss_fns_titles[[0, 1, 2, 11, 14, 8]]

        else:
            self.loss_fns = np.array([self.neg_avg_act_log_likelihood, self.neg_avg_trans_log_likelihood,
                                      self.td_err, self.td_err_sgq, self.td_err_sgt, self.llt_weighted_td_err,
                                      self.lqsafer, self.lla_weighted_td_err, self.bidirectional_KL,
                                      self.kl_weighted_td_err, self.kl_hinge])  # , self.true_q_err])
            self.loss_fns_titles = np.array(['nall', 'ntll', 'tde', 'tde_t', 'tde_q', 'llt_tde', 'lqsafe', 'lla_tde',
                                             'trans_kl_dist', 'kl_tde', 'kl'])  # , 'true_q_err'])

            self.log_losses = self.loss_fns[[0, 1, 2, 8]]
            self.log_loss_titles = self.loss_fns_titles[[0, 1, 2, 8]]

        # # True Qs for testing
        self.test_q_obs_t_feats_ph = tf.placeholder(tf.int32, [None, n_obs_feats], name="tqotf")
        self.test_qs_t = build_mlp(self.test_q_obs_t_feats_ph,
                                    n_act_dim, q_scope,
                                    n_layers=q_n_layers, size=q_layer_size,
                                    activation=q_layer_activation, output_activation=q_output_activation,
                                    layer_norm=q_layer_norm, weight_norm=weight_norm, reuse=True)
        self.true_q_err = tf.reduce_sum((self.true_qs_ph - self.test_qs_t)**2)



        self.regime = None
        self.validation_freq = None
        self.kl_ball_schedule = None
        self.br_ball_schedule = None


    def initialize_training_regime(self, regime, regime_params, validation_freq=20):
        assert(regime in ["MGDA", "coordinate", "weighted"])

        if self.regime is not None:
            print("Already initialized, create new learner instead")
            return

        self.regime = regime
        self.validation_freq = validation_freq
        self.kl_ball_schedule = regime_params.get('kl_ball_schedule', None)
        self.br_ball_schedule = regime_params.get('br_ball_schedule', None)

        if regime == "weighted":
            losses = regime_params['losses']
            loss_weights = regime_params['loss_weights']
            assert len(losses) == len(loss_weights)
            assert not any(['observed' in title for title in self.loss_fns_titles[losses]])
            self.weighted_loss = sum([self.loss_fns[losses[i]] * loss_weights[i] for i in range(len(losses))])
            self.log_losses = np.append(self.log_losses, [self.weighted_loss])
            self.log_loss_titles = np.append(self.log_loss_titles, ["weighted_loss"])
            if regime_params['clip_global'] is not None:
                optimizer = tf.train.AdamOptimizer(self.alpha, self.beta1, self.beta2)
                gradients, variables = zip(*optimizer.compute_gradients(self.weighted_loss))
                gradients, _ = tf.clip_by_global_norm(gradients, regime_params['clip_global'])
                self.update = [optimizer.apply_gradients(zip(gradients, variables))]
            else:
                self.update = tf.train.AdamOptimizer(self.alpha, self.beta1, self.beta2).minimize(self.weighted_loss)

        if regime == "coordinate":
            # Coordinate descent training regime
            self.alphas = regime_params['alphas']
            hs = regime_params["horizons"]
            self.update_horizons = hs if type(hs) == list else [hs] * len(self.alphas)
            self.improvement_proportions = regime_params["improvement_proportions"]
            self.prev_bests = [np.inf for _ in range(len(self.improvement_proportions))]
            self.switch_frequency = regime_params["switch_frequency"]
            self.observed_updates_progression = regime_params.get('observed_updates_progression', None)
            self.model_save_loss = sum(self.log_losses * np.array(regime_params["model_save_weights"]))
            # Update progression is a list of list of ints in the range(len(self.loss_fns))
            self.loss_progression = [sum(self.loss_fns[config]) for config in regime_params["update_progression"]]
            self.loss_progression_titles = [" ".join(self.loss_fns_titles[config]) for config in regime_params["update_progression"]]
            #TODO Check for equivalent losses
            if regime_params['clip_global'] is not None:
                self.update_progression = []
                for i,loss in enumerate(self.loss_progression):
                    optimizer = tf.train.AdamOptimizer(self.alpha, self.beta1, self.beta2)
                    gradients, variables = zip(*optimizer.compute_gradients(loss))
                    gradients, _ = tf.clip_by_global_norm(gradients, regime_params['clip_global'])
                    self.update_progression += [optimizer.apply_gradients(zip(gradients, variables))]
            else:
                self.update_progression = [tf.train.AdamOptimizer(self.alphas[i], self.beta1, self.beta2).minimize(
                        loss, name="opt_{}".format(i)) for i,loss in enumerate(self.loss_progression)]

            self.curr_update_index = 0
            self.curr_loss = self.loss_progression[self.curr_update_index]
            self.curr_update = self.update_progression[self.curr_update_index]



        if regime == "MGDA":
            # Update progression is a list of list of ints in the range(len(self.loss_fns))
            self.num_tasks = len(regime_params["loss_configurations"])
            self.scales = [tf.placeholder(tf.float32, [1], name="scale{}".format(i)) for i in range(self.num_tasks)]
            self.task_losses = [sum(self.loss_fns[config]) for config in regime_params["loss_configurations"]]

            q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.q_scope)
            t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.dyn_scope)
            all_vars = q_vars + t_vars
            opts = [tf.train.AdamOptimizer(self.alpha, self.beta1, self.beta2) for _ in range(self.num_tasks)]
            self.all_gvs = [opts[i].compute_gradients(self.task_losses[i], all_vars) for i in range(self.num_tasks)]
            sep_gvs = [[a for a in zip(*gv_set)] for gv_set in self.all_gvs]

            self.grads = [[tf.clip_by_norm(sep_gvs[i][0][j], self.task_losses[i]) for j in range(len(sep_gvs[i][0]))]
                                for i in range(self.num_tasks)]
            self.gvars = [gv[1] for gv in sep_gvs]

            self.scaled_grads = [[tf.multiply(g, self.scales[i]) for g in self.grads[i]] for i in range(self.num_tasks)]
            self.updates = [opts[i].apply_gradients(zip(self.scaled_grads[i], self.gvars[i])) for i in range(self.num_tasks)]


    def train(self, n_training_iters, rollouts, train_idxes, batch_size, constraints, val_demo_batch, out_dir,
                    q_states, adt_samples, target_update_freq=50, dyn_pretrain_iters=0, tab_save_freq = 1000,
                    _run=None, true_qs=None, verbose=True, q_source_path=None, dyn_source_path=None):

        assert(self.regime is not None)

        # Have to put up here before variables are initialized
        if dyn_pretrain_iters > 0:
            temp_ntll_update = tf.train.AdamOptimizer(self.alpha, self.beta1, self.beta2).minimize(
                self.neg_avg_trans_log_likelihood)
            temp_nall_update = tf.train.AdamOptimizer(self.alpha, self.beta1, self.beta2).minimize(
                self.neg_avg_act_log_likelihood)
            temp_q_update = tf.train.AdamOptimizer(self.alpha, self.beta1, self.beta2).minimize(self.td_err_sgt)
            if self.use_observed_policy:
                temp_pi_q_update = tf.train.AdamOptimizer(self.alpha, self.beta1, self.beta2).minimize(self.pi_td_err_sgt)


        tf.global_variables_initializer().run(session=self.sess)

        # Tabular logging setup
        tab_model_out_dir = os.path.join(out_dir, "tab")
        if not os.path.exists(tab_model_out_dir):
            os.makedirs(tab_model_out_dir)
        if true_qs is not None:
            pkl.dump(true_qs, open(os.path.join(tab_model_out_dir, 'true_q_vals.pkl'), 'wb'))
        pkl.dump(self.mdp.adt_mat, open(os.path.join(tab_model_out_dir, 'true_adt_probs.pkl'), 'wb'))

        # Load pretrained models with same scope
        if q_source_path is not None:
            load_tf_vars(self.sess, self.q_scope, q_source_path)
        if dyn_source_path is not None:
            load_tf_vars(self.sess, self.dyn_scope, dyn_source_path)

        q_f = os.path.join(out_dir, "q_fn")
        dyn_f = os.path.join(out_dir, "dyn_fn")

        train_time = 0
        # Initialize train_logs to have an array for train and validation evaluations
        full_train_logs = {}
        for k in self.log_loss_titles:
            full_train_logs[k] = []
            full_train_logs["val_" + k] = []

        if self.regime == "coordinate":
            total_loss = []
            best_model_score = np.inf

        if self.regime == "MGDA":
            full_train_logs["frank_wolfe"] = []

        val_feed = self._get_feed_dict(val_demo_batch, constraints, true_qs=true_qs)

        if self.kl_ball_schedule is not None:
            val_feed[self.kl_ball_ph] = self.kl_ball_schedule(train_time)
        if self.br_ball_schedule is not None:
            val_feed[self.br_ball_ph] = self.br_ball_schedule(train_time)

        if dyn_pretrain_iters > 0:
            print("Pretraining the dynamics model with baseline method.")
            for i in range(int(dyn_pretrain_iters/2)):
                demo_batch = sample_batch(rollouts, train_idxes, batch_size)
                feed_dict = self._get_feed_dict(demo_batch, constraints)
                self.sess.run(temp_ntll_update, feed_dict=feed_dict)
                if i % 1000 == 0:
                    val_loss = self.sess.run(self.neg_avg_trans_log_likelihood, feed_dict=val_feed)
                    print("{} Transition Loss: {}".format(i, val_loss))
            self.sess.run(self.update_baseline_fn)
            if self.use_observed_policy:
                for i in range(int(dyn_pretrain_iters/2)):
                    demo_batch = sample_batch(rollouts, train_idxes, batch_size)
                    feed_dict = self._get_feed_dict(demo_batch, constraints)
                    self.sess.run(temp_nall_update, feed_dict=feed_dict)
                    if i % 1000 == 0:
                        val_loss = self.sess.run(self.neg_avg_act_log_likelihood, feed_dict=val_feed)
                        print("{} Observed Policy Learning Loss: {}".format(i, val_loss))
                self.sess.run(self.update_observed_policy_fn)
            for i in range(dyn_pretrain_iters+1):
                demo_batch = sample_batch(rollouts, train_idxes, batch_size)
                feed_dict = self._get_feed_dict(demo_batch, constraints)
                if self.use_observed_policy:
                    cqs, pcqs, _, _ = self.sess.run([self.constraint_qs_t, self.pi_constraint_qs_t,
                                                     temp_q_update, temp_pi_q_update], feed_dict=feed_dict)

                    if i % target_update_freq == 0:
                        self.sess.run([self.update_target_fn, self.update_pi_target_fn])

                    if i % 1000 == 0:
                        val_loss, pi_val_loss = self.sess.run([self.td_err_sgt, self.pi_td_err_sgt], feed_dict=val_feed)
                        print("{} Q-Learning Losses: {} \t {}".format(i, val_loss, pi_val_loss))
                        print(np.max(cqs), np.min(cqs), np.max(pcqs), np.min(pcqs))
                else:
                    cqs, _ = self.sess.run([self.constraint_qs_t, temp_q_update], feed_dict=feed_dict)
                    if i % target_update_freq == 0:
                        self.sess.run(self.update_target_fn)

                    if i % 1000 == 0:
                        val_loss = self.sess.run(self.td_err_sgt, feed_dict=val_feed)
                        print("{} Q-Learning Losses: {}".format(i, val_loss))
                        print(np.max(cqs), np.min(cqs))

        try:

            for train_time in range(n_training_iters):

                if train_time % self.validation_freq == 0:
                    val_loss_data = self.sess.run(list(self.log_losses), feed_dict=val_feed)
                    for i, loss in enumerate(val_loss_data):
                        metric_name = "val_" + self.log_loss_titles[i]
                        full_train_logs[metric_name] += [loss]
                        if _run is not None:
                            _run.log_scalar(metric_name, loss, train_time)

                if train_time % 500 == 0 and verbose:
                    print(str(train_time) + "\t" + "\t".join([str(k) + ": " +
                                                str(round(full_train_logs["val_" + k][-1],7)) for k in self.log_loss_titles]))
                    # if self.regime == "MGDA":
                    #     print(sol)

                if train_time % tab_save_freq == 0:
                    # self.mdp only used in this section, can remove that dependency if we want
                    q_vals = self.sess.run([self.test_qs_t], feed_dict={self.test_q_obs_t_feats_ph: q_states})[0]
                    print(train_time, np.max(q_vals), np.min(q_vals))
                    adt_logits = self.sess.run([self.pred_obs], feed_dict={self.demo_tile_t_ph: adt_samples[:,0][np.newaxis].T,
                                                                    self.demo_act_t_ph: adt_samples[:,1][np.newaxis].T})[0]
                    adt_probs = softmax(adt_logits)
                    adt_probs = adt_probs.reshape(self.mdp.tile_types, self.mdp.num_actions, self.mdp.num_directions)
                    pkl.dump(q_vals, open(os.path.join(tab_model_out_dir, 'q_vals_{}.pkl'.format(train_time)), 'wb'))
                    pkl.dump(adt_probs, open(os.path.join(tab_model_out_dir, 'adt_probs_{}.pkl'.format(train_time)), 'wb'))
                    if self.regime == "coordinate":
                        _run.log_scalar("coordinate_regime", self.loss_progression_titles[self.curr_update_index], train_time)


                demo_batch = sample_batch(rollouts, train_idxes, batch_size)
                feed_dict = self._get_feed_dict(demo_batch, constraints, true_qs=true_qs)

                if self.kl_ball_schedule is not None:
                    feed_dict[self.kl_ball_ph] = self.kl_ball_schedule(train_time)
                    val_feed[self.kl_ball_ph] = self.kl_ball_schedule(train_time)

                if self.br_ball_schedule is not None:
                    feed_dict[self.br_ball_ph] = self.br_ball_schedule(train_time)
                    val_feed[self.br_ball_ph] = self.br_ball_schedule(train_time)

                if self.regime == "weighted":
                    # loss_data = self.sess.run(list(self.log_losses) + [self.update], feed_dict=feed_dict)[:-1]
                    self.sess.run(self.update, feed_dict=feed_dict)

                if self.regime == "coordinate":
                    loss_data = self.sess.run([self.curr_loss, self.curr_update], feed_dict=feed_dict)[0]
                    total_loss.append(loss_data)

                if self.regime == "MGDA":
                    loss_and_gvs = self.sess.run(list(self.log_losses) + [self.all_gvs], feed_dict=feed_dict)
                    loss_data, gvs = loss_and_gvs[:-1], loss_and_gvs[-1]

                    grad_arrays = {}
                    for t in range(self.num_tasks):
                        # Compute gradients of each loss function wrt parameters
                        grad_arrays[t] = np.concatenate([gvs[t][i][0].flatten() for i in range(len(gvs))])

                    # Frank-Wolfe iteration to compute scales.
                    sol, min_norm = find_min_norm_element(grad_arrays)
                    full_train_logs["frank_wolfe"] += [sol]

                    # Scaled back-propagation
                    for i, s in enumerate(self.scales):
                        feed_dict[s] = [sol[i]]

                    # Inefficient, takes gradients twice when we could just feed the gradients back in
                    self.sess.run(self.updates, feed_dict=feed_dict)

                # for i, loss in enumerate(loss_data):
                #     full_train_logs[self.log_loss_titles[i]] += [loss]

                # Check for update_switching
                if self.regime == "coordinate" and train_time % self.switch_frequency and self._update_switcher(total_loss):
                    total_loss = []

                if train_time % target_update_freq == 0:
                    if self.use_observed_policy:
                        self.sess.run([self.update_target_fn, self.update_pi_target_fn])
                    else:
                        self.sess.run(self.update_target_fn)

            # Do some Q-learning post-touches
            for i in range(int(dyn_pretrain_iters/4)):
                demo_batch = sample_batch(rollouts, train_idxes, batch_size)
                feed_dict = self._get_feed_dict(demo_batch, constraints)
                cqs, _ = self.sess.run([self.constraint_qs_t, temp_q_update], feed_dict=feed_dict)

                if i % target_update_freq == 0:
                    self.sess.run(self.update_target_fn)

                if i % 1000 == 0:
                    val_loss = self.sess.run(self.td_err_sgt, feed_dict=val_feed)
                    print("{} Q-Learning Loss: {}".format(i, val_loss))
                    print(np.max(cqs), np.min(cqs))

                if i % tab_save_freq == 0:
                    # self.mdp only used in this section, can remove that dependency if we want
                    q_vals = self.sess.run([self.test_qs_t], feed_dict={self.test_q_obs_t_feats_ph: q_states})[0]
                    print(train_time, np.max(q_vals), np.min(q_vals))
                    adt_logits = self.sess.run([self.pred_obs], feed_dict={self.demo_tile_t_ph: adt_samples[:,0][np.newaxis].T,
                                                                    self.demo_act_t_ph: adt_samples[:,1][np.newaxis].T})[0]
                    adt_probs = softmax(adt_logits)
                    adt_probs = adt_probs.reshape(self.mdp.tile_types, self.mdp.num_actions, self.mdp.num_directions)
                    pkl.dump(q_vals, open(os.path.join(tab_model_out_dir, 'q_vals_{}.pkl'.format(train_time+i)), 'wb'))
                    pkl.dump(adt_probs, open(os.path.join(tab_model_out_dir, 'adt_probs_{}.pkl'.format(train_time+i)), 'wb'))
                    if self.regime == "coordinate":
                        _run.log_scalar("coordinate_regime", "Q_tuning", train_time + i)


        except KeyboardInterrupt:
            print("Experiment Interrupted at timestep {}".format(train_time))
            pass

        # Save as file
        q_vals = self.sess.run([self.test_qs_t], feed_dict={self.test_q_obs_t_feats_ph: q_states})[0]
        adt_logits = self.sess.run([self.pred_obs], feed_dict={self.demo_tile_t_ph: adt_samples[:, 0][np.newaxis].T,
                                                               self.demo_act_t_ph: adt_samples[:, 1][np.newaxis].T})[0]
        adt_probs = softmax(adt_logits)
        adt_probs = adt_probs.reshape(self.mdp.tile_types, self.mdp.num_actions, self.mdp.num_directions)
        pkl.dump(q_vals, open(os.path.join(tab_model_out_dir, 'final_q_vals.pkl'), 'wb'))
        pkl.dump(adt_probs, open(os.path.join(tab_model_out_dir, 'final_adt_probs.pkl'), 'wb'))

        pkl.dump(self.mdp, open(os.path.join(out_dir, 'mdp.pkl'), 'wb'))

        save_tf_vars(self.sess, self.q_scope, q_f)
        save_tf_vars(self.sess, self.dyn_scope, dyn_f)
        return q_f, dyn_f if _run is None else q_f, dyn_f, full_train_logs


    def _update_switcher(self, losses, running_dist=50):

        if len(losses) < self.update_horizons[self.curr_update_index]:
            return False

        if self.prev_bests[self.curr_update_index] == np.inf:
            self.prev_bests[self.curr_update_index] = (sum(losses[:running_dist])/running_dist)

        recent_loss = (sum(losses[-running_dist:])/running_dist)
        improvement = self.prev_bests[self.curr_update_index] / (recent_loss + 1e-12) - 1
        switch = improvement < self.improvement_proportions[self.curr_update_index]

        if switch:
            self.prev_bests[self.curr_update_index] = np.inf
            if self.use_observed_policy and self.observed_updates_progression[self.curr_update_index]:
                self.sess.run(self.update_observed_policy_fn)
            self.curr_update_index = (self.curr_update_index + 1) % len(self.update_progression)
            print("Loss improvement at {}, switching to loss config {}".format(improvement, self.curr_update_index))
            self.curr_loss = self.loss_progression[self.curr_update_index]
            self.curr_update = self.update_progression[self.curr_update_index]
        else:
            self.prev_bests[self.curr_update_index] = recent_loss

        return switch


    def _get_feed_dict(self, batch, constraints, true_qs=None):

        feed_dict = {
            self.demo_obs_t_feats_ph: batch[1],
            self.demo_act_t_ph: batch[2],
            self.demo_dir_t_ph: batch[5],
            self.demo_tile_t_ph: batch[6],
            self.demo_batch_size_ph: batch[2].shape[0],
            self.constraint_obs_t_feats_ph: constraints[0],
            self.constraint_act_t_ph: constraints[1],
            self.constraint_rew_t_ph: constraints[2],
            self.constraint_next_obs_t_feats_ph: constraints[3],
            self.constraint_tile_t_ph: constraints[4],
            self.constraint_batch_size_ph: constraints[0].shape[0]}

        if true_qs is not None:
            feed_dict[self.true_qs_ph] = true_qs

        return feed_dict
