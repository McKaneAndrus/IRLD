import tensorflow as tf
from utils.tf_utils import build_mlp
from utils.learning_utils import sample_batch, softmax
import numpy as np
from utils.frank_wolfe_utils import find_min_norm_element
from utils.tf_utils import save_tf_vars, load_tf_vars
import os
import pickle as pkl


class InverseDynamicsLearner():

    def __init__(self, mdp, sess, gamma=0.99, adt=True, mellowmax=None, boltz_beta=50, mlp_params=None,
                    alpha=1e-4, beta1=0.9, beta2=0.999999, seed=0, dyn_scope="Dynamics", q_scope="Qs"):

        mlp_params = {} if mlp_params is None else mlp_params

        self.mdp = mdp
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.seed = seed

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

        q_n_layers = mlp_params.get('q_n_layers', 2)
        q_layer_size = mlp_params.get('q_layer_size', 2048)
        q_layer_activation = mlp_params.get('q_layer_activation', tf.nn.tanh)
        q_output_activation = mlp_params.get('q_output_activation', None)

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
                             activation=q_layer_activation, output_activation=q_output_activation) * boltz_beta

        demo_v_t = tf.reduce_logsumexp(demo_q_t, axis=1)

        action_indexes = tf.concat([tf.expand_dims(tf.range(self.demo_batch_size_ph), 1), self.demo_act_t_ph], axis=1)

        act_log_likelihoods = tf.gather_nd(demo_q_t, action_indexes) - demo_v_t

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
                activation=dyn_layer_activation, output_activation=dyn_output_activation)
        else:
            self.pred_obs = build_mlp(
                tf.concat((demo_tile_t_one_hot, self.demo_act_t_ph), axis=1),
                n_dirs, dyn_scope,
                n_layers=dyn_n_layers, size=dyn_layer_size,
                activation=dyn_layer_activation, output_activation=dyn_output_activation)


        trans_log_likelihoods = tf.gather_nd(self.pred_obs, dir_indexes) - tf.reduce_logsumexp(self.pred_obs, axis=1)

        self.neg_avg_trans_log_likelihood = -tf.reduce_mean(trans_log_likelihoods)



        ca_indexes = tf.concat([tf.expand_dims(tf.range(self.constraint_batch_size_ph), 1), self.constraint_act_t_ph], axis=1)

        self.constraint_qs_t = build_mlp(self.constraint_obs_t_feats_ph,
                                    n_act_dim, q_scope,
                                    n_layers=q_n_layers, size=q_layer_size,
                                    activation=q_layer_activation, output_activation=q_output_activation,
                                    reuse=True)

        constraint_q_t = tf.gather_nd(self.constraint_qs_t, ca_indexes)

        # Predicted constraint next state given inv dyns
        if adt:
            constraint_pred_obs = build_mlp(
                tf.concat((self.constraint_act_t_ph, constraint_tile_t_one_hot), axis=1),
                n_dirs, dyn_scope,
                n_layers=dyn_n_layers, size=dyn_layer_size,
                activation=dyn_layer_activation, output_activation=dyn_output_activation,
                reuse=True)
        else:
            constraint_pred_obs = build_mlp(
                tf.concat((self.constraint_obs_t_feats_ph, self.constraint_act_t_ph), axis=1),
                n_dirs, dyn_scope,
                n_layers=dyn_n_layers, size=dyn_layer_size,
                activation=dyn_layer_activation, output_activation=dyn_output_activation,
                reuse=True)

        constraint_sprimes_reshaped = tf.reshape(self.constraint_next_obs_t_feats_ph,
                                                 (self.constraint_batch_size_ph * n_dirs, n_obs_feats))

        # Q-values used to calculate 'V' in the bellman-residual
        cqtp1_misshaped = build_mlp(constraint_sprimes_reshaped,
                                    n_act_dim, scope="target",
                                    n_layers=q_n_layers, size=q_layer_size,
                                    activation=q_layer_activation, output_activation=q_output_activation,
                                    reuse=False)

        constraint_q_tp1 = tf.reshape(tf.stop_gradient(cqtp1_misshaped), (self.constraint_batch_size_ph, n_dirs, n_act_dim))

        # cqtp1_misshaped = build_mlp(constraint_sprimes_reshaped,
        #                             n_act_dim, scope=q_scope,
        #                             n_layers=q_n_layers, size=q_layer_size,
        #                             activation=q_layer_activation, output_activation=q_output_activation,
        #                             reuse=True)
        #
        # constraint_q_tp1 = tf.reshape(cqtp1_misshaped, (self.constraint_batch_size_ph, n_dirs, n_act_dim))

        if mellowmax is None:
            # constraint_v_tp1 = tf.reduce_logsumexp(constraint_q_tp1, axis=2)
            constraint_pi_tp1 = tf.nn.softmax(constraint_q_tp1 * boltz_beta, axis=2)
            constraint_v_tp1 = tf.reduce_sum(constraint_q_tp1 * constraint_pi_tp1, axis=2)
        else:
            constraint_v_tp1 = (tf.reduce_logsumexp(constraint_q_tp1 * mellowmax, axis=2) - np.log(n_act_dim)) / mellowmax



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

        #Experimental
        # self.best_ntll = 1.0
        # self.delta_ll_td_err = self.td_err * self.neg_avg_act_log_likelihood * tf.minimum(self.neg_avg_trans_log_likelihood - self.best_ntll, 1e-12)
        self.lqsafer = self.neg_avg_act_log_likelihood / (1000 - self.td_err + 1/self.neg_avg_trans_log_likelihood)
        self.llt_weighted_td_err = self.td_err_sgq * self.neg_avg_trans_log_likelihood ** 5
        self.lla_weighted_td_err = self.td_err * self.neg_avg_act_log_likelihood * self.neg_avg_trans_log_likelihood ** 5
        # self.td_err_weighted_ll = 1/(self.ll_weighted_td_err + 1e-8)

        # Set up target network
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=q_scope)
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target")
        update_target_fn_vars = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_fn_vars.append(var_target.assign(var))
        self.update_target_fn = tf.group(*update_target_fn_vars)

        # True Qs for testing
        self.test_q_obs_t_feats_ph = tf.placeholder(tf.int32, [None, n_obs_feats], name="tqotf")
        self.test_qs_t = build_mlp(self.test_q_obs_t_feats_ph,
                                    n_act_dim, q_scope,
                                    n_layers=q_n_layers, size=q_layer_size,
                                    activation=q_layer_activation, output_activation=q_output_activation,
                                    reuse=True)
        self.true_q_err = tf.reduce_sum((self.true_qs_ph - self.test_qs_t)**2)

        self.loss_fns = np.array([self.neg_avg_act_log_likelihood, self.neg_avg_trans_log_likelihood,
                         self.td_err, self.td_err_sgq, self.td_err_sgt, self.llt_weighted_td_err, self.lqsafer, self.lla_weighted_td_err]) #, self.true_q_err])
        self.loss_fns_titles = np.array(['nall', 'ntll', 'tde', 'tde_sg_q', 'tde_sg_t', 'llt_tde', 'lqsafe', 'lla_tde']) #, 'true_q_err'])

        self.log_losses = self.loss_fns[[0,1,2,5,6,7]]
        self.log_loss_titles = self.loss_fns_titles[[0,1,2,5,6,7]]

        self.regime = None


    def initialize_training_regime(self, regime, regime_params, validation_freq=20):
        assert(regime in ["MGDA", "coordinate", "weighted"])

        if self.regime is not None:
            print("Already initialized, create new learner instead")
            return

        self.regime = regime
        self.validation_freq = validation_freq

        if regime == "weighted":
            losses = regime_params['losses']
            loss_weights = regime_params['loss_weights']
            assert len(losses) == len(loss_weights)
            self.weighted_loss = sum([self.loss_fns[losses[i]] * loss_weights[i] for i in range(len(losses))])
            self.log_losses = np.append(self.log_losses, [self.weighted_loss])
            self.log_loss_titles = np.append(self.log_loss_titles, ["weighted_loss"])
            if regime_params['clip_global']:
                optimizer = tf.train.AdamOptimizer(self.alpha, self.beta1, self.beta2)
                gradients, variables = zip(*optimizer.compute_gradients(self.weighted_loss))
                gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                self.update = [optimizer.apply_gradients(zip(gradients, variables))]
            else:
                self.update = tf.train.AdamOptimizer(self.alpha, self.beta1, self.beta2).minimize(self.weighted_loss)

        if regime == "coordinate":
            # Coordinate descent training regime
            self.update_horizon = regime_params["horizon"]
            self.improvement_proportions = regime_params["improvement_proportions"]
            self.prev_bests = [np.inf for _ in range(len(self.improvement_proportions))]
            self.switch_frequency = regime_params["switch_frequency"]
            self.alphas = regime_params['alphas']
            self.model_save_loss = sum(self.log_losses * np.array(regime_params["model_save_weights"]))
            # Update progression is a list of list of ints in the range(len(self.loss_fns))
            self.loss_progression = [sum(self.loss_fns[config]) for config in regime_params["update_progression"]]
            #TODO Check for equivalent losses
            if regime_params['clip_global']:
                self.update_progression = []
                for i,loss in enumerate(self.loss_progression):
                    optimizer = tf.train.AdamOptimizer(self.alpha, self.beta1, self.beta2)
                    gradients, variables = zip(*optimizer.compute_gradients(loss))
                    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                    self.update_progression += [optimizer.apply_gradients(zip(gradients, variables))]
                else:
                    self.update_progression = [tf.train.AdamOptimizer(self.alphas[i], self.beta1, self.beta2).minimize(
                            loss, name="opt_{}".format(i)) for i,loss in enumerate(self.loss_progression)]
            if regime_params["initial_update"] is not None:
                self.curr_update_index = -1
                self.curr_loss = sum(self.loss_fns[regime_params["initial_update"]])
                self.curr_update = tf.train.AdamOptimizer(self.alpha, self.beta1, self.beta2).minimize(self.curr_loss)
            else:
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

        if dyn_pretrain_iters > 0:
            temp_update1 = tf.train.AdamOptimizer(self.alpha, self.beta1, self.beta2).minimize(
                self.neg_avg_trans_log_likelihood)
            temp_update2 = tf.train.AdamOptimizer(self.alpha, self.beta1, self.beta2).minimize(self.td_err_sgt)

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

        if dyn_pretrain_iters > 0:
            print("Pretraining the dynamics model with baseline method.")
            for i in range(int(dyn_pretrain_iters/2)):
                demo_batch = sample_batch(rollouts, train_idxes, batch_size)
                feed_dict = self._get_feed_dict(demo_batch, constraints)
                self.sess.run(temp_update1, feed_dict=feed_dict)

                if i % self.validation_freq == 0:
                    val_loss = self.sess.run(self.neg_avg_trans_log_likelihood, feed_dict=val_feed)

                if i % 1000 == 0:
                    print("{} Transition Loss: {}".format(i, val_loss))
            for i in range(dyn_pretrain_iters+1):
                demo_batch = sample_batch(rollouts, train_idxes, batch_size)
                feed_dict = self._get_feed_dict(demo_batch, constraints)
                cqs, _ = self.sess.run([self.constraint_qs_t, temp_update2], feed_dict=feed_dict)

                if i % self.validation_freq == 0:
                    val_loss = self.sess.run(self.td_err_sgt, feed_dict=val_feed)

                if i % target_update_freq == 0:
                    self.sess.run(self.update_target_fn)

                if i % 1000 == 0:
                    print("{} Q-Learning Loss: {}".format(i, val_loss))
                    print(np.max(cqs), np.min(cqs))


        try:

            for train_time in range(n_training_iters):

                demo_batch = sample_batch(rollouts, train_idxes, batch_size)
                feed_dict = self._get_feed_dict(demo_batch, constraints, true_qs=true_qs)

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
                    if self.regime == "MGDA":
                        print(sol)

                if train_time % tab_save_freq == 0 and train_time != 0:
                    # self.mdp only used in this section, can remove that dependency if we want
                    q_vals = self.sess.run([self.test_qs_t], feed_dict={self.test_q_obs_t_feats_ph: q_states})[0]
                    print(train_time, np.max(q_vals), np.min(q_vals))
                    adt_logits = self.sess.run([self.pred_obs], feed_dict={self.demo_tile_t_ph: adt_samples[:,0][np.newaxis].T,
                                                                    self.demo_act_t_ph: adt_samples[:,1][np.newaxis].T})[0]
                    adt_probs = softmax(adt_logits)
                    adt_probs = adt_probs.reshape(self.mdp.tile_types, self.mdp.num_actions, self.mdp.num_directions)
                    pkl.dump(q_vals, open(os.path.join(tab_model_out_dir, 'q_vals_{}.pkl'.format(train_time)), 'wb'))
                    pkl.dump(adt_probs, open(os.path.join(tab_model_out_dir, 'adt_probs_{}.pkl'.format(train_time)), 'wb'))

                # Check for update_switching
                if self.regime == "coordinate" and train_time % self.switch_frequency and self._update_switcher(total_loss):
                    total_loss = []
                    model_score = self.sess.run(self.model_save_loss, feed_dict=val_feed)
                    if model_score < best_model_score:
                        print("Current model ({}) is better than previous best ({})".format(model_score, best_model_score))
                        # Save as file
                        q_vals = self.sess.run([self.test_qs_t], feed_dict={self.test_q_obs_t_feats_ph: q_states})[0]
                        adt_logits = \
                        self.sess.run([self.pred_obs], feed_dict={self.demo_tile_t_ph: adt_samples[:, 0][np.newaxis].T,
                                                                  self.demo_act_t_ph: adt_samples[:, 1][np.newaxis].T})[
                            0]
                        adt_probs = softmax(adt_logits)
                        adt_probs = adt_probs.reshape(self.mdp.tile_types, self.mdp.num_actions,
                                                      self.mdp.num_directions)
                        pkl.dump(q_vals, open(os.path.join(tab_model_out_dir, 'final_q_vals.pkl'), 'wb'))
                        pkl.dump(adt_probs, open(os.path.join(tab_model_out_dir, 'final_adt_probs.pkl'), 'wb'))

                        pkl.dump(self.mdp, open(os.path.join(out_dir, 'mdp.pkl'), 'wb'))

                        save_tf_vars(self.sess, self.q_scope, q_f)
                        save_tf_vars(self.sess, self.dyn_scope, dyn_f)
                        best_model_score = model_score

                if train_time % target_update_freq == 0:
                    self.sess.run(self.update_target_fn)



        except KeyboardInterrupt:
            print("Experiment Interrupted at timestep {}".format(train_time))
            pass

        if self.regime == "coordinate":
            if best_model_score == np.inf:
                save_tf_vars(self.sess, self.q_scope, q_f)
                save_tf_vars(self.sess, self.dyn_scope, dyn_f)
            return q_f, dyn_f if _run is None else q_f, dyn_f, full_train_logs

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

        if len(losses) < self.update_horizon:
            return False

        if self.prev_bests[self.curr_update_index] == np.inf:
            self.prev_bests[self.curr_update_index] = (sum(losses[:running_dist])/running_dist)
            return False


        recent_loss = (sum(losses[-running_dist:])/running_dist)
        improvement = 1 - recent_loss/self.prev_bests[self.curr_update_index]
        switch = improvement > self.improvement_proportions[self.curr_update_index]

        if switch:
            self.prev_bests[self.curr_update_index] = recent_loss
            self.curr_update_index = (self.curr_update_index + 1) % len(self.update_progression)
            print("Loss improvement at {}, switching to loss config {}".format(improvement, self.curr_update_index))
            self.curr_loss = self.loss_progression[self.curr_update_index]
            self.curr_update = self.update_progression[self.curr_update_index]

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


# class WeightedMGDAInverseDynamicsLearner(InverseDynamicsLearner):
#
#     def initialize_training_regime(self, regime_params, validation_freq=20):
#
#
#         self.validation_freq = validation_freq
#
#         # Update switching initilization
#         self.update_horizon = regime_params["horizon"]
#         self.improvement_proportionss = regime_params["improvement_proportionss"]
#         self.switch_frequency = regime_params["switch_frequency"]
#         self.model_save_loss = sum(self.log_losses * np.array(regime_params["model_save_weights"]))
#         # Update progression is a list of list of ints in the range(len(self.loss_fns))
#         self.method_progression = regime_params["method_progression"]
#         self.curr_method_index = 0
#         self.curr_method = self.method_progression[self.curr_method_index]
#
#
#         # Weighted Training Initialization Section
#         losses = regime_params['losses']
#         loss_weights = regime_params['loss_weights']
#         assert len(losses) == len(loss_weights)
#         self.weighted_loss = sum([self.loss_fns[losses[i]] * loss_weights[i] for i in range(len(losses))])
#         self.log_losses = np.append(self.log_losses, [self.weighted_loss])
#         self.log_loss_titles = np.append(self.log_loss_titles, ["weighted_loss"])
#         self.weighted_update = tf.train.AdamOptimizer(self.alpha, self.beta1, self.beta1).minimize(self.weighted_loss)
#
#
#         # MGDA Training Initialization Section
#         self.num_tasks = len(regime_params["loss_configurations"])
#         self.scales = [tf.placeholder(tf.float32, [1], name="scale{}".format(i)) for i in range(self.num_tasks)]
#         self.mgda_task_losses = [sum(self.loss_fns[config]) for config in regime_params["loss_configurations"]]
#
#         q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.q_scope)
#         t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.dyn_scope)
#         all_vars = q_vars + t_vars
#         opts = [tf.train.AdamOptimizer(self.alpha, self.beta1, self.beta1) for _ in range(self.num_tasks)]
#         self.all_gvs = [opts[i].compute_gradients(self.mgda_task_losses[i], all_vars) for i in range(self.num_tasks)]
#         sep_gvs = [[a for a in zip(*gv_set)] for gv_set in self.all_gvs]
#
#         self.grads = [[tf.clip_by_norm(sep_gvs[i][0][j], self.mgda_task_losses[i]) for j in range(len(sep_gvs[i][0]))]
#                             for i in range(self.num_tasks)]
#         self.gvars = [gv[1] for gv in sep_gvs]
#
#         self.scaled_grads = [[tf.multiply(g, self.scales[i]) for g in self.grads[i]] for i in range(self.num_tasks)]
#         self.mgda_updates = [opts[i].apply_gradients(zip(self.scaled_grads[i], self.gvars[i])) for i in range(self.num_tasks)]
#
#
#     def train(self, n_training_iters, rollouts, train_idxes, batch_size, constraints, val_demo_batch, out_dir,
#                     q_states, adt_samples, dyn_pretrain_iters=0, tab_save_freq = 1000, _run=None, true_qs=None,
#                     verbose=True, q_source_path=None, dyn_source_path=None):
#
#         assert(self.regime is not None)
#
#         if dyn_pretrain_iters > 0:
#             temp_update = tf.train.AdamOptimizer(self.alpha, self.beta1, self.beta1).minimize(self.neg_avg_trans_log_likelihood)
#
#         tf.global_variables_initializer().run(session=self.sess)
#
#         # Tabular logging setup
#         tab_model_out_dir = os.path.join(out_dir, "tab")
#         if not os.path.exists(tab_model_out_dir):
#             os.makedirs(tab_model_out_dir)
#         if true_qs is not None:
#             pkl.dump(true_qs, open(os.path.join(tab_model_out_dir, 'true_q_vals.pkl'), 'wb'))
#         pkl.dump(self.mdp.adt_mat, open(os.path.join(tab_model_out_dir, 'true_adt_probs.pkl'), 'wb'))
#
#         # Load pretrained models with same scope
#         if q_source_path is not None:
#             load_tf_vars(self.sess, self.q_scope, q_source_path)
#         if dyn_source_path is not None:
#             load_tf_vars(self.sess, self.dyn_scope, dyn_source_path)
#
#         q_f = os.path.join(out_dir, "q_fn")
#         dyn_f = os.path.join(out_dir, "dyn_fn")
#
#         train_time = 0
#         # Initialize train_logs to have an array for train and validation evaluations
#         full_train_logs = {}
#         for k in self.log_loss_titles:
#             full_train_logs[k] = []
#             full_train_logs["val_" + k] = []
#
#         if self.regime == "coordinate":
#             total_loss = []
#             best_model_score = np.inf
#
#         if self.regime == "MGDA":
#             full_train_logs["frank_wolfe"] = []
#
#         val_feed = self._get_feed_dict(val_demo_batch, constraints, true_qs=true_qs)
#
#         if dyn_pretrain_iters > 0:
#             print("Pretraining the dynamics model with baseline method.")
#             for i in range(dyn_pretrain_iters):
#                 demo_batch = sample_batch(rollouts, train_idxes, batch_size)
#                 feed_dict = self._get_feed_dict(demo_batch, constraints)
#                 self.sess.run(temp_update, feed_dict=feed_dict)
#
#                 if i % self.validation_freq == 0:
#                     val_loss = self.sess.run(self.neg_avg_trans_log_likelihood, feed_dict=val_feed)
#
#                 if i % 1000 == 0:
#                     print("{}: {}".format(i, val_loss))
#
#
#         try:
#
#             for train_time in range(n_training_iters):
#
#                 demo_batch = sample_batch(rollouts, train_idxes, batch_size)
#                 feed_dict = self._get_feed_dict(demo_batch, constraints, true_qs=true_qs)
#
#                 if self.regime == "weighted":
#                     # loss_data = self.sess.run(list(self.log_losses) + [self.update], feed_dict=feed_dict)[:-1]
#                     self.sess.run(self.update, feed_dict=feed_dict)
#
#                 if self.regime == "coordinate":
#                     loss_data = self.sess.run([self.curr_loss, self.curr_update], feed_dict=feed_dict)[0]
#                     total_loss.append(loss_data)
#
#                 if self.regime == "MGDA":
#                     loss_and_gvs = self.sess.run(list(self.log_losses) + [self.all_gvs], feed_dict=feed_dict)
#                     loss_data, gvs = loss_and_gvs[:-1], loss_and_gvs[-1]
#
#                     grad_arrays = {}
#                     for t in range(self.num_tasks):
#                         # Compute gradients of each loss function wrt parameters
#                         grad_arrays[t] = np.concatenate([gvs[t][i][0].flatten() for i in range(len(gvs))])
#
#                     # Frank-Wolfe iteration to compute scales.
#                     sol, min_norm = find_min_norm_element(grad_arrays)
#                     full_train_logs["frank_wolfe"] += [sol]
#
#                     # Scaled back-propagation
#                     for i, s in enumerate(self.scales):
#                         feed_dict[s] = [sol[i]]
#
#                     # Inefficient, takes gradients twice when we could just feed the gradients back in
#                     self.sess.run(self.mgda_updates, feed_dict=feed_dict)
#
#                 # for i, loss in enumerate(loss_data):
#                 #     full_train_logs[self.log_loss_titles[i]] += [loss]
#
#                 if train_time % self.validation_freq == 0:
#                     val_loss_data = self.sess.run(list(self.log_losses), feed_dict=val_feed)
#                     for i, loss in enumerate(val_loss_data):
#                         metric_name = "val_" + self.log_loss_titles[i]
#                         full_train_logs[metric_name] += [loss]
#                         if _run is not None:
#                             _run.log_scalar(metric_name, loss, train_time)
#
#                 if train_time % 500 == 0 and verbose:
#                     print(str(train_time) + "\t" + "\t".join([str(k) + ": " +
#                                                 str(round(full_train_logs["val_" + k][-1],7)) for k in self.log_loss_titles]))
#                     if self.regime == "MGDA":
#                         print(sol)
#
#                 if train_time % tab_save_freq == 0 and train_time != 0:
#                     # self.mdp only used in this section, can remove that dependency if we want
#                     q_vals = self.sess.run([self.test_qs_t], feed_dict={self.test_q_obs_t_feats_ph: q_states})[0]
#                     adt_logits = self.sess.run([self.pred_obs], feed_dict={self.demo_tile_t_ph: adt_samples[:,0][np.newaxis].T,
#                                                                     self.demo_act_t_ph: adt_samples[:,1][np.newaxis].T})[0]
#                     adt_probs = softmax(adt_logits)
#                     adt_probs = adt_probs.reshape(self.mdp.tile_types, self.mdp.num_actions, self.mdp.num_directions)
#                     pkl.dump(q_vals, open(os.path.join(tab_model_out_dir, 'q_vals_{}.pkl'.format(train_time)), 'wb'))
#                     pkl.dump(adt_probs, open(os.path.join(tab_model_out_dir, 'adt_probs_{}.pkl'.format(train_time)), 'wb'))
#
#                 # Check for update_switching
#                 if self.regime == "coordinate" and train_time % self.switch_frequency and self._update_switcher(total_loss):
#                     total_loss = []
#                     model_score = self.sess.run(self.model_save_loss, feed_dict=val_feed)
#                     if model_score < best_model_score:
#                         print("Current model ({}) is better than previous best ({})".format(model_score, best_model_score))
#                         # Save as file
#                         q_vals = self.sess.run([self.test_qs_t], feed_dict={self.test_q_obs_t_feats_ph: q_states})[0]
#                         adt_logits = \
#                         self.sess.run([self.pred_obs], feed_dict={self.demo_tile_t_ph: adt_samples[:, 0][np.newaxis].T,
#                                                                   self.demo_act_t_ph: adt_samples[:, 1][np.newaxis].T})[
#                             0]
#                         adt_probs = softmax(adt_logits)
#                         adt_probs = adt_probs.reshape(self.mdp.tile_types, self.mdp.num_actions,
#                                                       self.mdp.num_directions)
#                         pkl.dump(q_vals, open(os.path.join(tab_model_out_dir, 'final_q_vals.pkl'), 'wb'))
#                         pkl.dump(adt_probs, open(os.path.join(tab_model_out_dir, 'final_adt_probs.pkl'), 'wb'))
#
#                         pkl.dump(self.mdp, open(os.path.join(out_dir, 'mdp.pkl'), 'wb'))
#
#                         save_tf_vars(self.sess, self.q_scope, q_f)
#                         save_tf_vars(self.sess, self.dyn_scope, dyn_f)
#                         best_model_score = model_score
#
#         except KeyboardInterrupt:
#             print("Experiment Interrupted at timestep {}".format(train_time))
#             pass
#
#         if self.regime == "coordinate":
#             if best_model_score == np.inf:
#                 save_tf_vars(self.sess, self.q_scope, q_f)
#                 save_tf_vars(self.sess, self.dyn_scope, dyn_f)
#             return q_f, dyn_f if _run is None else q_f, dyn_f, full_train_logs
#
#         # Save as file
#         q_vals = self.sess.run([self.test_qs_t], feed_dict={self.test_q_obs_t_feats_ph: q_states})[0]
#         adt_logits = self.sess.run([self.pred_obs], feed_dict={self.demo_tile_t_ph: adt_samples[:, 0][np.newaxis].T,
#                                                                self.demo_act_t_ph: adt_samples[:, 1][np.newaxis].T})[0]
#         adt_probs = softmax(adt_logits)
#         adt_probs = adt_probs.reshape(self.mdp.tile_types, self.mdp.num_actions, self.mdp.num_directions)
#         pkl.dump(q_vals, open(os.path.join(tab_model_out_dir, 'final_q_vals.pkl'), 'wb'))
#         pkl.dump(adt_probs, open(os.path.join(tab_model_out_dir, 'final_adt_probs.pkl'), 'wb'))
#
#         pkl.dump(self.mdp, open(os.path.join(out_dir, 'mdp.pkl'), 'wb'))
#
#         save_tf_vars(self.sess, self.q_scope, q_f)
#         save_tf_vars(self.sess, self.dyn_scope, dyn_f)
#         return q_f, dyn_f if _run is None else q_f, dyn_f, full_train_logs
#
#     def _update_switcher(self, losses):
#
#         if len(losses) < self.update_horizons[self.curr_update_index]:
#             return False
#
#         slope = np.polyfit(np.arange(self.update_horizon), losses[-self.update_horizon:], 1)[0]
#         switch = -slope < self.improvement_proportions
#
#         if switch:
#             print("Loss delta at {}, switching to loss config {}".format(-slope, self.curr_update_index))
#
#         return switch
#
