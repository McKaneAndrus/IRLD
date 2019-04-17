import tensorflow as tf
from utils.tf_utils import build_mlp
from utils.learning_utils import sample_batch, update_switcher, softmax
from envs.environment_visualization_utils import plot_values
import matplotlib.pyplot as plt
import numpy as np
from utils.frank_wolfe_utils import find_min_norm_element
from copy import deepcopy as copy


def create_tf_model(sess, mdp, q_scope, invsas_scope, invadt_scope,
                    gamma, alpha, beta1, beta2, sq_td_err_penalty, trans_penalty,
                    t_err_penalty, q_err_penalty, constraint_batch_size,
                    q_n_layers, q_activation, q_output_activation, invdyn_n_layers,
                    invdyn_layer_size, invdyn_output_activation, n_act_dim, featurize_acts,
                    n_dirs, boltz_beta):
    gamma = 0.99
    alpha = 1e-4
    beta1 = 0.9
    beta2 = 0.999999
    sq_td_err_penalty = 1
    trans_penalty = 1
    t_err_penalty = 1e0
    q_err_penalty = 1e0
    constraint_batch_size = 512

    q_n_layers = 2
    q_layer_size = 2048
    q_activation = tf.nn.tanh
    q_output_activation = None

    invdyn_n_layers = 1
    invdyn_layer_size = 256
    invdyn_activation = tf.nn.relu
    invdyn_output_activation = None


    n_act_dim = 5

    featurize_acts = lambda a: a
    n_dirs = 5

    # Boltz-beta determines the "rationality" of the agent being modeled.
    # Setting it to higher values corresponds to "pure rationality"
    boltz_beta = 50

    n_obs_feats = mdp.nrow + mdp.ncol
    # Demo placeholders are for the action-likelihood and transition-likelihood portions of the loss
    # Constraint placeholders are for the bellman-residual penalty

    demo_tile_t_ph = tf.placeholder(tf.int32, [None, 1], name="dtt")
    demo_act_t_ph = tf.placeholder(tf.int32, [None, 1], name="dat")
    demo_obs_t_feats_ph = tf.placeholder(tf.int32, [None, n_obs_feats], name="dotf")
    demo_next_obs_t_feats_ph = tf.placeholder(tf.int32, [None, n_obs_feats], name="dnotf")
    demo_dir_t_ph = tf.placeholder(tf.int32, [None, 1], name="ddt")
    demo_batch_size_ph = tf.placeholder(tf.int32, name="dbs")

    constraint_tile_t_ph = tf.placeholder(tf.int32, [None, 1], name="ctt")
    constraint_obs_t_feats_ph = tf.placeholder(tf.int32, [None, n_obs_feats], name="cotf")
    constraint_act_t_ph = tf.placeholder(tf.int32, [None, 1], name="cat")
    constraint_rew_t_ph = tf.placeholder(tf.float32, [None], name="crt")
    constraint_next_obs_t_feats_ph = tf.placeholder(tf.int32, [None, n_dirs, n_obs_feats], name="cnotf")
    constraint_batch_size_ph = tf.placeholder(tf.int32, name="cbs")

    demo_q_t = build_mlp(demo_obs_t_feats_ph,
                         n_act_dim, q_scope,
                         n_layers=q_n_layers, size=q_layer_size,
                         activation=q_activation, output_activation=q_output_activation) * boltz_beta

    demo_v_t = tf.reduce_logsumexp(demo_q_t, axis=1)

    action_indexes = tf.concat([tf.expand_dims(tf.range(demo_batch_size_ph), 1), demo_act_t_ph], axis=1)

    act_log_likelihoods = tf.gather_nd(demo_q_t, action_indexes) - demo_v_t

    neg_avg_act_log_likelihood = -tf.reduce_mean(act_log_likelihoods)

    # Encoded the potential to use "sas" transitions, which are simply state, action, state observations,
    # and "adt" transitions -- post-processed to determine (a)ction taken, (d)irection moved, and starting (t)ile-type
    # Learning an "adt" transition model should be much easier

    sas_pred_obs = build_mlp(
        tf.concat((demo_obs_t_feats_ph, demo_act_t_ph), axis=1),
        n_dirs, invsas_scope,
        n_layers=invdyn_n_layers, size=invdyn_layer_size,
        activation=invdyn_activation, output_activation=invdyn_output_activation)

    adt_pred_dir = build_mlp(
        tf.concat((demo_act_t_ph, demo_tile_t_ph), axis=1),
        n_dirs, invadt_scope,
        n_layers=invdyn_n_layers, size=invdyn_layer_size,
        activation=invdyn_activation, output_activation=invdyn_output_activation)

    dir_indexes = tf.concat([tf.expand_dims(tf.range(demo_batch_size_ph), 1), demo_dir_t_ph], axis=1)

    adt_log_likelihoods = tf.gather_nd(adt_pred_dir, dir_indexes) - tf.reduce_logsumexp(adt_pred_dir, axis=1)

    neg_avg_adt_log_likelihood = -tf.reduce_mean(adt_log_likelihoods)

    sas_log_likelihoods = tf.gather_nd(sas_pred_obs, dir_indexes) - tf.reduce_logsumexp(sas_pred_obs, axis=1)

    neg_avg_sas_log_likelihood = -tf.reduce_mean(sas_log_likelihoods)

    ca_indexes = tf.concat([tf.expand_dims(tf.range(constraint_batch_size_ph), 1), constraint_act_t_ph], axis=1)

    constraint_q_ts = build_mlp(constraint_obs_t_feats_ph,
                                n_act_dim, q_scope,
                                n_layers=q_n_layers, size=q_layer_size,
                                activation=q_activation, output_activation=q_output_activation,
                                reuse=True)

    constraint_q_t = tf.gather_nd(constraint_q_ts, ca_indexes)

    # Predicted constraint next state given inv dyns
    constraint_sas_pred_obs = build_mlp(
        tf.concat((constraint_obs_t_feats_ph, constraint_act_t_ph), axis=1),
        n_dirs, invsas_scope,
        n_layers=invdyn_n_layers, size=invdyn_layer_size,
        activation=invdyn_activation, output_activation=invdyn_output_activation,
        reuse=True)

    constraint_adt_pred_dir = build_mlp(
        tf.concat((constraint_act_t_ph, constraint_tile_t_ph), axis=1),
        n_dirs, invadt_scope,
        n_layers=invdyn_n_layers, size=invdyn_layer_size,
        activation=invdyn_activation, output_activation=invdyn_output_activation,
        reuse=True)

    constraint_sprimes_reshaped = tf.reshape(constraint_next_obs_t_feats_ph,
                                             (constraint_batch_size_ph * n_dirs, n_obs_feats))

    # Q-values used to calculate 'V' in the bellman-residual
    cqtp1_misshaped = build_mlp(constraint_sprimes_reshaped,
                                n_act_dim, q_scope,
                                n_layers=q_n_layers, size=q_layer_size,
                                activation=q_activation, output_activation=q_output_activation,
                                reuse=True)

    constraint_q_tp1 = tf.reshape(cqtp1_misshaped, (constraint_batch_size_ph, n_dirs, n_act_dim))

    constraint_v_tp1 = tf.reduce_logsumexp(constraint_q_tp1,
                                           axis=2)  # (tf.reduce_logsumexp(constraint_q_tp1 * boltz_beta, axis=2) - np.log(n_act_dim)) / boltz_beta

    # sas bellman residual penalty error
    constraint_sas_pred_probs = tf.nn.softmax(constraint_sas_pred_obs, axis=1)
    sas_V = tf.multiply(constraint_v_tp1, constraint_sas_pred_probs)
    sas_target_t = constraint_rew_t_ph + gamma * tf.reduce_sum(sas_V, axis=1)
    sas_td_err = tf.reduce_mean((constraint_q_t - sas_target_t) ** 2)

    # adt bellman residual penalty error
    constraint_adt_pred_probs = tf.nn.softmax(constraint_adt_pred_dir, axis=1)
    adt_V = tf.multiply(constraint_v_tp1, constraint_adt_pred_probs)
    adt_target_t = constraint_rew_t_ph + gamma * tf.reduce_sum(adt_V, axis=1)
    adt_td_err = tf.reduce_mean((constraint_q_t - adt_target_t) ** 2)

    # sas and adt bellman residual penalty error with a stop gradient on transitions to prevent the bellman update from
    # 'hacking' the transition function to overly explain the demonstrations. This happens a lot early on before the
    # Q-values have obtained much meaning

    sas_V_sgt = tf.multiply(constraint_v_tp1, tf.stop_gradient(constraint_sas_pred_probs))
    sas_target_sgt = constraint_rew_t_ph + gamma * tf.reduce_sum(sas_V_sgt, axis=1)
    sas_td_err_sgt = tf.reduce_mean((constraint_q_t - sas_target_sgt) ** 2)

    adt_V_sgt = tf.multiply(constraint_v_tp1, tf.stop_gradient(constraint_adt_pred_probs))
    adt_target_sgt = constraint_rew_t_ph + gamma * tf.reduce_sum(adt_V_sgt, axis=1)
    adt_td_err_sgt = tf.reduce_mean((constraint_q_t - adt_target_sgt) ** 2)

    # test_constraint_v_tp1 = tf.reduce_logsumexp(test_constraint_q_tp1_ph, axis=2) #- np.log(5)
    # test_adt_V = tf.multiply(test_constraint_v_tp1, test_constraint_adt_pred_probs)
    # test_adt_target = constraint_rew_t_ph + gamma * tf.reduce_sum(test_adt_V, axis=1)
    # indiv_errs = test_constraint_q_t_ph - test_adt_target
    # test_adt_td_err = tf.reduce_mean((test_constraint_q_t_ph - test_adt_target)**2)

    # sas and adt bellman residual penalty error with a stop gradient on q-fn to allow the bellman residual loss to update
    # differences in learned q-vals w.r.t. the dynamics model

    adt_V_sgq = tf.multiply(tf.stop_gradient(constraint_v_tp1), constraint_adt_pred_probs)
    adt_target_sgq = constraint_rew_t_ph + gamma * tf.reduce_sum(adt_V_sgq, axis=1)
    adt_td_err_sgq = tf.reduce_mean((tf.stop_gradient(constraint_q_t) - adt_target_sgq) ** 2)

    # HACK TO GET NON-NONE GRADIENTS
    adt_td_err_sgt += 1e-20 * adt_td_err_sgq
    adt_td_err_sgq += 1e-20 * adt_td_err_sgt

    # Total loss function for the sas formulation
    sas_loss = neg_avg_act_log_likelihood + trans_penalty * neg_avg_sas_log_likelihood + sq_td_err_penalty * sas_td_err

    # Total loss function for the adt formulation
    adt_loss = neg_avg_act_log_likelihood + trans_penalty * neg_avg_adt_log_likelihood + sq_td_err_penalty * adt_td_err

    # Loss function for the adt formulation that only optimizes over the q-value approximation
    adt_q_loss = neg_avg_act_log_likelihood + q_err_penalty * adt_td_err_sgt

    # Loss function for the adt formulation that only optimizes over the dynamics model
    adt_t_loss = neg_avg_adt_log_likelihood + t_err_penalty * adt_td_err_sgq

    # Loss function for the adt formulation that learns unseen dynamics
    adt_learn_loss = neg_avg_act_log_likelihood + t_err_penalty * adt_td_err_sgq

    # Loss function for the adt formulation that regularizes over observed and learned quantities
    adt_regularize_loss = neg_avg_adt_log_likelihood + q_err_penalty * adt_td_err_sgt

    # Total loss function for the adt formulation with a stop-gradient on the transition function for the bellman residual
    adt_brsgt_loss = neg_avg_act_log_likelihood + trans_penalty * neg_avg_adt_log_likelihood + sq_td_err_penalty * adt_td_err_sgt

    # HACK TO GET NON-NONE GRADIENTS
    nall_loss = neg_avg_act_log_likelihood + 1e-20 * neg_avg_adt_log_likelihood
    ntll_loss = neg_avg_adt_log_likelihood + 1e-20 * neg_avg_act_log_likelihood

    sas_update_op = tf.train.AdamOptimizer(alpha, beta1, beta2).minimize(sas_loss)

    adt_update_op = tf.train.AdamOptimizer(alpha, beta1, beta2).minimize(adt_loss)

    adt_brsgt_update_op = tf.train.AdamOptimizer(alpha, beta1, beta2).minimize(adt_brsgt_loss)

    adt_q_update_op = tf.train.AdamOptimizer(alpha, beta1, beta2).minimize(adt_q_loss)

    adt_t_update_op = tf.train.AdamOptimizer(alpha, beta1, beta2).minimize(adt_t_loss)

    adt_learn_update_op = tf.train.AdamOptimizer(alpha, beta1, beta2).minimize(adt_learn_loss)

    adt_regularize_update_op = tf.train.AdamOptimizer(alpha, beta1, beta2).minimize(adt_regularize_loss)

    adt_trans_only_update_op = tf.train.AdamOptimizer(alpha, beta1, beta2).minimize(neg_avg_adt_log_likelihood)

    adt_q_br_update_op = tf.train.AdamOptimizer(alpha, beta1, beta2).minimize(adt_td_err_sgt)

    q_nll_update_op = tf.train.AdamOptimizer(alpha, beta1, beta2).minimize(neg_avg_act_log_likelihood)

    # Uncertainty weighted loss function motivated by https://arxiv.org/pdf/1705.07115.pdf

    sigma_a, sigma_t, sigma_br = tf.Variable(0.0), tf.Variable(0.0), tf.Variable(0.0)

    # Simple sum is sufficient so long as all losses use means
    sigma_log_losses = sigma_a + sigma_t  # + sigma_br

    weighted_na_act_ll = tf.exp(-sigma_a) * neg_avg_act_log_likelihood

    weighted_na_adt_ll = tf.exp(-sigma_t) * neg_avg_adt_log_likelihood

    weighted_na_br = (
                                 adt_td_err * sq_td_err_penalty + adt_td_err_sgt * 3 * sq_td_err_penalty) / 4  # tf.exp(-sigma_br) *

    adt_uncertainty_loss = weighted_na_act_ll + weighted_na_adt_ll + weighted_na_br + sigma_log_losses

    adt_un_update_op = tf.train.AdamOptimizer(alpha, beta1, beta2).minimize(adt_uncertainty_loss)

    # Add in ability to train the q-fn approximator on the true values to test model capacity
    true_qs_ph = tf.placeholder(tf.float32, [None], name="tq")
    true_q_errs = constraint_q_t - true_qs_ph
    true_q_err = tf.reduce_sum((true_q_errs) ** 2)
    true_q_update_op = tf.train.AdamOptimizer(alpha, beta1, beta2).minimize(true_q_err)

    def compute_batch_loss(demo_batch, constraints, step=False, update="adt", true_qs=None):
        feed_dict = {
            demo_obs_t_feats_ph: demo_batch[1],
            demo_act_t_ph: demo_batch[2],
            demo_next_obs_t_feats_ph: demo_batch[4],
            demo_dir_t_ph: demo_batch[5],
            demo_tile_t_ph: demo_batch[6],
            demo_batch_size_ph: demo_batch[2].shape[0],
            constraint_obs_t_feats_ph: constraints[0],
            constraint_act_t_ph: constraints[1],
            constraint_rew_t_ph: constraints[2],
            constraint_next_obs_t_feats_ph: constraints[3],
            constraint_tile_t_ph: constraints[4],
            constraint_batch_size_ph: constraints[0].shape[0]
        }

        # Naive case of only optimizing over observed transitions -- useful to initialize the dynamics model
        if update == "trans":
            [trans_likelihood_eval] = sess.run([neg_avg_adt_log_likelihood], feed_dict=feed_dict)
            update_op = adt_trans_only_update_op
            if step:
                sess.run(update_op, feed_dict=feed_dict)
            d = {'loss': trans_likelihood_eval}

        # Optimization over the full adt loss function
        if update == "adt":
            [loss_eval, act_likelihood_eval, td_err_eval, trans_likelihood_eval] = sess.run(
                [adt_loss, neg_avg_act_log_likelihood, adt_td_err, neg_avg_adt_log_likelihood], feed_dict=feed_dict)
            update_op = adt_update_op
            d = {'loss': loss_eval,
                 'nall': act_likelihood_eval,
                 'tde': td_err_eval,
                 'ntll': trans_likelihood_eval}

        # Optimization over the adt uncertainty-weighted loss function
        if update == "adt_un":
            [loss_eval, act_likelihood_eval, td_err_eval, trans_likelihood_eval, sig_a, sig_t, sig_br] = sess.run(
                [adt_uncertainty_loss, neg_avg_act_log_likelihood, adt_td_err, neg_avg_adt_log_likelihood,
                 sigma_a, sigma_t, sigma_br], feed_dict=feed_dict)
            update_op = adt_un_update_op
            d = {'loss': loss_eval,
                 'nall': act_likelihood_eval,
                 'tde': td_err_eval,
                 'ntll': trans_likelihood_eval,
                 'sigmas': (sig_a, sig_t, sig_br)}

        # Optimizing only the q-fn over the bellman residual, equivalent to soft Q-learning
        if update == "adt_br":
            [loss_eval] = sess.run([adt_td_err_sgt], feed_dict=feed_dict)
            update_op = adt_q_br_update_op
            d = {'loss': loss_eval}

        # Optimization over the full adt loss function with a stop gradient on the dynamics model for the bellman residual
        # Useful in getting the q-fn to match both the observed dynamics and the observed actions taken
        if update == "adt_brsgt":
            [loss_eval, act_likelihood_eval, td_err_eval, trans_likelihood_eval] = sess.run(
                [adt_loss, neg_avg_act_log_likelihood, adt_td_err_sgt, neg_avg_adt_log_likelihood], feed_dict=feed_dict)
            update_op = adt_brsgt_update_op
            d = {'loss': loss_eval,
                 'nall': act_likelihood_eval,
                 'tde': td_err_eval,
                 'ntll': trans_likelihood_eval}

        # Optimization over the action likelihood and br with a stop gradient on the dynamics model for the br
        if update == "adt_q":
            [loss_eval, act_likelihood_eval, td_err_eval] = sess.run(
                [adt_q_loss, neg_avg_act_log_likelihood, adt_td_err_sgt], feed_dict=feed_dict)
            update_op = adt_q_update_op
            d = {'loss': loss_eval,
                 'nall': act_likelihood_eval,
                 'tde': td_err_eval}

        # Optimization over the action likelihood and br with a stop gradient on the q-fn for the br
        if update == "adt_learn":
            [loss_eval, trans_likelihood_eval, td_err_eval, act_likelihood_eval] = sess.run(
                [adt_learn_loss, neg_avg_adt_log_likelihood, adt_td_err_sgq, neg_avg_act_log_likelihood],
                feed_dict=feed_dict)
            update_op = adt_learn_update_op
            d = {'loss': loss_eval,
                 'tde': td_err_eval,
                 'ntll': trans_likelihood_eval,
                 'nall': act_likelihood_eval}

        # Optimization over the transition likelihood and br with a stop gradient on the dynamics for the br
        if update == "adt_regularize":
            [loss_eval, trans_likelihood_eval, td_err_eval] = sess.run(
                [adt_regularize_loss, neg_avg_adt_log_likelihood, adt_td_err_sgt], feed_dict=feed_dict)
            update_op = adt_regularize_update_op
            d = {'loss': loss_eval,
                 'ntll': trans_likelihood_eval,
                 'tde': td_err_eval}

        # Optimization over the transition likelihood and br with a stop gradient on the q-fn for the br
        if update == "adt_t":
            [loss_eval, td_err_eval, trans_likelihood_eval] = sess.run(
                [adt_t_loss, adt_td_err_sgq, neg_avg_adt_log_likelihood], feed_dict=feed_dict)
            update_op = adt_t_update_op
            d = {'loss': loss_eval,
                 'tde': td_err_eval,
                 'ntll': trans_likelihood_eval}

        # Optimization over the full sas loss function
        if update == "sas":
            [loss_eval, act_likelihood_eval, td_err_eval, trans_likelihood_eval] = sess.run(
                [sas_loss, neg_avg_act_log_likelihood, sas_td_err, neg_avg_sas_log_likelihood], feed_dict=feed_dict)
            update_op = sas_update_op
            d = {'loss': loss_eval,
                 'nall': act_likelihood_eval,
                 'tde': td_err_eval,
                 'ntll': trans_likelihood_eval}

        # For debugging and testing model capacity
        if update == "true_qs":
            feed_dict[true_qs_ph] = true_qs
            [loss_eval] = sess.run([true_q_err], feed_dict=feed_dict)
            update_op = true_q_update_op
            d = {'loss': loss_eval}

        if step:
            sess.run(update_op, feed_dict=feed_dict)

        return d

    def idl_train(n_training_iters, nn_rollouts, train_idxes, constraints, true_qs, val_demo_batch,
                 states, adt_samples, batch_size):

        tf.global_variables_initializer().run(session=sess)

        train_log_base = {
            'loss_evals': [],
            'ntll_evals': [],
            'nall_evals': [],
            'tde_evals': [],
            'val_loss_evals': [],
            'val_ntll_evals': [],
            'val_nall_evals': [],
            'val_tde_evals': [],
            'sigmas_evals': []
        }

        mode_logs = {}

        val_log = None

        # The best order of optimization seems to be "trans" -> "adt_br" -> "adt_brsgt" -> "adt" [->"adt-br"->"adt"]

        # update = "trans"
        # ugit pdate = "adt_q"
        # update = "adt_t"

        # update = "adt_learn"
        # update = "trans"
        # update = "adt_br"
        # update = "adt_regularize"

        # update_progression = ["adt_regularize", "adt_learn"]

        update_progression = ["true_qs"]

        for up in update_progression:
            mode_logs[up] = copy(train_log_base)

        update = "true_qs"

        mode_logs[update] = copy(train_log_base)

        full_train_logs = mode_logs[update]

        # update = "adt_br"
        # update = "adt_brsgt"
        # update = "adt"
        # update = "adt_un"

        # I built in a simple update-switcher so I no longer needed to manually stop training and switch update fns
        i=0
        total_train_time = 0
        while total_train_time < n_training_iters:
            demo_batch = sample_batch(nn_rollouts, train_idxes, batch_size)
            train_log = compute_batch_loss(demo_batch, constraints, step=True, update=update, true_qs=true_qs)
            if i % 20 == 0:
                val_log = compute_batch_loss(val_demo_batch, constraints, step=False, update=update, true_qs=true_qs)
                for k, v in train_log.items():
                    full_train_logs['%s_evals' % k].append(v)
                for k, v in val_log.items():
                    if 'val_%s_evals' % k in full_train_logs:
                        full_train_logs['val_%s_evals' % k].append(v)

            if i % 1000 == 0:
                if i > 5000:
                    pu = update
                    losses = full_train_logs['val_loss_evals'][(-i // 20):]
                    update = update_switcher(update, update_progression, losses, slope_threshold=5e-5)
                    if pu != update:
                        full_train_logs = mode_logs[update]
                        print("========= Switching to {} =========".format(update))
                        i = 0
                    else:
                        print([(key, val_log[key]) for key in val_log.keys()])
                else:
                    print([(key, val_log[key]) for key in val_log.keys()])

            if i % 5000 == 0:
                q_vals = sess.run([constraint_q_ts], feed_dict={constraint_obs_t_feats_ph: states})[0]
                plot_values(mdp, q_vals)
                plt.show()
                adt_probs = sess.run([adt_pred_dir], feed_dict={demo_tile_t_ph: adt_samples[:, 0][np.newaxis].T,
                                                                demo_act_t_ph: adt_samples[:, 1][np.newaxis].T})[0]
                print(softmax(adt_probs))

                for k in ['val_loss_evals', 'val_ntll_evals', 'val_nall_evals', 'val_tde_evals']:
                    if len(full_train_logs[k]) > 0:
                        plt.xlabel('Iterations')
                        plt.ylabel(k.split('_')[1])
                        plt.plot(full_train_logs[k][10:])
                        plt.show()
            i += 1
            total_train_time += 1

    def frank_wofe_train(sess, n_training_iters, nn_rollouts, train_idxes, batch_size, constraints, num_tasks,
                         val_demo_batch,
                         states, adt_samples, MAX_ITER, STOP_CRIT):
        # scales = tf.placeholder(tf.float32, [num_tasks], name="scale")
        scales = [tf.placeholder(tf.float32, [1], name="scale{}".format(i)) for i in range(num_tasks)]
        clip_norms = [tf.placeholder(tf.float32, [1], name="clipnorm{}".format(i)) for i in range(num_tasks)]
        opts = [tf.train.AdamOptimizer(alpha, beta1, beta2) for _ in range(num_tasks)]
        loss_fns = np.array([nall_loss, ntll_loss, adt_td_err, adt_td_err_sgq, adt_td_err_sgt])
        loss_fns_titles = np.array(['nall', 'ntll', 'tde', 'tde_sq', 'tde_st', 'loss'])
        # loss_configs = [[0],[1],[3],[4]]
        loss_configs = [[0, 3], [1, 4]]

        task_losses = [sum(loss_fns[config]) for config in loss_configs]
        # # add total loss to the list of losses
        # losses = [[sum(losses[t])] + list(losses[t]) for t in tasks]
        loss_titles = [loss_fns_titles[config] for config in loss_configs]

        q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope)
        t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=invadt_scope)
        all_vars = q_vars + t_vars
        all_gvs = [opts[i].compute_gradients(task_losses[i], all_vars) for i in range(num_tasks)]
        sep_gvs = [[a for a in zip(*gv_set)] for gv_set in all_gvs]
        grads = [[tf.clip_by_norm(sep_gvs[i][0][j], clip_norms[i]) for j in range(len(sep_gvs[i][0]))] for i in
                 range(num_tasks)]
        # grads = [gv[0] for gv in sep_gvs]
        gvars = [gv[1] for gv in sep_gvs]

        # print([v.shape for v in all_vars])

        def replace_none_with_zero(grad, var):
            return [tf.zeros_like(var[i]) if grad[i] == None else grad[i] for i in range(len(grad))]

            # print(grads)

        # grads = [replace_none_with_zero(grad, all_vars) for grad in grads]
        # print(grads)

        scaled_grads = [[tf.multiply(g, scales[i]) for g in grads[i]] for i in range(num_tasks)]
        updates = [opts[i].apply_gradients(zip(scaled_grads[i], gvars[i])) for i in range(num_tasks)]

        gn = "l+"



        tf.global_variables_initializer().run(session=sess)

        total_train_time = 0

        full_train_logs = {
            'loss_evals': [],
            'ntll_evals': [],
            'nall_evals': [],
            'tde_evals': [],
            'val_loss_evals': [],
            'val_ntll_evals': [],
            'val_nall_evals': [],
            'val_tde_evals': [],
            'sigmas_evals': []
        }

        train_iter = 0
        mgda = True
        while total_train_time < n_training_iters:

            demo_batch = sample_batch(nn_rollouts, train_idxes, batch_size)
            feed_dict = {
                demo_obs_t_feats_ph: demo_batch[1],
                demo_act_t_ph: demo_batch[2],
                demo_next_obs_t_feats_ph: demo_batch[4],
                demo_dir_t_ph: demo_batch[5],
                demo_tile_t_ph: demo_batch[6],
                demo_batch_size_ph: demo_batch[2].shape[0],
                constraint_obs_t_feats_ph: constraints[0],
                constraint_act_t_ph: constraints[1],
                constraint_rew_t_ph: constraints[2],
                constraint_next_obs_t_feats_ph: constraints[3],
                constraint_tile_t_ph: constraints[4],
                constraint_batch_size_ph: constraints[0].shape[0]
            }

            # Scaling the loss functions based on the algorithm choice
            loss_data = {}
            gvs_data = {}
            scale = {}
            mask = None
            masks = {}

            # This is MGDA
            grad_arrays = {}
            for t in range(num_tasks):
                # Compute gradients of each loss function wrt parameters
                loss_data[t], gvs_data[t] = sess.run([task_losses[t], all_gvs[t]], feed_dict=feed_dict)
                grad_arrays[t] = np.concatenate([gvs_data[t][i][0].flatten() for i in range(len(gvs_data[t]))])
                # g_loss_data[t] = loss_data[t][1:]

            if gn == "l2":
                for i, c in enumerate(clip_norms):
                    feed_dict[c] = [1.0]
            elif gn == "l+":
                for i, c in enumerate(clip_norms):
                    feed_dict[c] = [loss_data[i]]

            # Normalize all gradients
            #     gn = gradient_normalizers(grad_arrays, loss_data, "loss+")
            #     for t in range(num_tasks):
            #         for gr_i in range(len(grad_arrays[t])):
            #             grad_arrays[t][gr_i] = grad_arrays[t][gr_i] / gn[t]

            # Frank-Wolfe iteration to compute scales.
            sol, min_norm = find_min_norm_element(grad_arrays)
            #     print(sol)

            # Scaled back-propagation
            for i, s in enumerate(scales):
                feed_dict[s] = [sol[i]]
            for t in range(num_tasks):
                # Inefficient, takes gradients twice when we could just feed the gradients back in
                sess.run(updates, feed_dict=feed_dict)

            if total_train_time % 20 == 0:
                val_log = compute_batch_loss(val_demo_batch, constraints, step=False, update="adt")
                for k, v in val_log.items():
                    if 'val_%s_evals' % k in full_train_logs:
                        full_train_logs['val_%s_evals' % k].append(v)
            #         print([(key,val_log[key]) for key in val_log.keys()])

            if total_train_time % 100 == 0:
                print(sol)
                print([(key, val_log[key]) for key in val_log.keys()])

            if train_iter % 1000 == 0:
                q_vals = sess.run([constraint_q_ts], feed_dict={constraint_obs_t_feats_ph: states})[0]
                plot_values(mdp, q_vals)
                plt.show()
                adt_probs = sess.run([adt_pred_dir], feed_dict={demo_tile_t_ph: adt_samples[:, 0][np.newaxis].T,
                                                                demo_act_t_ph: adt_samples[:, 1][np.newaxis].T})[0]
                print(softmax(adt_probs))

                for k in ['val_loss_evals', 'val_ntll_evals', 'val_nall_evals', 'val_tde_evals']:
                    if len(full_train_logs[k]) > 0:
                        plt.xlabel('Iterations')
                        plt.ylabel(k.split('_')[1])
                        plt.plot(full_train_logs[k][10:])
                        plt.show()
            train_iter += 1
            total_train_time += 1


    return compute_batch_loss, idl_train, frank_wofe_train



