from sacred import Experiment
from sacred.observers import FileStorageObserver
from utils.data_utils import initialize_data_files, load_data
import os
from utils.tf_utils import os_setup
import tensorflow as tf
from utils.model_setup import create_tf_model
from envs.environment_setup_utils import get_mdp
from utils.demos_utils import get_demos

ex = Experiment()
ex.observers.append(FileStorageObserver.create('logs'))


@ex.config
def default_config():
    mdp_num = 0

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



    #DEMO Config
    gamma_demo = 0.99
    n_demos = 200
    demo_time_steps = 40
    temp_boltz_beta = 50


    #Frank Wolfe Config
    batch_size = 200
    n_training_iters = 1000000
    MAX_ITER = 250
    STOP_CRIT = 1e-6
    num_tasks = 2

@ex.automain
def frank_wolfe_train(mdp_num, gamma, alpha, beta1, beta2, sq_td_err_penalty,
             trans_penalty, t_err_penalty, q_err_penalty, constraint_batch_size, q_n_layers, q_activation,
            q_output_activation, invdyn_n_layers, invdyn_layer_size, invdyn_output_activation, n_act_dim,
            featurize_acts, n_dirs, boltz_beta, gamma_demo, temp_boltz_beta, n_demos, demo_time_steps,
            n_training_iters, batch_size, num_tasks, MAX_ITER, STOP_CRIT):

    os_setup()
    data_dir = os.path.join('data', '1.1')
    im_scope, q_scope, invsas_scope, invadt_scope = load_data(data_dir)
    sess = tf.Session()

    mdp = get_mdp(mdp_num)

    _, _, frank_wolfe_train = create_tf_model(sess, mdp, q_scope, invsas_scope, invadt_scope,
                                                                       gamma, alpha, beta1, beta2, sq_td_err_penalty,
                                                                       trans_penalty,
                                                                       t_err_penalty, q_err_penalty,
                                                                       constraint_batch_size,
                                                                       q_n_layers, q_activation, q_output_activation,
                                                                       invdyn_n_layers,
                                                                       invdyn_layer_size, invdyn_output_activation,
                                                                       n_act_dim, featurize_acts,
                                                                       n_dirs, boltz_beta)

    constraints, nn_rollouts, train_idxes, val_demo_batch, true_qs, states, adt_samples = get_demos(mdp, gamma_demo,
                                                                                                    temp_boltz_beta,
                                                                                                    n_demos,
                                                                                                    demo_time_steps)

    frank_wolfe_train(sess, n_training_iters, nn_rollouts, train_idxes, batch_size, constraints, num_tasks,
                      val_demo_batch, states, adt_samples, MAX_ITER, STOP_CRIT)

