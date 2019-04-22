from sacred import Experiment
from sacred.observers import FileStorageObserver
from utils.data_utils import initialize_scopes, load_scopes
import os
from utils.tf_utils import os_setup
import tensorflow as tf
from envs.environment_setup_utils import get_mdp
from utils.demos_utils import get_demos
from utils.experiment_utils import current_milli_time
from utils.models import InverseDynamicsLearner

ex = Experiment()
ex.observers.append(FileStorageObserver.create('logs/sacred'))


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
    constraint_batch_size = None

    q_n_layers = 2
    q_layer_size = 2048
    q_activation = tf.nn.tanh
    q_output_activation = None

    dyn_n_layers = 1
    dyn_layer_size = 256
    dyn_activation = tf.nn.relu
    dyn_output_activation = None


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
    n_training_iters = 1000

    loss_configurations = [[0,3],[1,4]]

@ex.automain
def mgda_train(_run, mdp_num, gamma, alpha, beta1, beta2, constraint_batch_size, q_n_layers, q_layer_size, q_activation,
            q_output_activation, dyn_n_layers, dyn_layer_size, dyn_activation, dyn_output_activation, boltz_beta,
            gamma_demo, temp_boltz_beta, n_demos, demo_time_steps, n_training_iters, batch_size, loss_configurations):

    os_setup()
    data_dir = os.path.join('data', '1.1')
    # q_scope, dyn_scope = load_scopes(data_dir)
    sess = tf.Session()

    mdp = get_mdp(mdp_num)

    mlp_params = {'q_n_layers':q_n_layers,
                  'q_layer_size':q_layer_size,
                  'q_activation': q_activation,
                  'q_output_activation':q_output_activation,
                  'dyn_n_layers':dyn_n_layers,
                  'dyn_layer_size':dyn_layer_size,
                  'dyn_activation':dyn_activation,
                  'dyn_output_activation':dyn_output_activation}

    model = InverseDynamicsLearner(mdp, sess, mlp_params=mlp_params, boltz_beta=boltz_beta, gamma=gamma) #, q_scope=q_scope, dyn_scope=dyn_scope)

    regime_params = {'loss_configurations':loss_configurations}
    model.initialize_training_regime("MGDA", regime_params=regime_params)

    constraints, rollouts, train_idxes, val_demo_batch, true_qs, states, adt_samples = get_demos(mdp, gamma_demo,
                                                                                                    temp_boltz_beta,
                                                                                                    n_demos,
                                                                                                    demo_time_steps)

    out_dir = os.path.join("logs", "models", str(current_milli_time()))

    return model.train(n_training_iters, rollouts, train_idxes, batch_size, constraints, val_demo_batch, out_dir)


