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

mgda_model_train_ex = Experiment("mgda_model_train")
mgda_model_train_ex.observers.append(FileStorageObserver.create('logs/sacred'))


@mgda_model_train_ex.config
def default_config():
    mdp_num = 0

    gamma = 0.99
    alpha = 1e-4
    beta1 = 0.9
    beta2 = 0.999999

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
    mellowmax = False

    #DEMO Config
    gamma_demo = 0.99
    n_demos = 200
    demo_time_steps = 40
    temp_boltz_beta = 50

    #Frank Wolfe Config
    batch_size = 200
    n_training_iters = 100000
    dyn_pretrain_iters = 0


    loss_configurations = [[0,3],[1,4]]

    tab_save_freq = 50

@mgda_model_train_ex.named_config
def simple_map_config_no_mellow_pretrain():
    mdp_num = 0

    gamma = 0.99
    alpha = 1e-5
    beta1 = 0.9
    beta2 = 0.999999

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
    mellowmax = False

    #DEMO Config
    gamma_demo = 0.99
    n_demos = 200
    demo_time_steps = 40
    temp_boltz_beta = 50

    #Frank Wolfe Config
    batch_size = 200
    n_training_iters = 500000
    dyn_pretrain_iters = 20000

    loss_configurations = [[0,3],[1,4]]

    tab_save_freq = 50

@mgda_model_train_ex.named_config
def simple_map_config_mellow_pretrain():
    mdp_num = 0

    gamma = 0.99
    alpha = 1e-5
    beta1 = 0.9
    beta2 = 0.999999

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
    mellowmax = False




    #DEMO Config
    gamma_demo = 0.99
    n_demos = 200
    demo_time_steps = 40
    temp_boltz_beta = 50


    #Frank Wolfe Config
    batch_size = 200
    n_training_iters = 500000
    dyn_pretrain_iters = 20000

    # Config made up of ['nall', 'ntll', 'tde', 'tde_sg_q', 'tde_sg_t']
    loss_configurations = [[0,3],[1,4]]

    tab_save_freq = 50
    seed = 0

@mgda_model_train_ex.automain
def mgda_train(_run, mdp_num, gamma, alpha, beta1, beta2, constraint_batch_size, q_n_layers, q_layer_size, q_activation,
            q_output_activation, dyn_n_layers, dyn_layer_size, dyn_activation, dyn_output_activation, boltz_beta, mellowmax,
            gamma_demo, temp_boltz_beta, n_demos, demo_time_steps, n_training_iters, dyn_pretrain_iters, batch_size,
            loss_configurations, tab_save_freq, seed):

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

    model = InverseDynamicsLearner(mdp, sess, mlp_params=mlp_params, boltz_beta=boltz_beta, gamma=gamma,
                                mellowmax=mellowmax, alpha=alpha, beta1=beta1, beta2=beta2, seed=seed) #, q_scope=q_scope, dyn_scope=dyn_scope)

    regime_params = {'loss_configurations':loss_configurations}
    model.initialize_training_regime("MGDA", regime_params=regime_params)

    constraints, rollouts, train_idxes, val_demo_batch, true_qs, states, adt_samples = get_demos(mdp, gamma_demo,
                                                                                                    temp_boltz_beta,
                                                                                                    n_demos,
                                                                                                    demo_time_steps)

    out_dir = os.path.join("logs", "models", str(_run._id))

    model.train(n_training_iters, rollouts, train_idxes, batch_size, constraints, val_demo_batch, out_dir, states,
                    adt_samples, dyn_pretrain_iters, tab_save_freq, _run, true_qs)

    return _run._id


