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

coordinate_model_train_ex = Experiment("coordinate_model_train")
coordinate_model_train_ex.observers.append(FileStorageObserver.create('logs/sacred'))


@coordinate_model_train_ex.config
def default_config():
    mdp_num = 0

    gamma = 0.99
    # TODO decide what to do with these learning params
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


    #Coordinate Config
    batch_size = 200
    n_training_iters = 500000
    dyn_pretrain_iters = 0
    horizon = 1000
    slope_threshold = 1e-4
    switch_frequency = 500
    # Config made up of ['nall', 'ntll', 'tde', 'tde_sg_q', 'tde_sg_t']
    initial_update = [1]
    update_progression = [[4],[0,1,3]]

    tab_save_freq = 50

@coordinate_model_train_ex.named_config
def simple_map_config_mellow():
    mdp_num = 0

    gamma = 0.99
    alpha = 5e-5
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
    mellowmax = True



    #DEMO Config
    gamma_demo = 0.99
    n_demos = 200
    demo_time_steps = 40
    temp_boltz_beta = 50


    #Coordinate Config
    batch_size = 200
    n_training_iters = 500000
    dyn_pretrain_iters = 0
    horizon = 5000
    slope_threshold = 1e-6
    switch_frequency = 500
    # Config made up of ['nall', 'ntll', 'tde', 'tde_sg_q', 'tde_sg_t']
    initial_update = [1]
    update_progression = [[1,4],[0,1,3],[0,2]]
    model_save_weights = [1.0, 1.0, 0.5]

    tab_save_freq = 200

    seed = 0



@coordinate_model_train_ex.automain
def coordinate_train(_run, mdp_num, gamma, alpha, beta1, beta2, constraint_batch_size, q_n_layers, q_layer_size, q_activation,
            q_output_activation, dyn_n_layers, dyn_layer_size, dyn_activation, dyn_output_activation, boltz_beta, mellowmax,
            gamma_demo, temp_boltz_beta, n_demos, demo_time_steps, n_training_iters, dyn_pretrain_iters, batch_size,
            horizon, slope_threshold, switch_frequency, initial_update, update_progression, model_save_weights, tab_save_freq, seed):

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

    regime_params = {"horizon": horizon,
                     'slope_threshold':slope_threshold,
                     'switch_frequency': switch_frequency,
                     'initial_update': initial_update,
                     'update_progression':update_progression,
                     'model_save_weights': model_save_weights}

    model.initialize_training_regime("coordinate", regime_params=regime_params)

    constraints, rollouts, train_idxes, val_demo_batch, true_qs, states, adt_samples = get_demos(mdp, gamma_demo,
                                                                                                    temp_boltz_beta,
                                                                                                    n_demos,
                                                                                                    demo_time_steps,
                                                                                                    seed)

    out_dir = os.path.join("logs", "models", str(_run._id))

    model.train(n_training_iters, rollouts, train_idxes, batch_size, constraints, val_demo_batch, out_dir,
                states, adt_samples, dyn_pretrain_iters, tab_save_freq,  _run, true_qs)

    return _run._id
