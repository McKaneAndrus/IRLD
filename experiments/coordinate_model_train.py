from sacred import Experiment
from sacred.observers import FileStorageObserver
from utils.data_utils import initialize_scopes, load_scopes
import os
from utils.tf_utils import os_setup
import tensorflow as tf
from envs.environment_setup_utils import get_mdp, get_tile_map
from utils.demos_utils import get_demos
from utils.experiment_utils import current_milli_time
from utils.models import InverseDynamicsLearner
from envs.mars_map_gen import make_map, get_mdp_from_map
import numpy as np

coordinate_model_train_ex = Experiment("coordinate_model_train")
coordinate_model_train_ex.observers.append(FileStorageObserver.create('logs/sacred'))
coordinate_model_train_ex.add_source_file('utils/models.py')
coordinate_model_train_ex.add_source_file('utils/demos_utils.py')
coordinate_model_train_ex.add_source_file('utils/tf_utils.py')
coordinate_model_train_ex.add_source_file('envs/environment_setup_utils.py')


@coordinate_model_train_ex.config
def default_config():


    gamma = 0.99
    alpha = 1e-2
    beta1 = 0.9
    beta2 = 0.999999

    constraint_batch_size = None

    q_n_layers = 1
    q_layer_size = 128
    q_activation = tf.nn.relu
    q_output_activation = None
    q_layer_norm = False
    target_update_freq = 25


    dyn_n_layers = 1
    dyn_layer_size = 64
    dyn_activation = tf.nn.relu
    dyn_output_activation = None
    dyn_layer_norm = False

    weight_norm = False


    # Boltz-beta determines the "rationality" of the agent being modeled.
    # Setting it to higher values corresponds to "pure rationality"
    boltz_beta = 50
    mellowmax = None
    lse_softmax = None


    #DEMO Config
    gamma_demo = 0.99
    n_demos = 200
    demo_time_steps = 40
    temp_boltz_beta = 50


    #Coordinate Config
    batch_size = 256
    n_training_iters = 150000
    dyn_pretrain_iters = 20000
    horizons = 5000
    alphas = [5e-3, 5e-3] #[1e-4, 1e-4,1e-2,1e-4]
    improvement_proportions = [-np.inf, -np.inf] #[0.1, -1, 0.1]
    switch_frequency = 500
    # Config made up of ['nall', 'ntll', 'tde', 'tde_sg_q', 'tde_sg_t']
    initial_update = None
    update_progression = [[0,1,4],[5,4]] #[[0],[5],[4],[7]] #[[4],[0,4,5]]
    # update_weights = [[1.],[1.],[10000,1.]]
    model_save_weights = [1.0, 1.0, 1.0, 0.0, 0.0]

    tab_save_freq = 250
    clip_global = True

    seed = 0
    map_seed = 1
    gpu_num = 0

    random_mdp = False

    map_height = 15
    map_width = 15
    clustering_iterations = 10
    mdp_num = 0

    if random_mdp:
        mdp_map = make_map(map_height, map_width, clustering_iterations, map_seed)
    else:
        mdp_map = get_tile_map(mdp_num)

@coordinate_model_train_ex.named_config
def concise_boi():

    dyn_layer_norm = False
    q_layer_norm = False

    horizons=2500
    switch_frequency = 500
    alphas = [5e-3, 1e-3, 5e-3]
    improvement_proportions = [-np.inf, -np.inf, 0.5]  # [0.1, -1, 0.1]
    # Config made up of ['nall', 'ntll', 'tde', 'tde_sg_q', 'tde_sg_t']
    update_progression = [[0], [5], [4]]  # [[0],[5],[4],[7]] #[[4],[0,4,5]]
    model_save_weights = [0.0, 0.0, 1.0, 0.0, 0.0]

@coordinate_model_train_ex.named_config
def safer_concise_boi():

    dyn_layer_norm = False
    q_layer_norm = False

    horizons=2500
    alphas = [5e-3, 1e-3, 5e-3]
    improvement_proportions = [-np.inf, -np.inf, 0.5]  # [0.1, -1, 0.1]
    # Config made up of ['nall', 'ntll', 'tde', 'tde_sg_q', 'tde_sg_t']
    update_progression = [[0, 6], [5], [4]]  # [[0],[5],[4],[7]] #[[4],[0,4,5]]
    model_save_weights = [0.0, 0.0, 1.0, 0.0, 0.0]

@coordinate_model_train_ex.named_config
def extended_boi():

    dyn_layer_norm = False
    q_layer_norm = False

    horizons=[500,1000,500,1000,3000]
    alphas = [5e-3, 2e-4, 2e-4, 2e-4, 5e-3]
    improvement_proportions = [-np.inf, -np.inf, -np.inf, -np.inf, 0.5]  # [0.1, -1, 0.1]
    # Config made up of ['nall', 'ntll', 'tde', 'tde_sg_q', 'tde_sg_t']
    update_progression = [[0],[1,5],[7],[1,5],[4]]  # [[0],[5],[4],[7]] #[[4],[0,4,5]]
    model_save_weights = [0.0, 0.0, 1.0, 0.0, 0.0]

@coordinate_model_train_ex.named_config
def speedy_concise_boi():

    dyn_layer_norm = False
    q_layer_norm = False
    weight_norm=True
    mellowmax=None

    horizons=[500,2000,3000]
    switch_frequency = 500
    alphas = [5e-3, 5e-4, 1e-2]
    improvement_proportions = [-np.inf, -np.inf, 0.25]  # [0.1, -1, 0.1]
    # Config made up of ['nall', 'ntll', 'tde', 'tde_sg_q', 'tde_sg_t']
    update_progression = [[0], [5], [4]]  # [[0],[5],[4],[7]] #[[4],[0,4,5]]
    model_save_weights = [0.0, 0.0, 1.0, 0.0, 0.0]

@coordinate_model_train_ex.named_config
def corrected_speedy_boi():
    dyn_layer_norm = False
    q_layer_norm = False
    weight_norm = True
    mellowmax = None

    horizons = [500, 1500, 3000]
    switch_frequency = 500
    alphas = [5e-3, 2e-4, 5e-3]
    improvement_proportions = [-np.inf, -np.inf, 0.25]  # [0.1, -1, 0.1]
    # Config made up of ['nall', 'ntll', 'tde', 'tde_sg_q', 'tde_sg_t']
    update_progression = [[0], [1] * 20 + [5], [4]]  # [[0],[5],[4],[7]] #[[4],[0,4,5]]
    model_save_weights = [0.0, 0.0, 1.0, 0.0, 0.0]


@coordinate_model_train_ex.named_config
def kl_boi():
    dyn_layer_norm = False
    q_layer_norm = False
    weight_norm = True
    mellowmax = None
    lse_softmax = None

    horizons = [500, 1000, 1000, 3000]
    switch_frequency = 500
    alphas = [5e-3, 2e-4, 2e-4, 5e-3]
    improvement_proportions = [-np.inf, -np.inf, -np.inf, 0.25]  # [0.1, -1, 0.1]
    # Config made up of ['nall', 'ntll', 'tde', 'tde_sg_q', 'tde_sg_t']
    update_progression = [[0], [5], [9], [4]]  # [[0],[5],[4],[7]] #[[4],[0,4,5]]
    model_save_weights = [0.0, 0.0, 1.0, 0.0, 1.0]



@coordinate_model_train_ex.automain
def coordinate_train(_run, mdp_map, gamma, alpha, beta1, beta2, constraint_batch_size, q_n_layers, q_layer_size, q_activation,
            q_output_activation, q_layer_norm, target_update_freq, dyn_n_layers, dyn_layer_size, dyn_activation,
            dyn_output_activation, dyn_layer_norm, weight_norm, boltz_beta, mellowmax, lse_softmax, gamma_demo, temp_boltz_beta, n_demos,
            demo_time_steps, n_training_iters, dyn_pretrain_iters, batch_size, horizons, alphas, improvement_proportions,
            switch_frequency, initial_update, update_progression, model_save_weights, tab_save_freq, clip_global, gpu_num, seed):

    os_setup(gpu_num)
    tf.reset_default_graph()
    data_dir = os.path.join('data', '1.1')
    # q_scope, dyn_scope = load_scopes(data_dir)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    mdp = get_mdp_from_map(mdp_map)



    mlp_params = {'q_n_layers':q_n_layers,
                  'q_layer_size':q_layer_size,
                  'q_activation': q_activation,
                  'q_output_activation':q_output_activation,
                  'q_layer_norm':q_layer_norm,
                  'dyn_n_layers':dyn_n_layers,
                  'dyn_layer_size':dyn_layer_size,
                  'dyn_activation':dyn_activation,
                  'dyn_output_activation':dyn_output_activation,
                  'dyn_layer_norm': dyn_layer_norm,
                  'weight_norm':weight_norm}

    with sess.as_default():
        model = InverseDynamicsLearner(mdp, sess, mlp_params=mlp_params, boltz_beta=boltz_beta, gamma=gamma, lse_softmax=lse_softmax,
                                       mellowmax=mellowmax, alpha=alpha, beta1=beta1, beta2=beta2, seed=seed)
                                                                    #, q_scope=q_scope, dyn_scope=dyn_scope)

        regime_params = {"horizons": horizons,
                         'improvement_proportions':improvement_proportions,
                         'switch_frequency': switch_frequency,
                         'initial_update': initial_update,
                         'update_progression':update_progression,
                         'model_save_weights': model_save_weights,
                         'alphas': alphas,
                         'clip_global':clip_global}

        model.initialize_training_regime("coordinate", regime_params=regime_params)

        constraints, rollouts, train_idxes, val_demo_batch, true_qs, states, adt_samples = get_demos(mdp, gamma_demo,
                                                                                                        temp_boltz_beta,
                                                                                                        n_demos,
                                                                                                        demo_time_steps,
                                                                                                     seed)

        out_dir = os.path.join("logs", "models", str(_run._id))

        model.train(n_training_iters, rollouts, train_idxes, batch_size, constraints, val_demo_batch, out_dir,
                    states, adt_samples, target_update_freq, dyn_pretrain_iters, tab_save_freq,  _run, true_qs)

    return _run._id
