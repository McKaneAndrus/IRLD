from sacred import Experiment
from sacred.observers import FileStorageObserver
from utils.data_utils import initialize_scopes, load_scopes
import os
from utils.tf_utils import os_setup
import tensorflow as tf
from envs.environment_setup_utils import get_mdp, get_tile_map
from utils.tab_models import TabularInverseDynamicsLearner
from envs.mars_map_gen import make_map, get_mdp_from_map
from utils.tab_learning_utils import *
from utils.demos_utils import *

tab_model_train_ex = Experiment("tab_model_train")
tab_model_train_ex.observers.append(FileStorageObserver.create('logs/sacred'))
tab_model_train_ex.add_source_file('utils/tab_models.py')
tab_model_train_ex.add_source_file('utils/demos_utils.py')
tab_model_train_ex.add_source_file('utils/tab_learning_utils.py')
tab_model_train_ex.add_source_file('envs/environment_setup_utils.py')



@tab_model_train_ex.config
def default_config():

    gamma = 0.99
    alpha = 2.0

    # Boltz-beta determines the "rationality" of the agent being modeled.
    # Setting it to higher values corresponds to "pure rationality"
    boltz_beta = 50

    #DEMO Config
    gamma_demo = 0.99
    n_demos = 200
    demo_time_steps = 40
    temp_boltz_beta = 50

    #tab Config
    batch_size = 256
    n_training_iters = 2000

    tab_save_freq = 5

    transition_likelihood_weight = 10.0
    mdp_num=12

    seed = 0

    random_mdp = False

    map_height = 15
    map_width = 15
    clustering_iterations = 10
    mdp_num = 12

    t0 = (0.6, 0.2, 0.0, 0.0)
    t1 = (0.0, 0.0, 0.0, 1.0)  # (0.1,0.15,0.5,0.1)

    trans_dict = {b'F': t0,
                  b'1': t0,
                  b'2': t0,
                  b'3': t0,
                  b'S': t0,
                  b'U': t1}

    if random_mdp:
        mdp_map = make_map(map_height, map_width, clustering_iterations, seed)
    else:
        mdp_map = get_tile_map(mdp_num)

    verbose = True

    serd=False





@tab_model_train_ex.automain
def tab_train(_run, mdp_map, trans_dict, gamma, alpha, transition_likelihood_weight, boltz_beta, gamma_demo,
              temp_boltz_beta, n_demos, demo_time_steps, n_training_iters, batch_size, tab_save_freq, seed,
              verbose, serd):

    # q_scope, dyn_scope = load_scopes(data_dir)

    mdp = get_mdp_from_map(mdp_map, trans_dict)

    model = TabularInverseDynamicsLearner(mdp, serd, gamma, boltz_beta, alpha=alpha, seed=seed)

    train_sas, train_adt, val_sas, val_adt, true_qs = get_demos(mdp, gamma_demo, temp_boltz_beta, n_demos,
                                                                demo_time_steps, seed, tabular=True)


    out_dir = os.path.join("logs", "models", str(_run._id))

    model.train(n_training_iters, train_sas, train_adt,  batch_size, val_sas, val_adt, out_dir,
                transition_likelihood_weight, _run, true_qs, tab_save_freq, verbose)

    return _run._id
