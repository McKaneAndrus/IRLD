import pickle as pkl
import numpy as np
import glob


def get_all_dynamics_distances(file_directory, true_dynamics, observed_indexes, unobserved_indexes):
    dyn_models = [pkl.load(dyn_model_fn) for dyn_model_fn in glob.glob(file_directory)]
    observed_distances, unobserved_distances = [], []
    for dyn_model in dyn_models:
        observed_distances += [dyn_model[observed_indexes] - true_dynamics[observed_indexes]]
        unobserved_distances += [dyn_model[unobserved_indexes] - true_dynamics[unobserved_indexes]]
    return observed_distances, unobserved_distances




