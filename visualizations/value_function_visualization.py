import numpy as np

from sacred import Experiment
from sacred.observers import FileStorageObserver

import pickle as pkl
import os

value_function_visualization = Experiment('value_function_visualization')
value_function_visualization.observers.append(FileStorageObserver.create('logs/sacred'))


def visualize_value_function(data, out_file):
    print(data)


@value_function_visualization.config
def config():
    out_dir = "logs/generated_images_"

    _ = locals()  # quieten flake8 unused variable warning
    del _


def load_dynamics(experiment_num):
    data = pkl.load(open(os.path.join("logs", "models", str(experiment_num), "q_vals.pkl"), "rb"))

    return [data]


@value_function_visualization.automain
def main(out_dir, _run, experiment_num):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data = load_dynamics(experiment_num)
    out_file = os.path.join(out_dir, "value_function_visualization_{}_from_{}.png".format(_run._id,  experiment_num))
    visualize_value_function(data, out_file)
