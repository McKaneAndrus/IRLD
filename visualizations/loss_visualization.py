import numpy as np
import json


from sacred import Experiment
from sacred.observers import FileStorageObserver

import pickle as pkl
import os

loss_visualization = Experiment('loss_visualization')
loss_visualization.observers.append(FileStorageObserver.create('logs/sacred'))


def visualize_value_function(data, out_file):
    print(data)


@loss_visualization.config
def config():
    out_dir = "logs/generated_images_"

    _ = locals()  # quieten flake8 unused variable warning
    del _


def load_dynamics(experiment_num):
    with open(os.path.join('logs', 'sacred', str(experiment_num), 'metrics.json')) as file:
        data = json.load(file)

    return data


@loss_visualization.automain
def main(out_dir, _run, experiment_num):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data = load_dynamics(experiment_num)
    out_file = os.path.join(out_dir, "loss_visualization_{}_from_{}.png".format(_run._id,  experiment_num))
    visualize_value_function(data, out_file)
