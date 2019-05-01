import numpy as np

from sacred import Experiment
from sacred.observers import FileStorageObserver

import pickle as pkl
import os

import seaborn as sns
import matplotlib.pyplot as plt

import glob

OBSERVED_TTS = 0
UNOBSERVED_TTS = 1

q_diff_visualization = Experiment('q_diff_visualization')
q_diff_visualization.observers.append(FileStorageObserver.create('logs/sacred'))
sns.set(style="darkgrid")




def visualize_q_diff(data, out_file):
    iters, models, true_qs = data
    q_diffs = [np.linalg.norm(true_qs - model) for model in models]
    plot = sns.lineplot(iters, q_diffs)
    fig = plot.get_figure()
    plt.title("Difference between Learned and True Q-Values")
    plt.xlabel("Training Iterations")
    plt.ylabel("L2 Distance")
    fig.savefig(out_file)
    plt.clf()


@q_diff_visualization.config
def config():
    out_dir = "logs/generated_images_"

    _ = locals()  # quieten flake8 unused variable warning
    del _


def load_dynamics(experiment_num):

    experiment_dir = os.path.join("logs", "models", str(experiment_num), "tab")
    true_dyn = pkl.load(open(os.path.join(experiment_dir, "true_q_vals.pkl"), "rb"))

    file_header = os.path.join(experiment_dir, "q_vals*")
    model_files = glob.glob(file_header)
    # Handles files of form %s_%s_%d.pkl, retrieves %d
    iters_models = [(int(file.split("_")[-1].split(".")[0]), pkl.load(open(file, 'rb'))) for file in model_files]
    iters_models = sorted(iters_models, key=lambda x: x[0])
    iters, models = zip(*iters_models)

    return [iters, models, true_dyn]


@q_diff_visualization.automain
def main(out_dir, _run, experiment_num):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data = load_dynamics(experiment_num)
    out_file = os.path.join(out_dir, "q_diff_visualization_{}_from_{}.png".format(_run._id,  experiment_num))
    visualize_q_diff(data, out_file)
