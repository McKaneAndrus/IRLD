import numpy as np

from sacred import Experiment
from sacred.observers import FileStorageObserver

import pickle as pkl
import os

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import glob


q_diff_visualization = Experiment('q_diff_visualization')
q_diff_visualization.observers.append(FileStorageObserver.create('logs/sacred/viz'))
sns.set(style="darkgrid")




def visualize_q_diff(data, out_file, plot_all):
    all_iters, all_models, true_qs = data
    plot_iters, q_diffs, baseline_q_diffs, experiment_i = [], [], [], []
    for i in range(len(all_iters)):
        iters, models, true_q = all_iters[i], all_models[i], true_qs[i]
        baseline_q_diffs += [np.linalg.norm(true_q - models[0])] * len(models)
        q_diffs += [np.linalg.norm(true_q - model) for model in models]
        plot_iters += iters
        experiment_i += [i] * len(models)

    df = pd.DataFrame(data={
        'q_diffs': q_diffs,
        'baseline_q_diffs': baseline_q_diffs,
        'experiment_i': experiment_i,
        'step_num': plot_iters})

    extra_kwargs = {}
    if not plot_all:
        extra_kwargs = {'units': 'experiment_i', 'estimator': None, 'hue': 'experiment_i'}

    plot = sns.lineplot(x='step_num', y='q_diffs', data=df, label='IRLD', **extra_kwargs)
    plot = sns.lineplot(x='step_num', y='baseline_q_diffs', data=df, label='Baseline', **extra_kwargs)
    fig = plot.get_figure()
    plt.title("Difference between Learned and True Q-Values")
    plt.xlabel("Training Iterations")
    plt.ylabel("L2 Distance")
    plt.legend()
    fig.savefig(out_file)
    plt.clf()


@q_diff_visualization.config
def config():
    out_dir = "logs/generated_images_"
    plot_all = True

    _ = locals()  # quieten flake8 unused variable warning
    del _


def load_qs(experiment_nums):

    all_iters, all_models, true_qs = [], [], []
    for experiment_num in experiment_nums:
        experiment_dir = os.path.join("logs", "models", str(experiment_num), "tab")
        true_q = pkl.load(open(os.path.join(experiment_dir, "true_q_vals.pkl"), "rb"))
        file_header = os.path.join(experiment_dir, "q_vals*")
        model_files = glob.glob(file_header)
        # Handles files of form %s_%s_%d.pkl, retrieves %d
        iters_models = [(int(file.split("_")[-1].split(".")[0]), pkl.load(open(file, 'rb'))) for file in model_files]
        iters_models = sorted(iters_models, key=lambda x: x[0])
        iters, models = zip(*iters_models)
        all_iters += [iters]
        all_models += [models]
        true_qs += [true_q]

    return [all_iters, all_models, true_qs]


@q_diff_visualization.automain
def main(out_dir, _run, experiment_nums, plot_all):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data = load_qs(experiment_nums)
    out_file = os.path.join(out_dir, "{}_q_diff_visualization_from_{}.png".format( "_".join(str(x) for x in experiment_nums), _run._id))
    visualize_q_diff(data, out_file, plot_all)
