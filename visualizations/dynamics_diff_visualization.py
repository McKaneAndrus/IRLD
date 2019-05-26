import numpy as np
import pandas as pd

from sacred import Experiment
from sacred.observers import FileStorageObserver

import pickle as pkl
import os

import seaborn as sns
import matplotlib.pyplot as plt

import glob

OBSERVED_TTS = 0
UNOBSERVED_TTS = 1

dynamics_diff_visualization = Experiment('dynamics_diff_visualization')
dynamics_diff_visualization.observers.append(FileStorageObserver.create('logs/sacred/viz'))
sns.set(style="darkgrid")




def visualize_dynamics_diff(data, out_file, plot_all):

    all_iters, all_models, true_dyns = data
    plot_iters, observed_diffs, unobserved_diffs,  o_to_u_diffs, experiment_i = [], [], [], [], []
    baseline_observed, baseline_unobserved, baseline_o_to_u = [], [], []
    for i in range(len(all_iters)):
        iters, models, true_dyn = all_iters[i], all_models[i], true_dyns[i]
        baseline_observed += [np.linalg.norm(true_dyn[OBSERVED_TTS] - models[0][OBSERVED_TTS])] * len(models)
        baseline_unobserved += [np.linalg.norm(true_dyn[UNOBSERVED_TTS] - models[0][UNOBSERVED_TTS])] * len(models)
        baseline_o_to_u += [np.linalg.norm(models[0][OBSERVED_TTS] - models[0][UNOBSERVED_TTS])] * len(models)
        observed_diffs += [np.linalg.norm(true_dyn[OBSERVED_TTS] - model[OBSERVED_TTS]) for model in models]
        unobserved_diffs += [np.linalg.norm(true_dyn[UNOBSERVED_TTS] - model[UNOBSERVED_TTS]) for model in models]
        o_to_u_diffs += [np.linalg.norm(model[OBSERVED_TTS] - model[UNOBSERVED_TTS]) for model in models]
        experiment_i += [i] * len(models)
        plot_iters += iters

    df = pd.DataFrame(data={
        'baseline_observed': baseline_observed,
        'baseline_unobserved': baseline_unobserved,
        'baseline_o_to_u':baseline_o_to_u,
        'observed_diffs': observed_diffs,
        'unobserved_diffs': unobserved_diffs,
        'o_to_u_diffs': o_to_u_diffs,
        'experiment_i': experiment_i,
        'step_num': plot_iters})

    extra_kwargs = {}
    if not plot_all:
        extra_kwargs = {'units': 'experiment_i', 'estimator': None, 'hue': 'experiment_i'}
    plot = sns.lineplot(x='step_num', y='observed_diffs', data=df, label='IRLD', **extra_kwargs)
    plot = sns.lineplot(x='step_num', y='baseline_observed', data=df, label='Baseline', **extra_kwargs)
    fig = plot.get_figure()
    plt.title("Difference between Learned and True Observed Dynamics")
    plt.xlabel("Training Iterations")
    plt.ylabel("L2 Distance")
    plt.legend()
    fig.savefig(out_file+"_observed.png")
    plt.clf()
    plot = sns.lineplot(x='step_num', y='unobserved_diffs', label='IRLD', data=df, **extra_kwargs)
    plot = sns.lineplot(x='step_num', y='baseline_unobserved', data=df, label='Baseline', **extra_kwargs)
    fig = plot.get_figure()
    plt.title("Difference between Learned and True Unobserved Dynamics")
    plt.ylabel("L2 Distance")
    plt.xlabel("Training Iterations")
    plt.legend()
    fig.savefig(out_file+"_unobserved.png")
    plt.clf()
    plot = sns.lineplot(x='step_num', y='o_to_u_diffs',  label='IRLD', data=df, **extra_kwargs)
    plot = sns.lineplot(x='step_num', y='baseline_o_to_u', data=df, label='Baseline', **extra_kwargs)
    fig = plot.get_figure()
    plt.title("Difference between Learned Observed and Unobserved Dynamics")
    plt.ylabel("L2 Distance")
    plt.xlabel("Training Iterations")
    plt.legend()
    fig.savefig(out_file+"_o_to_u.png")
    plt.clf()


@dynamics_diff_visualization.config
def config():
    out_dir = "logs/generated_images_"
    plot_all = True
    cutoff = None

    _ = locals()  # quieten flake8 unused variable warning
    del _


def load_dynamics(experiment_nums, cutoff=None):

    all_iters, all_models, true_dyns = [], [], []
    for experiment_num in experiment_nums:
        experiment_dir = os.path.join("logs", "models", str(experiment_num), "tab")
        true_dyn = pkl.load(open(os.path.join(experiment_dir, "true_adt_probs.pkl"), "rb"))
        file_header = os.path.join(experiment_dir, "adt_probs*")
        model_files = glob.glob(file_header)
        # Handles files of form %s_%s_%d.pkl, retrieves %d
        iters_models = [(int(file.split("_")[-1].split(".")[0]), pkl.load(open(file, 'rb'))) for file in model_files]
        if cutoff is not None:
            iters_models = [iter_model for iter_model in iters_models if iter_model[0] < cutoff]
        iters_models = sorted(iters_models, key=lambda x: x[0])
        iters, models = zip(*iters_models)
        all_iters += [iters]
        all_models += [models]
        true_dyns += [true_dyn]

    return [all_iters, all_models, true_dyns]


@dynamics_diff_visualization.automain
def main(out_dir, _run, experiment_nums, plot_all, cutoff):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data = load_dynamics(experiment_nums, cutoff)
    out_file = os.path.join(out_dir, "{}_dynamics_diff_visualization_from_{}".format("_".join(str(x) for x in experiment_nums), _run._id))
    visualize_dynamics_diff(data, out_file, plot_all)
