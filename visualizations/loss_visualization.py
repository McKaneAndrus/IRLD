import numpy as np
import pandas as pd

from sacred import Experiment
from sacred.observers import FileStorageObserver

import pickle as pkl
import os

import seaborn as sns
import matplotlib.pyplot as plt

import glob
import json

loss_visualization = Experiment('loss_visualization')
loss_visualization.observers.append(FileStorageObserver.create('logs/sacred'))


def visualize_likelihood(data, out_file, plot_all):

    step_nums, joint_likelihoods, nalls, ntlls, experiment_nums = [], [], [], [], []

    for experiment_num, loss_data in data.items():
        nall = loss_data['val_nall']['values']
        ntll = loss_data['val_ntll']['values']
        steps = loss_data['val_nall']['steps']
        joint_likelihoods += [nall[i] + ntll[i] for i in range(len(steps))]
        nalls += nall
        ntlls += ntll
        step_nums += steps
        experiment_nums += [experiment_num] * len(steps)

    print(len(nalls), len(ntlls), len(joint_likelihoods), len(step_nums), len(experiment_nums))

    df = pd.DataFrame(data={
        'joint_likelihoods': joint_likelihoods,
        'nalls':nalls,
        'ntlls':ntlls,
        'experiment_nums': experiment_nums,
        'step_num': step_nums})

    extra_kwargs = {}
    if not plot_all:
        extra_kwargs = {'units': 'experiment_nums', 'estimator': None, 'hue': 'experiment_nums'}

    plot = sns.lineplot(x='step_num', y='joint_likelihoods', data=df, **extra_kwargs)
    fig = plot.get_figure()
    plt.title("Validation Joint Negative Log-Likelihood")
    plt.xlabel("Training Iterations")
    plt.ylabel("Negative Log-Likelihood")
    fig.savefig(out_file + '_joint.png')
    plt.clf()

    plot = sns.lineplot(x='step_num', y='ntlls', data=df, **extra_kwargs)
    fig = plot.get_figure()
    plt.title("Validation Transition Negative Log-Likelihood")
    plt.xlabel("Training Iterations")
    plt.ylabel("Negative Log-Likelihood")
    fig.savefig(out_file + '_ntll.png')
    plt.clf()

    plot = sns.lineplot(x='step_num', y='nalls', data=df, **extra_kwargs)
    fig = plot.get_figure()
    plt.title("Validation Action Negative Log-Likelihood")
    plt.xlabel("Training Iterations")
    plt.ylabel("Negative Log-Likelihood")
    fig.savefig(out_file + '_nall.png')
    plt.clf()






@loss_visualization.config
def config():
    out_dir = "logs/generated_images_"
    plot_all = False

    _ = locals()  # quieten flake8 unused variable warning
    del _


def load_loss_data(experiment_num):
    print("Loading experiment {}".format(experiment_num))
    with open(os.path.join('logs', 'sacred', str(experiment_num), 'metrics.json')) as file:
        data = json.load(file)

    return data


@loss_visualization.automain
def main(out_dir, _run, experiment_nums, plot_all):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data = {experiment_num: load_loss_data(experiment_num) for experiment_num in experiment_nums}
    out_file = os.path.join(out_dir, "{}_loss_visualization_from_{}".format("_".join(str(x) for x in experiment_nums), _run._id))
    visualize_likelihood(data, out_file, plot_all)
