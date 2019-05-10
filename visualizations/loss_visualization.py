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
loss_visualization.observers.append(FileStorageObserver.create('logs/sacred/viz'))


def visualize_likelihood(data, out_file, plot_all, error_cap):

    step_nums, joint_likelihoods, nalls, ntlls, brs, experiment_nums = [], [], [], [], [], []

    for experiment_num, loss_data in data.items():
        nall = loss_data['val_nall']['values']
        ntll = loss_data['val_ntll']['values']
        br = loss_data['val_tde']['values']
        steps = loss_data['val_nall']['steps']
        joint_likelihoods += [nall[i] + ntll[i] for i in range(len(steps))]
        nalls += nall
        ntlls += ntll
        brs += br
        step_nums += steps
        experiment_nums += [experiment_num] * len(steps)
        print(experiment_num, min(nall), min(ntll))

    print(len(nalls), len(ntlls), len(joint_likelihoods), len(step_nums), len(experiment_nums))

    df = pd.DataFrame(data={
        'joint_likelihoods': joint_likelihoods,
        'nalls':nalls,
        'ntlls':ntlls,
        'brs': brs,
        'experiment_nums': experiment_nums,
        'step_num': step_nums})

    extra_kwargs = {}
    if not plot_all:
        extra_kwargs = {'units': 'experiment_nums', 'estimator': None, 'hue': 'experiment_nums'}

    fig = plt.figure(figsize=(16, 16))
    ax = plt.gca()
    ax.set_axis_off()

    fig.add_subplot(2, 2, 1)
    plot = sns.lineplot(x='step_num', y='joint_likelihoods', data=df, **extra_kwargs)
    plt.title("Validation Joint Negative Log-Likelihood")
    plt.xlabel("Training Iterations")
    plt.ylabel("Negative Log-Likelihood")
    if error_cap is not None:
        axes = plot.axes
        axes.set_ylim(-0.1,error_cap)


    fig.add_subplot(2, 2, 2)
    plot = sns.lineplot(x='step_num', y='ntlls', data=df, **extra_kwargs)
    plt.title("Validation Transition Negative Log-Likelihood")
    plt.xlabel("Training Iterations")
    plt.ylabel("Negative Log-Likelihood")

    fig.add_subplot(2, 2, 3)
    plot = sns.lineplot(x='step_num', y='nalls', data=df, **extra_kwargs)
    plt.title("Validation Action Negative Log-Likelihood")
    plt.xlabel("Training Iterations")
    plt.ylabel("Negative Log-Likelihood")
    if error_cap is not None:
        axes = plot.axes
        axes.set_ylim(-0.1,error_cap)

    fig.add_subplot(2, 2, 4)
    plot = sns.lineplot(x='step_num', y='brs', data=df, **extra_kwargs)
    plt.title("Bellman Residuals")
    plt.xlabel("Training Iterations")
    plt.ylabel("Average Bellman Error")
    if error_cap is not None:
        axes = plot.axes
        axes.set_ylim(-0.1,error_cap)

    plt.savefig(out_file + '_all.png')


def compare_likelihood(data, out_file):

    assert(len(data)==2)


    exp_data = [loss_data for experiment_num, loss_data in data.items()]

    nall = np.array(exp_data[0]['val_nall']['values']) - np.array(exp_data[1]['val_nall']['values'])
    ntll = np.array(exp_data[0]['val_ntll']['values']) - np.array(exp_data[1]['val_ntll']['values'])
    br = np.array(exp_data[0]['val_tde']['values']) - np.array(exp_data[1]['val_tde']['values'])

    steps = exp_data[0]['val_nall']['steps']
    joint_likelihoods = nall + ntll

    df = pd.DataFrame(data={
        'joint_likelihoods': joint_likelihoods,
        'nalls':nall,
        'ntlls':ntll,
        'brs': br,
        'step_num': steps})

    fig = plt.figure(figsize=(16, 16))
    ax = plt.gca()
    ax.set_axis_off()

    fig.add_subplot(2, 2, 1)
    plot = sns.lineplot(x='step_num', y='joint_likelihoods', data=df)
    plt.title("Validation Joint Negative Log-Likelihood Difference")
    plt.xlabel("Training Iterations")
    plt.ylabel("Negative Log-Likelihood")

    fig.add_subplot(2, 2, 2)
    plot = sns.lineplot(x='step_num', y='ntlls', data=df)
    plt.title("Validation Transition Negative Log-Likelihood Difference")
    plt.xlabel("Training Iterations")
    plt.ylabel("Negative Log-Likelihood")

    fig.add_subplot(2, 2, 3)
    plot = sns.lineplot(x='step_num', y='nalls', data=df)
    plt.title("Validation Action Negative Log-Likelihood Difference")
    plt.xlabel("Training Iterations")
    plt.ylabel("Negative Log-Likelihood")

    fig.add_subplot(2, 2, 4)
    plot = sns.lineplot(x='step_num', y='brs', data=df)
    plt.title("Bellman Residual Difference")
    plt.xlabel("Training Iterations")
    plt.ylabel("Average Bellman Error")

    plt.savefig(out_file + '_comparison.png')



@loss_visualization.config
def config():
    out_dir = "logs/generated_images_"
    plot_all = False
    compare = False
    error_cap = None

    _ = locals()  # quieten flake8 unused variable warning
    del _


def load_loss_data(experiment_num):
    print("Loading experiment {}".format(experiment_num))
    with open(os.path.join('logs', 'sacred', str(experiment_num), 'metrics.json')) as file:
        data = json.load(file)

    return data


@loss_visualization.automain
def main(out_dir, _run, experiment_nums, plot_all, compare, error_cap):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data = {experiment_num: load_loss_data(experiment_num) for experiment_num in experiment_nums}
    out_file = os.path.join(out_dir, "{}_loss_visualization_from_{}".format("_".join(str(x) for x in experiment_nums), _run._id))
    if compare:
        compare_likelihood(data, out_file)
    else:
        visualize_likelihood(data, out_file, plot_all, error_cap)
