import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from sacred import Experiment
from sacred.observers import FileStorageObserver

from utils.learning_utils import softmax

import pickle as pkl
import os
import json
import glob


value_function_visualization = Experiment('value_function_visualization')
value_function_visualization.observers.append(FileStorageObserver.create('logs/sacred/viz'))


MAP_COLORS = {b'F': "#ffffff", # Normal Square
              b'S': "#ffffff", # Normal Square
              b'U': "#000000", # Pit
              b'1': "reward", # Reward Square
              b'2': "reward", # Reward Square
              b'3': "reward"} # Reward Square


def add_boxed_text(x, y, width, height, label, fill_color, plt, ax, font_size=8):
    """This adds text within a box on a plot.
    """
    plt.text(x + width * .5, y + height * .5, label, size=font_size, ha="center", va="center")
    rect = Rectangle((x, y), width, height, facecolor=fill_color)
    ax.add_patch(rect)


def green_fill(proportion):
    """Given a proportion from 0 to 1, return a color value on a green colorbar..
    """
    cmap = cm.get_cmap('Greens')
    return cmap(1./12 + proportion*2/3)


def visualize_temporal_value_function(q_val_dyn_list, mdp, show_art, out_dir, metrics, coordinate):
    """This creates a grid of grid-world layouts representing the Q-values.

    Args:
        q_val_dyn_list: A list of q_vals, each of which is of shape [state * action]
        mdp: A MarsExplorerEnv
        q_val_labels: Human-readable labels associated with the first dimension of q_val_data above.
        out_file: A local path to where the visualization should be saved.
    """
    # Extract the data arguments.
    num_rows = mdp.nrow
    num_cols = mdp.ncol
    num_actions = mdp.num_actions

    fig = plt.figure(figsize=(num_cols, num_rows/2))
    ax = plt.gca()
    ax.set_axis_off()

    if coordinate:
        regime_steps = {metrics["coordinate_regime"]["steps"][i]: metrics["coordinate_regime"]["values"][i] for i in range(len(metrics["coordinate_regime"]["values"]))}

    if show_art:
        plt.title('Map', fontsize=30, weight='bold')

        art_map = np.zeros([num_rows, num_cols, 3])
        max_reward = np.max(mdp.reward_map)
        # Keep track of the minimum rewards of a reward square,
        # which will be higher than np.min(mdp.reward_map)
        min_reward = max_reward
        for y in range(num_rows):
            for x in range(num_cols):
                color = MAP_COLORS[mdp.tile_map[y][x]]
                if color == 'reward':
                    min_reward = min(min_reward, mdp.reward_map[y][x])
                    color = cm.get_cmap("Blues")(mdp.reward_map[y][x] / max_reward)
                art_map[y][x] = colors.to_rgb(color)

        img = plt.imshow(art_map)
        plt.savefig(os.path.join(out_dir, "art_map.png"))
        plt.clf()

    for step_num, q_val, dyn_model in q_val_dyn_list:
        fig.add_subplot(1, 2, 1)
        # ------ Plot the q-values
        plot_title = str(step_num)
        if coordinate and step_num in regime_steps:
            plot_title += " " + regime_steps[step_num]

        plt.title(plot_title, fontsize=30, weight='bold')

        q_val = q_val.reshape([num_rows, num_cols, num_actions])
        max_val = np.max(q_val, axis=2)
        softmaxed_val = np.max(softmax(q_val, axis=2), axis=2)
        max_arg_val = np.argmax(q_val, axis=2)
        action_to_uv = {
            0: (-1, 0),
            1: (0, -1),
            2: (1, 0),
            3: (0, 1),
            4: (0, 0),
        }
     
        for y in range(num_rows):
            for x in range(num_cols):
                u, v = action_to_uv[max_arg_val[y, x]]
                arrow_mag = softmaxed_val[y, x]
                # If the policy indicates going in any direction, plot a variable
                # length arrow based on that confidence.
                if u or v:
                    plt.arrow(x, y, u * .5 * arrow_mag, -v * .5 * arrow_mag,
                        color = 'm', head_width = 0.2, head_length = 0.1)
                # Otherwise, just plot a point to indicate staying.
                else:
                    plt.plot(x, y, 'mo')
        imshow = plt.imshow(max_val, cmap='gray')

        # ------ Plot the dynamics
        ax = fig.add_subplot(1, 2, 2)
        ax.set_axis_off()
        margin = .1
        dyn_data = dyn_model.round(2)
        num_algs = 1
        num_tiles = dyn_model.shape[0]

        dyn_num_rows = (1 + num_algs) * num_tiles
        dyn_num_cols = 5

        plt.xlim([-margin, dyn_num_cols + margin])
        plt.ylim([-margin, dyn_num_rows + margin])

        for tile_i in range(num_tiles):
            top_row_i = dyn_num_rows - 1 - (tile_i * (1 + num_algs))
            plt.text(1 - margin, top_row_i + margin, str(tile_i), fontdict={'weight': 'bold'}, size=50, ha='right', va='bottom')
            for act_i, act in enumerate(['left', 'down', 'right', 'up', 'stay']):
                plt.text(act_i + 0.5, top_row_i + margin, act, size=15, ha='center', va='bottom')

            # Plot the cross and the bounding rectangle.
            for act_i, act in enumerate(dyn_data[tile_i]):
                add_boxed_text(act_i + 3. / 8, top_row_i - 3. / 8, 1. / 4, 1. / 4,
                               act[3], green_fill(act[3]), plt, ax)
                add_boxed_text(act_i + 1. / 8, top_row_i - 5. / 8, 1. / 4, 1. / 4,
                               act[0], green_fill(act[0]), plt, ax)
                add_boxed_text(act_i + 3. / 8, top_row_i - 5. / 8, 1. / 4, 1. / 4,
                               act[4], green_fill(act[4]), plt, ax)
                add_boxed_text(act_i + 5. / 8, top_row_i - 5. / 8, 1. / 4, 1. / 4,
                               act[2], green_fill(act[2]), plt, ax)
                add_boxed_text(act_i + 3. / 8, top_row_i - 7. / 8, 1. / 4, 1. / 4,
                               act[1], green_fill(act[1]), plt, ax)
                ax.add_patch(Rectangle((act_i, top_row_i - 1), 1, 1, fill=False, linewidth=3))

        plt.savefig(os.path.join(out_dir, str(step_num) + ".png"))
        plt.clf()
        fig = plt.figure(figsize=(num_cols, num_rows/2))



@value_function_visualization.config
def config():
    out_dir = "logs/generated_images_"
    show_art = None
    coordinate = False

    _ = locals()  # quieten flake8 unused variable warning
    del _


def load_loss_data(experiment_num):
    print("Loading experiment {}".format(experiment_num))
    with open(os.path.join('logs', 'sacred', str(experiment_num), 'metrics.json')) as file:
        data = json.load(file)

    return data

def load_single_exp_data(experiment_num):
    file_path = os.path.join("logs", "models", str(experiment_num), "tab")
    data = [(-1, pkl.load(open(os.path.join(file_path, "true_q_vals.pkl"), "rb")), pkl.load(open(os.path.join(file_path, "true_adt_probs.pkl"), "rb")))]

    val_file_header = os.path.join(file_path, "q_vals*")
    val_model_files = glob.glob(val_file_header)
    # Handles files of form %s_%s_%d.pkl, retrieves %d
    value_iters_models = [(int(file.split("_")[-1].split(".")[0]), pkl.load(open(file, 'rb'))) for file in val_model_files]
    value_iters_models = list(sorted(value_iters_models, key=lambda x: x[0]))
    dyn_file_header = os.path.join(file_path, "adt_probs*")
    dyn_model_files = glob.glob(dyn_file_header)
    dyn_iters_models = [(int(file.split("_")[-1].split(".")[0]), pkl.load(open(file, 'rb'))) for file in dyn_model_files]
    dyn_iters_models = list(sorted(dyn_iters_models, key=lambda x: x[0]))
    iters_models = [(value_iters_models[i][0], value_iters_models[i][1], dyn_iters_models[i][1]) for i in range(len(dyn_iters_models))]
    data += iters_models
    mdp = pkl.load(open(os.path.join("logs", "models", str(experiment_num), "mdp.pkl"), "rb"))
    return data, mdp


@value_function_visualization.automain
def main(out_dir, _run, experiment_num, show_art, coordinate):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    metrics = load_loss_data(experiment_num)
    q_val_dyn_list, mdp = load_single_exp_data(experiment_num)
    out_dir = os.path.join(out_dir, "{}_temporal_value_function_vis_{}".format(experiment_num, _run._id))
    os.makedirs(out_dir)
    visualize_temporal_value_function(q_val_dyn_list, mdp, show_art, out_dir, metrics, coordinate)
