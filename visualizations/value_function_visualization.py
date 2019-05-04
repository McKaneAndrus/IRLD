import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sacred import Experiment
from sacred.observers import FileStorageObserver

from utils.learning_utils import softmax

import pickle as pkl
import os

value_function_visualization = Experiment('value_function_visualization')
value_function_visualization.observers.append(FileStorageObserver.create('logs/sacred'))


MAP_COLORS = {b'F': "#ffffff", # Normal Square
              b'S': "#ffffff", # Normal Square
              b'U': "#000000", # Pit
              b'1': "reward", # Reward Square
              b'2': "reward", # Reward Square
              b'3': "reward"} # Reward Square


def visualize_value_function(q_val_list, mdp, q_val_labels, show_art, out_file):
    """This creates a grid of grid-world layouts representing the Q-values.

    Args:
        q_val_list: A list of q_vals, each of which is of shape [state * action]
        mdp: A MarsExplorerEnv
        q_val_labels: Human-readable labels associated with the first dimension of q_val_data above.
        out_file: A local path to where the visualization should be saved.
    """
    # Extract the data arguments.
    num_rows = mdp.nrow
    num_cols = mdp.ncol
    num_actions = mdp.num_actions
    num_plots = len(q_val_list)

    if show_art:
        num_plots += 1

    plt_num_rows = int(np.ceil((num_rows + 1) * num_plots / 2))
    plt_num_cols = (num_cols + 1) * 2
    fig = plt.figure(figsize=(plt_num_cols, plt_num_rows))
    ax = plt.gca()
    ax.set_axis_off()

    i_offset = 0
    if show_art:
        i_offset = 1

        sub_ax = fig.add_subplot(int(np.ceil(num_plots / 2)), 2, 1)
        sub_ax.set_xticks(np.arange(num_rows) - .5)
        sub_ax.set_yticks(np.arange(num_cols) - .5)
        sub_ax.set_xticklabels([])
        sub_ax.set_yticklabels([])

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
        divider = make_axes_locatable(sub_ax)
        cax = divider.append_axes("left", size="5%", pad=0.15)
        cax.set_axis_off()

        legend_colors = [
            Rectangle((0, 0), 1, 1, facecolor=MAP_COLORS[b'F'], edgecolor='#000000'),
            Rectangle((0, 0), 1, 1, facecolor=MAP_COLORS[b'U'], edgecolor='#000000'),
            Rectangle((0, 0), 1, 1, facecolor=cm.get_cmap('Blues')(min_reward / max_reward), edgecolor='#000000'),
            Rectangle((0, 0), 1, 1, facecolor=cm.get_cmap('Blues')(1.), edgecolor='#000000'),
        ]
        leg = cax.legend(legend_colors, ['Regular', 'Pit', 'Low Reward', 'High Reward'], fontsize=20)
        leg.get_frame().set_edgecolor('#000000')

    for i, q_val in enumerate(q_val_list):
        # Get the x/y offset for this particular visualization within
        # the larger grid.
        # i_offset depends on whether we're rendering the art version
        # of the environment.
        x_offset = ((i + i_offset) % 2) * (num_cols + 1)
        y_offset = ((i + i_offset) // 2) * (num_rows + 1)


        sub_ax = fig.add_subplot(int(np.ceil(num_plots / 2)), 2, i + i_offset + 1)
        sub_ax.set_xticks(np.arange(num_rows) - .5)
        sub_ax.set_yticks(np.arange(num_cols) - .5)
        sub_ax.set_xticklabels([])
        sub_ax.set_yticklabels([])

        label = q_val_labels[i]
        plt.title(label, fontsize=30, weight='bold')

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

        # Weird hardcoding for the poster, yay!
        if i + i_offset == 2:
            divider = make_axes_locatable(sub_ax)
            cax = divider.append_axes("left", size="5%", pad=0.15)
            cbar = fig.colorbar(imshow, cax=cax, ticks=[np.min(max_val), np.max(max_val)])
            cbar.ax.yaxis.set_ticks_position('left')
            cbar.ax.set_yticklabels(['Min Q-Val', 'Max Q-Val'], fontsize=20)
        else:
            divider = make_axes_locatable(sub_ax)
            cax = divider.append_axes("left", size="5%", pad=0.15)
            cax.set_axis_off()

    plt.savefig(out_file, bbox_inches='tight')


@value_function_visualization.config
def config():
    out_dir = "logs/generated_images_"
    labels = "Experiment"
    show_art = None

    _ = locals()  # quieten flake8 unused variable warning
    del _


def load_single_exp_data(experiment_nums, labels):
    data = [pkl.load(open(os.path.join("logs", "models", str(experiment_nums[0]), "tab", "true_q_vals.pkl"), "rb"))]
    labels.insert(0, 'Ground Truth')
    data.extend([pkl.load(open(os.path.join("logs", "models", str(experiment_num), "tab", "final_q_vals.pkl"), "rb")) for experiment_num in experiment_nums])
    mdp = pkl.load(open(os.path.join("logs", "models", str(experiment_nums[0]), "mdp.pkl"), "rb"))
    return np.array(data), mdp


@value_function_visualization.automain
def main(out_dir, _run, experiment_nums, labels, show_art):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    labels = labels.split(",")
    if len(experiment_nums) != len(labels):
        raise Exception("Was given {} experiments and {} labels, these must match".format(len(experiment_nums),
                                                                                          len(labels)))
    if show_art is None:
        show_art = len(labels) > 2

    q_val_list, mdp = load_single_exp_data(experiment_nums, labels)
    out_file = os.path.join(out_dir, "value_function_visualization_{}_from_{}.png".format(_run._id,  experiment_nums))
    visualize_value_function(q_val_list, mdp, labels, show_art, out_file)
