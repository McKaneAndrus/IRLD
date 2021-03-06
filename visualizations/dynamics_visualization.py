import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from matplotlib.patches import Rectangle
from matplotlib.pyplot import figure

from sacred import Experiment
from sacred.observers import FileStorageObserver

import pickle as pkl
import os

dynamics_visualization = Experiment('dynamics_visualization')
dynamics_visualization.observers.append(FileStorageObserver.create('logs/sacred/viz'))


def add_boxed_text(x, y, width, height, label, fill_color, plt, ax, font_size=25):
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


def visualize_dynamics(data, alg_labels, tile_labels, out_file='tmp.png', fig_width=8, margin = .1):
    """This creates the cross-style dynamics visualization and saves it as a .png

    Args:
      data: Data of shape [alg * tile_type * action * result]
      alg_labels: A list of labels representing the algorithms being visualized.
      tile_labels: A list of labels representing the tile types being visualized.
      out_file: A local path to where the visualization should be saved.
      fig_width: The length of the output image's width in inches.
      margin: A margin value used to keep things looking a little clean.
    """
    data = data.round(2)

    num_algs = len(alg_labels)
    num_tiles = len(tile_labels)

    # For each tile, we have a title "row" and 1 row for each algorithm.
    num_rows = (1 + num_algs) * num_tiles
    num_cols = 6  # 1 + the number of directions, which is currently 5.

    figure(num=None, figsize=(fig_width, fig_width*(1.*num_rows/num_cols)), dpi=80)
    fig = plt.figure()
    ax = fig.add_axes([-margin, -margin, num_cols + 2*margin, num_rows + 2*margin])
    # Set it up so that each full cross takes up a 1x1 space in the axes dimensions.
    plt.xlim([-margin, num_cols + margin])
    plt.ylim([-margin, num_rows + margin])
    plt.gca().set_aspect('equal', adjustable='box')

    for tile_i, tile in enumerate(tile_labels):
        # Confusingly, we build the graph from the top down due to default axes.
        # This represents the value for the top row of this tile's section.
        top_row_i = num_rows - 1 - (tile_i * (1 + num_algs))

        # Plot the first row, the title and column labels.
        plt.text(1 - margin, top_row_i + margin, tile, fontdict={'weight': 'bold'}, size=50, ha='right', va='bottom')
        for act_i, act in enumerate(['left', 'down', 'right', 'up', 'stay']):
            plt.text(act_i + 1.5, top_row_i + margin, act, size=30, ha='center', va='bottom')

        # Plot the subsequent rows, with a row label and then the appropriate cross visualizations.
        for alg_i, alg in enumerate(alg_labels):
            # First, plot the row label.
            plt.text(1 - margin, top_row_i - alg_i - 0.5, alg, size=30, ha='right', va='center')
            # Then, plot the cross and the bounding rectangle.
            for act_i, act in enumerate(data[alg_i][tile_i]):
                # TODO: Don't hard code this transposition of actions into up/left/right/down/stay
                add_boxed_text(act_i + 11. / 8, top_row_i - alg_i - 3. / 8, 1. / 4, 1. / 4,
                               act[3], green_fill(act[3]), plt, ax)
                add_boxed_text(act_i + 9. / 8, top_row_i - alg_i - 5. / 8, 1. / 4, 1. / 4,
                               act[0], green_fill(act[0]), plt, ax)
                add_boxed_text(act_i + 11. / 8, top_row_i - alg_i - 5. / 8, 1. / 4, 1. / 4,
                               act[4], green_fill(act[4]), plt, ax)
                add_boxed_text(act_i + 13. / 8, top_row_i - alg_i - 5. / 8, 1. / 4, 1. / 4,
                               act[2], green_fill(act[2]), plt, ax)
                add_boxed_text(act_i + 11. / 8, top_row_i - alg_i - 7. / 8, 1. / 4, 1. / 4,
                               act[1], green_fill(act[1]), plt, ax)
                ax.add_patch(Rectangle((act_i + 1, top_row_i - alg_i - 1), 1, 1, fill=False, linewidth=3))

    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    plt.savefig(out_file, bbox_inches='tight')


@dynamics_visualization.config
def config():
    fig_width = 8
    margin = .1
    alg_labels = ["Experiment"]
    tile_labels = ['0','1']
    out_dir = "logs/generated_images_"

    _ = locals()  # quieten flake8 unused variable warning
    del _



def load_dynamics(experiment_nums, alg_labels):
    data = [pkl.load(open(os.path.join("logs", "models", str(experiment_nums[0]), "tab", "true_adt_probs.pkl"), "rb"))]
    alg_labels.insert(0, 'Ground Truth')
    data.extend([pkl.load(open(os.path.join("logs", "models", str(experiment_num), "tab", "final_adt_probs.pkl"), "rb")) for experiment_num in experiment_nums])
    return np.array(data)


@dynamics_visualization.automain
def main(out_dir, _run, experiment_nums, alg_labels, tile_labels, fig_width, margin):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # alg_labels = alg_labels.split(",")
    # tile_labels = tile_labels.split(",")
    if len(experiment_nums) != len(alg_labels):
        raise Exception("Was given {} experiments and {} alg labels, these must match".format(len(experiment_nums),
                                                                                               len(alg_labels)))
    data = load_dynamics(experiment_nums, alg_labels)

    if len(data[0]) != len(tile_labels):
        raise Exception("Had {} tile types and {} tile labels, these must match".format(len(data[0]),
                                                                                        len(tile_labels)))
    
    out_file = os.path.join(out_dir, "dynamics_visualization_{}_from_{}.png".format(_run._id, experiment_nums))
    visualize_dynamics(data, alg_labels, tile_labels, out_file, fig_width, margin)
