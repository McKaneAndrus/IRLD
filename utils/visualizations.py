import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from matplotlib.patches import Rectangle
from matplotlib.pyplot import figure

def addBoxedText(x, y, width, height, label, fill_color, plt, ax, font_size=25):
  """This adds text within a box on a plot.
  """
  plt.text(x + width * .5, y + height * .5, label, size=font_size, ha="center", va="center")
  rect = Rectangle((x, y), width, height, facecolor=fill_color)
  ax.add_patch(rect)

def greenFill(proportion):
  """Gien a proportion from 0 to 1, return a color value on a green colorbar..
  """
  cmap = cm.get_cmap('Greens')
  return cmap(1./12 + proportion*2/3)

def dynamicsVis(data, algo_labels, tile_labels, out_file='tmp.png', fig_width=8, margin = .1):
  """This creates the cross-style dynamics visualization and saves it as a .png

  Args:
    data: Data of shape [algo * tile_type * action * result]
    algo_labels: A list of labels representing the algorithms being visualized.
    tile_labels: A list of labels representing the tile types being visualized.
    out_file: A local path to where the visualization should be saved.
    fig_width: The length of the output image's width in inches.
    margin: A margin value used to keep things looking a little clean.
  """
  data = data.round(2)

  num_algos = len(algo_labels)
  num_tiles = len(tile_labels)

  # For each tile, we have a title "row" and 1 row for each algorithm.
  num_rows = (1 + num_algos) * num_tiles
  num_cols = 6 # 1 + the number of directions, which is currently 5.

  figure(num=None, figsize=(fig_width, fig_width*(1.*num_rows/num_cols)), dpi=80)
  fig = plt.figure()
  ax = fig.add_axes([-margin, -margin, num_cols + 2*margin, num_rows + 2*margin])
  # Set it up so that each full cross takes up a 1x1 space in the axes dimensions.
  plt.xlim([-margin, num_cols + margin])
  plt.ylim([-margin, num_cols + margin])
  plt.gca().set_aspect('equal', adjustable='box')

  for tile_i, tile in enumerate(tile_labels):
    # Confusingly, we build the graph from the top down due to default axes.
    # This represents the value for the top row of this tile's section.
    top_row_i = num_rows - 1 - (tile_i * (1 + num_algos))

    # Plot the first row, the title and column labels.
    plt.text(0, top_row_i + margin, tile, size=50, ha='left', va='bottom')
    for act_i, act in enumerate(['up', 'left', 'right', 'down', 'stay']):
      plt.text(act_i + 1.5, top_row_i + margin, act, size=30, ha='center', va='bottom')

    # Plot the subsequent rows, with a row label and then the appropriate cross visualizations.
    for algo_i, algo in enumerate(algo_labels):
      # First, plot the row label.
      plt.text(1 - margin, top_row_i - algo_i - 0.5, algo, size=30, ha='right', va='center')
      # Then, plot the cross and the boudning rectangle.
      for act_i, act in enumerate(data[algo_i][tile_i]):
        # TODO: Don't hardcode this transposition of actions into up/left/right/down/stay
        addBoxedText(act_i + 11./8, top_row_i - algo_i - 3./8, 1./4, 1./4,
          act[0], greenFill(act[0]), plt, ax)
        addBoxedText(act_i + 9./8, top_row_i - algo_i - 5./8, 1./4, 1./4,
          act[1], greenFill(act[1]), plt, ax)
        addBoxedText(act_i + 11./8, top_row_i - algo_i - 5./8, 1./4, 1./4,
          act[4], greenFill(act[4]), plt, ax)
        addBoxedText(act_i + 13./8, top_row_i - algo_i - 5./8, 1./4, 1./4,
          act[2], greenFill(act[2]), plt, ax)
        addBoxedText(act_i + 11./8, top_row_i - algo_i - 7./8, 1./4, 1./4,
          act[3], greenFill(act[3]), plt, ax)
        ax.add_patch(Rectangle(
          (act_i + 1, top_row_i - algo_i - 1), 1, 1, fill=False, linewidth=3))

  plt.gca().set_aspect('equal', adjustable='box')
  ax.set_axis_off()
  plt.savefig(out_file, bbox_inches='tight')
