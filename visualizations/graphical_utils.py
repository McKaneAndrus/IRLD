import numpy as np, numpy.random as nr, gym
from itertools import product
import matplotlib.pyplot as plt
import time
import matplotlib.cm as cm
import matplotlib.colors as colors
import os
import seaborn as sns
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib._png import read_png
import glob

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
NOOP = 4

cdict = {'red':   ((0.0,  0.173, 0.173),
                   (1.0,  0.925, 0.925)),

         'green': ((0.0,  0.067, 0.067),
                   (1.0, 0.384, 0.384)),

         'blue':  ((0.0,  0.027, 0.027),
                   (1.0,  0.196, 0.196))}
plt.register_cmap(name='RustPlanet', data=cdict)
REWARD_COLORS = cm.get_cmap('RustPlanet')
AGENT_COLORS = cm.get_cmap('gray')
MAP_COLORS = {b'B': "#3a0e00",
              b'F': "#933111",
              b'S': "#933111",
              b'U': "#d65b33",
              b'1': "#956F52",
              b'2': "#3C2F34",
              b'3': "#644C42"}

ROVER_PNGS = {LEFT: "resources/rover_left.png",
              RIGHT: "resources/rover_right.png",
              DOWN: "resources/rover_down.png",
              UP: "resources/rover_up.png",
              NOOP: "resources/rover_sample.png"}


def plot_mars(game, pi, folder, t, a, V=None, s=None, b=None, title=None, counts = None):
    mdp = game.true_mdp
    if V is None:
        V = np.array([colors.to_rgb(MAP_COLORS[l]) for l in mdp.desc.flat])
    V = V.reshape(mdp.nrow, mdp.ncol,3)
    fig = plt.figure(figsize=(4,4))
    if title != None:
        plt.title(title)
    plt.imshow(V)
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    ax = plt.gca()
    # for y in range(V.shape[0]):
    #     for x in range(V.shape[1]):
    #         # a = Pi[y, x]
    #         # u, v = a2uv[a]
    #         # plt.arrow(x, y,u*.3, -v*.3, color='m', head_width=0.1, head_length=0.1)
    #         if mdp.desc[y,x] in b'123':
    #             plt.text(x, y, str(mdp.desc[y,x].item().decode()),
    #                  color='w', size=12,  verticalalignment='center',
    #                  horizontalalignment='center', fontweight='bold')
    if s is not None:
        s = mdp.get_state(s)
        img_path = ROVER_PNGS[a]
        arr_hand = read_png(img_path)
        imagebox = OffsetImage(arr_hand, zoom=.5)
        xy = [s%V.shape[0], s//V.shape[0]]  # coordinates to position this image

        ab = AnnotationBbox(imagebox, xy,
                            xybox=(0, 0),
                            xycoords='data',
                            boxcoords="offset points",
                            frameon=False)
        ax.add_artist(ab)
        # if b is not None:
        #     plt.scatter([s%V.shape[0]], [s//V.shape[0]], c = AGENT_COLORS(b), s=[400.0])
        # else:
        #     plt.plot(s % V.shape[0], s // V.shape[0], 'ro')
    # plt.grid(color='b', lw=2, ls='-')
    fig.text(0.13, 0.05, 't: {}'.format(t), ha='left', fontsize=10)
    if counts:
        fig.text(0.87, 0.05, 'F: {}  L/R: {}  B: {}  S: {}'.format(counts['F'], counts['LR'], counts['B'], counts['S']), ha='right', fontsize=10)
    sns.despine(bottom=True, left=True, right=True, top=True)
    plt.savefig(folder+"/"+str(t)+".png", format='png')
    return

# TODO: need to include value returns for all algorithms

def visualize_mars_history(game, history, folder, use_counts=False, prev_step=None):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        if prev_step is None:
            folder = folder + str(round(time.time()))
            os.makedirs(folder)
    prev_step = 0 if prev_step is None else prev_step
    I = len(game.mdps)
    T = len(history['states'])
    for t in range(prev_step,T):
        pol = history['policies'][t]
        s = history['states'][t]
        b = history['beliefs'][t][0]
        a = history['actions'][t]
        title = '           '.join(["R{}: {}".format(i+1, round(history['rewards'][t][i],2)) for i in range(I)])
        # title += '  Trust: {}'.format(round((history['beliefs'][t][1] -0.5)*2,2))
        counts = history['counts'][t] if use_counts else None
        plot_mars(game, pol, folder, t, a, b=b, s=s, title=title, counts=counts)
    gif_name = folder.split('/')[-1]
    file_list = glob.glob(folder + '/*.png')  # Get all the pngs in the current directory
    list.sort(file_list, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    image_file = folder + '/image_list.txt'
    with open(image_file, 'w') as file:
        for item in file_list:
            file.write("%s\n" % item)
    os.system('convert -delay 50 @{} plots/{}.gif'.format(image_file,gif_name))
