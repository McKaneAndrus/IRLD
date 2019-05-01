import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib._png import read_png
import seaborn as sns
import matplotlib.colors as colors
from utils.learning_utils import softmax

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
NOOP = 4

ROVER_PNGS = {LEFT: "resources/rover_left.png",
              RIGHT: "resources/rover_right.png",
              DOWN: "resources/rover_down.png",
              UP: "resources/rover_up.png",
              NOOP: "resources/rover_sample.png"}

MAP_COLORS = {b'B': "#3a0e00",
              b'F': "#933111",
              b'S': "#933B11",
              b'U': "#d65b33",
              b'1': "#956F52",
              b'2': "#3C2F34",
              b'3': "#644C42"}


def plot_mars(mdp, pi, term=40, title=None, counts=None, Qs=None):
    background = np.array([colors.to_rgb(MAP_COLORS[l]) for l in mdp.tile_map.flat]).reshape(mdp.nrow, mdp.ncol, 3)
    s = mdp._reset()
    t, r = 0, 0
    while s is not None and t < term:
        fig = plt.figure(figsize=(4, 4))
        if title != None:
            plt.title(title)
        plt.imshow(background)
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        ax = plt.gca()
        a = pi[s] if type(pi) == np.ndarray else pi(s)
        img_path = ROVER_PNGS[a]
        arr_hand = read_png(img_path)
        imagebox = OffsetImage(arr_hand, zoom=.5)
        xy = [s % background.shape[1], s // background.shape[1]]  # coordinates to position this image

        ab = AnnotationBbox(imagebox, xy,
                            xybox=(0, 0),
                            xycoords='data',
                            boxcoords="offset points",
                            frameon=False)
        ax.add_artist(ab)
        fig.text(0.13, 0.05, 't: {}      r: {}'.format(t, round(r, 4)), ha='left', fontsize=10)
        sns.despine(bottom=True, left=True, right=True, top=True)
        plt.show()
        if Qs is not None:
            print(Qs[s])
            print(a)
            print(softmax(Qs[s]))
        print(mdp.s_to_grid(s), mdp.tile_map.flatten()[s])
        s, rt, _, d = mdp._step(a)
        print(mdp.s_to_grid(s), mdp.tile_map.flatten()[s])
        t += 1
        r += rt
    if counts:
        fig.text(0.87, 0.05,
                 'F: {}  L/R: {}  B: {}  S: {}'.format(counts['F'], counts['LR'], counts['B'], counts['S']),
                 ha='right', fontsize=10)
    #     plt.savefig(folder+"/"+str(t)+".png", format='png')
    return


def plot_mars_history(mdp, hist, title=None, counts=None):
    background = np.array([colors.to_rgb(MAP_COLORS[l]) for l in mdp.tile_map.flat]).reshape(mdp.nrow, mdp.ncol, 3)
    t = 0
    for s, a, sprime in [h[0] for h in hist]:
        fig = plt.figure(figsize=(4, 4))
        if title != None:
            plt.title(title)
        plt.imshow(background)
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        ax = plt.gca()
        img_path = ROVER_PNGS[a]
        arr_hand = read_png(img_path)
        imagebox = OffsetImage(arr_hand, zoom=.5)
        xy = [s % background.shape[1], s // background.shape[1]]  # coordinates to position this image

        ab = AnnotationBbox(imagebox, xy,
                            xybox=(0, 0),
                            xycoords='data',
                            boxcoords="offset points",
                            frameon=False)
        ax.add_artist(ab)
        fig.text(0.13, 0.05, 't: {}'.format(t), ha='left', fontsize=10)
        sns.despine(bottom=True, left=True, right=True, top=True)
        plt.show()
        t += 1
    if counts:
        fig.text(0.87, 0.05,
                 'F: {}  L/R: {}  B: {}  S: {}'.format(counts['F'], counts['LR'], counts['B'], counts['S']),
                 ha='right', fontsize=10)
    #     plt.savefig(folder+"/"+str(t)+".png", format='png')
    return


def plot_values(mdp, Qs, s=None, title=None):
    V = np.max(Qs, axis=1).reshape((mdp.nrow, mdp.ncol))
    pi = np.argmax(Qs, axis=1)
    plt.figure(figsize=(8, 8))
    if title != None:
        plt.title(title)
    plt.imshow(V, cmap='gray')  # , clim=(0,1)) 'gist_ncar'
    ax = plt.gca()
    ax.set_xticks(np.arange(V.shape[1]) - .5)
    ax.set_yticks(np.arange(V.shape[0]) - .5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    Y, X = np.mgrid[0:V.shape[0], 0:V.shape[1]]
    a2uv = {0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1), 4: (-1, -1)}
    Pi = pi.reshape(V.shape)
    for y in range(V.shape[0]):
        for x in range(V.shape[1]):
            a = Pi[y, x]
            u, v = a2uv[a]
            plt.arrow(x, y, u * .3, -v * .3, color='m', head_width=0.2, head_length=0.1)
            plt.text(x, y, str(mdp.tile_map[y, x].item().decode()),
                     color='c', size=12, verticalalignment='center',
                     horizontalalignment='center', fontweight='bold')
    if s != None:
        plt.plot(s % V.shape[0], s // V.shape[0], 'ro')
    #     plt.grid(color='b', lw=2, ls='-')
    return


def plot_reward_map(mdp):
    plt.imshow(mdp.reward_map, cmap="Blues")
    plt.title("Goal Rewards")
    plt.show()


def plot_texture_map(mdp):
    plt.imshow(mdp.pure_texture_map, cmap="Blues")
    plt.title("'Texture' Rewards")
    plt.show()


def plot_tile_map(mdp):
    binary_map = np.ones(mdp.tile_map.shape)
    binary_map[np.where(mdp.tile_map == b'S')] = 255.0
    plt.imshow(binary_map, cmap="Blues")
    plt.title("Starting positions")
    plt.show()


def plot_mars_map(mdp, s=None):
    if s is None:
        s = mdp.state
    background = np.array([colors.to_rgb(MAP_COLORS[l]) for l in mdp.tile_map.flat]).reshape(mdp.nrow, mdp.ncol, 3)
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(background)
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    ax = plt.gca()
    img_path = ROVER_PNGS[4]
    arr_hand = read_png(img_path)
    imagebox = OffsetImage(arr_hand, zoom=.5)
    xy = [s % background.shape[1], s // background.shape[1]]  # coordinates to position this image
    ab = AnnotationBbox(imagebox, xy,
                        xybox=(0, 0),
                        xycoords='data',
                        boxcoords="offset points",
                        frameon=False)
    ax.add_artist(ab)
    sns.despine(bottom=True, left=True, right=True, top=True)
    plt.show()

