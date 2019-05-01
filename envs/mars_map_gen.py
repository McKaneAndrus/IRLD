import numpy as np
from collections import Counter
import pickle as pkl
import os
from envs.environment_visualization_utils import plot_mars_map
from envs.environment_setup_utils import get_mdp_from_map


TILE_TYPES = ['S', 'F', 'U', '1', '2', '3']

TILE_MINS = {'S': 0.05,
             '1': 0.01,
             '2': 0.01}

TILE_MAXES = {'S': 0.1,
              '1': 0.03,
              '2': 0.03,
              '3': 0.01}

TILE_PROXIMITY_BOOST = {'F':0.5,
                        'U':0.1}

DEFAULT_WEIGHT = 1.0

def make_map(height, width, clustering_iterations = 0, seed=0):

    np.random.seed(seed)
    total = float(height * width)
    tile_map = np.random.choice(TILE_TYPES, size=(height, width))
    tile_proportions = {tt:len(np.nonzero(tile_map == tt)[0])/total for tt in TILE_TYPES}

    maxed_out = [tt for tt in TILE_MAXES.keys() if tile_proportions[tt] > TILE_MAXES[tt] + 1/total]
    exchange_chars = [tt for tt in TILE_TYPES if tt not in maxed_out]
    while len(maxed_out) > 0:
        for tt in TILE_MAXES.keys():
            exchanges = max(int((tile_proportions[tt] - TILE_MAXES[tt]) * total), 0)
            for _ in range(exchanges):
                indexes = np.nonzero(np.isin(tile_map, [tt]))
                i = np.random.choice(indexes[0].shape[0])
                index = (indexes[0][i], indexes[1][i])
                tile_map[index] = np.random.choice(exchange_chars)
        tile_proportions = {tt: len(np.nonzero(tile_map == tt)[0]) / total for tt in TILE_TYPES}
        maxed_out = [tt for tt in TILE_MAXES.keys() if tile_proportions[tt] > TILE_MAXES[tt] + 1/total]
        exchange_chars = [tt for tt in exchange_chars if tt not in maxed_out]


    while any([tile_proportions[tt] < TILE_MINS[tt] for tt in TILE_MINS.keys()]):
        for tt in TILE_MINS.keys():
            exchanges = max(int((TILE_MINS[tt] - tile_proportions[tt]) * total), 0)
            for _ in range(exchanges):
                indexes = np.nonzero(np.isin(tile_map, TILE_MINS.keys(), invert=True))
                i = np.random.choice(indexes[0].shape[0])
                index = (indexes[0][i], indexes[1][i])
                tile_map[index] = tt
        tile_proportions = {tt: len(np.nonzero(tile_map == tt)[0]) / total for tt in TILE_TYPES}

    for _ in range(clustering_iterations):
        indexes = np.nonzero(np.isin(tile_map, exchange_chars))
        indexes = zip(indexes[0], indexes[1])
        for index in indexes:
            adjacents = _get_all_adjacent_tiles(tile_map, index[0], index[1])
            weights = [DEFAULT_WEIGHT + adjacents[tt] * TILE_PROXIMITY_BOOST[tt] for tt in exchange_chars]
            total_weight = sum(weights)
            weights = [w/total_weight for w in weights]
            tile_map[index] = np.random.choice(exchange_chars, p=weights)

    tile_proportions = {tt: len(np.nonzero(tile_map == tt)[0]) / total for tt in TILE_TYPES}
    print(tile_proportions)
    print(tile_map)
    string_tile_map = ["".join(row) for row in tile_map]
    return string_tile_map

def make_maps(height, width, n_maps, output_dir=None, clustering_iterations=0, seed=0):

    maps = [make_map(height,width, clustering_iterations=clustering_iterations) for _ in range(n_maps)]
    if output_dir is not None:
        pkl.dump(maps, open(os.path.join(output_dir, "mars_maps_{}_{}_{}.pkl".format(
                                                    height,width,clustering_iterations, seed))))
    return maps


def visualize_map(tile_map):
    mdp = get_mdp_from_map(tile_map)
    plot_mars_map(mdp)



def _get_all_adjacent_tiles(tile_map, h, w):

    adjacents = []
    height, width = len(tile_map), len(tile_map[0])
    for hdelta in range(-1, 2):
        for wdelta in range(-1, 2):
            hprime = (h+hdelta) % height
            wprime = (w+wdelta) % width
            adjacents += [tile_map[hprime][wprime]]
    return Counter(adjacents)


if __name__ == "__main__":
    # make_maps(15,15, 10,  clustering_iterations=10)
    tile_map = make_map(15,15, 10, 0)
    visualize_map(tile_map)