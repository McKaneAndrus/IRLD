import numpy as np

def gridify_states(mdp, states):
    return np.array([mdp.s_to_grid(state) for state in states])

def featurize_states(mdp, states):
    grid_points = gridify_states(mdp, states)
    row_one_hots, col_one_hots = np.eye(mdp.nrow), np.eye(mdp.ncol)
    return np.array([np.concatenate((row_one_hots[gp[0]], col_one_hots[gp[1]])) for gp in grid_points])