import numpy as np
import random
from envs.environment_utils import featurize_states, gridify_states


def softmax(x, axis=1):
    """Compute softmax values for each sets of scores in x."""
    axis = 0 if len(x.shape) == 1 else axis
    e_x = np.exp(x - np.max(x, axis=axis,keepdims=True))
    return (e_x / e_x.sum(axis=axis, keepdims = True))


def nn_vectorize_rollouts(mdp, sas_obs, adt_obs):
    states, acts, next_states = zip(*sas_obs)
    _, dirs, tiles = zip(*adt_obs)
    states_feats = featurize_states(mdp, states)
    states = np.expand_dims(np.array(states), axis=1)
    actions = np.expand_dims(np.array(acts), axis=1)
    next_states_feats = featurize_states(mdp, next_states)
    next_states = np.expand_dims(np.array(next_states), axis=1)
    dirs = np.expand_dims(np.array(dirs), axis=1)
    tiles = np.expand_dims(np.array(tiles), axis=1)
    return (states, states_feats, actions, next_states, next_states_feats, dirs, tiles)


def get_rollout_indexes(sas_obs):
    demo_example_idxes = list(range(len(sas_obs)))
    random.shuffle(demo_example_idxes)
    n_train_demo_examples = int(0.9 * len(demo_example_idxes))
    train_demo_example_idxes = demo_example_idxes[:n_train_demo_examples]
    val_demo_example_idxes = demo_example_idxes[n_train_demo_examples:]
    return (train_demo_example_idxes, val_demo_example_idxes)


def sample_batch(rollouts, indexes, size=None):
    idxes = random.sample(indexes, size) if size is not None else indexes
    demo_batch = [comp[idxes] for comp in rollouts]
    return demo_batch


def generate_constraints(mdp):
    #TODO remove reliance on global state
    s = np.arange(mdp.num_states)
    a = np.arange(mdp.num_actions)
    sa = np.transpose([np.tile(s, len(a)), np.repeat(a, len(s))])
    states, acts = sa[:,0], sa[:,1]
    feat_states = featurize_states(mdp, states)
    grid_states = gridify_states(mdp, states)
    tiles = np.expand_dims(np.array([mdp.get_tile_type(s) for s in states]), axis=1)
    acts_array = np.expand_dims(acts, axis=1)
    rewards = np.array([mdp.get_reward(states[i], acts[i]) for i in range(len(states))])
    sprimes = mdp.get_possible_sprimes(states)
    sps = sprimes.shape
    feat_next_states = featurize_states(mdp, sprimes.reshape((sps[0]*sps[1]))).reshape((sps[0], sps[1],
                                                                                        mdp.nrow + mdp.ncol))

    return feat_states, acts_array, rewards, feat_next_states, tiles


def update_switcher(update, update_progression, losses, slope_threshold=1e-4, horizon=100):
    if len(losses) <= 3:
        switch = False
    else:
        if len(losses) < horizon:
            horizon = len(losses)
        slope = np.polyfit(np.arange(horizon), losses[-horizon:], 1)[0]
        switch = -slope < slope_threshold

        print(-slope, switch)

    if switch:
        if update not in update_progression:
            update = update_progression[0]
        else:
            update = update_progression[(update_progression.index(update) + 1) % len(update_progression)]

    return update
