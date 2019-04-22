import numpy as np
from utils.learning_utils import nn_vectorize_rollouts, generate_constraints, get_rollout_indexes, sample_batch, featurize_states
from utils.soft_q_learning import tabsoftq_gen_pol, tabsoftq_learn_Qs, vectorize_rollouts, generate_demonstrations


def clean_demos(sas_obs, max_noops=15):
    demo_example_idxes = list(range(len(sas_obs)))
    unique_stays, stay_count = set([]), 0
    for i,sas in enumerate(sas_obs):
        sas = tuple(sas)
        if sas[1] == 4:
            stay_count += 1
            if sas in unique_stays:
                if stay_count > max_noops:
                    demo_example_idxes.remove(i)
            else:
                unique_stays.add(sas)
        else:
            stay_count = 0
    return demo_example_idxes


def get_demos(mdp, gamma, temp_boltz_beta, n_demos, demo_time_steps):
    exQs = tabsoftq_learn_Qs(mdp, gamma=gamma)
    # The rationality constant used to generate demos could certainly vary from the one used in the model.....
    # Ensure bad areas have not been visited in exes (this is unique to this experiment)
    demos = []
    while len(demos) < n_demos:
        new_demo = generate_demonstrations(mdp, tabsoftq_gen_pol(exQs, beta=temp_boltz_beta), 1, demo_time_steps)[0]
        more_sas, more_adt = list(zip(*new_demo))
        if len(set([adt[2] for adt in more_adt])) == 1:
            demos += [new_demo]

    sas_obs, adt_obs = vectorize_rollouts(demos)

    good_indexes = clean_demos(sas_obs)
    sas_obs, adt_obs = sas_obs[good_indexes], adt_obs[good_indexes]
    constraints = generate_constraints(mdp)
    nn_rollouts = nn_vectorize_rollouts(mdp, sas_obs, adt_obs)
    train_idxes, val_idxes = get_rollout_indexes(sas_obs)
    val_demo_batch = sample_batch(nn_rollouts, val_idxes)

    # True q-vals for debugging and comparison purposes
    Qs = tabsoftq_learn_Qs(mdp, gamma=gamma)
    sa = np.transpose(
        [np.tile(np.arange(mdp.num_states), mdp.num_actions), np.repeat(np.arange(mdp.num_actions), mdp.num_states)])
    states, acts = sa[:, 0], sa[:, 1]
    true_qs = Qs[states, acts]

    # Preprocessing for training update visualizations
    tts = np.arange(mdp.tile_types)
    acts = np.arange(mdp.num_actions)
    adt_samples = np.transpose([np.tile(tts, len(acts)), np.repeat(acts, len(tts))])
    adt_samples = adt_samples[adt_samples[:, 0].argsort()]
    states = featurize_states(mdp, np.arange(mdp.num_states))

    return constraints, nn_rollouts, train_idxes, val_demo_batch, true_qs, states, adt_samples