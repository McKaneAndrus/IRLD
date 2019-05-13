import numpy as np
from envs.mars_map_gen import make_map, get_mdp_from_map
from utils.learning_utils import nn_vectorize_rollouts, generate_constraints, get_rollout_indexes, sample_batch, featurize_states
from utils.tab_learning_utils import tabsoftq_gen_pol, tabsoftq_learn_Qs, vectorize_rollouts, generate_demonstrations
import random


def clean_demos(sas_obs, adt_obs, max_noops):
    # print('cleaning demos')
    stay_count_sum = 0
    demo_example_idxes = list(range(len(sas_obs)))
    unique_stays, stay_count = set([]), 0
    for i,(sas,adt) in enumerate(zip(sas_obs,adt_obs)):
        sas = tuple(sas)
        if adt[2] == 1:
            return None
        if sas[1] == 4:
            stay_count += 1
            if sas in unique_stays:
                if max_noops and stay_count > max_noops:
                    demo_example_idxes.remove(i)
            else:
                unique_stays.add(sas)
        else:
          stay_count_sum += stay_count
          stay_count = 0

    # print('successful_yay_add_horiz_stay_count_proportion', stay_count_sum, i, stay_count_sum / i)
    return demo_example_idxes


def get_demos(gamma, temp_boltz_beta, n_demos, demo_time_steps, seed=0, map_seed=0, max_noops=50, tabular=False):
    np.random.seed(map_seed)
    map_state = np.random.get_state()
    np.random.seed(seed)
    random.seed(seed)
    count = 0
    good_indexes = None
    while True:
      random_state = np.random.get_state()
      np.random.set_state(map_state)
      this_map_seed = np.random.randint(100000)

      mdp_map = make_map(15, 15, 10, this_map_seed)
      map_state = np.random.get_state()
      np.random.set_state(random_state)
      mdp = get_mdp_from_map(mdp_map)
      exQs = tabsoftq_learn_Qs(mdp, gamma=gamma)

      # The rationality constant used to generate demos could certainly vary from the one used in the model.....
      # Ensure bad areas have not been visited in exes (this is unique to this experiment)
      demos = []
      while len(demos) < n_demos:
          new_demo = generate_demonstrations(mdp, tabsoftq_gen_pol(exQs, beta=temp_boltz_beta), 1, demo_time_steps)[0]
          demos += [new_demo]

      sas_obs, adt_obs = vectorize_rollouts(demos)

      good_indexes = clean_demos(sas_obs, adt_obs, None)
      count += 1
      if good_indexes:
        # print(count)
        print('good seed', this_map_seed)
    # print('\n'.join(mdp_map))
    return count
