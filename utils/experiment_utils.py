import numpy as np
import time

current_milli_time = lambda: int(round(time.time() * 1000))

def run_simulation(algorithm, vis_history=True, verbose=True, sep_rewards=False, steps=float('inf'),
                   on_belief_update=False, history=None, cancel_updates=False, use_counts=False):
    game = algorithm.game
    vals = hasattr(algorithm, 'values')
    if history is None:
        s, done = game.reset(), False
        a = algorithm.getFirstAction(s)
        if vis_history:
            # Add probabilities?
            history = {'rewards': [game.separate_rewards.copy()],
                       'beliefs': [game.b.copy()],
                       'states': [s],
                       'actions': [a],
                       'reward': [game.true_reward],
                       'policies': [algorithm.policy]}
            if vals:
                history['values'] = [algorithm.values]
            if use_counts:
                history['counts'] = [{'F':0,
                                      'LR':0,
                                      'B':0,
                                      'S':0}]
    else:
        s, done = game.getCurrentState(), False
        a = history['actions'][-1]
    step, alt_done = 0.0, False
    while not done and not alt_done:
        step += 1
        sprime, r, done, p, dic = game.act(a)
        if cancel_updates:
            game.resetBelief()
        if verbose:
            print("On day {}         \r".format(game.time_step),)
            # print("At state {} took action {} to arrive at {} with probability {} for reward {}".format(s, a, sprime, p, r))
        a = algorithm.getNextAction(s, a, sprime)
        s = sprime
        if vis_history:
            history['rewards'] += [game.separate_rewards.copy()]
            history['beliefs'] += [game.b.copy()]
            history['states'] += [s]
            history['actions'] += [a]
            history['reward'] += [game.true_reward]
            history['policies'] += [algorithm.policy]
            if vals:
                history['values'] += [algorithm.values]
            if use_counts:
                curr_counts = history['counts'][-1].copy()
                # Only way there can be two or more is if the action had multiple ways to stall
                if dic['inc'] is not None:
                    if len(dic['inc']) == 1:
                        curr_counts[dic['inc'][0]] += 1
                    else:
                        curr_counts['S'] += 1
                history['counts'] += [curr_counts]
            # if verbose:
            #     print('Total reward {}, beliefs are {}'.format(game.true_reward, game.b))
        if step >= 2:
            alt_done = step > steps or (not np.array_equal(history['beliefs'][-1], history['beliefs'][-2]) and on_belief_update)

    end_reward = game.separate_rewards if sep_rewards else game.discounted_reward
    returns = [end_reward]
    if vis_history:
        returns += [history]
    if steps != float('inf') or on_belief_update:
        returns += [done]
    return returns


# TODO: need to include value returns for
def test_alg_settings(alg, n, save_times=False):
    n = int(n)
    rewards = np.zeros(n)
    if save_times:
        times = np.zeros(n)
    for i in range(n):
        if save_times:
            start = time.time()
            rewards[i] = run_simulation(alg)
            times[i] = time.time() - start
        else:
            rewards[i] = run_simulation(alg)
    # print("Mean: " + str(rewards.mean()) + " Std: " + str(rewards.std()))
    if save_times:
        return rewards, times
    else:
        return rewards
