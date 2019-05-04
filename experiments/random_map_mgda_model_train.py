from sacred import Experiment
from sacred.observers import FileStorageObserver
from experiments.mgda_model_train import mgda_model_train_ex
from envs.mars_map_gen import make_map

random_map_mgda_model_train_ex = Experiment("random_map_mgda_model_train", ingredients=[mgda_model_train_ex])
random_map_mgda_model_train_ex.observers.append(FileStorageObserver.create('logs/sacred'))


@random_map_mgda_model_train_ex.config
def config():
    samples = 5
    seed = 42
    map_height = 15
    map_width = 15
    cluserting_iterations = 10


@random_map_mgda_model_train_ex.config_hook
def hook(config, command_name, logger):
    global mgda_model_train_config
    mgda_model_train_config = config['mgda_model_train']
    return config


def update(map, key, val):
    map[key] = val
    return map


@random_map_mgda_model_train_ex.automain
def main(seed, samples, map_height, map_width, cluserting_iterations):

    return [mgda_model_train_ex.run(config_updates=update(mgda_model_train_config, 'mdp_map',
                                                          make_map(map_height, map_width, cluserting_iterations,
                                                                   seed+i)))._id for i in range(samples)]

