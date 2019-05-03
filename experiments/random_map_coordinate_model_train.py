from sacred import Experiment
from sacred.observers import FileStorageObserver
from experiments.coordinate_model_train import coordinate_model_train_ex, coordinate_train
from envs.mars_map_gen import make_map

random_map_coordinate_model_train_ex = Experiment("random_map_coordinate_model_train", ingredients=[coordinate_model_train_ex])
random_map_coordinate_model_train_ex.observers.append(FileStorageObserver.create('logs/sacred'))


@random_map_coordinate_model_train_ex.config
def config():
    samples = 3
    seed = 42
    map_height = 15
    map_width = 15
    cluserting_iterations = 10


@random_map_coordinate_model_train_ex.config_hook
def hook(config, command_name, logger):
    global coordinate_model_train_config
    coordinate_model_train_config = config['coordinate_model_train']
    return config


def update(map, key, val):
    map[key] = val
    return map


@random_map_coordinate_model_train_ex.automain
def main(seed, samples, map_height, map_width, cluserting_iterations):

    return [coordinate_model_train_ex.run(config_updates=update(coordinate_model_train_config, 'mdp_map',
                                                                make_map(map_height, map_width, cluserting_iterations,
                                                                         seed+i)))._id for i in range(samples)]

