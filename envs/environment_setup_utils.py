import numpy as np
from envs.marsexplorer import MarsExplorerEnv

tile_map = [
        "2FFFUFFF1",
        "FFFUUUFFF",
        "FFFFUFFFF",
        "FSFUUUFSF",
        "FSUU3UUSF",
        "FSFUUUFSF",
        "FFFFUFFFF",
        "FFFSSSFFF",
        "1FFFFFFF2"
    ]

tile_map2 = [
        "FFFFFFFF1",
        "FFFFFFFFF",
        "FFFFUFFFF",
        "FFFUUUFSF",
        "FFUU3UUFF",
        "FSFUUUFSF",
        "FSFFUFFSF",
        "FFFSSSFFF",
        "1FFFFFFF2"
    ]

tile_map3 = [
        "FFFFUFFFF",
        "FFFFFFFFF",
        "FUUFFFFFF",
        "FFFFFFFFF",
        "FFFFFFUFF",
        "FFFFFFUFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "SFFFFFFF3",
]

TILE_MAPS = {0: tile_map,
             1: tile_map2,
             2: tile_map3}

tile_rewards = {'F':0.05,
              '1':0.1,
              '2':0.1,
              '3':0.1,
              'S':0.01,
              'U':0.0}


goal_tile_rewards = {'F':0.0,
              '1':1.0,
              '2':2.0,
              '3':8.0,
              'S':0.0,
              'U':0.0}

t0 = (0.6,0.2,0.0,0.0)
t1 = (0.001,0.0,0.0,0.999)  #(0.1,0.15,0.5,0.1)

trans_dict = {b'F':t0,
              b'1':t0,
              b'2':t0,
              b'3':t0,
              b'S':t0,
              b'U':t1}


gamma = 0.95

T_theta_shape = (5,5,2)

time_penalty = 0.0

# Enables "texturizing" the environment such that certain areas
# will have greater reward despite being the same tile types
tile_reward_modifier = lambda r,x,y,mx,my: r #* 0.1 * ((x-(mx/2 + np.random.normal(scale=0.5)))**2 + (y - (mx/2 + np.random.normal(scale=0.5)))**2)


def build_reward_map(tile_map, tile_rewards, goal_tile_rewards, tile_reward_modifier):
    reward_map = np.zeros((len(tile_map),len(tile_map[0])))
    texture_map = np.zeros((len(tile_map),len(tile_map[0])))
    for y,row in enumerate(tile_map):
        for x,c in enumerate(row):
            reward_map[y,x] = texture_map[y,x] = tile_reward_modifier(tile_rewards[c],x,y,len(tile_map[0]),len(tile_map))
            reward_map[y,x] += goal_tile_rewards[c]
    return reward_map, texture_map


def get_tile_map(tile_map_index):
    return TILE_MAPS[tile_map_index]


def get_mdp(tile_map_index):
    reward_map, texture_map = build_reward_map(TILE_MAPS[tile_map_index], tile_rewards,
                                               goal_tile_rewards, tile_reward_modifier)
    return MarsExplorerEnv(tile_map, reward_map, texture_map, trans_dict, time_penalty)

def get_mdp_from_map(tile_map):
    reward_map, texture_map = build_reward_map(tile_map, tile_rewards,
                                               goal_tile_rewards, tile_reward_modifier)
    return MarsExplorerEnv(tile_map, reward_map, texture_map, trans_dict, time_penalty)
