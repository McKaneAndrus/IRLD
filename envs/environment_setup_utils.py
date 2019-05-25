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
        "UUUUUUUUU",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "2FFFFFFF1",
        "FFFFSFFFF",
        "FFFFSFFFF",
        "1FFFFFFF2",
        "FFFFFFFFF",
        "FFFFFFFFF"
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

tile_map4 = [
  "FFSUSFU3UF2FUFF",
  "FFFF3FUUUFFSFSU",
  "SFF2FFFFFFFFFFF",
  "FFUFFFF1SUFSFUF",
  "FSSFU2UFSFFFFFF",
  "FUFFF1USFFSSSFF",
  "FFFFFFFFFFFSFFF",
  "UF1FFUUFFFFFFSF",
  "FFFFFU1FUFFFFFF",
  "FFF22FFF1FFF3FF",
  "FF1FFFFF1SUUFFU",
  "FSUFFFFFFFUUFFF",
  "U2FFFFFUS2FFFFF",
  "FFFUSFFUFFFFFUF",
  "FFSUFFSFFSFFFFF",
]

tile_map5 = [
  "FUFFFFUFFFFFFUS",
  "FFFSUFFFFF1F2UF",
  "SFSFFSFFFSFSFFF",
  "S2FFFFFF2SFF1FF",
  "FFFFFSF2USS1FUF",
  "21FFFFUSFFFUFF3",
  "FUFF1FUFSFFF1FF",
  "FFFUUFFFUFFFUFF",
  "UFFFFFSFFFFFUFU",
  "FFUFFUUUSFFF3SF",
  "FFFFFU2F3FFFFFF",
  "F1FFFFSFFSUFFFF",
  "UFFSFFFFUSFFFFU",
  "FFFFFFSS2FFFFFF",
  "FFFFFFFFUFFUFFF",
]

tile_map6 = [
  "FUSSFFFF2FF3FFF",
  "SFF2SFUUSFFF3UF",
  "UFFUF1FFSFSUSFF",
  "FF2FFFFFUSFS1FS",
  "FFFFFFFFFFFFFFF",
  "UFFUUFFFFF2FFFF",
  "2FFFFFFFFFFFFUF",
  "FFFFFFFUFFFFF2F",
  "UF3U1SFSFFFFF1S",
  "FFUUUFFFFFFFFUF",
  "FFSFFFFUFFFSFUF",
  "FFFFFFFFFFFFFUF",
  "F1FFFF2FFSFUFFF",
  "SFSSUUFFSS1FFFU",
  "FFFFFFFFFUUF1SF",
]

tile_map7 = [
  "U1FFFFFFFFFFFSF",
  "FFFFFFFFUFFFFFF",
  "2FFFFFU1U122FFU",
  "FUUFSFUU2UFFSFS",
  "FFFFSFFFFFFFFSF",
  "FFSF1UFFFFFFFFF",
  "FFUSFSFFSF2F2FS",
  "FFSFUFFFFFFFU1U",
  "FSFFFFFFU3FFFFF",
  "FSFUFUFFFF1FFFF",
  "F1FFFUFFFFFUFSF",
  "FUUFF2FFFFFFFUF",
  "FSFFSFUFFFSFSUF",
  "FFFFF3FSFFF3SSF",
  "FFFFFFUFFUFFUFS",
]

tile_map8 = [
  "USFFFSFFFFFSFFU",
  "FFUSFFFFFFFFFFF",
  "FFUFFFFFFFSUUUF",
  "FFUFFFUS1FS1FFS",
  "FSFFSSUFFUUFFFU",
  "FFFF2FFFFS2FFFU",
  "FFFFU2FFFFFFUFF",
  "FFFFFFFFFF31FFF",
  "USFFFFFFFFFFFFF",
  "FFFUUUFFU2S1FFS",
  "FFSFFFU12F2SSFS",
  "FFFF1FFFFFSFFFU",
  "FFFFFFFFUFFFU1F",
  "FSFF2FFFUFFFUUF",
  "FFFUFFFSU3USF3F",
]

tile_map9 = [
  "S2UUFFSFFFFFSFF",
  "FFFFFU1FFFUFFFF",
  "FFFFFFFFFFFFUUF",
  "FFFUUF23FF2UUUF",
  "FUFFFS2SFFU1FFF",
  "FSFFFFFFFF1UFFS",
  "FFSU1FFFSFFFFUF",
  "FFFUFFFUFSFFFFF",
  "UFF22SFUSFFFUFS",
  "FSFFUFFFFFFFFFS",
  "FF1SFS2FFFUFFFF",
  "FFUUSFUFFFFFFFU",
  "FFUUFSF1FFFFUFF",
  "F3FFF13UUFSFSFF",
  "FFFFFFSFSFFFFFF",
]

tile_map10 = [
  "FSFFFU2FF3FFFU1",
  "UFFFFU1UF2FSFFF",
  "FFFFFFFFFFF1FFF",
  "UUFFUUFFUFFFSFF",
  "F1FSFFFFFFFF2FF",
  "FFFSFFF2FU3UFFU",
  "FFFSFFFUFFFFFUU",
  "FUUUUFFSFFSFFFF",
  "F2FSSF1FF13FFFF",
  "UFFFFFUFFFFFSFF",
  "FSSFF2SFUUFFUFF",
  "FFFFFSFFSFFFUFF",
  "FFSFFFFUFFFFFSS",
  "FUFFFUFFFFFFFF2",
  "FFSFFUUFSSFU1SF",
]

tile_map11 = [
  "FSU2FFFFFFFF2FF",
  "FFFFFFFUF1UFFUF",
  "FSFFSFFFFSU1FFF",
  "FFFSFFUFFFFFFFF",
  "UFFFFFFFFFF1UFS",
  "FFUFSFFF1UUFFFF",
  "FSF2FFFSSFFSFFU",
  "2SFFUFFFFFFFFF1",
  "FFFFSFFSFFSSFFF",
  "UUFFSUFFFFFSFFU",
  "FFFSFF2UFFFF2FF",
  "FFFFFFFFFFUFFFF",
  "F2SFU1SFFFFFFUF",
  "SFFF3UUUUU3FFFF",
  "UFF1FFFFFU3FSFU",
]

tile_map12 = [
  "FFFUF2FFFSFFUF1",
  "SFFSFSFS32FFSFS",
  "SFF1FFFFFSSFFFF",
  "FFF2FUFF3UFFFFF",
  "UFFFSFFFFUFFUFF",
  "FF3SSFFFFFF2SF1",
  "FFFFUFFFFFFFSFF",
  "FFFFFFFFFUUUFFF",
  "UUUUFFUFFFFFFFF",
  "FFFFFFFFFF2FFFF",
  "FFFFFUSFFFFUUUF",
  "UFFUFF1FFF2SSF1",
  "2FSFFFFFSUFFFFF",
  "FUFFFFFSF1FFFFF",
  "FFFFSUU1UFFSFFF",
]

tile_map13 = [
  "SFFFUFFFSFUUFFU",
  "F1F2FSF2SFFFFFF",
  "FFFUFF1FFF2FUUF",
  "UUFFFFSFFFUSFFF",
  "FFSFFSFUFFUSFFF",
  "SFU2UUFFFFUFFFU",
  "FSFFFFSFF2FF1FU",
  "FFFFFFSUFSFFFFF",
  "FFUFFSFFFFFUFFS",
  "2FFFFSFFF1SFFUF",
  "1UFU3UFFFUFFFFF",
  "F3FUSFFFFFFFFFF",
  "F1FFFFFFUFFF3UF",
  "F1FSSFFFFUSFUUF",
  "FFFFFFFUFFFSF2F",
]

TILE_MAPS = {0: tile_map,
             1: tile_map2,
             2: tile_map3,
             3: tile_map4,
             4: tile_map5,
             5: tile_map6,
             6: tile_map7,
             7: tile_map8,
             8: tile_map9,
             9: tile_map10,
             10: tile_map11,
             11: tile_map12,
             12: tile_map13,
             }

tile_rewards = {'F':0.0,
              '1':0.0,
              '2':0.0,
              '3':0.0,
              'S':0.0,
              'U':0.0}


goal_tile_rewards = {'F':0.0,
              '1':0.125,
              '2':0.25,
              '3':1.0,
              'S':0.0,
              'U':0.0}

t0 = (0.6,0.2,0.0,0.0)
t1 = (0.0,0.0,0.0,1.0)  #(0.1,0.15,0.5,0.1)

trans_dict = {b'F':t0,
              b'1':t0,
              b'2':t0,
              b'3':t0,
              b'S':t0,
              b'U':t1}

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

def get_mdp_from_map(tile_map, trans_dict=trans_dict):
    reward_map, texture_map = build_reward_map(tile_map, tile_rewards,
                                               goal_tile_rewards, tile_reward_modifier)
    return MarsExplorerEnv(tile_map, reward_map, texture_map, trans_dict, time_penalty)
