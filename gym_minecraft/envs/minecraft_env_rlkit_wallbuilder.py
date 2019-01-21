import logging
import time
import os

import numpy as np
import json
import xml.etree.ElementTree as ET
import gym
from gym import spaces, error
from gym.spaces import Dict, Box
from rlkit.envs.minecraft.commands import CommandParser
from lxml import etree
from pathlib import Path
from gym_minecraft.envs.minecraft_env_rlkit_base import MinecraftEnvRLKitBase

try:
    import minecraft_py
    import MalmoPython
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: install minecraft_py from https://github.com/tambetm/minecraft-py".format(e))

logger = logging.getLogger(__name__)

SINGLE_DIRECTION_DISCRETE_MOVEMENTS = [ "jumpeast", "jumpnorth", "jumpsouth", "jumpwest",
                                        "movenorth", "moveeast", "movesouth", "movewest",
                                        "jumpuse", "use", "attack", "jump" ]

MULTIPLE_DIRECTION_DISCRETE_MOVEMENTS = [ "move", "turn", "look", "strafe",
                                          "jumpmove", "jumpstrafe" ]

BLOCK_KEY = dict(air=0, brick_block=1, grass=2, dirt=3, clay=4, spruce_fence=2)

class MinecraftEnvRLKitWallBuilder(MinecraftEnvRLKitBase):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, mission_file):
        super(MinecraftEnvRLKitBase, self).__init__()

        self.agent_host = MalmoPython.AgentHost()
        assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets')
        mission_file = os.path.join(assets_dir, mission_file)

        xml = Path(mission_file).read_text()
        if not xml.startswith('<Mission'):
            i = xml.index("<Mission")
            xml = xml[i:]
        self.ns = '{http://ProjectMalmo.microsoft.com}'
        self.xml = etree.fromstring(xml)

        self.load_mission_file(mission_file)

        self.client_pool = None
        self.mc_process = None
        self.screen = None
        self.image_obs = None

        obs_from_grid = self.xml.findall('.//' + self.ns + 'ObservationFromGrid')[0].find(self.ns+'Grid')
        low = dict(obs_from_grid[0].attrib)
        high = dict(obs_from_grid[1].attrib)
        self.x_lim = (int(low['x']), int(high['x']))
        self.y_lim = (int(low['y']), int(high['y']))
        self.z_lim = (int(low['z']), int(high['z']))

        self.x = int(high['x']) - int(low['x']) + 1
        self.y = int(float(high['y']) - float(low['y']) + 1)
        self.z = int(high['z']) - int(low['z']) + 1
        self.voxel_shape = (self.y, self.x, self.z)
        self.obs_shape = (self.y+4, self.x, self.z)

        self.voxel_dim = np.prod(self.voxel_shape)
        self.obs_dim = np.prod(self.obs_shape)

        self.agent_pos_space = Box(low=-10, high=10, shape=(2,), dtype=np.float32)
        self.obs_space = Box(low=0, high=10, shape=(self.obs_dim, ), dtype=np.float32)
        self.goal_space = Box(low=0, high=10, shape=(self.voxel_dim, ), dtype=np.float32)
        self.achieved_goal_space = Box(low=0, high=10, shape=(self.voxel_dim, ), dtype=np.float32)

        self._state_goal = np.zeros((self.y, self.z, self.x))
        for y in range(1):
            #self._state_goal[y, 0, :] = BLOCK_KEY['brick_block']
            #self._state_goal[y, -1, :] = BLOCK_KEY['brick_block']
            #self._state_goal[y, :, 0] = BLOCK_KEY['brick_block']
            #self._state_goal[y, 3, -1] = BLOCK_KEY['brick_block']
            self._state_goal[y, :, -1] = BLOCK_KEY['brick_block']
        self._state_goal = self._state_goal.flatten()


        self.last_obs = None

        self.fix_goal = True

        self.init_state = None

    def _get_obs(self, world_state):
        msg = world_state.observations[-1].text
        state = json.loads(msg)
        state_obs = np.array([BLOCK_KEY[x] for x in state['board']])
        state_obs = state_obs.reshape((self.y, self.z, self.x))

        #self.last_obs = obs
        agent = state['entities'][0]
        agent_pos = np.array([agent['x'], agent['y'], agent['z'],
                              (agent['yaw']/270.0)])

        yaw_to_idx = {0:0, 90:1, 180:2, 270:3}
        full_obs = np.concatenate([state_obs, np.zeros((4, self.z, self.x))])
        x_idx = int(agent_pos[0] - self.x_lim[0])
        z_idx = int(agent_pos[2] - self.z_lim[0])
        full_obs[yaw_to_idx[agent['yaw']] + 2, z_idx, x_idx] = 1 #yaw_to_idx[agent['yaw']]

        state_obs = state_obs.flatten()
        full_obs = full_obs.flatten()
        self.agent_pos = np.array([agent_pos[0], agent_pos[2]])

        return dict(
            observation=full_obs,
            desired_goal=self._state_goal,
            achieved_goal=state_obs,
            state_desired_goal=self._state_goal,
            state_observation=full_obs,
            state_achieved_goal=state_obs,
            agent_pos=np.array([agent_pos[0], agent_pos[2]])
        )

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['achieved_goal']
        desired_goals = obs['desired_goal']
        r = np.sum(achieved_goals == desired_goals, axis=-1) - np.sum(desired_goals == 0, axis=-1)
        #r = -np.linalg.norm(obs['agent_pos'] - np.array([3.5, 3.5]), axis=-1)

        #r = np.sum(np.bitwise_and(achieved_goals == desired_goals, desired_goals != 0), -1)
        #distances = np.linalg.norm(achieved_goals - desired_goals, axis=1)
        return r


    def sample_goals(self, batch_size):

        if self.fix_goal:
            goals = np.repeat(np.expand_dims(self._state_goal, 0), batch_size, 0)
        else:
            goals = np.random.binomial(1, 0.5, (self.y, batch_size, self.z, self.x))
            floating = (goals[1] - goals[0]) == 1 # remove floating blocks
            goals[1][np.where(floating)] = 0
            goals = np.swapaxes(goals, 0, 1)
            goals = goals.reshape((batch_size, -1))

        return dict(
            desired_goal=goals,
            state_desired_goal=goals,
        )

if __name__ == '__main__':
    import gym_minecraft
    env = gym.make('MinecraftWallBuilder-v0')
    env.init(start_minecraft=False)
    #import pdb; pdb.set_trace()
    for i in range(3):
        print("reset " + str(i))
        obs = env.reset()
        steps = 0
        done = False
        while not done and (steps < 1000):

            #action = env.action_space.sample()
            #import pdb; pdb.set_trace()
            print(env.action_names)
            action = input('Enter action: ')
            key_to_action = dict(w=1, s=2, a=0, d=3)
            #key_to_action = dict(e=0, w=1, s=2, a=4, d=3)
            #key_to_action = dict(a=1, d=0, w=2)
            action = key_to_action[action]
            obs, reward, done, info = env.step(action)
            steps += 1

            # print("done: " + str(done))
            #print("obs: " + str(obs))
            # print(obs['state_observation'][:env.full_dim].reshape((env.y, env.x, env.z)))
            print(obs['state_observation'].reshape(env.obs_shape))
            #print(obs['state_observation'][env.full_dim+4:].reshape((env.y, 3)))
            #print(obs['state_observation'][env.full_dim:env.full_dim+4])

            print("reward: " + str(reward))

    env.reset()
    env.step(0)