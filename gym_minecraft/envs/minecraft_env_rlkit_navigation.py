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

class MinecraftEnvRLKitNavigation(MinecraftEnvRLKitBase):
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
        self.obs_shape = (self.y+6, self.x, self.z)

        self.voxel_dim = np.prod(self.voxel_shape)
        self.obs_dim = np.prod(self.obs_shape)

        self.agent_pos_space = Box(low=-10, high=10, shape=(2,), dtype=np.float32)
        self.obs_space = Box(low=0, high=10, shape=(self.obs_dim, ), dtype=np.float32)
        self.goal_space = Box(low=0, high=10, shape=(self.voxel_dim, ), dtype=np.float32)
        self.achieved_goal_space = Box(low=0, high=10, shape=(self.voxel_dim, ), dtype=np.float32)

        self._state_goal = np.zeros((self.y, self.z, self.x))
        for y in range(1):
            self._state_goal[y, 0, :] = BLOCK_KEY['brick_block']
            self._state_goal[y, -1, :] = BLOCK_KEY['brick_block']
            self._state_goal[y, :, 0] = BLOCK_KEY['brick_block']
            #self._state_goal[y, 3, -1] = BLOCK_KEY['brick_block']
            self._state_goal[y, :, -1] = BLOCK_KEY['brick_block']
        self._state_goal = self._state_goal.flatten()


        self.last_obs = None

        self.fix_goal = True

        self.init_state = None

    def init(self, client_pool=None, start_minecraft=None,
             continuous_discrete=True, add_noop_command=None,
             max_retries=90, retry_sleep=10, step_sleep=0.001, skip_steps=0,
             videoResolution=None, videoWithDepth=None,
             observeRecentCommands=None, observeHotBar=None,
             observeFullInventory=None, observeGrid=None,
             observeDistance=None, observeChat=None,
             allowContinuousMovement=None, allowDiscreteMovement=None,
             allowAbsoluteMovement=None, recordDestination=None,
             recordObservations=None, recordRewards=None,
             recordCommands=None, recordMP4=None,
             gameMode=None, forceWorldReset=None, image_obs=False):

        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.step_sleep = step_sleep
        self.skip_steps = skip_steps
        self.forceWorldReset = forceWorldReset
        self.continuous_discrete = continuous_discrete
        self.add_noop_command = add_noop_command
        self.image_obs = image_obs

        if videoResolution:
            if videoWithDepth:
                self.mission_spec.requestVideoWithDepth(*videoResolution)
            else:
                self.mission_spec.requestVideo(*videoResolution)

        if observeRecentCommands:
            self.mission_spec.observeRecentCommands()
        if observeHotBar:
            self.mission_spec.observeHotBar()
        if observeFullInventory:
            self.mission_spec.observeFullInventory()
        if observeGrid:
            self.mission_spec.observeGrid(*(observeGrid + ["grid"]))
        if observeDistance:
            self.mission_spec.observeDistance(*(observeDistance + ["dist"]))
        if observeChat:
            self.mission_spec.observeChat()

        if allowContinuousMovement or allowDiscreteMovement or allowAbsoluteMovement:
            # if there are any parameters, remove current command handlers first
            self.mission_spec.removeAllCommandHandlers()

            if allowContinuousMovement is True:
                self.mission_spec.allowAllContinuousMovementCommands()
            elif isinstance(allowContinuousMovement, list):
                for cmd in allowContinuousMovement:
                    self.mission_spec.allowContinuousMovementCommand(cmd)

            if allowDiscreteMovement is True:
                self.mission_spec.allowAllDiscreteMovementCommands()
            elif isinstance(allowDiscreteMovement, list):
                for cmd in allowDiscreteMovement:
                    self.mission_spec.allowDiscreteMovementCommand(cmd)

            if allowAbsoluteMovement is True:
                self.mission_spec.allowAllAbsoluteMovementCommands()
            elif isinstance(allowAbsoluteMovement, list):
                for cmd in allowAbsoluteMovement:
                    self.mission_spec.allowAbsoluteMovementCommand(cmd)

        if start_minecraft:
            # start Minecraft process assigning port dynamically
            self.mc_process, port = minecraft_py.start()
            logger.info("Started Minecraft on port %d, overriding client_pool.", port)
            client_pool = [('127.0.0.1', port)]

        if client_pool:
            if not isinstance(client_pool, list):
                raise ValueError("client_pool must be list of tuples of (IP-address, port)")
            self.client_pool = MalmoPython.ClientPool()
            for client in client_pool:
                self.client_pool.add(MalmoPython.ClientInfo(*client))

        # TODO: produce observation space dynamically based on requested features

        self.video_height = self.mission_spec.getVideoHeight(0)
        self.video_width = self.mission_spec.getVideoWidth(0)
        self.video_depth = self.mission_spec.getVideoChannels(0)
        #self.observation_space = spaces.Box(low=0, high=255,
        #        shape=(self.video_height, self.video_width, self.video_depth))
        self.observation_space = self.obs_space
        # self.observation_space = Dict([
        #     ('observation', self.obs_space),
        #     ('desired_goal', self.goal_space),
        #     ('achieved_goal', self.achieved_goal_space),
        #     ('state_observation', self.obs_space),
        #     ('state_desired_goal', self.goal_space),
        #     ('state_achieved_goal', self.achieved_goal_space),
        #     ('agent_pos', self.agent_pos_space),
        # ])
        # dummy image just for the first observation
        self.last_image = np.zeros((self.video_height, self.video_width, self.video_depth), dtype=np.uint8)

        self._create_action_space()

        # mission recording
        self.mission_record_spec = MalmoPython.MissionRecordSpec()  # record nothing
        if recordDestination:
            self.mission_record_spec.setDestination(recordDestination)
        if recordRewards:
            self.mission_record_spec.recordRewards()
        if recordCommands:
            self.mission_record_spec.recordCommands()
        if recordMP4:
            self.mission_record_spec.recordMP4(*recordMP4)

        if gameMode:
            if gameMode == "spectator":
                self.mission_spec.setModeToSpectator()
            elif gameMode == "creative":
                self.mission_spec.setModeToCreative()
            elif gameMode == "survival":
                logger.warn("Cannot force survival mode, assuming it is the default.")
            else:
                assert False, "Unknown game mode: " + gameMode

        #self._start_mission()

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
        full_obs = np.concatenate([state_obs, np.zeros((6, self.z, self.x))])
        x_idx = int(agent_pos[0] - self.x_lim[0])
        z_idx = int(agent_pos[2] - self.z_lim[0])
        full_obs[yaw_to_idx[agent['yaw']] + 2, z_idx, x_idx] = 1 #yaw_to_idx[agent['yaw']]
        self.agent_pos = np.array([agent_pos[0], agent_pos[2]])
        state_obs = state_obs.flatten()
        full_obs = full_obs.flatten()

        return full_obs

    def compute_rewards(self, actions, obs):

        r = -np.linalg.norm(self.agent_pos - np.array([3.5, 3.5]), axis=-1)
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
    env = gym.make('MinecraftWallNavigation-v0')
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
            key_to_action = dict(e=0, w=1, s=2, a=4, d=3)
            #key_to_action = dict(a=1, d=0, w=2)
            action = key_to_action[action]
            obs, reward, done, info = env.step(action)
            steps += 1

            # print("done: " + str(done))
            #print("obs: " + str(obs))
            # print(obs['state_observation'][:env.full_dim].reshape((env.y, env.x, env.z)))
            print(obs.reshape(env.obs_shape))
            #print(obs['state_observation'][env.full_dim+4:].reshape((env.y, 3)))
            #print(obs['state_observation'][env.full_dim:env.full_dim+4])

            print("reward: " + str(reward))

    env.reset()
    env.step(0)