import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os
import math


# you can completely modify this class for your MuJoCo environment by following the directions
class DoqQuadrupedEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 200,
    }

    # set default episode_len for truncate episodes
    def __init__(self, episode_len=500, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        # change shape of observation to your observation space size
        observation_space = Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float64)
        ## 29 Observations (15 qpos (7dof body, 8 joints) + 14 qvel (6dof body, 8 joints))

        # load your MJCF model with env and choose frames count between actions
        MujocoEnv.__init__(
            self,
            os.path.abspath("assets/doq.xml"),
            5, ## TODO: update frame count? what is this?
            observation_space=observation_space,
            **kwargs
        )
        self.step_number = 0
        self.episode_len = episode_len

        self.position_goal_weight = 1.0
        self.control_cost_weight = 0.1
        self.posture_cost_weight = 0.08
        self.velocity_cost_weight = 0.001

    def control_cost(self, action):
        # return self.control_cost_weight * np.sum(np.square(self.data.ctrl))
        return self.control_cost_weight * np.sum(np.abs(self.data.ctrl))

    def posture_cost(self):
        return self.posture_cost_weight * np.sum(np.square(self.data.qpos[7:]))


    # determine the reward depending on observation or other properties of the simulation
    def step(self, a):
        # reward = 1.0
        ## Check if the action is valid
        assert len(a) == 8
        # Make sure its not infinite or nan or somehting
        for i in a: 
            assert not math.isnan(i)
            assert not math.isinf(i)
            assert str(type(i)) == "<class 'numpy.float32'>"


        self.do_simulation(a, self.frame_skip)
        self.step_number += 1

        ## Reward the quadruped for traveling in x
        reward = self.position_goal_weight * self.data.qpos[0] \
                - self.control_cost(a) \
                - self.posture_cost() \
                - self.velocity_cost_weight * np.abs(self.data.qvel[7:]).mean()
        # print("\n qpos:", self.data.qpos)
        # print("\n reward:", reward)

        if (math.isnan(reward) or math.isinf(reward) or str(type(reward)) != "<class 'numpy.float64'>"):
            print("\n bad reward:", reward)

        obs = self._get_obs()
        # done = bool(not np.isfinite(obs).all() or (obs[2] < 0))
        done = bool(not np.isfinite(obs).all())
        truncated = self.step_number > self.episode_len
        return obs, reward, done, truncated, {}

    # define what should happen when the model is reset (at the beginning of each episode)
    def reset_model(self):
        self.step_number = 0


        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        ## Initialize qpos as zero
        qpos = np.zeros(self.model.nq)
        ## Make sure it starts above ground
        qpos[2] = 0.2
        qpos[3:7] = [1, 0, 0, 0]



        ## TODO: still not absolutely sure what init_qpos does
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )

        qvel = np.zeros(self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    # determine what should be added to the observation
    # for example, the velocities and positions of various joints can be obtained through their names, as stated here
    def _get_obs(self):
        # obs = np.concatenate((np.array(self.data.joint("ball").qpos[:3]),
        #                       np.array(self.data.joint("ball").qvel[:3]),
        #                       np.array(self.data.joint("rotate_x").qpos),
        #                       np.array(self.data.joint("rotate_x").qvel),
        #                       np.array(self.data.joint("rotate_y").qpos),
        #                       np.array(self.data.joint("rotate_y").qvel)), axis=0)
        # print("\n old obs.shape:", obs.shape)
        # print("\n old obs:", obs)


        obs = np.concatenate((self.data.qpos.flat, self.data.qvel.flat), axis=0)

        # print("\n new obs.shape:", obs.shape)
        # print("\n new obs:", obs)

        ## I want observation for quadruped to be 6dof body origin position and 8 dof joint angles
        return obs
