
import time
import numpy as np
import cv2
import gym
from gym import spaces

import ptan
from nes_py.wrappers import JoypadSpace
from Contra.actions import COMPLEX_MOVEMENT
# change some terms from ptan wrapper
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True
        self.was_real_reset = False

    def step(self, action):
        #print(action,"action")
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = info['life']
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
            self.was_real_reset = True
            lives = 3
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, info = self.env.step(0)
            self.was_real_reset = False
            lives = info['life']
        self.lives = lives
        return obs

class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


def wrap_dqn(env, stack_frames=4, episodic_life=True, reward_clipping=True):
    if episodic_life:
        env = EpisodicLifeEnv(env)
    env = ptan.common.wrappers.NoopResetEnv(env, noop_max=30)
    env = ptan.common.wrappers.MaxAndSkipEnv(env, skip=4)
    env = ProcessFrame84(env)
    env = ptan.common.wrappers.ImageToPyTorch(env)
    env = ptan.common.wrappers.FrameStack(env, stack_frames)
    if reward_clipping:
        env = ptan.common.wrappers.ClippedRewardsWrapper(env)
    return env


def create_gym(game_name='Contra-v0'):
    env = gym.make(game_name)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_dqn(env)
    return env





