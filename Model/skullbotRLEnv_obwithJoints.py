'''
Author: FANG Zhiwei
Email: zwfang@link.cuhk.edu.hk/ afjsliny@gmail.com
Date: 2024-03-18 16:20:57
LastEditTime: 2024-04-22 15:56:25
Description: 
'''
import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium import spaces, logger
import time

import sys
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from skullbotSimModel import skullbotSimModel

sys.path.append('../utils')
from imageProc import visionSensorImage2Canny, minEdgeDist, secondEdgeDist

class skullbotEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, action_type='continuous'):
        super(skullbotEnv, self).__init__()
        
        self.action_type = action_type
        # self.push_force = 0
        self.joint_advancement = [0.0, 0.000, 0.0, 0.000, 0.000]
        '''the obeservation/ state should be the joints and the image'''
        # [rotor1, slider1, rotor2, slider2, needle driver] && sim_OCT_vision_sensor and minimum distance
        q1 = np.array([0.0, 0.000, 0.0, 0.000, 0.000], dtype=np.float32)
        q2 = np.zeros((256,256), dtype=np.float32)
        q2 = q2.reshape(-1)
        q3 = np.array([128], dtype = np.float32)

        self.q = np.concatenate((q1, q2, q3), axis=0)
        self.q_last = np.concatenate((q1, q2, q3), axis=0)

        self.done_frame = 0


        '''The following parameters are for the cart-pole environment '''
        # self.theta_max = 10 * np.pi / 360
        # self.cart_pos_max = 0.2

        '''The following parameters are for the skullbot environment observation space'''
        self.rotor_pos_max = np.pi * 90 / 360       # rotor range
        self.slider_pos_max = 20e-3    # slider range
        self.needle_driver_max = 20e-3
        # self.dist_max = np.round(np.sqrt(128**2*2), 2)
        self.dist_max = 1e6

        box1 = np.array(
            [
                self.rotor_pos_max,
                self.slider_pos_max,
                self.rotor_pos_max,
                self.slider_pos_max,
            ],
            dtype=np.float32,
        ) 
        high_needleDriver = np.array([20e-3], dtype=np.float32)
        low_needleDriver = np.array([0], dtype=np.float32)

        high_imagePixel = np.ones((256, 256), dtype=np.float32) * 255
        high_imagePixel = high_imagePixel.reshape(-1)
        low_imagePixel = np.ones((256, 256), dtype=np.float32) * 0
        low_imagePixel = low_imagePixel.reshape(-1)

        high_edgeDist = np.array([self.dist_max], dtype=np.float32)
        low_edgeDist = np.array([0], dtype=np.float32)
        highBox = np.concatenate((box1, high_needleDriver, high_imagePixel, high_edgeDist), axis=0)
        lowBox = np.concatenate((-box1, low_needleDriver, low_imagePixel, low_edgeDist), axis=0)

        client = RemoteAPIClient()
        self.sim = client.require('sim')
        self.sim.setStepping(True) # false when coding, true when trainning

        if self.action_type == 'discrete':
            self.action_space = spaces.Discrete(5)
        elif self.action_type == 'continuous':
            self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        else:
            assert 0, "The action type \'" + self.action_type + "\' can not be recognized"

        self.observation_space = spaces.Box(low=lowBox, high=highBox, dtype=np.float32)

        self.seed()
        self.state1= None
        state1 = self.np_random.uniform(low=-0.05e-3, high=0.05e-3, size=(5,)) # rotor1, slider1, rotor2, slider2, needle driver
        state3 = self.np_random.uniform(low=0, high=1, size=(256, 256)) # image
        state3 = state3.reshape(-1)
        state4 = self.np_random.uniform(low=0, high=1, size=(1)) # edge distance
        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,)) origin statment in cart_pole
        self.state = np.concatenate((state1, state3, state4), axis=0)
        self.counts = 0
        self.steps_beyond_done = None

        self.sim.startSimulation()

        self.skullbot_sim_model = skullbotSimModel()
        self.skullbot_sim_model.initializeSimModel(self.sim)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if self.action_type == 'discrete':
            assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        q1 = [0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        q1[0] = self.skullbot_sim_model.getJointPosition('Rotor1_joint')
        q1[1] = self.skullbot_sim_model.getJointPosition('Slider1_joint')
        q1[2] = self.skullbot_sim_model.getJointPosition('Rotor2_joint')
        q1[3] = self.skullbot_sim_model.getJointPosition('Slider2_joint')
        q1[4] = self.skullbot_sim_model.getJointPosition('needle_driver_joint')
        q1 = np.array(q1, dtype=np.float32)
        q1 = np.round(q1, 5)

        image, resX, resY = self.skullbot_sim_model.getVisionSensorCharImage('sim_OCT_vision_sensor')
        cannyImg = visionSensorImage2Canny(image, resX, resY)
        _, minDist, _ = minEdgeDist(cannyImg, resX, resY)
        _, edgeDist = secondEdgeDist(cannyImg, resX, resY, minDist)
        holdingDist = 2.5*minDist
        # print(minDist, edgeDist, holdingDist)
        cannyImg = cannyImg.reshape(-1)
        q2 = cannyImg
        q3 = np.array([edgeDist], dtype=np.float32)

        q = np.concatenate((q1, q2, q3), axis=0)
        self.q_last = self.q
        self.q = q

        '''action space'''
        if self.action_type == 'discrete':
            if action == 0:
                self.push_force = 0
            elif action == 1:
                self.push_force = 1.0
            elif action == 2:
                self.push_force = -1.0

        elif self.action_type == 'continuous':
            # self.push_force = action[0] * 2.0 # The action is in [-1.0, 1.0], therefore the force is in [-2.0, 2.0]
            self.joint_advancement[0] = action[0] * 9.0 / 180 * np.pi / 10
            self.joint_advancement[1] = action[1] * 2.0e-3 / 10
            self.joint_advancement[2] = action[2] * 9.0 / 180 * np.pi / 10
            self.joint_advancement[3] = action[3] * 2.0e-3 / 10
            self.joint_advancement[4] = action[4] * 2.0e-3 / 10
        else:
            assert 0, "The action type \'" + self.action_type + "\' can not be recognized"

        # set action
        # self.skullbot_sim_model.setJointTorque(self.push_force)
        self.skullbot_sim_model.setJointPosition('Rotor1_joint', q[0] + self.joint_advancement[0])
        self.skullbot_sim_model.setJointPosition('Slider1_joint', q[1] + self.joint_advancement[1])
        self.skullbot_sim_model.setJointPosition('Rotor2_joint', q[2] + self.joint_advancement[2])
        self.skullbot_sim_model.setJointPosition('Slider2_joint', q[3] + self.joint_advancement[3])
        self.skullbot_sim_model.setJointPosition('needle_driver_joint', q[4] + self.joint_advancement[4])
        
        # done = (q[0] <= -self.cart_pos_max) or (q[0] >= self.cart_pos_max) or (q[1] < -self.theta_max) or (q[1] > self.theta_max) #or (self.counts >= 3000)
        
        step_done = (edgeDist <= minDist*1.1) or (edgeDist >= holdingDist*2) or (self.counts >= 3000)
        step_done = bool(step_done)
        if not step_done:
            self.done_frame = 0
        else:
            self.done_frame = self.done_frame + 1

        if self.done_frame >= 3:
            done = True


        '''reward function'''
        if not done:
            if not step_done:
                reward = 1 - abs((edgeDist - holdingDist)/holdingDist)
            else:
                reward = 0.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        dt = 0.005

        # self.state = (self.q[0], self.q[1], self.q[2], self.q[3])
        self.state = tuple(q)
        self.counts += 1

        self.sim.step()
        # print(reward, step_done, done)
        return np.array(self.state, dtype=np.float32), reward, done, False, {}
        # return np.array(q1, dtype=np.float32), reward, done, False, {}
    
    def reset(self, seed=None):
        # print('Reset the environment after {} counts'.format(self.counts))

        self.counts = 0
        self.joint_advancement = [0.0, 0.0, 0.0, 0.0, 0.0]
        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        state1 = self.np_random.uniform(low=-1e-2, high=1e-2, size=(5,)) # rotor1, slider1, rotor2, slider2, needle driver

        self.steps_beyond_done = None

        self.sim.stopSimulation() # stop the simulation
        time.sleep(0.1) # ensure the coppeliasim is stopped
        self.sim.setStepping(True)

        self.skullbot_sim_model.setJointPosition('Rotor1_joint', state1[0]*9 / 180 * np.pi / 10)
        self.skullbot_sim_model.setJointPosition('Slider1_joint', state1[1]*2e-3)
        self.skullbot_sim_model.setJointPosition('Rotor2_joint', state1[2]*9 / 180 * np.pi / 10)
        self.skullbot_sim_model.setJointPosition('Slider2_joint', state1[3]*2e-3)
        self.skullbot_sim_model.setJointPosition('needle_driver_joint', state1[4]*2e-3)
        self.sim.startSimulation()

        state1[0] = self.skullbot_sim_model.getJointPosition('Rotor1_joint')
        state1[1] = self.skullbot_sim_model.getJointPosition('Slider1_joint')
        state1[2] = self.skullbot_sim_model.getJointPosition('Rotor2_joint')
        state1[3] = self.skullbot_sim_model.getJointPosition('Slider2_joint')
        state1[4] = self.skullbot_sim_model.getJointPosition('needle_driver_joint')
        self.state1 = state1

        image, resX, resY = self.skullbot_sim_model.getVisionSensorCharImage('sim_OCT_vision_sensor')
        cannyImg = visionSensorImage2Canny(image, resX, resY)
        _, minDist, _ = minEdgeDist(cannyImg, resX, resY)
        _, edgeDist = secondEdgeDist(cannyImg, resX, resY, minDist)
        holdingDist = 2.5*minDist
        cannyImg = cannyImg.reshape(-1)
        state3 = cannyImg
        state4 = np.array([edgeDist], dtype=np.float32)
        
        self.done_frame = 0
        self.state = np.concatenate((state1, state3, state4), axis=0)

        return np.array(self.state, dtype=np.float32), {}
    
    def zeroingJoints(self):
        self.skullbot_sim_model.setJointPosition('Rotor1_joint', 0)
        self.skullbot_sim_model.setJointPosition('Slider1_joint', 0)
        self.skullbot_sim_model.setJointPosition('Rotor2_joint', 0)
        self.skullbot_sim_model.setJointPosition('Slider2_joint', 0)
        self.skullbot_sim_model.setJointPosition('needle_driver_joint', 0)
        return None
        
    def render(self):
        return None

    def close(self):
        self.sim.stopSimulation() # stop the simulation
        print('Close the environment')
        return None

if __name__ == "__main__":
    env = skullbotEnv()
    env.reset()
    # env.zeroingJoints()
    for _ in range(200):
        action = env.action_space.sample() # random action
        env.step(action)
        print(env.state[:5], env.state[-1])

    env.close()
