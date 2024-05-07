'''
Author: FANG Zhiwei
Email: zwfang@link.cuhk.edu.hk/ afjsliny@gmail.com
Date: 2024-03-18 16:20:57
LastEditTime: 2024-04-23 23:58:37
Description: RL env setup, this version without joints in the observation space
'''
import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium import spaces, logger
import time
import cv2
import matplotlib.pyplot as plt
import random

import sys
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from skullbotSimModel import skullbotSimModel

sys.path.append('../utils')
from imageProc import visionSensorImage2Canny, minEdgeDist, secondEdgeDist, calculateTangetVector

class skullbotEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, action_type='continuous', obs_type='joints_image', model_type='teacher'): #image/ joints_image
        super(skullbotEnv, self).__init__()
        
        self.action_type = action_type
        self.obs_type = obs_type
        self.model_type = model_type
        # self.push_force = 0
        self.joint_advancement = [0.0, 0.000, 0.0, 0.000]
        self.j = np.zeros((6,), dtype=np.float32) #joints

        self.targetImage = np.zeros((256,256), dtype=np.uint8) 
        self.imageLoss1 = 0 # loss of the last frameX and target frame(calculated based on the X)
        self.done_frame = 0
    
        client = RemoteAPIClient()
        self.sim = client.require('sim')
        self.sim.setStepping(True) # false when coding, true when trainning
        self.rotor_pos_max = np.pi * 90 / 360       # rotor range
        self.slider_pos_max = 20e-3    # slider range
        self.needle_driver_max = 20e-3
        jointBox = np.array(
            [
                self.rotor_pos_max,
                self.slider_pos_max,
                self.rotor_pos_max,
                self.slider_pos_max,
                1e6,
                1e6
            ],
            dtype=np.float32,
        ) 

        '''action space'''
        if self.action_type == 'discrete':
            self.action_space = spaces.Discrete(4)
        elif self.action_type == 'continuous':
            self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        else:
            assert 0, "The action type \'" + self.action_type + "\' can not be recognized"

        '''the obeservation/ state should be the joints and the image'''
        # [rotor1, slider1, rotor2, slider2, needle driver] && sim_OCT_vision_sensor and minimum distance
        
        self.q = np.zeros((3,256,256), dtype=np.uint8) # two frames for observation
        if obs_type == 'joints_image':
            print('The observation space is joints and image')
            obs_space = {
                'joints': spaces.Box(low=-jointBox, high=jointBox, dtype=self.j.dtype),
                'image': spaces.Box(low=0, high=255, shape=self.q.shape, dtype=self.q.dtype)
            }
            self.observation_space = spaces.Dict(obs_space)
            self.seed()
            statej= self.np_random.uniform(low=-1e3, high=1e3, size=self.j.shape) # image
            stateq= self.np_random.uniform(low=0, high=255, size=self.q.shape)
            # self.state = spaces.Dict({'joints': statej, 'image': stateq})
            self.state = {'joints': statej, 'image': stateq}
        elif obs_type == 'image':
            print('The observation space is image')
            self.observation_space = spaces.Box(low=0, high=255, shape= self.q.shape, dtype=self.q.dtype)
            self.seed()
            self.state= self.np_random.uniform(low=0, high=255, size=self.q.shape) # image
        # self.q_last = self.q # yinggai bu xuyao q.last, cause we have 3 image in the buffer
        # self.observation_space = spaces.Box(low=0, high=255, shape= self.q.shape, dtype=self.q.dtype)
        # '''Box(0, 255, (3, 256, 256), uint8)'''

        # spaces = {
        # 'position': gym.spaces.Box(low=0, high=100, shape=(2,)),
        # 'orientation': ...
        # }
        # dict_space = gym.spaces.Dict(spaces)

        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,)) origin statment in cart_pole
        self.counts = 0
        self.steps_beyond_done = None

        self.sim.startSimulation()

        self.skullbot_sim_model = skullbotSimModel()
        self.skullbot_sim_model.initializeSimModel(self.sim)
        # self.sim.setBoolParam(self.sim.boolparam_display_enabled,False)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        startTime = time.time()
        if self.action_type == 'discrete':
            assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        # print('action',action)

        '''imgae proc'''
        #region
        image, resX, resY = self.skullbot_sim_model.getVisionSensorCharImage('sim_OCT_vision_sensor')
        cannyImg = visionSensorImage2Canny(image, resX, resY)
        _, minDist, point1, _ = minEdgeDist(cannyImg, resX, resY)
        outerImage, edgeDist, nearest_point, edges = secondEdgeDist(cannyImg, resX, resY, minDist)
        holdingDist = 2*minDist
        tangentVector = calculateTangetVector(edges, nearest_point)*10

        imageLoss_2 = (np.sum(np.abs(outerImage - self.targetImage))) / 256

        q_new = cannyImg
        self.q[0] = self.q[1]
        self.q[1] = self.q[2]
        self.q[2] = q_new

        # print(self.imageLoss1, imageLoss_2)
        imageExp = ((self.imageLoss1 - imageLoss_2) / self.imageLoss1) if self.imageLoss1!=0 else 0.0
        imageReward = np.round(np.exp(2*imageExp)-0.8, 5) 
        # imageReward = 1 - imageLoss_2/resX/resY
        # print(imageReward)

        translation_matrix = np.float32([[1, 0, tangentVector[1]], [0, 1, tangentVector[0]]])
        target_image = cv2.warpAffine(outerImage, translation_matrix, (outerImage.shape[1], outerImage.shape[0]))
        self.targetImage = target_image
        self.imageLoss1 = imageLoss_2

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
            self.joint_advancement[0] = round(action[0] * 9.0 / 180 * np.pi /100, 4)
            self.joint_advancement[1] = round(action[1] * 2.0e-3 /100, 8)
            self.joint_advancement[2] = round(action[2] * 9.0 / 180 * np.pi /100, 4)
            self.joint_advancement[3] = round(action[3] * 2.0e-3 /100, 8)
            # self.joint_advancement[4] = round(action[4] * 2.0e-3 /100, 8)
        else:
            assert 0, "The action type \'" + self.action_type + "\' can not be recognized"
        # print('joint_advancement:', self.joint_advancement)
        self.j[0] = self.skullbot_sim_model.getJointPosition('Rotor1_joint')
        self.j[1] = self.skullbot_sim_model.getJointPosition('Slider1_joint')
        self.j[2] = self.skullbot_sim_model.getJointPosition('Rotor2_joint')
        self.j[3] = self.skullbot_sim_model.getJointPosition('Slider2_joint')
        self.skullbot_sim_model.setJointPosition('Rotor1_joint', self.j[0]+self.joint_advancement[0])
        self.skullbot_sim_model.setJointPosition('Slider1_joint', self.j[1]+self.joint_advancement[1])
        self.skullbot_sim_model.setJointPosition('Rotor2_joint', self.j[2]+self.joint_advancement[2])
        self.skullbot_sim_model.setJointPosition('Slider2_joint', self.j[3]+self.joint_advancement[3])


        '''step done?'''
        #region
        step_done = (edgeDist <= minDist*1.3) or (edgeDist >= holdingDist*2) or (self.counts >= 3000)
        step_done = bool(step_done)
        if not step_done:
            self.done_frame = 0
        else:
            self.done_frame = self.done_frame + 1
            # print((edgeDist <= minDist*1.1) , (edgeDist >= holdingDist*2) , (self.counts >= 3000))

        if self.done_frame >= 3:
            done = True
        else:
            done = False
        #endregion
        # time.sleep(0.1)

        distReward = (0.5 - (abs((edgeDist - holdingDist)/holdingDist)))if holdingDist!=0 else 0.0
        # the number means the percentage of the edgeDist can deviate from the holdingDist
        if abs(distReward) > 2:     # edgeDist is too big, mostly because the path is too far
            distReward = -2
        distReward = np.round(distReward, 5)
        # print('imageR', imageReward, 'distR', distReward, 'action', action)
        if not done:
            if not step_done:
                reward = distReward + imageReward
            else:
                reward = 0.0
        elif self.steps_beyond_done is None:
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

        dt = 0.025
        
        if self.obs_type == 'joints_image':
            self.state['joints'] = self.j
            self.state['image'] = self.q
        elif self.obs_type == 'image':
            self.state = self.q
        self.counts += 1

        self.sim.step()
        endTime= time.time()
        # print('Time for one step:', endTime-startTime)
        # print(reward, step_done, done)
        return self.state, reward, done, False, {}
        # return np.array(q1, dtype=np.float32), reward, done, False, {}
    
    def reset(self, seed=None):
        ''''设置不同旋转角度下的fiber model'''
        # 1. 随机选取一个 fiber model 设置为visible
        selected_object_handle = random.choice(self.Objects)
        self.skullbot_sim_model.setObjVisible(selected_object_handle)
        # 2. 随机设置 fiber model距离 needle dummy的相对位置，z 保持不变，
        #    沿着针的切向的距离可以在一个较大范围内随机设置，而沿着针的法向的距离可以在一个较小范围内随机设置 （或基本保持不变）
        
        self.skullbot_sim_model.setObjectPosition(selected_object_handle, [0, 0, 0])
        # 3. 在初始随机化之后， 如果在已知方向的视野范围内看不到edge，则让fiber model沿着x/y方向一直运动直到看到edge或超出count
        
        # 4. 在一个episode中还需要设置整体的goal吗，在一个step中还需要考虑图像的reward吗


        # print('Reset the environment after {} counts'.format(self.counts))
        self.counts = 0
        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        randomJoints = self.np_random.uniform(low=-1, high=1, size=(6,)) # rotor1, slider1, rotor2, slider2, needle driver
        self.steps_beyond_done = None

        self.sim.stopSimulation() # stop the simulation
        time.sleep(0.1) # ensure the coppeliasim is stopped
        self.sim.setStepping(True)

        self.skullbot_sim_model.setJointPosition('Rotor1_joint', randomJoints[0]*9 / 180 * np.pi / 10)
        self.skullbot_sim_model.setJointPosition('Slider1_joint', randomJoints[1]*2e-3 / 10)
        self.skullbot_sim_model.setJointPosition('Rotor2_joint', randomJoints[2]*9 / 180 * np.pi / 10)
        self.skullbot_sim_model.setJointPosition('Slider2_joint', randomJoints[3]*2e-3 / 10)
        # self.skullbot_sim_model.setJointPosition('needle_driver_joint', randomJoints[4]*2e-3 / 100)
        self.sim.startSimulation()
        # self.sim.setBoolParam(self.sim.boolparam_display_enabled,False)


        image, resX, resY= self.skullbot_sim_model.getVisionSensorCharImage('sim_OCT_vision_sensor')
        cannyImg = visionSensorImage2Canny(image, resX, resY)
        # _, minDist, _ = minEdgeDist(cannyImg, resX, resY)
        # _, edgeDist, _ = secondEdgeDist(cannyImg, resX, resY, minDist)
        # holdingDist = 2.5*minDist

        self.q = np.zeros((3,256,256), dtype=np.uint8)
        self.q[-1] = cannyImg
        
        if self.obs_type == 'joints_image':
            self.j = np.array(
                [
                    self.skullbot_sim_model.getJointPosition('Rotor1_joint'),
                    self.skullbot_sim_model.getJointPosition('Slider1_joint'),
                    self.skullbot_sim_model.getJointPosition('Rotor2_joint'),
                    self.skullbot_sim_model.getJointPosition('Slider2_joint'),
                    0,
                    0,
                ],
                dtype=np.float32,
            ) # joint
            self.state = {'joints': self.j, 'image': self.q}
            print('the obs_type is joints_image')
        elif self.obs_type == 'image':
            self.state = self.q
            print('the obs_type is image')

        self.done_frame = 0
        print('Reset the environment')
        return self.state, {}
    
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
        state = env.step(action)
        # print(env.state[:5], env.state[-1])

    env.close()
