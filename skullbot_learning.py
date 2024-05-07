'''
Author: FANG Zhiwei
Email: zwfang@link.cuhk.edu.hk
Date: 2024-03-18 16:20:57
LastEditTime: 2024-04-23 23:56:57
Description: start at 1845  1915-1924 500steps
'''

'''run headless mode in coppeliaSim
./coppeliaSim.sh -h ~/Documents/skullbot/OCTservo_ws/CoppeliaSimRL-skullbot-OCT/scene/OCT_servoing_pyCom2.ttt
'''


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.env_checker import check_env

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import EvalCallback

import torch as th
import cv2
import argparse


import sys
sys.path.append("./utils")
from callbackFunctions import VisdomCallback

sys.path.append("./Model")
from skullbotRLEnv import skullbotEnv

import os
parser=argparse.ArgumentParser()
parser.add_argument('--_obs_type', type=str, default='joints_image')
args = parser.parse_args()

if args._obs_type == 'joints_image':
    al_policy = 'MultiInputPolicy'
elif args._obs_type == 'image':
    al_policy = 'CnnPolicy'

# ---------------- Create environment
env = skullbotEnv(action_type='continuous', obs_type=args._obs_type) 
# action_type can be set as discrete or continuous
# obs_type = 'joints_image' or 'joints' or 'image'
check_env(env)

# ---------------- Callback functions
log_dir = "./Model/saved_models/tmp_3"
os.makedirs(log_dir, exist_ok=True)

env = Monitor(env, log_dir)

# callback_visdom = VisdomCallback(name='visdom_skullbot_rl', check_freq=100, log_dir=log_dir)
# callback_save_best_model = EvalCallback(env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=10000, deterministic=True, render=False, verbose=0)
# callback_list = CallbackList([callback_visdom, callback_save_best_model])

# ---------------- Model
'''Option 1: create a new model'''
print("create a new model")

policy = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(qf=[128, 128, 128], pi=[128, 128, 128]))
buffer_size = int(3e4)
model = DDPG(policy=al_policy, env=env, policy_kwargs=policy, buffer_size= buffer_size,learning_rate=5e-4, verbose=1, tensorboard_log=log_dir, device='cuda')

'''Option 2: load the model from files (note that the loaded model can be learned again)'''
# # # print("load the model from files")
# # # model = A2C.load("../CartPole/saved_models/tmp/best_model", env=env)
# # # model.learning_rate = 1e-4

'''Option 3: load the pre-trained model from files'''
# print("load the pre-trained model from files")
# if env.action_type == 'discrete':
#     model = DDPG.load("./CartPole/saved_models/best_model_discrete", env=env)
# else:
#     model = DDPG.load("./Model/saved_models/tmp_1/best_model", env=env)   


'''---------------- Learning'''
# print('Learning the model')
# model.learn(total_timesteps=300000, callback=callback_save_best_model)
# model.learn(total_timesteps=400000, callback=callback_list) # 'MlpPolicy' = Actor Critic Policy
model.learn(total_timesteps=300000) # 'MlpPolicy' = Actor Critic Policy
# print('Finished')
model.save(log_dir + '/best_model')

# del model # delete the model and load the best model to predict
# model = PPO.load("./Model/saved_models/tmp_1/best_model", env=env)


# ---------------- Prediction
print('Prediction')

observation, info = env.reset()
# while True:
#     action, _state = model.predict(observation, deterministic=True)
#     observation, reward, done, terminated, info = env.step(action)
#     # if done:
#     #     break

for _ in range(10):
    print('test episode')
    observation, info = env.reset()
    done = False
    episode_reward = 0.0

    while not done:
        action, _state = model.predict(observation, deterministic=True)
        observation, reward, done, terminated, info = env.step(action)
        episode_reward += reward
    
    print([episode_reward, env.counts])

env.close()

# print('wait key')
# cv2.waitKey('q')
# print('finished')