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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import TensorBoardOutputFormat

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
parser.add_argument('--_model_type', type=str, default='teacher')
parser.add_argument('--_tmp_dir_num', type=str, default=1)
args = parser.parse_args()


class lastEpiRewCallback(BaseCallback):
    def __init__(self, env, eval_freq: int):
        self.eval_freq = eval_freq
        self.env = env

    def _on_step(self) -> bool:
        # print(self.locals["rewards"])
        # print(self.locals)
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            _episode_reward = self.env.get_episode_rewards()
            _lastEpiRew = _episode_reward[-1]

            # _reward = self.locals['rewards'][0]
            # print(_reward)
            # self.logger.record('step reward', _reward)

        return True
    

if args._obs_type == 'joints_image':
    algo_policy = 'MultiInputPolicy'
elif args._obs_type == 'image':
    algo_policy = 'CnnPolicy'

# ---------------- Create environment
env = skullbotEnv(action_type='continuous', obs_type=args._obs_type, model_type=args._model_type) 
check_env(env)

log_dir = f"./Model/saved_models/tmp_{args._tmp_dir_num}"
while os.path.exists(log_dir):
    args._tmp_dir_num = 1 + int(args._tmp_dir_num)
    log_dir = f"./Model/saved_models/tmp_{args._tmp_dir_num}"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)
# env.get_episode_rewards()

# ---------------- Callback functions
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix='rl_model')
callback_save_best_model = EvalCallback(env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=10000, deterministic=True, render=False, verbose=0)

callback_list = CallbackList([checkpoint_callback, callback_save_best_model])

# ---------------- Model
'''Option 1: create a new model'''
print("create a new model")

policy = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(qf=[128, 128, 128], pi=[128, 128, 128]))
buffer_size = int(3e4)
model = PPO(policy=algo_policy, env=env,
             learning_rate=1e-4, verbose=1, tensorboard_log=log_dir, device='cuda')

'''Option 2: load the model from files (note that the loaded model can be learned again)'''
# # # print("load the model from files")
# # # model = A2C.load("../CartPole/saved_models/tmp/best_model", env=env)
# # # model.learning_rate = 1e-4

'''Option 3: load the pre-trained model from files'''
# print("load the pre-trained model from files")

# model = DDPG.load("./Model/saved_models/tmp_1/best_model", env=env)   


'''---------------- Learning'''
print('Learning the model')
# model.learn(total_timesteps=300000, callback=callback_save_best_model)
# model.learn(total_timesteps=400000, callback=callback_list) # 'MlpPolicy' = Actor Critic Policy
model.learn(total_timesteps=300000, callback=callback_list) # 'MlpPolicy' = Actor Critic Policy
model.save(log_dir + '/best_model')
print('Finished')
del model # delete the model and load the best model to predict
model = PPO.load(log_dir + '/best_model', env=env)


''' ---------------- Prediction'''
# print('Prediction')
# observation, info = env.reset()
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