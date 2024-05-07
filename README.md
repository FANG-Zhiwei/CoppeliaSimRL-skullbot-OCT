# Skullbot RL

The skullbotSimModel.py communicates with the Vrep

The skullbotRLEnv.py sets the Rl environment fot this simulation, where

* observation/ states space: 4 joints' positions && the sim_OCT_vision_sensor

  ```
  self.q = [0.0, 0.0, 0.0, 0.0]
  ```

  joint value is easy, the sim_OCT goes into a CNN about edge detection and output **sth**
* action space: the 4 joints' movements
* 

# UNet detection

* inner circle detection
* outer line detection

# Reinforcement Learning Workspace

The basic workspace for reinforcement learning with CoppeliaSim (VREP) simulation environments, including some demonstrated project for beginners.
Some tutorials can be found at:

- Zhihu: https://zhuanlan.zhihu.com/p/398874515
- WeChat: https://mp.weixin.qq.com/s/id7fw0eGBtLqEqr-zaKKPQ

I add gymnasium and ZeroMQ support compared with the original repository. All the demos have been tested on Windows 11, with python 3.10 environment.

---

## Environments setup

You have to install the following softwares and environments for this project, the recommend operating system is Ubuntu.

- CoppeliaSim 4.6 (https://www.coppeliarobotics.com/)
- Python 3.6+ (Anaconda is recommended)
- Gymnasium (https://github.com/Farama-Foundation/Gymnasium)
- Stable-baselines3 (https://github.com/DLR-RM/stable-baselines3)
- Pytorch (https://pytorch.org/)
- Visdom (pip install visdom)

---

## Demo 1: Cart-pole control with the A2C (modified SAC) algorithm

- Step 1: run CoppeliaSim, import cart_pole.ttt
- Step 2: run visdom in your terminal, open your browser, and visit link: localhost:8097
- Step 3: run the script named 'demo_cart_pole_learning.py' in the sub-path ./examples

Then we have:

![](pic/cart_pole_demo_1.gif)
