import torch

import gym
import numpy as np
import torch.nn as nn 
import torch.optim as optim 


env_name = 'CartPole-v0' # for now just run stuff on cartpole sim 
env = gym.make(env_name)

print(env.action_space)

print (env.action_space.n)

def run(): 
    

