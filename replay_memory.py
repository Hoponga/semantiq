import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim


Transition = namedtuple('Transition', 
('state', 'action', 'next_state', 'reward'))

# data structure for storing transitions for experience replay training
class ReplayMemory:
    def __init__(self, size): 
        self.memory = deque([], maxlen=size)

    def push(self, *args): 
        self.memory.append(Transition(*args))

    def sample(self, batch_size): 
        return random.sample(self.memory, batch_size)

    def __len__(self): 
        return len(self.memory)

