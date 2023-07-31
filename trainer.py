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


BATCH_SIZE = 128
GAMMA = 0.99 
EXP_REPLAY_SIZE = 10000


# IDEALLY, a quantum circuit (pennylane model) can replace the DQN for the policy_net & target_net 

# I'm not sure how soft updates would work for quantum version (but, if it has pytorch model wrapper, maybe can get state_dict in same way)
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
# have 10000 transitions stored in our memory for experienec replay 

memory = ReplayMemory(EXP_REPLAY_SIZE)


# Epsilon greedy sampling 
eps_i = 0.9 
eps_f = 0.05 
eps_decay = 1000 # time constant for epsilon decay 
def get_action(policy, state, steps_done): 
    sample = random.random()
    eps_current = eps_f + (eps_start - eps_end)*math.exp(-steps_done*1.0/eps_decay)

    if sample > eps_current: 
        with torch.no_grad(): 
            return policy_net(state).max(1)[1].view(1, 1)

    else: 
        return torch.tensor([[env.action_space.sample()]], device = device, dtype = torch.long)



def train_loop(): 
    if len(memory) > BATCH_SIZE: 
        return 
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda state: state is not None, batch.next_state)), device = device, dtype = torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)


    # Gather method takes values from the acting tensor on the given dimension (1st input) where the second parameter
    # represents the indices to sample from 
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device = device)

    with torch.no_grad(): 
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    # Q value update based on recursive Q eq. 
    # THESE are the targets for our Q values.. ie the yi = ri + gamma max_a Q(s', a)

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch 
    loss_func = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))


    optimizer.zero_grad()
    loss.backward()

    # I believe this is just PPO? though that limit may be different 
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step() 

if torch.cuda.is_available(): 
    num_episodes = 600
else: 
    num_episodes = 50 # will take forever lol 


episode_durations = [] 
for i_episode in range(num_episodes): 
    state, info = env.reset() 

    state = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
    for t in count(): 
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device = device)
        done = terminated or truncated 

        if terminated: 
            next_state = None 
        else: 
            next_state = torch.tensor(observation, dtype = torch.float32, device = device).unsqueeze(0)
        memory.push(state, action, next_state, reward)

        state = next_state
        train_loop()
        
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()

        # this seems highly inefficient for soft update but parallelize later maybe 
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            break






