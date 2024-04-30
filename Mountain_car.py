# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:06:15 2024

@author: jim
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple, deque
import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
import matplotlib
from itertools import count

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# named Tuple
# https://qiubite31.github.io/2017/03/02/%E4%BD%BF%E7%94%A8collections%E4%B8%AD%E7%9A%84namedtuple%E4%BE%86%E6%93%8D%E4%BD%9C%E7%B0%A1%E5%96%AE%E7%9A%84%E7%89%A9%E4%BB%B6%E7%B5%90%E6%A7%8B/
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 2
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000
TAU = 0.005
LR = 1e-4
device = "cuda"

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class MountainCar:
    def __init__(self):
        self.position = 0
        self.velocity = 0
        self.action_pool = [-1,0,1]
        self.position_bond = [-1.2, 0.5]
        self.velocity_bond = [-0.07, 0.07]
        
    def next_position(self, pos, velocity):
        new_pos = pos + velocity
        new_pos = max(self.position_bond[0], min(new_pos, self.position_bond[1]))
        
        return new_pos
    
    def next_velocity(self, pos, velocity, action):
        new_velocity = velocity + 0.001 * self.action_pool[action] - 0.0025 * math.cos(3 * pos)
        new_velocity = max(self.velocity_bond[0], min(new_velocity, self.velocity_bond[1]))
        return new_velocity
        
    def reset(self):
        rand = random.uniform(-0.6, -0.4)
        self.position = rand
        self.velocity = 0
        
    def return_state(self):
        return [self.position, self.velocity]
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

env = MountainCar()
policy_net = DQN(2, 3).to("cuda")
target_net = DQN(2, 3).to("cuda")
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)
steps_done = 0
    
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            print(policy_net(state), policy_net(state).max(1).indices.view(1, 1))
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.randint(0,2)]], device=device)

def select_action_no_epsilon(state):
    with torch.no_grad():
        print("X", policy_net(state), policy_net(state).max(1).indices.view(1, 1))
        return policy_net(state).max(1).indices.view(1, 1)



# episode_durations = []
    
    
# def plot_durations(show_result=False):
#     plt.figure(1)
#     durations_t = torch.tensor(episode_durations, dtype=torch.float)
#     if show_result:
#         plt.title('Result')
#     else:
#         plt.clf()
#         plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(durations_t.numpy())
#     # Take 100 episode averages and plot them too
#     if len(durations_t) >= 100:
#         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())

#     plt.pause(0.001)  # pause a bit so that plots are updated
#     if is_ipython:
#         if not show_result:
#             display.display(plt.gcf())
#             display.clear_output(wait=True)
#         else:
#             display.display(plt.gcf())
            
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    # print(batch.state)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
if torch.cuda.is_available():
    num_episodes = 100
else:
    num_episodes = 10

eval_graph_y = []
def evaluate():
    steps_done = 0
    env.reset()
    state = [env.position, env.velocity]
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        while(True):
            steps_done += 1
            action = select_action_no_epsilon(state)
            env.velocity = env.next_velocity(env.position, env.velocity, action)
            env.position = env.next_position(env.position, env.velocity)
            if env.position < env.position_bond[0]:
                env.reset()
            observation, reward = [env.position, env.velocity], -1
            reward = torch.tensor([reward], device=device)
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            state = next_state
            # print(env.velocity, env.position, action)
            if env.position >= env.position_bond[1] or steps_done >= 5000:
                print(steps_done)
                break
    return steps_done

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    env.reset()
    state = [env.position, env.velocity]
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        env.velocity = env.next_velocity(env.position, env.velocity, action)
        env.position = env.next_position(env.position, env.velocity)
        if env.position < env.position_bond[0]:
            env.reset()

        observation, reward = [env.position, env.velocity], -1
        reward = torch.tensor([reward], device=device)
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        # episode_durations.append(t + 1)
        # plot_durations()
        if env.position >= env.position_bond[1]:
            break
    eval_graph_y.append(evaluate())

print('Complete')
ep = torch.tensor(np.arange(num_episodes, dtype=int))
plt.plot(ep, eval_graph_y)
plt.show()

# plot_durations(show_result=True)
# plt.ioff()
# plt.show()

# TODO:Implement eval


    