# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:36:09 2024

@author: jim
"""

import os
import time
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import operator
import random

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            return din + T.autograd.Variable(T.randn(din.size()).cuda() * self.stddev)
        print(din)
        return din

class DDPGMemory:
    def __init__(self, batch_size):
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.states),\
                np.array(self.next_states),\
                np.array(self.actions),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, next_states, action, reward, done): # (s,a,s',r,d)
        self.states.append(state)
        self.next_states.append(next_states)
        self.actions.append(action)
        self.rewards.append(reward)
        if done == True:
            self.dones.append(1)
        else:
            self.dones.append(0)

    def clear_memory(self):
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        
class ActorNetwork(nn.Module):
    def __init__(self, input_dims, action_dims, alpha, gaussian,
            fc1_dims, fc2_dims, chkpt_dir='ddpg'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_ddpg')
        self.actor = None
        self.n_actions = action_dims
        if gaussian == True:
            self.actor = nn.Sequential(
                    nn.Linear(*input_dims, fc1_dims),
                    nn.LayerNorm(fc1_dims),
                    nn.ReLU(),
                    nn.Linear(fc1_dims, fc2_dims),
                    nn.LayerNorm(fc2_dims),
                    nn.ReLU(),
                    nn.Linear(fc2_dims, *self.n_actions),
                    nn.Tanh(),
                    GaussianNoise(0.2),
            )
        else:
            self.actor = nn.Sequential(
                    nn.Linear(*input_dims, fc1_dims),
                    nn.LayerNorm(fc1_dims),
                    nn.ReLU(),
                    nn.Linear(fc1_dims, fc2_dims),
                    nn.LayerNorm(fc2_dims),
                    nn.ReLU(),
                    nn.Linear(fc2_dims, *self.n_actions),
                    nn.Tanh(),
                    GaussianNoise(0.0),
            )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha, weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, action_dims, alpha, fc1_dims, fc2_dims,
            chkpt_dir='ddpg'):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_ddpg')
        self.critic_1 = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.LayerNorm(fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.LayerNorm(fc2_dims),
                nn.ReLU(),
        )
        self.action_value = nn.Linear(*action_dims, fc2_dims)
        self.critic_2 = nn.Sequential(
                nn.Linear(fc2_dims, 1),
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha, weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_value = self.critic_1(state)
        action_value = self.action_value(action)
        state_action_value = T.add(state_value, action_value)
        state_action_value = self.critic_2(state_action_value)
        return state_action_value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
        
class Agent:
    def __init__(self, a_min, a_max, input_dims, action_dims, fc1_dims, fc2_dims, gamma=0.995, actor_alpha=0.0001, critic_alpha=0.001, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10, rho=0.001, warm_up=0):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.rho = rho
        self.warm_up = warm_up
        self.batch_size = batch_size

        self.actor = ActorNetwork(input_dims=input_dims, action_dims=action_dims, fc1_dims=fc1_dims, fc2_dims=fc2_dims ,alpha=actor_alpha, gaussian=False) # (input + action)
        self.critic = CriticNetwork(input_dims, action_dims, fc1_dims=fc1_dims, fc2_dims=fc2_dims, alpha=actor_alpha)
        self.targ_actor = ActorNetwork(input_dims=input_dims, action_dims=action_dims, fc1_dims=fc1_dims, fc2_dims=fc2_dims ,alpha=actor_alpha, gaussian=True)
        self.targ_critic = CriticNetwork(input_dims, action_dims , fc1_dims=fc1_dims, fc2_dims=fc2_dims, alpha=critic_alpha)
        
        self.targ_actor.load_state_dict(dict(self.actor.named_parameters()))
        self.targ_critic.load_state_dict(dict(self.critic.named_parameters()))
        
        self.memory = DDPGMemory(batch_size)
        self.a_min = a_min
        self.a_max = a_max

       
    def remember(self, state, next_state, action, reward, done):
        self.memory.store_memory(state, next_state, action, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()   
        
    def choose_action(self, observation):
        if self.warm_up == 0:
            state = T.tensor([observation], dtype=T.float).to(self.actor.device)
            action = self.actor(state)
            action = action.clamp(self.a_min, self.a_max)
            action = T.squeeze(action).item()
            return action
        else:
            self.warm_up -= 1
            return random.random() * (self.a_max - self.a_min) + self.a_min
    
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, next_states_arr, action_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()
            
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                next_states = T.tensor(next_states_arr[batch], dtype=T.float).to(self.actor.device)
                actions = T.tensor(action_arr[batch], dtype=T.float).to(self.actor.device)
                rewards = T.tensor(reward_arr[batch], dtype=T.float).to(self.actor.device)
                dones = T.tensor(dones_arr[batch], dtype=T.int).to(self.actor.device)
                
                actions = actions.unsqueeze(0).transpose(0,1)
                target_actions = self.targ_actor.forward(next_states)
                critic_value_ = self.targ_critic.forward(next_states, target_actions)
                critic_value_[dones] = 0.0
                critic_value_ = critic_value_.view(-1)
                critic_value = self.critic.forward(states, actions)
                    
                target = rewards + self.gamma*critic_value_
                target = target.view(self.batch_size, 1)
                
                self.critic.optimizer.zero_grad()
                critic_loss = F.mse_loss(target, critic_value)
                critic_loss.backward()
                self.critic.optimizer.step()
                
                self.actor.optimizer.zero_grad()
                actor_loss = -self.critic.forward(states, self.actor.forward(states))
                actor_loss = T.mean(actor_loss)
                actor_loss.backward()
                self.actor.optimizer.step()
                
                self.update_network_parameters()

    def update_network_parameters(self, rho=None):
        if rho is None:
            rho = self.rho

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.targ_actor.named_parameters()
        target_critic_params = self.targ_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = rho*critic_state_dict[name].clone() + \
                                (1-rho)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = rho*actor_state_dict[name].clone() + \
                                 (1-rho)*target_actor_state_dict[name].clone()

        self.targ_critic.load_state_dict(critic_state_dict)
        self.targ_actor.load_state_dict(actor_state_dict)
        #self.target_critic.load_state_dict(critic_state_dict, strict=False)
        #self.target_actor.load_state_dict(actor_state_dict, strict=False)

                
        self.memory.clear_memory()   
                
