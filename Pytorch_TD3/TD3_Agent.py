# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import time
import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import operator

def GaussianNoise(data, stddev):
    data = data + (stddev**0.5)* (T.randn(data.shape).to("cuda"))
    return T.clamp_(data, -0.5, 0.5)

class TD3Memory:
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
    def __init__(self, input_dims, alpha,
            fc1_dims=16, fc2_dims=16, chkpt_dir='TD3'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_TD3')
        self.actor = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1),
                nn.ReLU(),
                nn.Tanh(),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
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
    def __init__(self, input_dims, action_dims, alpha, fc1_dims=16, fc2_dims=16,
            chkpt_dir='TD3', Q_tag=0):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_TD3_' + str(Q_tag))
        self.critic = nn.Sequential(
                nn.Linear(list(map(operator.add, input_dims, action_dims))[0], fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
        
class Agent:
    def __init__(self, a_min, a_max, input_dims, action_dims, gamma=0.995, actor_alpha=0.0001, critic_alpha=0.001,
            policy_clip=0.2, batch_size=64, n_epochs=10, rho=0.9995, warm_up=10000, policy_delay=2):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.rho = rho
        self.warm_up = warm_up

        self.actor = ActorNetwork(input_dims=input_dims, alpha=actor_alpha) # (input + action)
        self.critic_1 = CriticNetwork(input_dims, action_dims, actor_alpha, Q_tag=1)
        self.critic_2 = CriticNetwork(input_dims, action_dims, actor_alpha, Q_tag=2)
        self.targ_actor = ActorNetwork(input_dims=input_dims, alpha=critic_alpha)
        self.targ_critic_1 = CriticNetwork(input_dims, action_dims, critic_alpha)
        self.targ_critic_2 = CriticNetwork(input_dims, action_dims, critic_alpha)
        
        self.targ_actor.load_state_dict(dict(self.actor.named_parameters()))
        self.targ_critic_1.load_state_dict(dict(self.critic_1.named_parameters()))
        self.targ_critic_2.load_state_dict(dict(self.critic_2.named_parameters()))
        
        self.memory = TD3Memory(batch_size)
        self.a_min = a_min
        self.a_max = a_max
        self.count = 0
        self.policy_delay = policy_delay
       
    def remember(self, state, next_state, action, reward, done):
        self.memory.store_memory(state, next_state, action, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        
    def choose_action(self, observation):
        if self.warm_up == 0:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            action = self.actor(state) + 1.0
            action = action * (self.a_max - self.a_min)  + self.a_min 
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
                self.count += 1
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                next_states = T.tensor(next_states_arr[batch], dtype=T.float).to(self.actor.device)
                actions = T.tensor(action_arr[batch], dtype=T.float).to(self.actor.device)
                rewards = T.tensor(reward_arr[batch], dtype=T.float).to(self.actor.device)
                dones = T.tensor(dones_arr[batch], dtype=T.int).to(self.actor.device)

                actions = actions.unsqueeze(0).transpose(0,1).to(self.actor.device)
                state_actions = T.cat([states, actions], dim=1).to(self.actor.device)
                next_state_action = T.cat([next_states, GaussianNoise(self.targ_actor(next_states), stddev=0.2)], dim=1).to(self.actor.device)
                predicted_state_action = T.cat([states, self.actor(states)], dim=1).to(self.actor.device)
                
                
                tc1 = self.targ_critic_1(next_state_action)
                tc2 = self.targ_critic_2(next_state_action)
                # if self.warm_up < 1:
                #     print(next_state_action)
                min_Q_value = T.minimum(tc1, tc2)
                y_rsd = rewards + self.gamma * (1 - dones) * min_Q_value
                
                critic_1_loss = ((self.critic_1(state_actions) - y_rsd)**2).mean()
                critic_2_loss = ((self.critic_2(state_actions) - y_rsd)**2).mean()
                
                self.critic_1.optimizer.zero_grad()
                critic_1_loss.backward(retain_graph=True)
                self.critic_1.optimizer.step()
                
                self.critic_2.optimizer.zero_grad()
                critic_2_loss.backward()
                self.critic_2.optimizer.step()
                
                if self.count % self.policy_delay == 0:
                    # update target network
                    actor_loss = self.critic_1(predicted_state_action).mean()
                    self.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor.optimizer.step()
                    
                    actor_param = self.actor.named_parameters()
                    targ_actor_param = self.targ_actor.named_parameters()
                    
                    ap = dict(actor_param)
                    
                    for name1, param1 in targ_actor_param:
                        if name1 in ap:
                            ap[name1].data.copy_(self.rho * param1.data + (1.0-self.rho) * ap[name1].data)
                    
                    self.targ_actor.load_state_dict(ap)
                    
                    critic_1_param = self.critic_1.named_parameters()
                    targ_critic_1_param = self.targ_critic_1.named_parameters()
                    cp1 = dict(critic_1_param)
                    
                    for name1, param1 in targ_critic_1_param:
                        if name1 in cp1:
                            cp1[name1].data.copy_(self.rho * param1.data + (1.0-self.rho) * cp1[name1].data)
                    
                    self.targ_critic_1.load_state_dict(cp1)
                    
                    critic_2_param = self.critic_2.named_parameters()
                    targ_critic_2_param = self.targ_critic_2.named_parameters()
                    cp2 = dict(critic_2_param)
                    
                    for name1, param1 in targ_critic_2_param:
                        if name1 in cp2:
                            cp2[name1].data.copy_(self.rho * param1.data + (1.0-self.rho) * cp2[name1].data)
                    
                    self.targ_critic_2.load_state_dict(cp2)

                
        self.memory.clear_memory()   