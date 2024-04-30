# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 08:27:22 2024

@author: jim
"""

import gymnasium as gym
import numpy as np
from TD3_Agent import Agent
from TD3_util import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    N = 16
    batch_size = 4
    n_epochs = 4
    actor_alpha = 0.0005
    critic_alpha = 0.0005
    warm_up = 3000 # (step)
    policy_delay = 2
    agent = Agent(a_min=-1.0, a_max=1.0, batch_size=batch_size, 
                    actor_alpha=actor_alpha, critic_alpha=critic_alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape, action_dims=env.action_space.shape, warm_up=warm_up, policy_delay=policy_delay)
    n_games = 300

    figure_file = 'plots/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    
    for i in range(n_games):
        observation, info = env.reset()
        done = False
        score = 0
        game_step = 0
        max_game_step = 10000
        while not done:
            action = agent.choose_action(np.array(observation))
            observation_, reward, terminated, truncated, info = env.step(np.array([action]))
            done = terminated
            n_steps += 1
            game_step += 1
            if reward > 0:
                reward = 100
            score += (reward - 0.02)
            agent.remember(observation, observation_, action, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
            if game_step >= max_game_step:
                done = True
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
            
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)