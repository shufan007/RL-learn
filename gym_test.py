# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 22:16:59 2019

@author: Administrator
"""
import time
import numpy as np
import gym

#env = gym.make('CartPole-v0')
env = gym.make('Breakout-v0')
#env = gym.make('Breakout-v4')
#env = gym.make('MsPacman-v0')


def gym_env_test(env, n_episode=1, max_step=None):
    for i_episode in range(1):
        obs = env.reset()
        step = 0
        total_reward = 0
        while True:
            env.render()
            time.sleep(0.05)
            step += 1

            action = np.random.randint(0, env.action_space.n)
            observation, reward, done, info = env.step(action)
            # print(observation.shape)
            total_reward += reward
            print("Episoe:", i_episode, " ,total_reward:", total_reward, ", step:", step)

            if done:
                print("Episoe:", i_episode," ,total_reward:", total_reward, ", step:", step)
                break
            if max_step is not None and (step >= max_step):
                break
    env.close()


def gym_env_test_breakout():
    env = gym.make('Breakout-v0')
    gym_env_test(env)

def gym_env_test_breakout():
    env = gym.make('Breakout-v0')
    gym_env_test(env)

if __name__ == '__main__':
    gym_env_test_breakout()
