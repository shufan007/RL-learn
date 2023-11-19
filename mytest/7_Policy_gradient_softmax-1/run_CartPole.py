"""
Policy Gradient, Reinforcement Learning.

The cart pole example

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""
import os

import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time
SAVE_EPISODE = 10

ENV_NAME = 'CartPole-v0'

env = gym.make(ENV_NAME)
#env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

basePath = os.getcwd()
model_base_path = os.path.join(os.path.dirname(basePath), "model")
if os.path.isdir(model_base_path) == False:
    os.makedirs(model_base_path)
    
model_id = ENV_NAME+'-PG-1'
model_path = os.path.join(model_base_path, model_id)
if os.path.isdir(model_path) == False:
    os.makedirs(model_path)   

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
    # output_graph=True,
)

RL.restore_model(model_path)

EPISODE = 0
total_step = 0

for i_episode in range(EPISODE+1):

    observation = env.reset()
    step = 0
    while True:
        #if RENDER: env.render()
        if i_episode == EPISODE : env.render()
        
        step+=1
        total_step+=1

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward)

        if done:
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
            print("episode:", i_episode, ", step:", step, ",  reward:", int(running_reward))

            vt = RL.learn()
            
            if i_episode>0 and i_episode % SAVE_EPISODE == 0:
                RL.save_model(model_path, model_id)
            
            """
            if i_episode == 0:
                plt.plot(vt)    # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            """
            break

        observation = observation_
        
RL.save_model(model_path, model_id)
