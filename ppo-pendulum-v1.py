"""
@Date   ：2022/11/2
@Fun: 倒立摆控制
"""
import random
import gym
import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython import display
import time

env = gym.make("Pendulum-v1")
# 智能体状态
state = env.reset()
# 动作空间（连续性问题）
actions = env.action_space
print(state, actions)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(0))

N=2000
M=10
max_step=500

d_model = 64
discount = 0.95

# 演员模型：接收一个状态，使用抽样方式确定动作
class Model(torch.nn.Module):
    """
    继承nn.Module，必须实现__init__() 方法和forward()方法。其中__init__() 方法里创建子模块，在forward()方法里拼接子模块。
    """
    def __init__(self):
        super().__init__()
        self.fc_state = torch.nn.Sequential(torch.nn.Linear(3, d_model),
                                            torch.nn.ReLU(),
                                            )
        self.fc_mu = torch.nn.Sequential(torch.nn.Linear(d_model, 1),
                                         torch.nn.Tanh())
        self.fc_std = torch.nn.Sequential(torch.nn.Linear(d_model, 1),
                                          torch.nn.Softplus())

    def forward(self, state):
        state = self.fc_state(state)
        mu = self.fc_mu(state)
        std = self.fc_std(state)

        return mu, std

# 学一个针对state的概率密度函数mu，std
actor_model = Model().to(device)
#actor_model = actor_model.cuda()
# 评论员模型：评价一个状态的价值，给出多好的得分
critic_model = torch.nn.Sequential(torch.nn.Linear(3, d_model),
                                   torch.nn.ReLU(),
                                   torch.nn.Linear(d_model, 1))
critic_model = critic_model.to(device)

# 演员模型执行一个动作（采样获得）
def get_action(state):
    state = torch.FloatTensor(state).reshape(1, 3)
    state = state.to(device)
    mu, std = actor_model(state)
    # 通过服从(mu, std)的概率密度函数得到连续性动作
    action = torch.distributions.Normal(mu, std).sample().item()

    return action

# 获取一个回合的样本数据
def get_data():
    states = []
    rewards = []
    actions = []
    next_states = []
    dones = []

    state = env.reset()[0]
    done = False
    n_step=0
    while not done and n_step<max_step:
        action = get_action(state)
        next_state, reward, done, _, _ = env.step([action])
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        next_states.append(next_state)
        dones.append(done)

        state = next_state

        n_step +=1
        # print("n_step:", n_step)
    # 转换为tensor
    states = torch.FloatTensor(states).reshape(-1, 3)
    rewards = torch.FloatTensor(rewards).reshape(-1, 1)
    actions = torch.FloatTensor(actions).reshape(-1, 1)       # 动作连续
    next_states = torch.FloatTensor(next_states).reshape(-1, 3)
    dones = torch.LongTensor(dones).reshape(-1, 1)

    return states, actions, rewards, next_states, dones

def test():
    state = env.reset()[0]
    reward_sum = 0
    over = False

    n_step = 0
    while not over and n_step<max_step:
        action = get_action(state)
        state, reward, over, _,_ = env.step([action])
        reward_sum += reward

        n_step += 1

    return reward_sum

# 优势函数
def get_advantage(deltas):
    # 算法来源：GAE，广义优势估计方法。便于计算从后往前累积优势
    advantages = []
    s = 0
    for delta in deltas[::-1]:
        s = discount * 0.95 * s + delta
        advantages.append(s)
    advantages.reverse()

    return advantages

print(get_advantage([0.8, 0.9, 0.99, 1.00, 1.11, 1.12]))

start = time.time()

def train():
    optimizer = torch.optim.Adam(actor_model.parameters(), lr=1e-5)
    optimizer_td = torch.optim.Adam(critic_model.parameters(), lr=1e-3)

    # 玩N局游戏，每局游戏玩M次
    for epoch in range(N):
        # print("epoch:", epoch)

        states, actions, rewards, next_states, dones = get_data()
        #print("rewards:",rewards)
        #rewards = (rewards + 8) / 8

        #print("rewards:",rewards)
        # 计算values和targets
        states = states.to(device)
        next_states = next_states.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)

        # print("states:",states)
        # print("next_states:",next_states)

        values = critic_model(states)
        targets = critic_model(next_states).detach()    # 目标，不作用梯度

        # 结束状态价值为零
        targets *= (1- dones)
        # 计算总回报(奖励+下一状态)
        targets = rewards + targets * discount

        # 计算优势，类比策略梯度中的reward_sum
        deltas = (targets - values).squeeze().tolist()  # 标量数值
        advantages = get_advantage(deltas)
        advantages = torch.FloatTensor(advantages).reshape(-1, 1)

        # print("advantages:",advantages)

        # 取出每一步动作演员给的评分
        mu, std = actor_model(states)
        mu = mu.cpu()
        std = std.cpu()

        action_dist = torch.distributions.Normal(mu, std)
        # 找到当前连续动作在分布下的概率值，exp()做还原使用，就数据补参与梯度更新
        old_probs = action_dist.log_prob(actions).exp().detach()

        # print("mu, std:", mu, std)

        # 每批数据反复训练10次
        for _ in range(M):
            # 重新计算每一步动作概率
            mu, std = actor_model(states)
            mu = mu.cpu()
            std = std.cpu()
            new_action_dist = torch.distributions.Normal(mu, std)
            new_probs = new_action_dist.log_prob(actions).exp()
            # 概率变化率
            ratios = new_probs / old_probs
            # 计算不clip和clip中的loss，取较小值
            no_clip_loss = ratios * advantages
            clip_loss = torch.clamp(ratios, min=0.8, max=1.2) * advantages
            loss = -torch.min(no_clip_loss, clip_loss).mean()
            # 重新计算value，并计算时序差分loss
            values = critic_model(states)
            loss_td = torch.nn.MSELoss()(values, targets)

            # print("loss_td:", loss_td)

            # 更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer_td.zero_grad()
            loss_td.backward()
            optimizer_td.step()

        if epoch % 20 == 0 or epoch == N-1:
            result = sum([test() for _ in range(10)]) / 10
            print("time:", time.time() - start)
            print(epoch, result)


train()