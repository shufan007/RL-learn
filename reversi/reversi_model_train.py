import os
import sys
import time
from shutil import copyfile
import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3 import PPO
from typing import Callable
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from gym_reversi import ReversiEnv
from custom_feature_extractor import CustomCNN

# from utils import time2str
from datetime import datetime

# PolicyModel = PPO
# PolicyModel = MaskablePPO


def time2str(timestamp):
    d = datetime.fromtimestamp(timestamp)
    timestamp_str = d.strftime('%Y-%m-%d %H:%M:%S')
    return timestamp_str


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


class ReversiModelTrain(object):
    def __init__(self,
                 backbone='cnn',
                 n_channels=3,
                 n_steps=512,
                 n_epochs=4,
                 batch_size=64,
                 board_size=8,
                 lr_init=3e-4,
                 lr_decay_rate=0.95,
                 total_timesteps=1000_0000,
                 start_timesteps=0,
                 check_point_timesteps=10_0000,
                 n_envs=8,
                 model_path=None,
                 opponent_model_path="random",
                 opponent_update_timesteps=10_0000,
                 opponent_prob_decay_rate=0.9,
                 opponent_window_size=100,
                 tensorboard_log=None,
                 verbose=0,
                 archive_timesteps=20_0000,
                 archive_path=None):

        self.backbone = backbone
        self.n_channels = n_channels
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr_init = lr_init
        self.lr_decay_rate = lr_decay_rate

        self.board_size = board_size
        self.total_timesteps = total_timesteps
        self.start_timesteps = start_timesteps
        self.check_point_timesteps = check_point_timesteps
        self.n_envs = n_envs
        self.model_path = model_path
        self.opponent_model_path = opponent_model_path
        self.opponent_update_timesteps = opponent_update_timesteps
        self.opponent_prob_decay_rate = opponent_prob_decay_rate
        self.opponent_window_size = opponent_window_size
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        # self.PolicyModel = PPO
        self.PolicyModel = MaskablePPO
        self.archive_timesteps = archive_timesteps
        self.archive_path = archive_path
        print(f"self.archive_path: {self.archive_path}")
        self.opponent_model = 'random'

        self.current_timesteps = min(total_timesteps, start_timesteps)
        self.progress_remaining = 1
        self.learning_rate = lr_init
        self.min_learning_rate = 1e-5

        self.opponent_distribution = self._get_opponent_distribution()

    def _get_opponent_distribution(self):
        _distrib = self.opponent_prob_decay_rate ** np.arange(1, self.opponent_window_size+1)
        d1 = _distrib[::-1].cumsum()[::-1]
        opponent_distribution = d1/d1[0]
        return opponent_distribution

    def _opponent_sampling(self):
        rand = np.random.random()
        for i in range(self.opponent_window_size):
            if rand > self.opponent_distribution[i]:
                return i
        return self.opponent_window_size

    def reversi_model_train_step(self, check_point_timesteps, save_model_path=None):

        self.learning_rate_update()

        env = ReversiEnv(opponent=self.opponent_model, is_train=True,
                         n_channels=self.n_channels,
                         board_size=self.board_size,
                         greedy_rate=0, verbose=self.verbose)

        vec_env = env
        if self.n_envs > 1:
            # multi-worker training (n_envs=4 => 4 environments)
            vec_env = make_vec_env(ReversiEnv, n_envs=self.n_envs, seed=None,
                                   env_kwargs={
                                       "opponent": self.opponent_model,
                                       "n_channels": self.n_channels,
                                       "is_train": True,
                                       "board_size": self.board_size,
                                       "greedy_rate": 0,
                                       "verbose": self.verbose},
                                )

        # set policy model configs
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=256,
                                           backbone=self.backbone,
                                           # net_arch=[128, 128, 256, 256],
                                           # net_arch=[64, 128, 128],
                                           net_arch=[128, 128, 256],
                                           # net_arch=[32, 64, 64],
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           is_batch_norm=True,
                                           pool_type=None),
            net_arch=[256, 256],
            normalize_images=False
        )

        try:
            model = self.PolicyModel.load(self.model_path, env=vec_env, learning_rate=self.learning_rate)
        except Exception:
            print(f"load model from self.model_path: {self.model_path} error")
            model = self.PolicyModel(MaskableActorCriticPolicy, vec_env,
                          policy_kwargs=policy_kwargs,
                          learning_rate=self.learning_rate,  # linear_schedule(self.learning_rate), learning_rate=2.5e-4,
                          ent_coef=0.01,
                          n_steps=self.n_steps, # n_steps=128,
                          n_epochs=self.n_epochs,
                          batch_size=self.batch_size, # batch_size=256,
                          gamma=0.99,
                          gae_lambda=0.95,
                          clip_range=0.1,
                          vf_coef=0.5,
                          verbose=1,
                          tensorboard_log=self.tensorboard_log)

        t0 = time.time()
        # model.learn(int(2e4))
        model.learn(total_timesteps=check_point_timesteps)
        model.save(self.model_path)
        if save_model_path is not None:
            model.save(save_model_path)
        print(f"train time: {time.time()-t0}")

    def learning_rate_update(self):
        self.progress_remaining = (self.total_timesteps-self.current_timesteps)/self.total_timesteps
        # self.learning_rate = self.lr_init * self.progress_remaining + self.min_learning_rate
        self.learning_rate = max(self.learning_rate * self.lr_decay_rate, self.min_learning_rate)
        print(f"progress_remaining: {self.progress_remaining}, learning_rate: {self.learning_rate}")

    def update_opponent_model(self):
        if self.opponent_update_timesteps and (self.current_timesteps % self.opponent_update_timesteps == 0):
            print(f"current_timesteps: {self.current_timesteps}")
            opponent_index = self._opponent_sampling()
            print(f"opponent_index: {opponent_index}")
            opponent_model_timesteps = self.current_timesteps - self.opponent_update_timesteps * opponent_index
            model_str = f"model_{int(opponent_model_timesteps / 10000)}w.zip"
            opponent_model_path = os.path.join(self.tensorboard_log, model_str)
            print(f"opponent_model_path: {opponent_model_path}")
            if os.path.exists(opponent_model_path):
                print(f"opponent_model_path: {opponent_model_path} exists")
                try:
                    self.opponent_model = self.PolicyModel.load(opponent_model_path)
                except IOError as e:
                    print(f"load model from self.model_path: {opponent_model_path} error")

    def reversi_model_train(self):
        n_check_point = int(np.ceil(self.total_timesteps/self.check_point_timesteps))
        for i in range(n_check_point):
            self.update_opponent_model()
            self.current_timesteps += self.check_point_timesteps
            model_str = f"model_{int(self.current_timesteps/10000)}w"
            save_model_path = os.path.join(self.tensorboard_log, model_str)
            self.reversi_model_train_step(self.check_point_timesteps, save_model_path)

            print(f"self.archive_path: {self.archive_path}")
            print(f"current_timesteps: {self.current_timesteps}, self.archive_timesteps: {self.archive_timesteps}")
            if self.archive_path and (self.current_timesteps % self.archive_timesteps == 0):
                source = save_model_path+'.zip'
                target = os.path.join(self.archive_path, model_str)
                try:
                    copyfile(source, target)
                except IOError as e:
                    print("Unable to copy file. %s" % e)

    def sb3_model_to_pth_model(self, PolicyModel, model_path):
        ppo_model = PolicyModel.load(model_path)
        ## 保存pth模型
        torch.save(ppo_model.policy, model_path + '.pth')

    def save_pth_model(self, model, save_model_path):
        torch.save(model.policy, save_model_path + '.pth')

    def load_pth_model(self, pth_model_path):
        pth_model = torch.load(pth_model_path)
        return pth_model

    def save_policy_model_state_dict(self, model, save_model_path):
        th.save(model.policy.state_dict(), save_model_path + '_state_dict.pt')


def transfer_policy_model_to_state_dict(model_path):
    model = PPO.load(model_path)
    th.save(model.policy.state_dict(), model_path + '_state_dict.pt')

def load_state_dict(policy_model, state_dict_path):
    policy_model.load_state_dict(torch.load(state_dict_path))

def task_args_parser(argv, usage=None):
    """
    :param argv:
    :return:
    """
    import argparse

    parser = argparse.ArgumentParser(prog='main', usage=usage, description='reversi model train')

    # env config
    parser.add_argument('--backbone', type=str, default='cnn', help="特征提取骨干网络， [cnn, resnet]")
    parser.add_argument('--n_channels', type=int, default=3, help="observation channels [3,4,5]  0: 黑棋位置， 1: 白棋位置， 2：player 颜色, 3： 对手落子位置， 4: 当前可合法落子位置")
    parser.add_argument('--n_steps', type=int, default=512, help="策略网络更新步数")
    parser.add_argument('--n_epochs', type=int, default=4, help="训练n_epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--lr_init', type=float, default=5e-4, help="初始学习率")
    parser.add_argument('--lr_decay_rate', type=float, default=0.95, help="学习率衰减因子")
    parser.add_argument('--board_size', type=int, default=8, help="棋盘尺寸")

    parser.add_argument('--n_envs', type=int, default=4, help="并行环境个数")
    parser.add_argument('--total_timesteps', type=int, default=100_0000, help="训练步数")
    parser.add_argument('--cp_timesteps', type=int, default=10_0000, help="检查点步数")
    parser.add_argument('--start_timesteps', type=int, default=0, help="本次训练开始index")
    parser.add_argument('--opponent_model_path', type=str, default='random', help='对手模型路径')
    parser.add_argument('--opponent_update_timesteps', type=int, default=1000000, help='对手模型更新步数')
    parser.add_argument('--opponent_prob_decay_rate', type=float, default=0.9, help='对手选择概率衰减因子')
    parser.add_argument('--opponent_window_size', type=int, default=100, help='对手选择窗口大小')

    parser.add_argument('--tensorboard_log', type=str, help='tensorboard_log路径')
    parser.add_argument('--greedy_rate', type=int, default=0, help="贪心奖励比率，大于0时使用贪心比率，值越大越即时奖励越大")
    parser.add_argument('--archive_timesteps', type=int, default=50_0000, help="存档训练步数")
    parser.add_argument('--archive_path', type=str, default='/content/drive/MyDrive/models', help='存档模型路径')

    args = parser.parse_args()
    return args


def run_train(argv):
    usage = '''
    example:
    python reversi_model_train.py --backbone cnn --n_channels 3 --lr_init 0.0003 --lr_decay_rate 0.99 --n_steps 512 --n_epochs 4 --batch_size 64 --board_size 8 --total_timesteps 10000000 --cp_timesteps 50000 --n_envs 4 --opponent_update_timesteps 50000 --opponent_prob_decay_rate 0.9 --opponent_window_size 100 --start_timesteps 0 --tensorboard_log cnn_selfplay
    
    python reversi_model_train.py --backbone cnn --n_channels 3 --lr_init 0.00007 --lr_decay_rate 0.99 --n_steps 512 --n_epochs 4 --batch_size 64 --board_size 8 --total_timesteps 50000000 --cp_timesteps 50000 --n_envs 4 --opponent_update_timesteps 50000 --opponent_prob_decay_rate 0.9 --opponent_window_size 100 --start_timesteps 12950000 --tensorboard_log cnn_selfplay

    python reversi_model_train.py --backbone resnet --n_channels 3 --lr_init 0.0002 --lr_decay_rate 0.99 --n_steps 512 --n_epochs 4 --batch_size 64 --board_size 8 --total_timesteps 50000000 --cp_timesteps 50000 --n_envs 4 --opponent_update_timesteps 50000 --opponent_prob_decay_rate 0.85 --opponent_window_size 100 --start_timesteps 0 --tensorboard_log resnet_selfplay  --archive_path /content/drive/MyDrive/models --archive_timesteps 500000

    debug
    python reversi_model_train.py --backbone cnn --n_channels 3 --lr_init 0.0003 --n_steps 512 --n_epochs 4 --batch_size 64 --board_size 8 --total_timesteps 200000 --cp_timesteps 10000 --n_envs 4 --opponent_update_timesteps 10000 --start_timesteps 0 --tensorboard_log ppo_8x8_cnn_debug001

    '''
    args = task_args_parser(argv, usage)
    base_path = '/content/drive/MyDrive/'
    base_path = 'models'

    backbone = args.backbone
    n_channels = args.n_channels
    lr_init = args.lr_init
    lr_decay_rate = args.lr_decay_rate
    n_steps = args.n_steps
    n_epochs = args.n_epochs
    batch_size = args.batch_size

    board_size = args.board_size
    n_envs = args.n_envs
    total_timesteps = args.total_timesteps
    start_timesteps = args.start_timesteps
    check_point_timesteps = args.cp_timesteps
    opponent_model_path = args.opponent_model_path
    opponent_update_timesteps = args.opponent_update_timesteps
    opponent_prob_decay_rate = args.opponent_prob_decay_rate
    opponent_window_size = args.opponent_window_size
    archive_timesteps = args.archive_timesteps
    archive_path = args.archive_path

    tensorboard_log = args.tensorboard_log
    if not tensorboard_log:
        tensorboard_log = f"ppo_{board_size}x{board_size}_{backbone}/"
    tensorboard_log = os.path.join(base_path, tensorboard_log)
    print(f"tensorboard_log: {tensorboard_log}")
    print(f"archive_path: {archive_path}")
    if not os.path.isdir(tensorboard_log):
        os.makedirs(tensorboard_log)
    model_path = os.path.join(tensorboard_log, "model")
    print(f"model_path: {model_path}")

    train_obj = ReversiModelTrain(board_size=board_size,
                                  backbone=backbone,
                                  n_channels=n_channels,
                                  lr_init=lr_init,
                                  lr_decay_rate=lr_decay_rate,
                                  n_steps=n_steps,
                                  n_epochs=n_epochs,
                                  batch_size=batch_size,
                                  total_timesteps=total_timesteps,
                                  start_timesteps=start_timesteps,
                                  check_point_timesteps=check_point_timesteps,
                                  n_envs=n_envs,
                                  model_path=model_path,
                                  opponent_model_path=opponent_model_path,
                                  opponent_update_timesteps=opponent_update_timesteps,
                                  opponent_prob_decay_rate=opponent_prob_decay_rate,
                                  opponent_window_size=opponent_window_size,
                                  tensorboard_log=tensorboard_log,
                                  archive_timesteps=archive_timesteps,
                                  archive_path=archive_path)

    t0 = time.time()
    train_obj.reversi_model_train()
    print(f"end time: {time2str(time.time())}")
    print(f"total train time: {time.time() - t0}")


if __name__ == '__main__':
    run_train(sys.argv[1:])
