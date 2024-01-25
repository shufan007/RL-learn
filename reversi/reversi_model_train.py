import os
import sys
import time
import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from gym_reversi import ReversiEnv
# from utils import time2str
from datetime import datetime


def time2str(timestamp):
    d = datetime.fromtimestamp(timestamp)
    timestamp_str = d.strftime('%Y-%m-%d %H:%M:%S')
    return timestamp_str


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    # observation_space: spaces.Box
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256,
                 net_arch=[32, 64, 128], kernel_size=3, stride=1, padding='same', is_batch_norm=False):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        stride_list = stride
        if isinstance(stride, int):
            stride_list = [stride] * len(net_arch)

        kernel_size_list = kernel_size
        if isinstance(kernel_size, int):
            kernel_size_list = [kernel_size] * len(net_arch)

        # self.cnn = nn.Sequential(
        #     nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )

        conv_module_list = nn.ModuleList()

        in_channels = n_input_channels
        for i in range(len(net_arch)):
            out_channels = net_arch[i]
            kernel_size = kernel_size_list[i]
            stride = stride_list[i]
            conv_block = [
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding),
                # nn.BatchNorm2d(num_features=out_channels),
                # nn.ReLU(),
            ]

            if is_batch_norm:
                conv_block.append(nn.BatchNorm2d(num_features=out_channels))
            conv_block.append(nn.ReLU())

            conv_module_list.extend(conv_block)
            in_channels = out_channels

        conv_module_list.extend([nn.Flatten()])
        self.cnn = nn.Sequential(* conv_module_list)

        # Compute shape by doing one forward pass
        # 自动推导 n_flatten
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        # 手动计算
        # n_flatten = observation_shape[1] * observation_shape[2] * net_arch[-1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class ReversiModelTrain(object):
    def __init__(self, board_size=8, check_point_timesteps=100000, n_envs=16, model_path=None,
                 opponent_model_path="random", tensorboard_log=None, verbose=0):
        self.board_size = board_size
        self.check_point_timesteps = check_point_timesteps
        self.n_envs = n_envs
        self.model_path = model_path
        self.opponent_model_path = opponent_model_path
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.PolicyModel = PPO

    def reversi_model_train_step(self, check_point_timesteps, save_model_path=None):
        if self.opponent_model_path != "random":
            opponent_model = self.PolicyModel.load(self.opponent_model_path)
        else:
            opponent_model = "random"

        env = ReversiEnv(opponent=opponent_model, is_train=True, board_size=self.board_size,
                         greedy_rate=0, verbose=self.verbose)

        vec_env = env
        if self.n_envs > 1:
            # multi-worker training (n_envs=4 => 4 environments)
            vec_env = make_vec_env(ReversiEnv, n_envs=self.n_envs, seed=None,
                                   env_kwargs={
                                       "opponent": opponent_model,
                                       "is_train": True,
                                       "board_size": self.board_size,
                                       "greedy_rate": 0,
                                       "verbose": self.verbose},
                                )

        # set policy model configs
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=256,
                                           # net_arch=[64, 128, 256],
                                           net_arch=[64, 128, 128],
                                           # net_arch=[32, 64, 64],
                                           kernel_size=3,
                                           stride=1,
                                           padding='same',
                                           is_batch_norm=False),
            net_arch=[256, 256],
            normalize_images=False
        )

        try:
            model = self.PolicyModel.load(self.model_path, env=vec_env)
        except Exception:
            print(f"load model from self.model_path: {self.model_path} error")
            model = PPO('MlpPolicy', vec_env,
                          policy_kwargs=policy_kwargs,
                          learning_rate=2e-4,  # learning_rate=2.5e-4,
                          ent_coef=0.01,
                          n_steps=64, # n_steps=128,
                          n_epochs=4,
                          batch_size=64, # batch_size=256,
                          gamma=0.9,
                          gae_lambda=0.8,
                          clip_range=0.2,
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

    def reversi_model_train(self, total_timesteps, timesteps_start_index=0):
        n_check_point = int(np.ceil(total_timesteps/self.check_point_timesteps))
        _current_timestemps = min(total_timesteps, self.check_point_timesteps)
        _current_timestemps += timesteps_start_index
        for i in range(n_check_point):
            model_str = f"model_{int(_current_timestemps/10000)}w"
            save_model_path = os.path.join(self.tensorboard_log, model_str)
            _current_timestemps += self.check_point_timesteps
            self.reversi_model_train_step(self.check_point_timesteps, save_model_path)

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

    def game_play(self, model_path, opponent_model_path="random", player_color='black', max_round=100, verbose=0):

        # opponent_model = "random"
        # opponent_model = PPO.load("models/Reversi_ppo/model4x4_50w")
        # opponent_model = PPO.load("models/Reversi_ppo/model")
        if self.opponent_model_path != "random":
            opponent_model = PPO.load(opponent_model_path)
        else:
            opponent_model = "random"

        env = ReversiEnv(opponent=opponent_model, is_train=False, board_size=self.board_size, player_color=player_color,
                         verbose=verbose)

        model = PPO.load(model_path)
        # model = PPO.load("models/Reversi_ppo/model4x4_50w")

        total_round = 0
        total_win = 0
        total_failure = 0
        total_equal = 0

        t0 = time.time()
        obs, info = env.reset()
        while total_round < max_round:
            action, _states = model.predict(obs, deterministic=False)
            obs, rewards, dones, truncated, info = env.step(action)

            #     print(f"---- round:{total_round} --------")
            #     print(f"action: {action}")
            #     env.render("human")

            if dones:
                print(f"---- round:{total_round} --------")
                #         env.render("human")
                obs, info = env.reset()
                total_round += 1
                if rewards > 0:
                    total_win += 1
                elif rewards < 0:
                    total_failure += 1
                else:
                    total_equal += 1

                print(f"total_win:{total_win}, total_failure: {total_failure}, total_equal:{total_equal}\n")

        # print(f"total_win:{total_win}, total_failure: {total_failure}")
        print(f"train time: {time.time() - t0}")


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
    parser.add_argument('--board_size', type=int, default=8, help="棋盘尺寸")
    parser.add_argument('--n_envs', type=int, default=4, help="并行环境个数")
    parser.add_argument('--total_timesteps', type=int, default=10_0000, help="训练步数")
    parser.add_argument('--cp_timesteps', type=int, default=10_0000, help="检查点步数")
    parser.add_argument('--start_index', type=int, default=0, help="本次训练开始index")
    parser.add_argument('--opponent_model_path', type=str, default='random', help='对手模型路径')
    parser.add_argument('--greedy_rate', type=int, default=0, help="贪心奖励比率，大于0时使用贪心比率，值越大越即时奖励越大")

    args = parser.parse_args()
    return args


def run_train(argv):
    usage = '''
    example:
    python reversi_model_train.py --board_size 8 --total_timesteps 1000000 --cp_timesteps 200000 --n_envs 8 --opponent_model_path random --start_index 1000000
    python reversi_model_train.py --board_size 8 --total_timesteps 1000000 --cp_timesteps 200000 --n_envs 8 --opponent_model_path random --start_index 1000000

    '''
    args = task_args_parser(argv, usage)
    base_path = '/content/drive/MyDrive/'

    board_size = args.board_size
    n_envs = args.n_envs
    total_timesteps = args.total_timesteps
    start_index = args.start_index
    check_point_timesteps = args.cp_timesteps
    opponent_model_path = args.opponent_model_path
    tensorboard_log = f"models/ppo_{board_size}x{board_size}_cnn/"
    # tensorboard_log = os.path.join(base_path, tensorboard_log)

    if not os.path.isdir(tensorboard_log):
        os.makedirs(tensorboard_log)
    model_path = os.path.join(tensorboard_log, "model")

    train_obj = ReversiModelTrain(board_size=board_size,
                                  check_point_timesteps=check_point_timesteps,
                                  n_envs=n_envs,
                                  model_path=model_path,
                                  opponent_model_path=opponent_model_path,
                                  tensorboard_log=tensorboard_log)

    t0 = time.time()
    train_obj.reversi_model_train(total_timesteps, start_index)
    print(f"end time: {time2str(time.time())}")
    print(f"total train time: {time.time() - t0}")


if __name__ == '__main__':
    run_train(sys.argv[1:])
