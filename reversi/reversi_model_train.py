import os
import sys
import time
import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3 import PPO
from gym_reversi import ReversiEnv
from utils import time2str


class ReversiEnvWrapper(gym.Wrapper):
    def __init__(self, opponent="random", is_train=True, board_size=8, greedy_rate=0, verbose=0):
        env = ReversiEnv(opponent=opponent, is_train=is_train, board_size=board_size, 
                         greedy_rate=greedy_rate, verbose=verbose)
        super().__init__(env)
        self.env = env

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=None)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


# 运行报错，先不用
def make_vectorized_env(env_wrapper, dumm, n):
    if dumm:
        env = DummyVecEnv([env_wrapper] * n)
    else:
        env = SubprocVecEnv([env_wrapper] * n)
    return env


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

            # vec_env = make_vectorized_env(ReversiEnvWrapper, dumm=False, n=8)

        try:
            model = self.PolicyModel.load(self.model_path, env=vec_env)
        except Exception:
            print(f"load model from self.model_path: {self.model_path} error")
            model = PPO('MlpPolicy', vec_env,
                          policy_kwargs=dict(net_arch=[256, 256]),
                          learning_rate=2.5e-4,  # learning_rate=2.5e-4,
                          ent_coef=0.01,
                          n_steps=64, # n_steps=128,
                          n_epochs=4,
                          batch_size=32, # batch_size=256,
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


def task_args_parser(argv, usage=None):
    """
    :param argv:
    :return:
    """
    import argparse

    parser = argparse.ArgumentParser(prog='main', usage=usage, description='reversi model train')

    # env config
    parser.add_argument('--board_size', type=int, default=8, help="棋盘尺寸")
    parser.add_argument('--n_envs', type=int, default=8, help="并行环境个数")
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

    board_size = args.board_size
    n_envs = args.n_envs
    total_timesteps = args.total_timesteps
    start_index = args.start_index
    check_point_timesteps = args.cp_timesteps
    opponent_model_path = args.opponent_model_path
    tensorboard_log = f"models/Reversi_ppo_{board_size}x{board_size}/"
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
