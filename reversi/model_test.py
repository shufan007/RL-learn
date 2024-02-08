import os
import sys
import time
from shutil import copyfile
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib.ppo_mask import MaskablePPO
from gym_reversi import ReversiEnv


def get_possible_actions(board, player_color):
    actions = []
    d = board.shape[-1]
    opponent_color = 1 - player_color
    for pos_x in range(d):
        for pos_y in range(d):
            if board[0, pos_x, pos_y] or board[1, pos_x, pos_y]:
                continue
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx = pos_x + dx
                    ny = pos_y + dy
                    n = 0
                    if nx not in range(d) or ny not in range(d):
                        continue
                    while board[opponent_color, nx, ny] == 1:
                        tmp_nx = nx + dx
                        tmp_ny = ny + dy
                        if tmp_nx not in range(d) or tmp_ny not in range(d):
                            break
                        n += 1
                        nx += dx
                        ny += dy
                    if n > 0 and board[player_color, nx, ny] == 1:
                        action = pos_x * d + pos_y
                        if action not in actions:
                            actions.append(action)
    return actions

def set_possible_actions_place(board, possible_actions, channel_index=2):
    board[channel_index, :, :] = 0
    # possible_actions = ReversiEnv.get_possible_actions(board, player_color)
    possible_actions_coords = [ReversiEnv.action_to_coordinate(board, _action) for _action in possible_actions]
    for pos_x, pos_y in possible_actions_coords:
        board[channel_index, pos_x, pos_y] = 1
    return board

def get_test_observation(board_size=4, player_color=0):
    # init board setting
    N_CHANNELS = 4
    # channels： 0: 黑棋位置， 1: 白棋位置， 2: 当前可合法落子位置，3：player 颜色
    observation = np.zeros((N_CHANNELS, board_size, board_size), dtype=int)

    observation[3, :, :] = player_color

    centerL = int(board_size / 2 - 1)
    centerR = int(board_size / 2)
    # self.observation[2, :, :] = 1
    # self.observation[2, (centerL) : (centerR + 1), (centerL) : (centerR + 1)] = 0
    observation[0, centerR, centerL] = 1
    observation[0, centerL, centerR] = 1
    observation[1, centerL, centerL] = 1
    observation[1, centerR, centerR] = 1
    possible_actions = get_possible_actions(observation, player_color)

    # 设置主玩家合法位置
    set_possible_actions_place(observation, possible_actions)

    return observation


def action_to_coordinate(board, action):
    return action // board.shape[-1], action % board.shape[-1]


def valid_action_mask(board_size=8, possible_actions=[]):
    valid_actions = np.zeros((board_size**2, ), dtype=np.uint8)
    for idx in possible_actions:
        valid_actions[idx] = 1
    return valid_actions


def game_play(model_path, opponent_model_path="random", n_channels=3, max_round=500,
              board_size=8, verbose=0, deterministic=True):
    PolicyModel = MaskablePPO

    model = PolicyModel.load(model_path)

    if opponent_model_path != "random":
        opponent_model = PolicyModel.load(opponent_model_path)
    else:
        opponent_model = "random"

    t0 = time.time()

    n = 0
    _win = 0
    _failure = 0
    _equal = 0

    is_train = not deterministic
    env = ReversiEnv(opponent=opponent_model, n_channels=n_channels, is_train=is_train,
                     board_size=board_size, verbose=verbose)

    obs, info = env.reset()
    while n < max_round:
        possible_actions = get_possible_actions(obs, player_color=env.player_color)
        action_masks = valid_action_mask(board_size=board_size, possible_actions=possible_actions)
        action, _states = model.predict(obs, deterministic=deterministic, action_masks=action_masks)
        action = int(action)
        obs, rewards, dones, truncated, info = env.step(action)
        #     print(f"---- round:{n} --------")
        #     print(f"action: {action}")
        #     env.render("human")
        if dones:
            print(f"---- round:{n} --------")
            #         env.render("human")
            obs, info = env.reset()
            n += 1
            if rewards > 0:
                _win += 1
            elif rewards < 0:
                _failure += 1
            else:
                _equal += 1
            print(f"total_win:{_win}, total_failure: {_failure}, total_equal:{_equal}\n")

    _win_rate = round(_win/(_win + _failure) * 10000)/10000
    print(f"win: {_win}, failure: {_failure}, equal_cnt: {_equal}, win_rate: {_win_rate}\n")
    print(f" win_rate: {_win_rate}\n")
    print(f"total time: {time.time() - t0}")


