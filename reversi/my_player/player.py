
import os
import numpy as np
import random
import torch


class MyReversiPlayer:
    """
    """
    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """
        self.color = color
        self.colormap = {
            "X": 0,
            "O": 1,
        }
        self.empty = '.'  # 未落子状态
        self.board_size = 8
        self.player_color = self.colormap[color]
        self.observation_channels = 4

        current_path = os.path.dirname(os.path.abspath(__file__))
        # print(f"current_path: {current_path}")
        model_path = os.path.join(current_path, 'model.pth')
        self.model = self._get_model(model_path)

    def _get_model(self, model_path):
        model = torch.load(model_path)
        return model

    def get_move(self, board):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        observation, possible_actions = self._get_observation(board)
        _action, _states = self.model.predict(observation, deterministic=True)
        if len(possible_actions) > 0:
            if _action not in possible_actions:
                _action = random.choice(possible_actions)
        elif _action < self.board_size**2:
            _action = self.board_size**2

        # print(f"_action:{_action}")
        _action_coords = self.action_to_coordinate(_action)
        action = self.num_board(_action_coords)
        # print(f"action:{action}")
        return action

    def board_num(self, action):
        """
        棋盘坐标转化为数字坐标
        :param action:棋盘坐标，比如A1
        :return:数字坐标，比如 B3 --->(2,1)
        """
        y = 'ABCDEFGH'.index(action[0].upper())
        x = '12345678'.index(action[1].upper())
        return x, y

    def num_board(self, action):
        """
        数字坐标转化为棋盘坐标
        :param action:数字坐标 ,比如(0,0)
        :return:棋盘坐标，比如 （0,0）---> A1
        """
        row, col = action
        l = [0, 1, 2, 3, 4, 5, 6, 7]
        if col in l and row in l:
            return chr(ord('A') + col) + str(row + 1)

    @staticmethod
    def get_possible_actions(observation, player_color):
        actions = []
        d = observation.shape[-1]
        opponent_color = 1 - player_color
        for pos_x in range(d):
            for pos_y in range(d):
                if observation[0, pos_x, pos_y] or observation[1, pos_x, pos_y]:
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
                        while observation[opponent_color, nx, ny] == 1:
                            tmp_nx = nx + dx
                            tmp_ny = ny + dy
                            if tmp_nx not in range(d) or tmp_ny not in range(d):
                                break
                            n += 1
                            nx += dx
                            ny += dy
                        if n > 0 and observation[player_color, nx, ny] == 1:
                            action = pos_x * d + pos_y
                            if action not in actions:
                                actions.append(action)
        return actions

    def get_possible_actions_from_boad(self, board):
        action_list = list(board.get_legal_actions(self.color))
        possible_actions = []
        for _action in action_list:
            action_coords = self.board_num(_action)
            action = action_coords[0]*self.board_size + action_coords[1]
            possible_actions.append(action)
        return possible_actions

    def action_to_coordinate(self, action):
        return action // self.board_size, action % self.board_size

    def set_possible_actions_place(self, observation, possible_actions, channel_index=2):
        observation[channel_index, :, :] = 0
        possible_actions_coords = [self.action_to_coordinate(_action) for _action in possible_actions]
        for pos_x, pos_y in possible_actions_coords:
            observation[channel_index, pos_x, pos_y] = 1
        return observation

    def _get_observation(self, board):
        # transform board to observation
        # channels： 0: 黑棋位置， 1: 白棋位置， 2: 当前可合法落子位置，3：player 颜色
        observation = np.zeros((self.observation_channels, self.board_size, self.board_size), dtype=np.uint8)
        observation[3, :, :] = self.player_color
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board._board[i][j] != self.empty:
                    index = self.colormap[board._board[i][j]]
                    observation[index, i, j] = 1
        possible_actions = MyReversiPlayer.get_possible_actions(observation, self.player_color)
        # print(f"possible_actions ---1---: {possible_actions}")
        possible_actions = self.get_possible_actions_from_boad(board)
        # print(f"possible_actions ---2---: {possible_actions}")

        # 设置主玩家合法位置
        self.set_possible_actions_place(observation, possible_actions)
        return observation, possible_actions

