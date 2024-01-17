"""
Reversi with Gym Style
"""

from io import StringIO
import sys
import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium import error
from gymnasium.utils import seeding


def make_random_policy(np_random):
    def random_policy(state, player_color):
        possible_places = ReversiEnv.get_possible_actions(state, player_color)
        # No places left
        if len(possible_places) == 0:
            board_size = state.shape[1]
            # pass, 当没有合法落子位置时 pass
            return board_size**2
        a = np_random.integers(len(possible_places))
        return possible_places[a]

    return random_policy

class ReversiEnv(gym.Env):
    """
    Reversi environment. Play against a fixed opponent.
    """

    BLACK = 0
    WHITE = 1
    metadata = {"render.modes": ["ansi", "human"]}

    def __init__(self, opponent, is_train=True, board_size=8, player_color='black',
                 replace_invalid_action=False, only_game_finished_reward=True, verbose=0,
                 valid_place_reward=1, illegal_place_reward=-2, winner_reward=10
                 ):
        """
        Args:
            opponent: An opponent policy
            is_train: 是否为训练模式，如果是训练模式，player_color 可不设置
            board_size: size of the Reversi board
            player_color: Stone color for the agent. Either 'black' or 'white'
            replace_invalid_action: 是否将非法行动作替换为随机合法动作
            only_game_finished_reward: 是否只有游戏结束才能获得奖励
            TODO:
                1.设置二维 observation 为初始盘面状态，通过转换函数转换成模型输入self.state
                2.主玩家可支持2中颜色            [Done]
                3.self.observation 中添加可行位置
                4.模型保存及部署
        """
        assert (
            isinstance(board_size, int) and board_size >= 3
        ), "Invalid board size: {}".format(board_size)
        self.board_size = board_size
        self.is_train = is_train
        self.only_game_finished_reward = only_game_finished_reward
        self.replace_invalid_action = replace_invalid_action
        self.verbose = verbose
        # 合法落子奖励
        self.valid_place_reward = valid_place_reward
        # 非法落子惩罚
        self.illegal_place_reward = illegal_place_reward
        # 最终游戏输赢的额外回报
        self.winner_reward = winner_reward

        # 颜色的棋子，X-黑棋，O-白棋, '.'  # 未落子状态
        self.render_black = 'X'  # 'B'
        self.render_white = 'O'  # 'B'
        self.render_empty = '.'

        colormap = {
            "black": ReversiEnv.BLACK,
            "white": ReversiEnv.WHITE,
        }
        try:
            self.player_color = colormap[player_color]
        except KeyError:
            raise error.Error(
                "player_color must be 'black' or 'white', not {}".format(player_color)
            )

        self.opponent = opponent

        # One action for each board position and resign and pass
        self.action_space = spaces.Discrete(self.board_size**2)

        self.N_CHANNELS = 4
        self.HEIGHT = self.board_size
        self.WIDTH = self.board_size
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.N_CHANNELS, self.HEIGHT, self.WIDTH),
                                            dtype=np.uint8)
        self.seed()
        observation, info = self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        # Update the random policy if needed
        if isinstance(self.opponent, str):
            if self.opponent == "random":
                self.opponent_policy = make_random_policy(self.np_random)
            else:
                raise error.Error(
                    "Unrecognized opponent policy {}".format(self.opponent)
                )
        else:
            self.opponent_policy = self.opponent

        return [seed]

    def reset(self, seed=None, options=None):
        if self.verbose >= 1:
            print(f" --- episode start --- ")
        # init board setting
        # channels： 0: 黑棋位置， 1: 白棋位置， 2: 当前可合法落子位置，3：player 颜色
        self.observation = np.zeros((self.N_CHANNELS, self.board_size, self.board_size), dtype=int)

        # 训练模式下，每次棋盘重置时都随机生成主玩家颜色
        if self.is_train:
            # self.player_color = np.random.randint(2)
            self.player_color = self.np_random.integers(2)
            if self.verbose >= 1:
                print(f" --- self.player_color:{self.player_color} --- ")

        self.observation[3, :, :] = self.player_color

        centerL = int(self.board_size / 2 - 1)
        centerR = int(self.board_size / 2)
        # self.observation[2, :, :] = 1
        # self.observation[2, (centerL) : (centerR + 1), (centerL) : (centerR + 1)] = 0
        self.observation[0, centerR, centerL] = 1
        self.observation[0, centerL, centerR] = 1
        self.observation[1, centerL, centerL] = 1
        self.observation[1, centerR, centerR] = 1
        self.to_play = ReversiEnv.BLACK
        self.possible_actions = ReversiEnv.get_possible_actions(self.observation, self.to_play)
        self.done = False

        if self.verbose >= 1:
            print(f" --- start observation --- ")
            self.render("human")
        # print(f"self.observation 1:{self.observation}")
        # Let the opponent play if it's not the agent's turn
        if self.player_color != self.to_play:
            a = self.opponent_policy(self.observation, self.to_play)
            ReversiEnv.make_place(self.observation, a, ReversiEnv.BLACK)
            self.to_play = ReversiEnv.WHITE
            self.possible_actions = ReversiEnv.get_possible_actions(self.observation, self.to_play)
            if self.verbose >= 1:
                print(f" --- observation move by opponent --- ")
                self.render("human")
        # print(f"self.observation 2:{self.observation}")
        # 设置主玩家合法位置
        ReversiEnv.set_possible_actions_place(self.observation, self.possible_actions)
        # print(f"self.observation 3:{self.observation}")

        info = self._get_info()
        return self.observation, info

    def step(self, action):
        assert self.to_play == self.player_color
        truncated = False
        # If already terminal, then don't do anything
        if self.done:
            return self.observation, 0.0, True, truncated, {"state": self.observation}

        # score_diff_before = ReversiEnv.get_score_diff(self.observation)
        if self.verbose >= 1:
            print(f"step start: ")
            self.render("human")
            print(f"self.possible_actions: {self.possible_actions}")
            possible_actions_coords = [ReversiEnv.action_to_coordinate(self.observation, _action) for _action in self.possible_actions]
            print(f"possible_actions coords: {possible_actions_coords}")

        if self.verbose >= 1:
            coords = ReversiEnv.action_to_coordinate(self.observation, action)
            print(f" player action: {action}, coords: {coords}, self.player_color: {self.player_color}")
        # 有合法落子点时
        if len(self.possible_actions) > 0:
            # action 为非法落子
            if not ReversiEnv.valid_place(self.observation, action, self.player_color):
                if self.replace_invalid_action:
                    a_index = self.np_random.integers(len(self.possible_actions))
                    _action = self.possible_actions[a_index]
                    if self.verbose >= 1:
                        coords = ReversiEnv.action_to_coordinate(self.observation, _action)
                        print(f" replace_invalid_action, action: {_action}, coords: {coords}")
                    ReversiEnv.make_place(self.observation, _action, self.player_color)
                else:
                    return self.observation, -1.0, True, truncated, {"state": self.observation}
            # action 为合法落子
            else:
                ReversiEnv.make_place(self.observation, action, self.player_color)
            if self.verbose >= 1:
                self.render("human")
        # 没有合法落子点时，action 为‘pass’ 正确，否则结束
        else:
            if ReversiEnv.pass_place(self.board_size, action):
                if self.verbose >= 1:
                    print(f"=> pass_place, action valid, action: {action}")
                pass
            else:
                if self.replace_invalid_action:
                    action = self.observation.shape[-1] ** 2
                else:
                    return self.observation, -1.0, True, truncated, {"state": self.observation}

        # Opponent play
        a = self.opponent_policy(self.observation, 1 - self.player_color)
        if self.verbose >= 1:
            coords = ReversiEnv.action_to_coordinate(self.observation, a)
            print(f" opponent_policy action: {a}, coords: {coords}, player_color: {1 - self.player_color}")
        # Making place if there are places left
        if a is not None:
            if ReversiEnv.pass_place(self.board_size, a):
                pass
            elif not ReversiEnv.valid_place(self.observation, a, 1 - self.player_color):
                # Automatic loss on illegal place
                if self.verbose >= 1:
                    print(f" ** opponent_policy action invalid, action: {a}")
                return self.observation, 1.0, True, truncated, {"state": self.observation}
            else:
                ReversiEnv.make_place(self.observation, a, 1 - self.player_color)

        self.possible_actions = ReversiEnv.get_possible_actions(self.observation, self.player_color)
        # 设置主玩家合法位置
        ReversiEnv.set_possible_actions_place(self.observation, self.possible_actions)

        is_done, game_reward, score_diff = self.game_finished(self.observation)

        # player_score, opponent_score, score_diff = ReversiEnv.get_score_diff(self.observation)
        # # 黑子相对于本轮开始增加的棋子数
        # curr_score = score_diff_after - score_diff_before
        # if game_finish_reward == 1:
        #     reward = score_diff_after + self.winner_reward
        # elif game_finish_reward == -1:
        #     reward = score_diff_after - self.winner_reward
        # else:
        #     reward = curr_score

        if self.only_game_finished_reward:
            reward = game_reward

        if self.player_color == ReversiEnv.WHITE:
            reward = -reward

        self.done = is_done
        info = self._get_info()

        if self.verbose >= 1:
            self.render("human")
            print(f"step end")
            print(f"reward: {reward}, done: {self.done}")
            black_score, white_score, _score_diff = ReversiEnv.get_score_diff(self.observation)
            print(f"black_score: {black_score}, white_score: {white_score}, score_diff: {score_diff}")
            print(f"self.possible_actions: {self.possible_actions}")
            # possible_actions = [ReversiEnv.action_to_coordinate(self.observation, _action) for _action in self.possible_actions]
            # print(f"possible_actions coords: {possible_actions}")
            info = {"reward": reward,
                    "black_score": black_score,
                    "white_score": white_score,
                    "score_diff": score_diff
                    }
        return self.observation, reward, self.done, truncated, info

    # def _reset_opponent(self):
    #     if self.opponent == 'random':
    #         self.opponent_policy = random_policy
    #     else:
    #         raise error.Error('Unrecognized opponent policy {}'.format(self.opponent))

    def render(self, mode="human", close=False):
        if close:
            return
        board = self.observation
        outfile = StringIO() if mode == "ansi" else sys.stdout

        outfile.write(" " * 6)
        for j in range(board.shape[1]):
            outfile.write(" " + str(j) + "  | ")
        outfile.write("\n")
        outfile.write(" " + "-" * (board.shape[1] * 7 - 1))
        outfile.write("\n")
        for i in range(board.shape[1]):
            outfile.write(" " + str(i) + "  |")
            for j in range(board.shape[1]):
                if board[0, i, j] == 1:
                    outfile.write(f"  {self.render_black}  ")
                elif board[1, i, j] == 1:
                    outfile.write(f"  {self.render_white}  ")
                else:
                    outfile.write(f"  {self.render_empty}  ")
                outfile.write("|")
            outfile.write("\n")
            outfile.write(" " + "-" * (board.shape[1] * 7 - 1))
            outfile.write("\n")

        if mode != "human":
            return outfile

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_info(self):
        return {"state": self.observation}

    @staticmethod
    def pass_place(board_size, action):
        return action == board_size**2

    @staticmethod
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
        # if len(actions) == 0:
        #     actions = [d**2]
        return actions

    @staticmethod
    def valid_reverse_opponent(board, coords, player_color):
        """
        check whether there is any reversible places
        """
        d = board.shape[-1]
        opponent_color = 1 - player_color
        pos_x = coords[0]
        pos_y = coords[1]
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
                    return True
        return False

    @staticmethod
    def valid_place(board, action, player_color):
        d = board.shape[-1]
        if action >= d**2:
            return False
        coords = ReversiEnv.action_to_coordinate(board, action)
        # check whether there is any empty places
        if (board[0, coords[0], coords[1]] == 0) and (board[1, coords[0], coords[1]] == 0):
            # check whether there is any reversible places
            if ReversiEnv.valid_reverse_opponent(board, coords, player_color):
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def make_place(board, action, player_color):
        coords = ReversiEnv.action_to_coordinate(board, action)

        d = board.shape[-1]
        opponent_color = 1 - player_color
        pos_x = coords[0]
        pos_y = coords[1]

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
                    nx = pos_x + dx
                    ny = pos_y + dy
                    while board[opponent_color, nx, ny] == 1:
                        # board[2, nx, ny] = 0
                        board[player_color, nx, ny] = 1
                        board[opponent_color, nx, ny] = 0
                        nx += dx
                        ny += dy
                    # board[2, pos_x, pos_y] = 0
                    board[player_color, pos_x, pos_y] = 1
                    board[opponent_color, pos_x, pos_y] = 0
        return board

    @staticmethod
    def set_possible_actions_place(board, possible_actions, channel_index=2):
        board[channel_index, :, :] = 0
        # possible_actions = ReversiEnv.get_possible_actions(board, player_color)
        possible_actions_coords = [ReversiEnv.action_to_coordinate(board, _action) for _action in possible_actions]
        for pos_x, pos_y in possible_actions_coords:
            board[channel_index, pos_x, pos_y] = 1
        return board

    @staticmethod
    def coordinate_to_action(board, coords):
        return coords[0] * board.shape[-1] + coords[1]

    @staticmethod
    def action_to_coordinate(board, action):
        return action // board.shape[-1], action % board.shape[-1]

    def game_finished(self, board):
        # Returns 1 if player 1 wins, -1 if player 2 wins and 0 otherwise
        d = board.shape[-1]
        is_done = False   # 游戏是否结束
        player_score_x, player_score_y = np.where(board[0, :, :] == 1)
        player_score = len(player_score_x)
        opponent_score_x, opponent_score_y = np.where(board[1, :, :] == 1)
        opponent_score = len(opponent_score_x)
        score_diff = player_score - opponent_score
        if player_score == 0:
            is_done = True
        elif opponent_score == 0:
            is_done = True
        else:
            # free_x, free_y = np.where(board[2, :, :] == 1)
            if (player_score + opponent_score) == d**2:
                # 棋盘已被占满
                is_done = True
            else:
                # 棋盘仍有空位，检查是否双方都有可行位置，只要有一方仍有可行位置，reward为0
                if len(self.possible_actions) > 0:
                    is_done = False
                else:
                    # 检查对手是否可行位置
                    _possible_actions = ReversiEnv.get_possible_actions(self.observation, 1-self.player_color)
                    if len(_possible_actions) > 0:
                        is_done = False
                    else:
                        is_done = True
        reward = 0
        if is_done:
            if score_diff > 0:
                reward = 1
            elif score_diff == 0:
                reward = 0
            else:
                reward = -1
        return is_done, reward, score_diff

    @staticmethod
    def get_score_diff(board):
        # 统计黑子多于白子的个数
        player_score_x, player_score_y = np.where(board[0, :, :] == 1)
        player_score = len(player_score_x)
        opponent_score_x, opponent_score_y = np.where(board[1, :, :] == 1)
        opponent_score = len(opponent_score_x)
        score_diff = player_score - opponent_score
        return player_score, opponent_score, score_diff

