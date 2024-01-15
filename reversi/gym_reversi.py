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

    def __init__(self, player_color, opponent, observation_type, board_size,
                 replace_invalid_action=True, only_game_finished_reward=False,
                 valid_place_reward=1, illegal_place_reward=-2, winner_reward=10
                 ):
        """
        Args:
            player_color: Stone color for the agent. Either 'black' or 'white'
            opponent: An opponent policy
            observation_type: State encoding
            board_size: size of the Reversi board
            replace_invalid_action: 是否将非法行动作替换为随机合法动作
            only_game_finished_reward: 是否只有游戏结束才能获得奖励
        """
        assert (
            isinstance(board_size, int) and board_size >= 1
        ), "Invalid board size: {}".format(board_size)
        self.board_size = board_size

        self.only_game_finished_reward=only_game_finished_reward
        self.replace_invalid_action=replace_invalid_action
        # 合法落子奖励
        self.valid_place_reward = valid_place_reward
        # 非法落子惩罚
        self.illegal_place_reward = illegal_place_reward
        # 最终游戏输赢的额外回报
        self.winner_reward = winner_reward

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

        assert observation_type in ["numpy3c"]
        self.observation_type = observation_type

        if self.observation_type != "numpy3c":
            raise error.Error(
                "Unsupported observation type: {}".format(self.observation_type)
            )

        # One action for each board position and resign and pass
        self.action_space = spaces.Discrete(self.board_size**2)
        observation, info = self.reset()
        self.observation_space = spaces.Box(
            np.zeros(observation.shape), np.ones(observation.shape), dtype=int
        )

        self.seed()

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
        # init board setting
        self.state = np.zeros((3, self.board_size, self.board_size), dtype=int)
        centerL = int(self.board_size / 2 - 1)
        centerR = int(self.board_size / 2)
        self.state[2, :, :] = 1
        self.state[2, (centerL) : (centerR + 1), (centerL) : (centerR + 1)] = 0
        self.state[0, centerR, centerL] = 1
        self.state[0, centerL, centerR] = 1
        self.state[1, centerL, centerL] = 1
        self.state[1, centerR, centerR] = 1
        self.to_play = ReversiEnv.BLACK
        self.possible_actions = ReversiEnv.get_possible_actions(
            self.state, self.to_play
        )
        self.done = False

        # Let the opponent play if it's not the agent's turn
        if self.player_color != self.to_play:
            a = self.opponent_policy(self.state)
            ReversiEnv.make_place(self.state, a, ReversiEnv.BLACK)
            self.to_play = ReversiEnv.WHITE
        info = self._get_info()
        return self.state, info

    def step(self, action):
        assert self.to_play == self.player_color
        truncated = False
        # If already terminal, then don't do anything
        if self.done:
            return self.state, 0.0, True, truncated, {"state": self.state}
        score_diff_before = ReversiEnv.get_score_diff(self.state)
        place_reward = 0

        # 有合法落子点时
        if len(self.possible_actions) > 0:
            # action 为非法落子
            if not ReversiEnv.valid_place(self.state, action, self.player_color):
                place_reward = self.illegal_place_reward
                if self.replace_invalid_action:
                    if len(self.possible_actions) > 0:
                        a_index = self.np_random.integers(len(self.possible_actions))
                        _action = self.possible_actions[a_index]
                        ReversiEnv.make_place(self.state, _action, self.player_color)
            # action 为合法落子
            else:
                place_reward = self.valid_place_reward
                ReversiEnv.make_place(self.state, action, self.player_color)
        # 没有合法落子点时，action 为‘pass’ 正确，否则惩罚
        else:
            if ReversiEnv.pass_place(self.board_size, action):
                place_reward = self.valid_place_reward
            else:
                place_reward = self.illegal_place_reward

        # Opponent play
        a = self.opponent_policy(self.state, 1 - self.player_color)

        # Making place if there are places left
        if a is not None:
            if ReversiEnv.pass_place(self.board_size, a):
                # 让对手无处可下
                # return self.state, 2.0, False, truncated, {"state": self.state}
                pass
            if not ReversiEnv.valid_place(self.state, a, 1 - self.player_color):
                # Automatic loss on illegal place
                # self.done = False
                # return self.state, 1.0, self.done, truncated, {"state": self.state}
                pass
            else:
                ReversiEnv.make_place(self.state, a, 1 - self.player_color)

        self.possible_actions = ReversiEnv.get_possible_actions(self.state, self.player_color)

        score_diff_after = ReversiEnv.get_score_diff(self.state)
        curr_score = score_diff_after - score_diff_before

        game_finish_reward = ReversiEnv.game_finished(self.state)
        if game_finish_reward == 1:
            reward = score_diff_after + self.winner_reward
        elif game_finish_reward == -1:
            reward = score_diff_after - self.winner_reward
        else:
            reward = curr_score

        if self.only_game_finished_reward:
            reward = game_finish_reward
        if self.player_color == ReversiEnv.WHITE:
            reward = -reward

        if not self.only_game_finished_reward:
            reward += place_reward

        self.done = game_finish_reward != 0
        info = self._get_info()
        return self.state, reward, self.done, truncated, info

    # def _reset_opponent(self):
    #     if self.opponent == 'random':
    #         self.opponent_policy = random_policy
    #     else:
    #         raise error.Error('Unrecognized opponent policy {}'.format(self.opponent))

    def render(self, mode="human", close=False):
        if close:
            return
        board = self.state
        outfile = StringIO() if mode == "ansi" else sys.stdout

        outfile.write(" " * 6)
        for j in range(board.shape[1]):
            outfile.write(" " + str(j + 1) + "  | ")
        outfile.write("\n")
        outfile.write(" " + "-" * (board.shape[1] * 7 - 1))
        outfile.write("\n")
        for i in range(board.shape[1]):
            outfile.write(" " + str(i + 1) + "  |")
            for j in range(board.shape[1]):
                if board[2, i, j] == 1:
                    outfile.write("  O  ")
                elif board[0, i, j] == 1:
                    outfile.write("  B  ")
                else:
                    outfile.write("  W  ")
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
        return {"state": self.state}

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
                if board[2, pos_x, pos_y] == 0:
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
                            actions.append(pos_x * d + pos_y)
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
        if board[2, coords[0], coords[1]] == 1:
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
                        board[2, nx, ny] = 0
                        board[player_color, nx, ny] = 1
                        board[opponent_color, nx, ny] = 0
                        nx += dx
                        ny += dy
                    board[2, pos_x, pos_y] = 0
                    board[player_color, pos_x, pos_y] = 1
                    board[opponent_color, pos_x, pos_y] = 0
        return board

    @staticmethod
    def coordinate_to_action(board, coords):
        return coords[0] * board.shape[-1] + coords[1]

    @staticmethod
    def action_to_coordinate(board, action):
        return action // board.shape[-1], action % board.shape[-1]

    @staticmethod
    def game_finished(board):
        # Returns 1 if player 1 wins, -1 if player 2 wins and 0 otherwise
        d = board.shape[-1]

        player_score_x, player_score_y = np.where(board[0, :, :] == 1)
        player_score = len(player_score_x)
        opponent_score_x, opponent_score_y = np.where(board[1, :, :] == 1)
        opponent_score = len(opponent_score_x)
        if player_score == 0:
            return -1
        elif opponent_score == 0:
            return 1
        else:
            free_x, free_y = np.where(board[2, :, :] == 1)
            if free_x.size == 0:
                if player_score > (d**2) / 2:
                    return 1
                elif player_score == (d**2) / 2:
                    return 1
                else:
                    return -1
            else:
                return 0
        return 0

    @staticmethod
    def get_score_diff(board):
        player_score_x, player_score_y = np.where(board[0, :, :] == 1)
        player_score = len(player_score_x)
        opponent_score_x, opponent_score_y = np.where(board[1, :, :] == 1)
        opponent_score = len(opponent_score_x)
        score_diff = player_score - opponent_score
        return score_diff


