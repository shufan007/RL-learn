# !/usr/bin/Anaconda3/python
# -*- coding: utf-8 -*-
import datetime
from copy import deepcopy

# from func_timeout import FunctionTimedOut
# from func_timeout import func_timeout

from .board import Board
from .player import Player


class Game:
    def __init__(self, player1: Player, player2: Player, timeout = 150):
        """初始化对局

        如果player1与player同色,则player1为黑棋,先手

        :param player1:(Player) 玩家1
        :param player2:(Player) 玩家2
        """
        self.board = Board()  # 棋盘
        # 定义棋盘上当前下棋棋手，先默认是 None
        self.current_player = None
        if player1.color == player2.color:
            player1.color = 'X'
            player2.color = 'O'

        if player1.color == 'X':
            self.black_player = player1  # 黑棋一方
            self.white_player = player2
        else:
            self.black_player = player2  # 黑棋一方
            self.white_player = player1
        self.player1_color = player1.color
        
        self._move_timeout = 6
        self._cum_timeout = timeout
        
    def get_player_by_color(self, color: str) -> str:
        player_name = self.player1_color == color and "player1" or "player2"
        color_name = color == 'X' and 'black' or "white"
        return f"{player_name}({color_name})"
    
        
    def switch_player(self):
        """ 游戏过程中切换玩家

        如果当前玩家是 None 或者 white_player，则返回 黑棋一方 black_player;
        """
        if not self.current_player or self.current_player == self.white_player:
            self.current_player = self.black_player
        else:
            self.current_player = self.white_player

    def print_winner(self, winner):
        """
        打印赢家
        :param winner: [0,1,2] 分别代表黑棋获胜、白棋获胜、平局3种可能。
        :return:
        """
        print(['黑棋获胜!', '白棋获胜!', '平局'][winner])

    def force_loss(self, type='timeout'):
        """落子3个不合符规则和超时则结束游戏,修改棋盘也是输

        :param type:(str) 输的原因, type in ['timeout','board','legal','exception']
        :return: 赢家（0,1）,棋子差 0
        """

        if self.current_player == self.black_player:
            win_color = '白棋 - O'
            loss_color = '黑棋 - X'
            winner = 1
        else:
            win_color = '黑棋 - X'
            loss_color = '白棋 - O'
            winner = 0
        
        winner = self.get_player_by_color(win_color)
        loser = self.get_player_by_color(loss_color)
        error_map = {
            'timeout': f'{loser} think more than {self._move_timeout}s, {winner} wins',
            'legal': f'{loser} violate move rules, {winner} wins',
            'board': f'{loser} modifies the board, {winner} wins',
            'exception': f'{loser} error due to exception, {winner} wins'
        }
        error = error_map.has_key(type) and error_map(type) or ""
        diff = 64

        return winner, diff, error

    def run(self, timeout=60):
        """ 运行游戏 """
        # 定义统计双方下棋时间
        total_time = {"X": 0, "O": 0}
        # 定义双方每一步下棋时间
        step_time = {"X": 0, "O": 0}
        # 初始化胜负结果和棋子差
        winner = None
        diff = -1
        execption = None  # 判输的Exception
        boards = []  # 每步的棋盘
        error = ''  # 判输的原因

        # 棋盘初始化
        # self.board.display(step_time, total_time)
        while True:
            # 切换当前玩家,如果当前玩家是 None 或者白棋 white_player，则返回黑棋 black_player;
            #  否则返回 white_player。
            self.switch_player()
            start_time = datetime.datetime.now()
            # 当前玩家对棋盘进行思考后，得到落子位置
            # 判断当前下棋方
            color = "X" if self.current_player == self.black_player else "O"
            # 获取当前下棋方合法落子位置
            legal_actions = list(self.board.get_legal_actions(color))
            # print("%s合法落子坐标列表："%color,legal_actions)
            if len(legal_actions) == 0:
                # 判断游戏是否结束
                if self.game_over():
                    # 游戏结束，双方都没有合法位置
                    winner, diff = self.board.get_winner()  # 得到赢家 0,1,2
                    break
                else:
                    # 另一方有合法位置,切换下棋方
                    continue

            board = deepcopy(self.board._board)

            # legal_actions 不等于 0 则表示当前下棋方有合法落子位置
            try:
                for i in range(0, 3):
                    # 获取落子位置
                    timeout = max(self._cum_timeout - total_time[color], self._move_timeout)
                    #action = func_timeout(
                    #    timeout,
                    #    self.current_player.get_move,
                    #    kwargs={'board': self.board}
                    #)
                    action = self.current_player.get_move(self.board)

                    # 如果 action 是 Q 则说明人类想结束比赛
                    if action == "Q":
                        # 说明人类想结束游戏，即根据棋子个数定输赢。
                        break
                    if action not in legal_actions:
                        # 判断当前下棋方落子是否符合合法落子,如果不合法,则需要对方重新输入
                        continue
                    else:
                        # 落子合法则直接 break
                        break
                else:
                    # 落子3次不合法，结束游戏！
                    winner, diff, error = self.force_loss(type='legal')
                    execption = RuntimeError("invalid move after 3 retires!")
                    break
            #except FunctionTimedOut as e:
                # 落子超时，结束游戏
            #    execption = e
            #    winner, diff, error = self.force_loss(type='timeout')
            #    break

            except Exception as e:
                execption = e
                winner, diff, error = self.force_loss(type='exception')
                break
            end_time = datetime.datetime.now()
            if board != self.board._board:
                # 修改棋盘，结束游戏！
                winner, diff, error = self.force_loss(type='board')
                break
            if action == "Q":
                # 说明人类想结束游戏，即根据棋子个数定输赢。
                winner, diff = self.board.get_winner()  # 得到赢家 0,1,2
                break

            if action is None:
                continue
            else:
                # 统计一步所用的时间
                es_time = (end_time - start_time).total_seconds()
                if es_time > timeout:
                    # 该步超过60秒则结束比赛。
                    winner, diff, error = self.force_loss(type='timeout')
                    break

                # 当前玩家颜色，更新棋局
                self.board._move(action, color, append_to_history=True)

                # 统计每种棋子下棋所用总时间
                if self.current_player == self.black_player:
                    # 当前选手是黑棋一方
                    step_time["X"] = es_time
                    total_time["X"] += es_time
                else:
                    step_time["O"] = es_time
                    total_time["O"] += es_time

                # 判断游戏是否结束
                if self.game_over():
                    # 游戏结束
                    winner, diff = self.board.get_winner()  # 得到赢家 0,1,2
                    break

        self.board.display(step_time, total_time)

        # 返回'black_win','white_win','draw',棋子数差
        if winner is not None and diff > -1:
            result = {0: 'black_win', 1: 'white_win', 2: 'draw'}[winner]
            
            # Identify players
            black_player = self.player1_color == "X" and "player1" or "player2"
            white_player = self.player1_color == "O" and "player1" or "player2"
            if diff != 0:
                winner_name = winner == 0 and black_player or white_player
                win_reason = f"win by {diff} moves"
            else:
                # Check who is faster?
                winner_name = total_time['X'] < total_time['O'] and black_player or white_player
                win_reason = f"draw but it is faster by {abs(total_time['X'] - total_time['O'])} seconds"
            
            def get_player_by_color(color):
                return self.player1_color == color and "player1" or "player2"
            
            time_info = {get_player_by_color(color): total_time[color] for color in ["X", "O"]}
                
            def get_result(winner, diff, error):
                if error:
                    return error
                else:
                    result_map = {
                        0: '黑棋获胜',
                        1: '白棋获胜',
                        2: '平局',
                    }
                    return "{}, 领先棋子数：{}".format(result_map[winner], diff)

            return {
                'winner': winner_name,
                'result': f"{winner_name} wins with {win_reason}",
                'diff': diff,
                'exception': execption,
                'boards': self.board.move_history,
                'error': error,
                'time_info': time_info,
            }

    def game_over(self):
        """ 判断游戏是否结束

        :return: True/False 游戏结束/游戏没有结束
        """

        # 根据当前棋盘，判断棋局是否终止
        # 如果当前选手没有合法下棋的位子，则切换选手；如果另外一个选手也没有合法的下棋位置，则比赛停止。
        b_list = list(self.board.get_legal_actions('X'))
        w_list = list(self.board.get_legal_actions('O'))

        is_over = len(b_list) == 0 and len(w_list) == 0  # 返回值 True/False

        return is_over
