#!/usr/bin/env python3
from sdk import RandomPlayer
from sdk import Game
from deployModule import MyPlayer

import importlib
import datetime


def do_play(player1, player2):
    game = Game(player1, player2)
    start = datetime.datetime.now()
    result = game.run()
    end = datetime.datetime.now()
    spent = (end - start).total_seconds()
    resultStr = result["result"]
    print(f"{resultStr}, time spent = {spent} seconds")
    return result['winner']


def play_test(n):
    win_cnt = 0
    for i in range(n):
        player1 = MyPlayer('X')
        player2 = RandomPlayer('O')
        winner = do_play(player1, player2)
        if winner == 'player1':
            win_cnt += 1
    for i in range(n):
        player1 = MyPlayer('O')
        player2 = RandomPlayer('X')
        winner = do_play(player1, player2)
        if winner == 'player1':
            win_cnt += 1
    print(f"total win: {win_cnt}/({2*n}), win_rate: {win_cnt/(2*n)}")


if __name__ == '__main__':
    play_test(10)

