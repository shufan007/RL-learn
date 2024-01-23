#!/usr/bin/env python3
from sdk import RandomPlayer
from sdk import Game
from myModule import MyPlayer

import importlib
import datetime

if __name__ == '__main__': 
    player1 = MyPlayer('O')
    # player1 = RandomPlayer('X')
    player2 = RandomPlayer('X')
    game = Game(player1, player2)

    start = datetime.datetime.now()
    result = game.run()
    end = datetime.datetime.now()
    spent = (end - start).total_seconds()
    resultStr = result["result"]
    print(f"{resultStr}, time spent = {spent} seconds")
