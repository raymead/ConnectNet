import abc
import os.path
import random
from typing import Optional

import numpy as np
import torch

import connect_four
import connect_net
import mcts
import utils


class Strategy(abc.ABC):
    @abc.abstractmethod
    def setup(self, game_cache: connect_four.GameCache) -> None:
        pass

    @abc.abstractmethod
    def get_move(self, state: torch.Tensor, rep: str, move: int) -> int:
        pass


class NetworkStrategy(Strategy):
    def __init__(self, nnet: torch.nn.Module, c: float, num_mcts_sims: int, random_moves: int) -> None:
        self.c = c
        self.num_mcts_sims = num_mcts_sims
        self.random_moves = random_moves

        self.nnet = nnet

        self.game_cache = None
        self.network_cache = None
        self.mcts = None

    def setup(self, game_cache: connect_four.GameCache):
        self.game_cache = game_cache
        self.network_cache = connect_net.NetworkCache(nnet=self.nnet)
        self.mcts: mcts.NetworkMCTS = mcts.NetworkMCTS(
            c=self.c,
            game_cache=self.game_cache, network_cache=self.network_cache,
        )

    def get_move(self, state: torch.Tensor, rep: str, move: int) -> int:
        self.simulate(state=state, rep=rep)
        if move <= self.random_moves * 2:
            a, _ = self.mcts.get_random_action(rep=rep)
        else:
            a = self.mcts.get_best_action(rep=rep)
        return a

    def simulate(self, state: torch.Tensor, rep: str):
        self.mcts.simulate_moves(nnet=self.nnet, num_mcts_sims=self.num_mcts_sims, state=state, rep=rep)

    def get_eval(self, rep: str) -> torch.Tensor:
        return self.mcts.Q[rep]

    def get_best_eval(self, rep: str) -> torch.Tensor:
        return self.mcts.Q[rep][self.mcts.get_best_action(rep=rep)]

    def get_sims(self, rep: str) -> torch.Tensor:
        return self.mcts.N[rep]


class RMCTSStrategy(Strategy):
    def __init__(self, c: float, num_mcts_sims: int, random_move: int) -> None:
        self.c = c
        self.num_mcts_sims = num_mcts_sims
        self.random_move = random_move

        self.game_cache = None
        self.mcts = None

    def setup(self, game_cache: connect_four.GameCache) -> None:
        self.game_cache = game_cache
        self.mcts = mcts.RandomMCTS(c=self.c, game_cache=game_cache)

    def get_move(self, state: torch.Tensor, rep: str, move: int) -> int:
        self.mcts.simulate_games(num_mcts_sims=self.num_mcts_sims, state=state, rep=rep)
        if move <= self.random_move * 2:
            a, _ = self.mcts.get_random_action(rep=rep)
        else:
            a = self.mcts.get_best_action(rep=rep)
        return a


class RandomStrategy(Strategy):
    def __init__(self):
        self.game_cache = None

    def setup(self, game_cache: connect_four.GameCache) -> None:
        self.game_cache = game_cache

    def get_move(self, state: torch.Tensor, rep: str, move: int) -> int:
        a = int(random.choice(connect_four.possible_moves(state=state).tolist()))
        return a


def competition_game(player1: Strategy, player2: Strategy, log_games: bool, offset: Optional[int] = None):
    # Setup
    game_cache = connect_four.GameCache()
    player1.setup(game_cache=game_cache)
    player2.setup(game_cache=game_cache)

    # Logging
    game_info = [] if log_games else None

    # Game state
    move = 1
    offset = random.randint(0, 1) if offset is None else offset
    state = connect_four.start_state()
    rep = connect_four.to_rep(state=state)
    while True:
        if (move + offset) % 2 == 1:
            a = player1.get_move(state=state, rep=rep, move=move)
        else:
            a = player2.get_move(state=state, rep=rep, move=move)

        next_state, next_rep = game_cache.next_state_rep(state=state, rep=rep, action=a)
        ge = game_cache.game_ended(state=next_state, rep=next_rep)

        if ge is not None:
            if log_games:
                game_info.append(utils.MoveInfo(state=state, rep=rep, move=move, action=a, ge=-ge))
            winner = -ge
            break
        elif log_games:
            game_info.append(utils.MoveInfo(state=state, rep=rep, move=move, action=a))

        move += 1
        state, rep = next_state, next_rep

    first_move = "player1" if offset % 2 == 0 else "player2"
    winner = "Draw" if winner == 0 else ("player1" if (move + offset) % 2 == 1 else "player2")
    if log_games:
        return winner, move, first_move, state
    else:
        return winner, move, first_move, state, game_info


def move8_evaluation(nnet: torch.nn.Module):
    data_folder = utils.get_data_folder()
    boards = np.load(os.path.join(data_folder, "move8_boards.npy"))
    winners = np.load(os.path.join(data_folder, "move8_winner.npy"))
    boards = torch.Tensor(boards)
    winners = torch.Tensor(winners)
    ones = torch.ones_like(winners)

