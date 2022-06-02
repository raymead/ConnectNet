import abc
import os.path
import random
import time
from typing import Optional, Tuple, Dict, Any, Type

import pandas
import torch
import torch.multiprocessing as mp

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

    def setup(self, game_cache: connect_four.GameCache) -> None:
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
            a, _ = self.mcts.get_best_action(rep=rep)
        return a

    def simulate(self, state: torch.Tensor, rep: str) -> None:
        self.mcts.simulate_moves(num_mcts_sims=self.num_mcts_sims, state=state, rep=rep)


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
            a, _ = self.mcts.get_best_action(rep=rep)
        return a


class RandomStrategy(Strategy):
    def __init__(self) -> None:
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


def parallel_competition(
        scores: Dict[Any, float], strategies: Dict[Any, Strategy],
        num_processes: int, num_games: int, k: float) -> Tuple[Dict[Any, float], pandas.DataFrame]:
    list_df = []
    with mp.Pool(processes=num_processes) as pool:
        for g in range(num_games):
            start_time = time.time()

            sorted_players = sorted(scores.items(), key=lambda x: x[1])
            pairs = [tuple(sorted_players[2 * i:2 * i + 2]) for i in range(len(sorted_players) // 2)]
            arguments = [(strategies[p1], strategies[p2], False) for (p1, rating1), (p2, rating2) in pairs]

            map_results = pool.starmap(competition_game, arguments, chunksize=1)
            for ((p1, rating1), (p2, rating2)), result in zip(pairs, map_results):
                if result[0] == "player1":
                    s1, s2 = 1, 0
                elif result[0] == "player2":
                    s1, s2 = 0, 1
                else:
                    s1, s2 = 0.5, 0.5
                scores[p1], scores[p2] = next_rating(rating1=rating1, rating2=rating2, score1=s1, score2=s2, k=k)

            if g % 10 == 0:
                print(f"ITER::{g}  TIME::{time.time() - start_time:.3f}")

            df_tmp = pandas.DataFrame(list(scores.items()), columns=["Player", "ELO"])
            df_tmp["iteration"] = g
            list_df.append(df_tmp)
    df_all = pandas.concat(list_df)
    return scores, df_all


def evaluate_batch(
        klass: Type[torch.nn.Module], trial: str, c: float, num_mcts_sims: int, random_moves: int,
        num_games: int, num_processes: int, max_iteration: Optional[int], k: float) -> None:
    if max_iteration is None:
        max_iteration = max([int(f.split("training")[1]) for f in utils.get_training_folders(trial=trial)])
    strategies = {}
    for iteration in range(1, max_iteration + 1):
        training_folder = utils.get_training_folder(trial=trial, iteration=iteration)
        training_path = utils.get_model_path(folder=training_folder, iteration=iteration)
        nnet = connect_net.load_model(path=training_path, klass=klass, log=False)
        nnet.share_memory()
        strategies[iteration] = NetworkStrategy(nnet=nnet, c=c, num_mcts_sims=num_mcts_sims, random_moves=random_moves)
    scores = {key: 1600 for key in strategies.keys()}
    print(f"NUM_STRAT::{len(strategies)}")

    scores, df_all = parallel_competition(
        scores=scores, strategies=strategies,
        num_processes=num_processes, num_games=num_games, k=k,
    )

    df_scores = pandas.DataFrame(list(scores.items()), columns=["Player", "ELO"])

    results_folder = utils.get_results_folder(trial=trial)
    df_all.to_csv(os.path.join(results_folder, "all_scores.csv"), index=False)
    df_scores.to_csv(os.path.join(results_folder, "fin_scores.csv"), index=False)


# Helpers
def next_rating(rating1: float, rating2: float, score1: float, score2: float, k: float) -> Tuple[float, float]:
    r1 = 10 ** (rating1 / 400)
    r2 = 10 ** (rating2 / 400)
    r_total = r1 + r2
    e1 = r1 / r_total
    e2 = r2 / r_total
    new_rating1 = rating1 + k * (score1 - e1)
    new_rating2 = rating2 + k * (score2 - e2)
    return new_rating1, new_rating2


if __name__ == '__main__':
    evaluate_batch(
        klass=connect_net.ConnectNet2,
        trial="conv501", num_processes=6, max_iteration=84,
        num_games=1000, k=16,
        c=1, num_mcts_sims=32, random_moves=8,
    )
