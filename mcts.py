import abc
import os
import random
import pickle
import time
from typing import List, Tuple, Union

import torch
import torch.multiprocessing

import connect_four
import connect_net
import utils


class MCTS(abc.ABC):
    def __init__(self, game_cache: connect_four.GameCache) -> None:
        self.game_cache = game_cache

        self.visited = set()
        self.Q = {}
        self.N = {}

    def simulate_moves(self, num_mcts_sims: int, state: torch.Tensor, rep: str):
        for i in range(num_mcts_sims):
            self.search(state=state, rep=rep)

    @abc.abstractmethod
    def search(self, state: torch.Tensor, rep: str) -> Union[torch.Tensor, int]:
        pass

    def search_action(self, state: torch.Tensor, rep: str, action: int) -> Union[torch.Tensor, int]:
        next_state, next_rep = self.game_cache.next_state_rep(state=state, rep=rep, action=action)
        v = self.search(state=next_state, rep=next_rep)

        # Total values / visits
        self.Q[rep][action] = (self.N[rep][action] * self.Q[rep][action] + v) / (self.N[rep][action] + 1)
        self.N[rep][action] += 1
        return -v

    def init_node(self, rep: str) -> None:
        self.visited.add(rep)
        self.N[rep] = torch.zeros(7)
        self.Q[rep] = torch.zeros(7)

    def get_random_action(self, rep: str) -> Tuple[int, torch.Tensor]:
        mcts_n = self.N[rep]
        pi_val = mcts_n / mcts_n.sum()
        a = int(torch.multinomial(pi_val, 1))
        return a, pi_val

    def get_best_action(self, rep: str) -> int:
        return int(torch.argmax(self.N[rep]))


class RandomMCTS(MCTS):
    def __init__(self, c: float, game_cache: connect_four.GameCache) -> None:
        super().__init__(game_cache=game_cache)
        self.c = c

    def search(self, state: torch.Tensor, rep: str) -> Union[torch.Tensor, int]:
        ge = self.game_cache.game_ended(state=state, rep=rep)
        if ge is not None:
            return -ge

        if rep not in self.visited:
            self.init_node(rep=rep)
            a = random.choice(connect_four.possible_moves(state=state).tolist())
        else:
            n = self.N[rep]
            visits = 1 + n
            u_explore = (n.sum() + 1).sqrt() / visits
            u = 1 + self.Q[rep] + self.c * u_explore
            u = u * (1 - state[0, :].abs())
            a = int(torch.argmax(u))

        return self.search_action(state=state, rep=rep, action=a)


class NetworkMCTS(MCTS):
    def __init__(self, c: float, game_cache: connect_four.GameCache, network_cache: connect_net.NetworkCache) -> None:
        super().__init__(game_cache=game_cache)
        self.c = c
        self.network_cache = network_cache

    def search(self, state: torch.Tensor, rep: str) -> Union[torch.Tensor, int]:
        ge = self.game_cache.game_ended(state=state, rep=rep)
        if ge is not None:
            return -ge

        if rep not in self.visited:
            v, prob_a = self.network_cache.network(state=state, rep=rep)
            self.init_node(rep=rep)
            return -v

        else:
            n = self.N[rep]
            u_explore = (n.sum() + 1).sqrt() / (n + 1)
            u = 1 + self.Q[rep] + self.c * self.network_cache.P[rep] * u_explore
            u = u * (1 - state[0, :].abs())
            a = int(torch.argmax(u))

            return self.search_action(state=state, rep=rep, action=a)


def generate_examples(
        nnet: torch.nn.Module, folder: str, iteration: int, process: int, c: float,
        num_episodes: int, num_mcts_sims: int, batch_size: int) -> None:
    print(f"PROC::LAUNCH::{process}")
    examples = []
    total_time = 0
    nc = connect_net.NetworkCache(nnet=nnet)
    gc = connect_four.GameCache()
    with torch.no_grad():
        for e in range(num_episodes):
            # Beginning of batch
            if e % batch_size == 0:
                print(f"PROC::Checkpoint::{process}::{e}::{num_episodes}")
                nc = connect_net.NetworkCache(nnet=nnet)
                gc = connect_four.GameCache()

            start_time = time.time()
            examples.append(execute_episode(c=c, num_mcts_sims=num_mcts_sims, net_cache=nc, game_cache=gc))
            total_time += time.time() - start_time

            # End of batch
            if e % batch_size == (batch_size - 1):
                batch = e // 16
                filename = batch_path(
                    folder=folder, iteration=iteration, process=process,
                    num_episodes=num_episodes, num_mcts_sims=num_mcts_sims, batch=batch,
                )
                with open(filename, "wb") as f:
                    pickle.dump(examples, f)
                examples = []

        print(f"PROC::{process}::{num_episodes}::{num_mcts_sims}::{total_time / num_episodes}")


def execute_episode(
        num_mcts_sims: int, c: float,
        net_cache: connect_net.NetworkCache, game_cache: connect_four.GameCache) -> List[utils.MoveInfo]:
    mcts_data = NetworkMCTS(c=c, game_cache=game_cache, network_cache=net_cache)

    move = 1
    game_info = []
    state = connect_four.start_state()
    rep = connect_four.to_rep(state=state)
    while True:
        mcts_data.simulate_moves(num_mcts_sims=num_mcts_sims, state=state, rep=rep)
        a, pi_val = mcts_data.get_random_action(rep=rep)

        move_info = utils.MoveInfo(state=state, rep=rep, move=move, action=a, pi_val=pi_val)
        game_info.append(move_info)

        state, rep = game_cache.next_state_rep(state=state, rep=rep, action=a)
        ge = game_cache.game_ended(state=state, rep=rep)
        if ge is not None:
            return match_rewards(game_info=game_info, reward=-ge)

        move += 1


def batch_path(
        folder: str, iteration: int, process: int,
        num_episodes: int, num_mcts_sims: int, batch: int) -> str:
    return os.path.join(
        folder,
        f"examples-{iteration}-{process}-{num_episodes}-{num_mcts_sims}-{batch}.pickle",
    )


def match_rewards(game_info: List[utils.MoveInfo], reward: int) -> List[utils.MoveInfo]:
    assigned_examples = []
    for move_info in reversed(game_info):
        move_info.set_game_ended(ge=reward)
        reward = -reward
        assigned_examples.append(move_info)
    return assigned_examples
