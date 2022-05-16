import os
import pickle
import time
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.multiprocessing

import connect_four
import connect_net
import utils


class MCTS:
    def __init__(self, c: float, game_cache: connect_four.GameCache, network_cache: connect_net.NetworkCache) -> None:
        self.c = c
        self.game_cache = game_cache
        self.network_cache = network_cache

        self.visited = set()
        self.Q = {}
        self.N = {}

    def simulate_moves(self, num_mcts_sims: int, state: torch.Tensor, rep: str, nnet: torch.nn.Module):
        for i in range(num_mcts_sims):
            self.search(state=state, rep=rep, nnet=nnet)

    def search(self, state: torch.Tensor, rep: str, nnet: torch.nn.Module) -> Union[torch.Tensor, int]:
        ge = self.game_cache.game_ended(state=state, rep=rep)
        if ge is not None:
            return -ge

        if rep not in self.visited:
            v, prob_a = self.network_cache.network(state=state, rep=rep)
            self.visited.add(rep)
            self.N[rep] = torch.zeros(7)
            self.Q[rep] = torch.zeros(7)
            return -v

        else:
            n = self.N[rep]
            visits = 1 + n
            u_explore = (n.sum() + 1).sqrt() / visits
            u = 1 + self.Q[rep] / visits + self.c * self.network_cache.P[rep] * u_explore
            u = u * (1 - state[0, :].abs())
            a = int(torch.argmax(u))

            next_state, next_rep = self.game_cache.next_state_rep(state=state, rep=rep, action=a)
            v = self.search(state=next_state, rep=next_rep, nnet=nnet)

            # Total values / visits
            self.Q[rep][a] = self.Q[rep][a] + v
            self.N[rep][a] += 1
            return -v

    def get_random_action(self, rep: str) -> Tuple[int, torch.Tensor]:
        mcts_n = self.N[rep]
        pi_val = mcts_n / mcts_n.sum()
        a = int(torch.multinomial(pi_val, 1))
        return a, pi_val

    def get_best_action(self, rep: str) -> int:
        a = int(torch.argmax(self.N[rep]))
        return a


def find_policy_examples(
        nnet: torch.nn.Module, folder: str, iteration: int, process: int,
        num_episodes: int, num_mcts_sims: int, batch_size: int = 16) -> None:
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
            examples.append(execute_episode(nnet=nnet, num_mcts_sims=num_mcts_sims, net_cache=nc, game_cache=gc))
            total_time += time.time() - start_time

            # End of batch
            if e % batch_size == (batch_size - 1):
                batch = e // 16
                filename = name_policy_batch(
                    folder=folder, iteration=iteration, process=process,
                    num_episodes=num_episodes, num_mcts_sims=num_mcts_sims, batch=batch,
                )
                with open(filename, "wb") as f:
                    pickle.dump(examples, f)
                examples = []

        print(f"PROC::{process}::{num_episodes}::{num_mcts_sims}::{total_time / num_episodes}")


def name_policy_batch(
        folder: str, iteration: int, process: int,
        num_episodes: int, num_mcts_sims: int, batch: int) -> str:
    return os.path.join(
        folder,
        f"examples-{iteration}-{process}-{num_episodes}-{num_mcts_sims}-{batch}.pickle",
    )


def execute_episode(
        nnet: torch.nn.Module, num_mcts_sims: int,
        net_cache: connect_net.NetworkCache, game_cache: connect_four.GameCache) -> List[utils.MoveInfo]:
    mcts_data = MCTS(c=1, game_cache=game_cache, network_cache=net_cache)
    state = connect_four.start_state()
    rep = connect_four.to_rep(state=state)

    game_info = []
    move = 1
    while True:
        mcts_data.simulate_moves(num_mcts_sims=num_mcts_sims, state=state, rep=rep, nnet=nnet)
        a, pi_val = mcts_data.get_random_action(rep=rep)

        move_info = utils.MoveInfo(state=state, rep=rep, move=move, action=a, pi_val=pi_val)
        game_info.append(move_info)

        state, rep = game_cache.next_state_rep(state=state, rep=rep, action=a)
        ge = game_cache.game_ended(state=state, rep=rep)
        if ge is not None:
            return match_rewards(game_info=game_info, reward=-ge)

        move += 1


def match_rewards(game_info: List[utils.MoveInfo], reward: int) -> List[utils.MoveInfo]:
    assigned_examples = []
    for move_info in reversed(game_info):
        move_info.set_game_ended(ge=reward)
        reward = -reward
        assigned_examples.append(move_info)
    return assigned_examples
