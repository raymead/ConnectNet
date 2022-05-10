import pickle
import time
from typing import Optional, List, Tuple, Union

import numpy as np
import torch
import torch.multiprocessing

import connect_four

T_EXAMPLE = List[List[Union[torch.Tensor, torch.Tensor, float]]]


class CalculationCache:
    def __init__(self, nnet: torch.nn.Module) -> None:
        self.nnet = nnet
        self.V = {}
        self.P = {}

        self.GE = {}

        self.NS = {}
        self.NR = {}

    def network(self, state: torch.Tensor, rep: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if rep not in self.V:
            v, prob_a = self.nnet(state.view(1, 1, 6, 7))
            self.V[rep] = v
            self.P[rep] = prob_a[0]
        return self.V[rep], self.P[rep]

    def game_ended(self, state: torch.Tensor, rep: str):
        if rep not in self.GE:
            self.GE[rep] = connect_four.game_ended(state)
        return self.GE[rep]

    def next_state_rep(self, state: torch.Tensor, rep: str, action: int):
        if (rep, action) not in self.NS:
            next_state = -1 * connect_four.next_state(state, action)
            next_rep = connect_four.to_rep(next_state)
            self.NS[rep, action] = next_state
            self.NR[rep, action] = next_rep
        return self.NS[rep, action], self.NR[rep, action]


class MCTS:
    def __init__(self, c: float, nc: CalculationCache) -> None:
        self.c = c
        self.nc = nc

        self.visited = set()
        self.Q = {}
        self.N = {}

    def search(self, state: torch.Tensor, rep: str, nnet: torch.nn.Module) -> Union[torch.Tensor, int]:
        ge = self.nc.game_ended(state=state, rep=rep)
        if ge is not None:
            return -ge

        if rep not in self.visited:
            v, prob_a = self.nc.network(state=state, rep=rep)
            self.visited.add(rep)
            self.N[rep] = torch.zeros(7)
            self.Q[rep] = torch.zeros(7)
            return -v

        else:
            n = self.N[rep]
            visits = 1 + n
            u_explore = (n.sum() + 1).sqrt() / visits
            u = 1 + self.Q[rep] / visits + self.c * self.nc.P[rep] * u_explore
            u = u * (1 - state[0, :].abs())
            a = int(torch.argmax(u))

            next_state, next_rep = self.nc.next_state_rep(state=state, rep=rep, action=a)
            v = self.search(state=next_state, rep=next_rep, nnet=nnet)

            # Total values / visits
            self.Q[rep][a] = self.Q[rep][a] + v
            self.N[rep][a] += 1
            return -v

    def pi(self, rep):
        return self.N[rep] / (self.N[rep].sum() + 1e-3)


def policy_iteration(nnet: torch.nn.Module, num_iterations: int, num_episodes: int, num_mcts_sims: int) -> T_EXAMPLE:
    examples = []
    for i in range(num_iterations):
        with torch.no_grad():
            nc = CalculationCache(nnet=nnet)
            print(f"\nIteration: {i}::", end="")
            for e in range(num_episodes):
                print(e, end=",")
                examples += execute_episode(nnet=nnet, num_mcts_sims=num_mcts_sims, nc=nc)
    return examples


def find_policy_examples(
        nnet: torch.nn.Module, folder: str, iteration: int, process: int,
        num_episodes: int, num_mcts_sims: int, batch_size: int = 16) -> None:
    print(f"PROC::LAUNCH::{process}")
    examples = []
    total_time = 0
    nc = CalculationCache(nnet=nnet)
    with torch.no_grad():
        for e in range(num_episodes):
            # Beginning of batch
            if e % batch_size == 0:
                print(f"PROC::Checkpoint::{process}::{e}::{num_episodes}")
                nc = CalculationCache(nnet=nnet)

            start_time = time.time()
            examples += execute_episode(nnet=nnet, num_mcts_sims=num_mcts_sims, nc=nc)
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
    return f"{folder}/examples-{iteration}-{process}-{num_episodes}-{num_mcts_sims}-{batch}.pickle"


def execute_episode(nnet: torch.nn.Module, num_mcts_sims: int, nc: CalculationCache) -> T_EXAMPLE:
    examples = []
    mcts = MCTS(c=1, nc=nc)
    state = connect_four.start_state()
    rep = connect_four.to_rep(state=state)
    while True:
        for i in range(num_mcts_sims):
            mcts.search(state=state, rep=rep, nnet=nnet)

        # Calculate action
        mcts_n = mcts.N[rep]
        pi_val = mcts_n / (mcts_n.sum() + 1e-3)
        a = int(np.argmax(np.random.multinomial(1, pvals=pi_val)))
        examples.append([state, pi_val, a, None])

        state, rep = nc.next_state_rep(state=state, rep=rep, action=a)
        ge = nc.game_ended(state=state, rep=rep)
        if ge is not None:
            return match_rewards(examples, -ge)


def match_rewards(examples: List[List[Union[torch.Tensor, torch.Tensor, Optional[float]]]], reward: float) -> T_EXAMPLE:
    assigned_examples = []
    for example in reversed(examples):
        example[3] = reward
        reward = -reward
        assigned_examples.append(example)
    return assigned_examples
