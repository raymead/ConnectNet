from typing import Optional, List, Tuple, Union

import numpy as np
import torch
import torch.multiprocessing

import connect_four

T_EXAMPLE = List[List[Union[torch.Tensor, torch.Tensor, float]]]


class NetworkCache:
    def __init__(self, nnet):
        self.nnet = nnet
        self.V = {}
        self.P = {}

    def run_network(self, state, rep):
        if rep not in self.V:
            v, prob_a = self.nnet(state)
            self.V[rep] = v
            self.P[rep] = prob_a
        return self.V[rep], self.P[rep]


class MCTS:
    def __init__(self, c: float, nc: NetworkCache):
        self.c = c
        self.nc = nc

        self.visited = set()
        self.Q = {}
        self.N = {}

    def search(self, state: torch.Tensor, rep: str, nnet: torch.nn.Module):
        ge = connect_four.game_ended(state)
        if ge is not None:
            return -ge

        if rep not in self.visited:
            v, prob_a = self.nc.run_network(state=state, rep=rep)
            self.visited.add(rep)
            self.N[rep] = torch.zeros(7)
            self.Q[rep] = torch.zeros(7)
            return -v

        else:
            max_u, best_a = -torch.inf, -1
            for a in connect_four.get_valid_actions(state):
                u_explore = self.N[rep].sum().sqrt() / (1 + self.N[rep][a])
                u = self.Q[rep][a] + self.c * self.nc.P[rep][0, a] * u_explore
                if u > max_u:
                    max_u = u
                    best_a = a
            a = best_a

            next_state = connect_four.next_state(state, a)
            next_rep = connect_four.to_rep(next_state)
            v = self.search(state=next_state * -1, rep=next_rep, nnet=nnet)

            self.Q[rep][a] = (self.N[rep][a] + v) / (self.N[rep][a] + 1)
            self.N[rep][a] += 1
            return -v

    def pi(self, rep):
        return self.N[rep] / (self.N[rep].sum() + 1e-3)


def policy_iteration(
        nnet: torch.nn.Module, num_iterations: int, num_episodes: int, num_mcts_sims: int,
        optimizer: torch.optim.Optimizer) -> None:
    examples = []
    for i in range(num_iterations):
        with torch.no_grad():
            nc = NetworkCache(nnet=nnet)
            print(f"\nIteration: {i}::", end="")
            for e in range(num_episodes):
                print(e, end=",")
                examples += execute_episode(nnet=nnet, num_mcts_sims=num_mcts_sims, nc=nc)

        train_nnet(nnet=nnet, examples=examples, optimizer=optimizer)


def train_nnet(nnet: torch.nn.Module, examples: T_EXAMPLE, optimizer: torch.optim.Optimizer) -> None:
    loss = torch.zeros(1)
    for e in examples:
        v, proba = nnet(e[0])
        loss += torch.square(v - e[3]) - torch.dot(e[1], torch.log(proba.squeeze()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def execute_episode(nnet: torch.nn.Module, num_mcts_sims: int, nc: NetworkCache) -> T_EXAMPLE:
    examples = []
    mcts = MCTS(c=1, nc=nc)
    state = connect_four.start_state()
    while True:
        rep = connect_four.to_rep(state=state)
        for i in range(num_mcts_sims):
            mcts.search(state=state, rep=rep, nnet=nnet)

        # Calculate action
        mcts_n = mcts.N[rep]
        pi_val = mcts_n / (mcts_n.sum() + 1e-3)
        a = int(np.argmax(np.random.multinomial(1, pvals=pi_val)))
        examples.append([state, pi_val, a, None])

        state = connect_four.next_state(state, a)
        ge = connect_four.game_ended(state)

        if ge is not None:
            return match_rewards(examples, ge)
        state = -state


def match_rewards(examples: List[List[Union[torch.Tensor, torch.Tensor, Optional[float]]]], reward: float) -> T_EXAMPLE:
    assigned_examples = []
    for example in reversed(examples):
        example[3] = reward
        reward = -reward
        assigned_examples.append(example)
    return assigned_examples
