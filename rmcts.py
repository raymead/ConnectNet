import random
from typing import Union, Tuple

import torch

import connect_four


class RandomMCTS:
    def __init__(self, c: float, game_cache: connect_four.GameCache) -> None:
        self.c = c
        self.game_cache = game_cache

        self.visited = set()
        self.Q = {}
        self.N = {}

    def search(self, state: torch.Tensor, rep: str) -> Union[torch.Tensor, int]:
        ge = self.game_cache.game_ended(state=state, rep=rep)
        if ge is not None:
            return -ge

        if rep not in self.visited:
            self.visited.add(rep)
            self.N[rep] = torch.zeros(7)
            self.Q[rep] = torch.zeros(7)

            a = random.choice(connect_four.possible_moves(state=state).tolist())
        else:
            n = self.N[rep]
            visits = 1 + n
            u_explore = (n.sum() + 1).sqrt() / visits
            u = 1 + self.Q[rep] / visits + self.c * u_explore
            u = u * (1 - state[0, :].abs())
            a = int(torch.argmax(u))

        next_state, next_rep = self.game_cache.next_state_rep(state=state, rep=rep, action=a)
        v = self.search(state=next_state, rep=next_rep)

        self.Q[rep][a] = self.Q[rep][a] + v
        self.N[rep][a] += 1
        return -v

    def simulate_games(self, num_mcts_sims: int, state: torch.Tensor, rep: str):
        for i in range(num_mcts_sims):
            self.search(state=state, rep=rep)

    def get_random_action(self, rep: str) -> Tuple[int, torch.Tensor]:
        mcts_n = self.N[rep]
        pi_val = mcts_n / mcts_n.sum()
        a = int(torch.multinomial(pi_val, 1))
        return a, pi_val

    def get_best_action(self, rep: str) -> int:
        return int(torch.argmax(self.N[rep]))